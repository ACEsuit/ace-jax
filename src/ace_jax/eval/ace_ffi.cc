// Lean linear ACE energies and forces on the CPU, one XLA FFI call on lammps-jax's neighbour matrix.
// Per row, pairs within the cutoff only: radial table, harmonics, A, product-basis DAG and its adjoint.
// Two SIMD axes, written with GCC/clang vector types (any target: the compiler splits what the CPU lacks):
// four neighbours of a site for the harmonics, A and the force contraction, and kLanes sites of one centre
// species for the product basis.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr double kPi = 3.14159265358979323846;

typedef double v4 __attribute__((vector_size(32)));     // four neighbours of one site
typedef double v4u __attribute__((vector_size(32), aligned(8), may_alias));   // four doubles anywhere
typedef long long v4i __attribute__((vector_size(32)));
#if defined(__clang__)
#define ACE_SHUF(a, b, i, j, k, l) __builtin_shufflevector(a, b, i, j, k, l)
#else
#define ACE_SHUF(a, b, i, j, k, l) __builtin_shuffle(a, b, (v4i){i, j, k, l})
#endif
// a[j][c] -> o[c][j]: four neighbours' values of four columns, to four columns of the four neighbours.
inline void transpose(const v4 a[4], v4 *o) {
  const v4 t0 = ACE_SHUF(a[0], a[1], 0, 4, 2, 6), t1 = ACE_SHUF(a[0], a[1], 1, 5, 3, 7);
  const v4 t2 = ACE_SHUF(a[2], a[3], 0, 4, 2, 6), t3 = ACE_SHUF(a[2], a[3], 1, 5, 3, 7);
  o[0] = ACE_SHUF(t0, t2, 0, 1, 4, 5);
  o[1] = ACE_SHUF(t1, t3, 0, 1, 4, 5);
  o[2] = ACE_SHUF(t0, t2, 2, 3, 6, 7);
  o[3] = ACE_SHUF(t1, t3, 2, 3, 6, 7);
}

// ace_jax.eval.harmonics, Y[l * l + l + m], and their gradients from the harmonics one degree down: for
// C_lm = r^l P_lm e^(i m phi), d/dz C_lm = (l + m) C_l-1,m, (d/dx + i d/dy) C_lm = -C_l-1,m+1 and
// (d/dx - i d/dy) C_lm = (l + m)(l + m - 1) C_l-1,m-1, so grad Y_i is at most five terms (`grad`: per i and
// component two (index, coefficient) pairs, one for z), exact for the solid harmonics; the spherical ones are
// r^-l times them: grad Y = (those terms - l rhat Y) / r.  No derivative is stored per neighbour.
struct Harmonics {
  int L;
  bool solid;
  std::vector<double> norm, ca, cb, gc;                   // gc[5 * i + t]: x, x, y, y, z terms of grad Y_i
  std::vector<int32_t> gi;

  Harmonics(int L_, bool solid_)
      : L(L_), solid(solid_), norm((L_ + 1) * (L_ + 1)), ca((L_ + 1) * (L_ + 1)),
        cb((L_ + 1) * (L_ + 1)), gc(5 * (L_ + 1) * (L_ + 1)), gi(5 * (L_ + 1) * (L_ + 1)) {
    for (int l = 0; l <= L; ++l)
      for (int m = -l; m <= l; ++m) {
        const int am = std::abs(m);
        const double c = std::exp(0.5 * (std::log(2.0 * l + 1) - std::log(4.0 * kPi) +
                                         std::lgamma(l - am + 1.0) - std::lgamma(l + am + 1.0)));
        norm[l * l + l + m] = m ? c * std::sqrt(2.0) : c;
        if (m >= 0 && l >= m + 2) {                       // the recurrence ratios in l
          ca[l * (L + 1) + m] = (2.0 * l - 1) / (l - m);
          cb[l * (L + 1) + m] = (l + m - 1.0) / (l - m);
        }
      }
    for (int l = 1; l <= L; ++l)
      for (int m = -l; m <= l; ++m) {
        const int am = std::abs(m), sg = m < 0 ? -1 : 1, i = l * l + l + m, b = (l - 1) * l;
        const double c = norm[l * l + l + am];
        const double gz = am < l ? (l + am) * c / norm[b + am] : 0.0;
        const double gp = am + 1 < l ? (am ? -0.5 : -1.0) * c / norm[b + am + 1] : 0.0;   // m = 0: both conjugates
        const double gm = am >= 1 ? 0.5 * (l + am) * (l + am - 1) * c / norm[b + am - 1] : 0.0;
        const bool up = am + 1 < l, dn = am > 0;
        const bool sm = dn && (sg > 0 || am > 1), om = dn && (sg < 0 || am > 1);          // no sine at m = 0
        auto put = [&](int t, bool on, int idx, double coef) {
          gi[5 * i + t] = on ? idx : 0;
          gc[5 * i + t] = on ? coef : 0.0;
        };
        put(0, up, b + sg * (am + 1), gp);
        put(1, sm, b + sg * (am - 1), gm);
        put(2, up, b - sg * (am + 1), sg * gp);
        put(3, om, b - sg * (am - 1), -sg * gm);
        put(4, am < l, b + sg * am, gz);
      }
  }

  // The harmonics of (x, y, z), r = |v| and ir = 1 / r given: T = double, or v4 for four vectors at once.
  // Per m, the recurrence in l in registers.
  template <class T> void eval(const T x, const T y, const T z, const T r, const T ir, T *Y) const {
    const T zero = T{}, one = zero + 1.0, u = z * ir;
    const int W = L + 1;
    T Am = one, Bm = zero, rm = one;                      // A_m + i B_m = (x + i y)^m; spherical: r^-m
    double dfact = 1.0;
    for (int m = 0; m <= L; ++m) {
      if (m > 0) {
        const T a1 = Am;
        Am = x * a1 - y * Bm;
        Bm = x * Bm + y * a1;
        dfact *= 2 * m - 1;
        if (!solid) rm = rm * ir;
      }
      T fa = rm * Am, fb = rm * Bm, P1 = zero + dfact, P2 = zero;   // solid: times r^(l - m)
      for (int l = m; l <= L; ++l) {
        const T P = l == m ? P1 : l == m + 1 ? ((2 * m + 1) * dfact) * u
                                             : ca[l * W + m] * u * P1 - cb[l * W + m] * P2;
        Y[l * l + l + m] = (norm[l * l + l + m] * P) * fa;
        if (m) Y[l * l + l - m] = (norm[l * l + l - m] * P) * fb;
        if (l > m) P2 = P1;
        P1 = P;
        if (solid) {
          fa = fa * r;
          fb = fb * r;
        }
      }
    }
  }
};

// A site's A (or its dE/dY) over one run of `len` consecutive harmonics, unrolled for the lengths of whole
// l blocks; `ffi.plan` lists the runs by length, so the switch is predicted.
template <int N> inline void axpy(v4 *a, const v4 R, const v4 *Y) {
  for (int m = 0; m < N; ++m) a[m] += R * Y[m];
}
inline void run_a(v4 *a, const v4 R, const v4 *Y, int32_t len) {
  switch (len) {
    case 1: return axpy<1>(a, R, Y);
    case 3: return axpy<3>(a, R, Y);
    case 5: return axpy<5>(a, R, Y);
    case 7: return axpy<7>(a, R, Y);
    case 9: return axpy<9>(a, R, Y);
    case 11: return axpy<11>(a, R, Y);
    case 13: return axpy<13>(a, R, Y);
    default: for (int32_t m = 0; m < len; ++m) a[m] += R * Y[m];
  }
}
template <int N> inline void axpy2(v4 *q, v4 *dq, const double *w, const v4 R, const v4 dR) {
  for (int m = 0; m < N; ++m) {
    q[m] += w[m] * R;
    dq[m] += w[m] * dR;
  }
}
inline void run_q(v4 *q, v4 *dq, const double *w, const v4 R, const v4 dR, int32_t len) {
  switch (len) {
    case 1: return axpy2<1>(q, dq, w, R, dR);
    case 3: return axpy2<3>(q, dq, w, R, dR);
    case 5: return axpy2<5>(q, dq, w, R, dR);
    case 7: return axpy2<7>(q, dq, w, R, dR);
    case 9: return axpy2<9>(q, dq, w, R, dR);
    case 11: return axpy2<11>(q, dq, w, R, dR);
    case 13: return axpy2<13>(q, dq, w, R, dR);
    default:
      for (int32_t m = 0; m < len; ++m) {
        q[m] += w[m] * R;
        dq[m] += w[m] * dR;
      }
  }
}

// radial.spline_eval and d/dr on one species pair's table `coef`, zero from its cutoff on: the first of the four
// rows r reads, and their weights w and dw (no division: ih = 1 / h).
inline const double *table_rows(const double *coef, int64_t F, double x0, double ih, int64_t n, double r, double rc,
                                double *w, double *dw) {
  if (!(r < rc)) {
    for (int i = 0; i < 4; ++i) w[i] = dw[i] = r >= rc ? 0.0 : r;   // a NaN stays one
    return coef;
  }
  const double k = (r - x0) * ih + 1.0, sixth = 1.0 / 6.0;
  int64_t ix = static_cast<int64_t>(std::floor(k));
  ix = ix < 1 ? 1 : (ix > n - 1 ? n - 1 : ix);
  const double t = k - ix, s = 1.0 - t;
  w[0] = s * s * s * sixth; w[1] = (4.0 - 6.0 * t * t + 3.0 * t * t * t) * sixth;
  w[2] = (4.0 - 6.0 * s * s + 3.0 * s * s * s) * sixth; w[3] = t * t * t * sixth;
  dw[0] = -0.5 * s * s * ih; dw[1] = (1.5 * t * t - 2.0 * t) * ih; dw[2] = (2.0 * s - 1.5 * s * s) * ih;
  dw[3] = 0.5 * t * t * ih;
  return coef + (ix - 1) * F;
}

// Sites in batches of kLanes of one centre species, a site per SIMD lane: A, its adjoint, the DAG values and
// their adjoints are [entry][lane], so a batch reads each coefficient and index of the model (readout, DAG
// factors and groups: ~1 MB at o4d20) once for kLanes sites.  Per site that was 12 instructions a leaf, and
// on Zen 2 ~10 cycles (the dE/dA update's load waits for the previous store); a batch of 4 halves the
// kernel's time on an i9-14900K and an EPYC 7742 (docs/dev/accuracy-cost-results.md).  8 lanes need AVX-512.
//
// D is the type of the product-basis DAG, its adjoint and the readout: double, or float when the readout
// buffer is float32 (ffi.kernel_inputs(dag="float32")), with twice the sites a register.  The radials, the
// harmonics, A and the force contraction are double in both.  In float the energy is sum_a A_a w_a in double,
// w_a the adjoint of A_a as a DAG node: it is good to ~1e-6 eV/atom, with noise from the float node values
// (0.1 meV in 200 atoms at o4d20), so a line search on energies can fail near a minimum.
#ifndef ACEJAX_FFI_LANES
#define ACEJAX_FFI_LANES 4
#endif
constexpr int kLanes = ACEJAX_FFI_LANES;

// One batch's energies and dE/dv for each site's live neighbour vectors; model buffers and scratch.
template <typename D>
struct Site {
  static constexpr int V = kLanes * static_cast<int>(sizeof(double) / sizeof(D));
  const double *tab, *rc, *E0;
  const D *ct;
  const int32_t *run, *grp, *fac;
  int64_t form, nz, ncoef, F, n_rnl, n_runs, n_groups, n_nodes, N, n_a, nY, n, n_leaf, g_leaf, K;
  double x0, ih;
  Harmonics harm;
  std::vector<double> hat, ir4, A, wl;
  std::vector<D> wA, wB, L, w;
  std::vector<v4> R, dR, Y, at, q, dq;                   // [lane][block of 4 neighbours][feature]
  int64_t Kb;
  std::vector<double> vin, G;                             // per lane: neighbour vectors in, dE/dv out
  std::vector<int32_t> zin, idx;                          // per lane: neighbour species, and atom or slot
  double e[V];
  int64_t nl[V], nzc;

  Site(ffi::Buffer<ffi::F64> rtab, ffi::Buffer<ffi::F64> rcut, ffi::Buffer<ffi::S32> runs,
       ffi::Buffer<ffi::S32> groups, ffi::Buffer<ffi::S32> factor, ffi::AnyBuffer ctz,
       ffi::Buffer<ffi::F64> e0, double x0_, double h_, int64_t n_, int64_t n_rnl_, int64_t lmax,
       int64_t solid, int64_t n_leaf_, int64_t form_, int64_t K_)
      : tab(rtab.typed_data()), rc(rcut.typed_data()), E0(e0.typed_data()), ct(ctz.typed_data<D>()),
        run(runs.typed_data()), grp(groups.typed_data()), fac(factor.typed_data()),
        form(form_), nz(rtab.dimensions()[0]), ncoef(rtab.dimensions()[2]), F(rtab.dimensions()[3]), n_rnl(n_rnl_),
        n_runs(runs.dimensions()[0]), n_groups(groups.dimensions()[0]), n_nodes(factor.element_count()),
        N(ctz.dimensions()[1]), n_a(form_ ? grp[0] : N - n_nodes), nY((lmax + 1) * (lmax + 1)), n(n_), n_leaf(n_leaf_),
        g_leaf(0), K(K_), x0(x0_), ih(1.0 / h_), harm(static_cast<int>(lmax), solid != 0), hat(V * K_ * 3), ir4(V * K_), A(n_a * V), wl(n_a),
        wA(n_a * V), wB(n_a * V), L((form_ ? grp[0] + grp[1] : N - n_leaf) * V), w(L.size()),
        R(V * ((K_ + 3) / 4) * F), dR(R.size()), Y(V * ((K_ + 3) / 4) * nY), at(n_a), q(nY), dq(nY),
        Kb((K_ + 3) / 4), vin(V * K * 3), G(V * K * 3),
        zin(V * K), idx(V * K), nzc(e0.element_count()) {
    while (!form && g_leaf < n_groups && grp[2 * g_leaf + 1] <= n_nodes - n_leaf) ++g_leaf;   // the first leaf group
  }

  // Lane v's radials, harmonics and A column for its nl neighbour vectors V3 (species zj); e[v] = E0 + pair.
  void neighbours(int v, int64_t zc, int64_t nl, const double *V3, const int32_t *zj) {
    double ei = E0[zc];
    v4 *Rv = &R[v * Kb * F], *dRv = &dR[v * Kb * F], *Yv = &Y[v * Kb * nY];
    double *hv = &hat[v * K * 3];
    const int64_t nb = (nl + 3) / 4;
    std::fill(at.begin(), at.end(), v4{});
    v4 *a4 = at.data();
    for (int64_t b = 0; b < nb; ++b) {                    // four neighbours: a padding one has R = 0
      v4 x = {0.6, 0.6, 0.6, 0.6}, y = {0.48, 0.48, 0.48, 0.48}, z = {0.64, 0.64, 0.64, 0.64};
      v4 rr = {1.0, 1.0, 1.0, 1.0}, ir = rr;
      const double *row[4] = {tab, tab, tab, tab};
      double w[4][4] = {}, dw[4][4] = {};
      for (int j = 0; j < 4 && 4 * b + j < nl; ++j) {
        const int64_t s = 4 * b + j;
        const double *u = V3 + 3 * s;
        const double r = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), ri = 1.0 / r;
        const int64_t pz = zc * nz + zj[s];
        row[j] = table_rows(tab + pz * ncoef * F, F, x0, ih, n, r, rc[pz], w[j], dw[j]);
        for (int d = 0; d < 3; ++d) hv[s * 3 + d] = u[d] * ri;
        ir4[v * K + s] = ri;
        x[j] = u[0]; y[j] = u[1]; z[j] = u[2]; rr[j] = r; ir[j] = ri;
      }
      v4 *Rb = Rv + b * F, *dRb = dRv + b * F, *Yb = Yv + b * nY;
      int64_t c = 0;
      for (; c + 4 <= F; c += 4) {                        // four columns of each neighbour, then transposed
        v4 a[4], d[4];
        for (int j = 0; j < 4; ++j) {
          const double *p = row[j] + c;
          const v4 l0 = *(const v4u *)p, l1 = *(const v4u *)(p + F), l2 = *(const v4u *)(p + 2 * F),
                   l3 = *(const v4u *)(p + 3 * F);
          a[j] = (w[j][0] * l0 + w[j][1] * l1) + (w[j][2] * l2 + w[j][3] * l3);
          d[j] = (dw[j][0] * l0 + dw[j][1] * l1) + (dw[j][2] * l2 + dw[j][3] * l3);
        }
        transpose(a, Rb + c);
        transpose(d, dRb + c);
      }
      for (; c < F; ++c)
        for (int j = 0; j < 4; ++j) {
          const double *p = row[j] + c;
          Rb[c][j] = (w[j][0] * p[0] + w[j][1] * p[F]) + (w[j][2] * p[2 * F] + w[j][3] * p[3 * F]);
          dRb[c][j] = (dw[j][0] * p[0] + dw[j][1] * p[F]) + (dw[j][2] * p[2 * F] + dw[j][3] * p[3 * F]);
        }
      v4 ep = {};
      for (c = n_rnl; c < F; ++c) ep += Rb[c];            // the folded pair radial
      ei += (ep[0] + ep[1]) + (ep[2] + ep[3]);
      harm.eval<v4>(x, y, z, rr, ir, Yb);
      for (int64_t u = 0; u < n_runs; ++u) {
        const int32_t *ru = run + 4 * u;
        run_a(a4 + ru[0], Rb[ru[1]], Yb + ru[2], ru[3]);
      }
    }
    e[v] = ei;
    for (int64_t a = 0; a < n_a; ++a) A[a * V + v] = (a4[a][0] + a4[a][1]) + (a4[a][2] + a4[a][3]);
  }

  // The product basis of every lane (centre species zc): energies into e, dE/dA into wA.
  typedef D vd __attribute__((vector_size(sizeof(D) * V), aligned(sizeof(D)), may_alias));
  static inline vd ld(const D *p) { return *(const vd *)p; }
  static inline void st(D *p, vd x) { *(vd *)p = x; }
  void dag(int64_t zc) {
    if (form) return coupled(zc);
    const int64_t n_in = n_nodes - n_leaf, M = n_a + n_in;   // leaves: the last level, never stored
    const int32_t *gp = grp, *fa = fac;
    constexpr bool dbl = sizeof(D) == sizeof(double);
    D *Lv = L.data(), *wv = w.data(), *wa = wA.data(), *wb = wB.data();
    const D *cz = ct + zc * N, *cn = cz + n_a, *Av = Lv;   // A in the DAG's type
    std::copy(A.data(), A.data() + n_a * V, Lv);

    int64_t k = 0;
    for (int64_t g = 0; g < g_leaf; ++g) {                // a group: one parent's children, together
      const vd lp = ld(Lv + gp[2 * g] * V);
      for (const int64_t end = gp[2 * g + 1]; k < end; ++k) st(Lv + (n_a + k) * V, lp * ld(Av + fa[k] * V));
    }
    vd s0 = {}, s1 = {}, s2 = {}, s3 = {};                 // four sums a lane: no add chain over M
    int64_t j = 0;
    for (; dbl && j + 3 < M; j += 4) {                    // (float: the energy is summed at the end)
      const vd c0 = cz[j] + vd{}, c1 = cz[j + 1] + vd{}, c2 = cz[j + 2] + vd{}, c3 = cz[j + 3] + vd{};
      s0 += c0 * ld(Lv + j * V); s1 += c1 * ld(Lv + (j + 1) * V);
      s2 += c2 * ld(Lv + (j + 2) * V); s3 += c3 * ld(Lv + (j + 3) * V);
    }
    for (; dbl && j < M; ++j) {
      const vd c = cz[j] + vd{};
      s0 += c * ld(Lv + j * V);
    }
    std::fill(wa, wa + n_a * V, D(0));
    std::fill(wb, wb + n_a * V, D(0));
    std::fill(wv, wv + M * V, D(0));                       // adjoints less their seeds (the readout, added on read)
    vd el = {};
    for (int64_t g = g_leaf; g < n_groups; ++g) {         // leaves: energy and both adjoints in one pass
      const int32_t p = gp[2 * g];
      const vd lp = ld(Lv + p * V);
      vd acc = {};
      const int64_t end = gp[2 * g + 1];
      vd ac2 = {};
      for (; k + 1 < end; k += 2) {                       // two dE/dA accumulators (wa, wb), a code site each:
        // on Zen 2 one accumulator's load waits for the previous leaf's store (1.16x, EPYC 7742)
        const vd c = cn[k] + vd{}, c2 = cn[k + 1] + vd{};
        const int64_t f = (int64_t)fa[k] * V, f2 = (int64_t)fa[k + 1] * V;
        acc += c * ld(Av + f);
        ac2 += c2 * ld(Av + f2);
        st(wa + f, ld(wa + f) + c * lp);
        st(wb + f2, ld(wb + f2) + c2 * lp);
      }
      if (k < end) {
        const vd c = cn[k] + vd{};
        const int64_t f = (int64_t)fa[k] * V;
        acc += c * ld(Av + f);
        st(wa + f, ld(wa + f) + c * lp);
        ++k;
      }
      acc += ac2;
      st(wv + p * V, ld(wv + p * V) + acc);
      if (dbl) el += acc * lp;
    }
    const vd tot = ((s0 + s1) + (s2 + s3)) + el;
    for (int v = 0; dbl && v < V; ++v) e[v] += tot[v];
    for (int64_t g = g_leaf - 1; g >= 0; --g) {           // children before their parents
      const int32_t p = gp[2 * g];
      const vd lp = ld(Lv + p * V);
      vd acc = {};
      const int64_t end = gp[2 * g + 1];
      vd ac2 = {};
      for (k = g ? gp[2 * g - 1] : 0; k + 1 < end; k += 2) {
        const vd wk = ld(wv + (n_a + k) * V) + cn[k], wk2 = ld(wv + (n_a + k + 1) * V) + cn[k + 1];
        const int64_t f = (int64_t)fa[k] * V, f2 = (int64_t)fa[k + 1] * V;
        acc += wk * ld(Av + f);
        ac2 += wk2 * ld(Av + f2);
        st(wa + f, ld(wa + f) + wk * lp);
        st(wb + f2, ld(wb + f2) + wk2 * lp);
      }
      if (k < end) {
        const vd wk = ld(wv + (n_a + k) * V) + cn[k];
        const int64_t f = (int64_t)fa[k] * V;
        acc += wk * ld(Av + f);
        st(wa + f, ld(wa + f) + wk * lp);
      }
      acc += ac2;
      st(wv + p * V, ld(wv + p * V) + acc);
    }
    for (int64_t a = 0; !dbl && a < n_a; ++a)             // E = sum_a A_a w_a, in double
      for (int v = 0; v < V; ++v) e[v] += A[a * V + v] * (static_cast<double>(wv[a * V + v]) + cz[a]);
    for (int64_t a = 0; a < n_a; ++a) st(wa + a * V, (ld(wa + a * V) + ld(wb + a * V)) + ld(wv + a * V) + cz[a]);
  }

  // g_U[m0 .. m0 + NC) of one block row: sum over its entries of w z_V, the weight shared by the components;
  // returns the row's z . g of these components.
  template <int NC>
  inline vd row_chunk(const D *gw, const int32_t *ent, int64_t k0, int64_t end, const D *zv, D *gu, const D *zu) {
    vd a0 = {}, a1 = {}, a2 = {}, a3 = {};
    for (int64_t k = k0; k < end; ++k) {
      const vd wk = gw[k] + vd{};
      const D *zp = zv + (int64_t)ent[k] * V;
      a0 += wk * ld(zp);
      if constexpr (NC > 1) a1 += wk * ld(zp + V);
      if constexpr (NC > 2) a2 += wk * ld(zp + 2 * V);
      if constexpr (NC > 3) a3 += wk * ld(zp + 3 * V);
    }
    vd es = a0 * ld(zu);
    st(gu, a0);
    if constexpr (NC > 1) { st(gu + V, a1); es += a1 * ld(zu + V); }
    if constexpr (NC > 2) { st(gu + 2 * V, a2); es += a2 * ld(zu + 2 * V); }
    if constexpr (NC > 3) { st(gu + 3 * V, a3); es += a3 * ld(zu + 3 * V); }
    return es;
  }

  // The product basis in coupled form (eval/coupled.py; `ffi.coupled_plan`'s arrays in grp and ct): z = [A; X],
  // X = sum c A A, g = W z by block rows, E += lin . A + z . g / 2, dE/dA = lin + g_A + sum c g_X A into wA.
  void coupled(int64_t zc) {
    const int32_t *h = grp;
    const int64_t nx = h[1], nxe = h[2], nrows = h[3], nent = h[4], np3 = h[5];
    const int32_t *xptr = h + 8, *xa = xptr + nx + 1, *xb = xa + nxe, *rows = xb + nxe, *ent = rows + 3 * nrows;
    const int32_t *aptr = ent + nent, *ax = aptr + n_a + 1, *ap = ax + np3;
    const D *lin = ct + zc * N, *gw = lin + n_a, *xc = gw + nent, *ac = xc + nxe;
    D *zv = L.data(), *gv = w.data(), *wa = wA.data();
    std::copy(A.data(), A.data() + n_a * V, zv);
    for (int64_t x = 0; x < nx; ++x) {
      vd s0 = {}, s1 = {};
      int64_t k = xptr[x];
      const int64_t end = xptr[x + 1];
      for (; k + 1 < end; k += 2) {
        s0 += xc[k] * ld(zv + (int64_t)xa[k] * V) * ld(zv + (int64_t)xb[k] * V);
        s1 += xc[k + 1] * ld(zv + (int64_t)xa[k + 1] * V) * ld(zv + (int64_t)xb[k + 1] * V);
      }
      if (k < end) s0 += xc[k] * ld(zv + (int64_t)xa[k] * V) * ld(zv + (int64_t)xb[k] * V);
      st(zv + (n_a + x) * V, s0 + s1);
    }
    std::fill(gv, gv + n_a * V, D(0));                    // an A block without a weight has no row
    double eq[V] = {};
    int64_t k0 = 0;
    for (int64_t r = 0; r < nrows; ++r) {
      const int64_t u = rows[3 * r], K = rows[3 * r + 1], end = rows[3 * r + 2];
      vd es = {};
      int64_t m = 0;
      for (; m + 4 <= K; m += 4) es += row_chunk<4>(gw, ent, k0, end, zv + m * V, gv + (u + m) * V, zv + (u + m) * V);
      if (K - m == 3) es += row_chunk<3>(gw, ent, k0, end, zv + m * V, gv + (u + m) * V, zv + (u + m) * V);
      else if (K - m == 2) es += row_chunk<2>(gw, ent, k0, end, zv + m * V, gv + (u + m) * V, zv + (u + m) * V);
      else if (K - m == 1) es += row_chunk<1>(gw, ent, k0, end, zv + m * V, gv + (u + m) * V, zv + (u + m) * V);
      for (int v = 0; v < V; ++v) eq[v] += es[v];        // rows are short: the sum over them is in double
      k0 = end;
    }
    for (int v = 0; v < V; ++v) {
      double el = 0.0;
      for (int64_t a = 0; a < n_a; ++a) el += A[a * V + v] * static_cast<double>(lin[a]);
      e[v] += el + 0.5 * eq[v];
    }
    for (int64_t a = 0; a < n_a; ++a) {
      vd s0 = {}, s1 = {};
      int64_t k = aptr[a];
      const int64_t end = aptr[a + 1];
      for (; k + 1 < end; k += 2) {
        s0 += ac[k] * ld(gv + (int64_t)ax[k] * V) * ld(zv + (int64_t)ap[k] * V);
        s1 += ac[k + 1] * ld(gv + (int64_t)ax[k + 1] * V) * ld(zv + (int64_t)ap[k + 1] * V);
      }
      if (k < end) s0 += ac[k] * ld(gv + (int64_t)ax[k] * V) * ld(zv + (int64_t)ap[k] * V);
      st(wa + a * V, (s0 + s1) + ld(gv + a * V) + lin[a]);
    }
  }

  // Lane v's dE/dv for its nl neighbours, from wA (after dag).
  void grad(int v, int64_t nl, double *G) {
    const v4 *Rv = &R[v * Kb * F], *dRv = &dR[v * Kb * F], *Yv = &Y[v * Kb * nY];
    const double *hv = &hat[v * K * 3];
    for (int64_t a = 0; a < n_a; ++a) wl[a] = wA[a * V + v];
    v4 *q4 = q.data(), *dq4 = dq.data();
    for (int64_t b = 0; 4 * b < nl; ++b) {
      const v4 *Rb = Rv + b * F, *dRb = dRv + b * F, *Yb = Yv + b * nY;
      std::fill(q4, q4 + nY, v4{});                      // q_y = sum_a wA R_r(a), dq_y with R'
      std::fill(dq4, dq4 + nY, v4{});
      for (int64_t u = 0; u < n_runs; ++u) {
        const int32_t a0 = run[4 * u], rr = run[4 * u + 1], y0 = run[4 * u + 2], len = run[4 * u + 3];
        const v4 Rr = Rb[rr], dRr = dRb[rr];
        run_q(q4 + y0, dq4 + y0, &wl[a0], Rr, dRr, len);
      }
      v4 sc = {}, gx = {}, gy = {}, gz = {}, sl = {};
      for (int64_t c = n_rnl; c < F; ++c) sc += dRb[c];
      sc += dq4[0] * Yb[0];
      const int32_t *gi = harm.gi.data();
      const double *gc = harm.gc.data();
      for (int64_t yy = 1; yy < nY; ++yy) {               // grad Y from the degree below (Harmonics)
        const int32_t *i5 = gi + 5 * yy;
        const double *c5 = gc + 5 * yy;
        const v4 qy = q4[yy];
        sc += dq4[yy] * Yb[yy];
        gx += qy * (c5[0] * Yb[i5[0]] + c5[1] * Yb[i5[1]]);
        gy += qy * (c5[2] * Yb[i5[2]] + c5[3] * Yb[i5[3]]);
        gz += qy * (c5[4] * Yb[i5[4]]);
      }
      if (!harm.solid)                                    // sum_y l q_y Y_y, along rhat
        for (int64_t l = 1, yy = 1; l <= harm.L; ++l)
          for (int64_t m = 0; m < 2 * l + 1; ++m, ++yy) sl += double(l) * (q4[yy] * Yb[yy]);
      for (int j = 0; j < 4 && 4 * b + j < nl; ++j) {
        const int64_t s = 4 * b + j;
        const double ir = harm.solid ? 1.0 : ir4[v * K + s], rad = sc[j] - ir * sl[j];
        G[3 * s] = rad * hv[s * 3] + ir * gx[j];
        G[3 * s + 1] = rad * hv[s * 3 + 1] + ir * gy[j];
        G[3 * s + 2] = rad * hv[s * 3 + 2] + ir * gz[j];
      }
    }
  }
};

// -DACEJAX_FFI_PROF: seconds in the three stages (A, the product basis, the forces), printed per call to stderr.
#ifdef ACEJAX_FFI_PROF
#include <chrono>
#include <cstdio>
static std::atomic<int64_t> prof_ns[3], prof_rows;
#define PROF_T(t) const auto t = std::chrono::steady_clock::now()
#define PROF_ADD(t0, t1, t2, nr)                                                                     \
  do {                                                                                              \
    const auto t3 = std::chrono::steady_clock::now();                                               \
    prof_ns[0] += (t1 - t0).count(); prof_ns[1] += (t2 - t1).count(); prof_ns[2] += (t3 - t2).count(); \
    prof_rows += nr;                                                                                \
  } while (0)
static void prof_report() {
  const double n = std::max<int64_t>(1, prof_rows.exchange(0));
  std::fprintf(stderr, "ace_ffi us/site: A %.3f product %.3f forces %.3f\n", prof_ns[0].exchange(0) / n * 1e-3,
               prof_ns[1].exchange(0) / n * 1e-3, prof_ns[2].exchange(0) / n * 1e-3);
}
#else
#define PROF_T(t)
#define PROF_ADD(t0, t1, t2, nr)
static void prof_report() {}
#endif

// Rows [i0, i1) in batches of the site's V lanes of one centre species z[i].  gather(i, vec, zs, ix) writes row i's live
// neighbour vectors, species and atom (or slot) indices and returns their count; emit(i, e, nl, G, ix) takes
// the row's energy and dE/dv.  A batch runs when it fills, and the remainders at the end with empty lanes
// (no neighbours, results discarded).
template <class S, class Gather, class Emit>
void batched(S &site, int64_t i0, int64_t i1, const int32_t *z, Gather gather, Emit emit) {
  constexpr int V = S::V;
  const int64_t K = site.K;
  std::vector<std::vector<int64_t>> pend(site.nzc);
  auto run = [&](int64_t zc, std::vector<int64_t> &rows) {
    const int nr = static_cast<int>(rows.size());
    PROF_T(t0);
    for (int v = 0; v < V; ++v) {
      double *vec = &site.vin[v * K * 3];
      int32_t *zs = &site.zin[v * K], *ix = &site.idx[v * K];
      site.nl[v] = v < nr ? gather(rows[v], vec, zs, ix) : 0;
      site.neighbours(v, zc, site.nl[v], vec, zs);
    }
    PROF_T(t1);
    site.dag(zc);
    PROF_T(t2);
    for (int v = 0; v < nr; ++v) {
      double *g = &site.G[v * K * 3];
      site.grad(v, site.nl[v], g);
      emit(rows[v], site.e[v], site.nl[v], g, &site.idx[v * K]);
    }
    PROF_ADD(t0, t1, t2, nr);
    rows.clear();
  };
  for (int64_t i = i0; i < i1; ++i) {
    if (gather(i, &site.vin[0], &site.zin[0], &site.idx[0]) == 0) {   // no neighbour (a padding row): E0 alone
      emit(i, site.E0[z[i]], 0, nullptr, nullptr);
      continue;
    }
    auto &p = pend[z[i]];
    p.push_back(i);
    if (static_cast<int>(p.size()) == V) run(z[i], p);
  }
  for (int64_t zc = 0; zc < site.nzc; ++zc)
    if (!pend[zc].empty()) run(zc, pend[zc]);
}

ffi::Error AceDirectImpl(ffi::Buffer<ffi::F64> pos, ffi::Buffer<ffi::S32> species, ffi::Buffer<ffi::S32> nbr,
                         ffi::Buffer<ffi::S32> num, ffi::Buffer<ffi::F64> weight, ffi::Buffer<ffi::F64> rtab,
                         ffi::Buffer<ffi::F64> rcut,
                         ffi::Buffer<ffi::S32> runs, ffi::Buffer<ffi::S32> groups,
                         ffi::Buffer<ffi::S32> factor, ffi::AnyBuffer ct, ffi::Buffer<ffi::F64> e0,
                         double x0, double h, int64_t n, int64_t n_rnl, int64_t lmax, int64_t solid,
                         int64_t n_leaf, int64_t form, double rc2, ffi::ResultBuffer<ffi::F64> e,
                         ffi::ResultBuffer<ffi::F64> f) {
  const int64_t atoms = pos.dimensions()[0], K = nbr.dimensions()[0], rows = nbr.dimensions()[1];
  const double *x = pos.typed_data(), *wt = weight.typed_data();
  const int32_t *z = species.typed_data(), *nb = nbr.typed_data(), *cnt = num.typed_data();
  double *eo = e->typed_data(), *fo = f->typed_data();
  if (ct.element_type() != ffi::F64 && ct.element_type() != ffi::F32)
    return ffi::Error::InvalidArgument("the ACE kernel's readout is float64, or float32 for a float DAG");
  std::memset(eo, 0, atoms * sizeof(double));
  std::memset(fo, 0, atoms * 3 * sizeof(double));
  bool broken = false;
  auto gather = [&](int64_t i, double *V, int32_t *zs, int32_t *js) {
    int64_t nl = 0;
    for (int64_t k = 0; k < cnt[i] && k < K; ++k) {
      const int32_t j = nb[k * rows + i];
      if (j < 0 || j >= atoms) continue;                  // padding
      double d[3], r2 = 0.0;
      for (int c = 0; c < 3; ++c) {
        d[c] = x[3 * j + c] - x[3 * i + c];
        r2 += d[c] * d[c];
      }
      broken |= std::isnan(r2);
      if (!(r2 <= rc2)) continue;                         // skin pairs
      std::memcpy(&V[3 * nl], d, 3 * sizeof(double));
      zs[nl] = z[j];
      js[nl++] = j;
    }
    return nl;
  };
  auto emit = [&](int64_t i, double ei, int64_t nl, const double *G, const int32_t *js) {
    eo[i] = ei;
    const double wi = wt[i];                              // G is d e_i / dv alone: -sum_i w_i grad e_i
    for (int64_t s = 0; s < nl; ++s)
      for (int c = 0; c < 3; ++c) {
        fo[3 * i + c] += wi * G[3 * s + c];
        fo[3 * js[s] + c] -= wi * G[3 * s + c];
      }
  };
  const auto run = [&](auto d) {                          // the DAG's type, from the readout's
    Site<decltype(d)> site(rtab, rcut, runs, groups, factor, ct, e0, x0, h, n, n_rnl, lmax, solid, n_leaf, form, K);
    batched(site, 0, std::min(rows, atoms), z, gather, emit);
  };
  if (ct.element_type() == ffi::F32) run(float{});
  else run(double{});
  prof_report();
  if (broken) {                                           // a NaN position: NaN out, not a dropped pair
    std::fill(eo, eo + atoms, std::numeric_limits<double>::quiet_NaN());
    std::fill(fo, fo + atoms * 3, std::numeric_limits<double>::quiet_NaN());
  }
  return ffi::Error::Success();
}

// Rows [0, rows) in chunks, on XLA's intra-op pool and the calling thread.  Every worker, the caller
// included, claims chunks from one counter until none are left, so a pool of one thread (or none: the FFI
// then runs the task on the caller) cannot deadlock; the caller then waits only for chunks already running.
template <class F>
void parallel_rows(ffi::ThreadPool &pool, int64_t rows, int64_t max_threads, F body) {
  const int64_t avail = std::max<int64_t>(1, pool.num_threads() + 1);   // the pool and the caller
  const int64_t threads = max_threads > 0 ? std::min(max_threads, avail) : avail;
  const int64_t chunk = std::max<int64_t>(16, (rows + 4 * threads - 1) / (4 * threads));
  const int64_t nchunk = (rows + chunk - 1) / chunk;
  if (threads == 1 || nchunk <= 1) {
    body(0, rows);
    return;
  }
  struct Shared {
    std::atomic<int64_t> next{0}, done{0};
    std::mutex m;
    std::condition_variable cv;
  };
  auto sh = std::make_shared<Shared>();
  auto run = [sh, nchunk, chunk, rows, &body]() {   // a late task finds no chunk and never touches body
    for (int64_t c; (c = sh->next.fetch_add(1)) < nchunk;) {
      body(c * chunk, std::min(rows, (c + 1) * chunk));
      if (sh->done.fetch_add(1) + 1 == nchunk) {
        std::lock_guard<std::mutex> lk(sh->m);
        sh->cv.notify_all();
      }
    }
  };
  for (int64_t t = 1; t < std::min(threads, nchunk); ++t) pool.Schedule(decltype(run)(run));   // an rvalue copy: Schedule owns it
  run();
  std::unique_lock<std::mutex> lk(sh->m);
  sh->cv.wait(lk, [&] { return sh->done.load() == nchunk; });
}

// The ASE path (calc.ACECalculator(ffi=...)): the dense edge input of `EdgeSiteModel.energy_forces_virial_dense`,
// rij (rows, K, 3) with node_z (rows,), zj (rows, K) and mask (rows, K), to site energies (rows,) and
// G = dE_i / drij (rows, K, 3), zero on masked slots.  Rows are independent and write only their own outputs,
// so they run in parallel on XLA's intra-op pool (`max_threads` > 0 caps it), each chunk in its own batches.
ffi::Error AceEdgesImpl(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> rij, ffi::Buffer<ffi::S32> zi,
                        ffi::Buffer<ffi::S32> zj, ffi::Buffer<ffi::PRED> mask, ffi::Buffer<ffi::F64> rtab,
                        ffi::Buffer<ffi::F64> rcut, ffi::Buffer<ffi::S32> runs, ffi::Buffer<ffi::S32> groups,
                        ffi::Buffer<ffi::S32> factor, ffi::AnyBuffer ct, ffi::Buffer<ffi::F64> e0,
                        double x0, double h, int64_t n, int64_t n_rnl, int64_t lmax, int64_t solid,
                        int64_t n_leaf, int64_t form, int64_t max_threads, ffi::ResultBuffer<ffi::F64> e,
                        ffi::ResultBuffer<ffi::F64> g) {
  const int64_t rows = rij.dimensions()[0], K = rij.dimensions()[1];
  const double *r = rij.typed_data();
  const int32_t *z = zi.typed_data(), *zn = zj.typed_data();
  const bool *live = mask.typed_data();
  double *eo = e->typed_data(), *go = g->typed_data();
  if (ct.element_type() != ffi::F64 && ct.element_type() != ffi::F32)
    return ffi::Error::InvalidArgument("the ACE kernel's readout is float64, or float32 for a float DAG");
  double rc2 = 0.0;
  for (size_t p = 0; p < rcut.element_count(); ++p) rc2 = std::max(rc2, rcut.typed_data()[p] * rcut.typed_data()[p]);
  parallel_rows(pool, rows, max_threads, [&](int64_t i0, int64_t i1) {
    auto gather = [&](int64_t i, double *V, int32_t *zs, int32_t *slot) {
      int64_t nl = 0;
      for (int64_t k = 0; k < K; ++k)
        if (live[i * K + k]) {
          const double *u = r + 3 * (i * K + k);           // a skin pair is zero in every table: not a neighbour
          if (u[0] * u[0] + u[1] * u[1] + u[2] * u[2] >= rc2) continue;   // (a NaN stays, and gives NaN)
          std::memcpy(&V[3 * nl], r + 3 * (i * K + k), 3 * sizeof(double));
          zs[nl] = zn[i * K + k];
          slot[nl++] = static_cast<int32_t>(k);
        }
      return nl;
    };
    auto emit = [&](int64_t i, double ei, int64_t nl, const double *G, const int32_t *slot) {
      eo[i] = ei;
      double *gi = go + 3 * i * K;
      std::memset(gi, 0, 3 * K * sizeof(double));
      for (int64_t s = 0; s < nl; ++s) std::memcpy(gi + 3 * slot[s], &G[3 * s], 3 * sizeof(double));
    };
    const auto run = [&](auto d) {
      Site<decltype(d)> site(rtab, rcut, runs, groups, factor, ct, e0, x0, h, n, n_rnl, lmax, solid, n_leaf, form, K);
      batched(site, i0, i1, z, gather, emit);
    };
    if (ct.element_type() == ffi::F32) run(float{});
    else run(double{});
  });
  prof_report();
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(AceEdges, AceEdgesImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::F64>>()     // rij (rows, K, 3)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // centre species (rows,)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // neighbour species (rows, K)
                                  .Arg<ffi::Buffer<ffi::PRED>>()    // live slots (rows, K)
                                  .Arg<ffi::Buffer<ffi::F64>>()     // radial table (NZ, NZ, ncoef, F)
                                  .Arg<ffi::Buffer<ffi::F64>>()     // pair cutoffs (NZ, NZ)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // A runs (n_runs, 4): a0, r, y0, len
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG groups (n_groups, 2): parent, end; or the coupled plan
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG A factor per node
                                  .Arg<ffi::AnyBuffer>()            // readout (NZ, n_A + nodes), or the coupled plan's reals: its type is the DAG's
                                  .Arg<ffi::Buffer<ffi::F64>>()     // E0 (NZ,)
                                  .Attr<double>("x0")
                                  .Attr<double>("h")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("n_rnl")
                                  .Attr<int64_t>("lmax")
                                  .Attr<int64_t>("solid")
                                  .Attr<int64_t>("n_leaf")
                                  .Attr<int64_t>("form")            // 0: the DAG; 1: the coupled form
                                  .Attr<int64_t>("max_threads")
                                  .Ret<ffi::Buffer<ffi::F64>>()     // site energies (rows,)
                                  .Ret<ffi::Buffer<ffi::F64>>());   // dE_i / drij (rows, K, 3)

XLA_FFI_DEFINE_HANDLER_SYMBOL(AceDirect, AceDirectImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()     // positions (atoms, 3)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // model species (atoms,)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // neighbour matrix (K, rows)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // neighbour counts (rows,)
                                  .Arg<ffi::Buffer<ffi::F64>>()     // site weights (atoms,): f = -sum_i w_i grad e_i
                                  .Arg<ffi::Buffer<ffi::F64>>()     // radial table (NZ, NZ, ncoef, F)
                                  .Arg<ffi::Buffer<ffi::F64>>()     // pair cutoffs (NZ, NZ)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // A runs (n_runs, 4): a0, r, y0, len
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG groups (n_groups, 2): parent, end; or the coupled plan
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG A factor per node
                                  .Arg<ffi::AnyBuffer>()            // readout (NZ, n_A + nodes), or the coupled plan's reals: its type is the DAG's
                                  .Arg<ffi::Buffer<ffi::F64>>()     // E0 (NZ,)
                                  .Attr<double>("x0")
                                  .Attr<double>("h")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("n_rnl")
                                  .Attr<int64_t>("lmax")
                                  .Attr<int64_t>("solid")
                                  .Attr<int64_t>("n_leaf")
                                  .Attr<int64_t>("form")            // 0: the DAG; 1: the coupled form
                                  .Attr<double>("rc2")
                                  .Ret<ffi::Buffer<ffi::F64>>()     // atom energies (atoms,)
                                  .Ret<ffi::Buffer<ffi::F64>>());   // -sum_i w_i grad e_i (atoms, 3): forces at w = 1
