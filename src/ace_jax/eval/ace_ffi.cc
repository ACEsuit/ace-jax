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
inline double vsqrt(double x) { return std::sqrt(x); }
inline v4 vsqrt(v4 x) { v4 r; for (int i = 0; i < 4; ++i) r[i] = std::sqrt(x[i]); return r; }

// ace_jax.eval.harmonics with gradients: Y[l * l + l + m], dY[d * nY + i].
struct Harmonics {
  template <class T> struct Scratch {
    std::vector<T> A, B, P, dP, rp;
    explicit Scratch(int W) : A(W), B(W), P(W * W), dP(W * W), rp(W) {}
  };
  Scratch<double> s1;
  Scratch<v4> s4;
  Scratch<double> &scr(double) { return s1; }
  Scratch<v4> &scr(v4) { return s4; }
  int L;
  bool solid;
  std::vector<double> norm, ca, cb;

  Harmonics(int L_, bool solid_)
      : s1(L_ + 1), s4(L_ + 1), L(L_), solid(solid_), norm((L_ + 1) * (L_ + 1)), ca((L_ + 1) * (L_ + 1)), cb((L_ + 1) * (L_ + 1)) {
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
  }

  void eval(const double *v, double *Y, double *dY) { eval<double>(v[0], v[1], v[2], Y, dY); }

  // T = double, or v4 for four vectors at once (Y[i], dY[d * nY + i] then hold the four)
  template <class T> void eval(const T x, const T y, const T z, T *Y, T *dY) {
    auto &S = scr(T{});
    T *A = S.A.data(), *B = S.B.data(), *P = S.P.data(), *dP = S.dP.data(), *rp = S.rp.data();
    const T zero = T{}, one = zero + 1.0;
    const T r2 = x * x + y * y + z * z, r = vsqrt(r2), ir = 1.0 / r, ir2 = ir * ir, u = z * ir;
    const T du[3] = {-z * x * ir2 * ir, -z * y * ir2 * ir, (x * x + y * y) * ir2 * ir};
    const int W = L + 1, nY = W * W;
    A[0] = one;
    B[0] = zero;
    for (int m = 1; m <= L; ++m) {                       // A_m + i B_m = (x + i y)^m
      A[m] = x * A[m - 1] - y * B[m - 1];
      B[m] = x * B[m - 1] + y * A[m - 1];
    }
    double dfact = 1.0;                                   // P(l, m) = Rbar_l^m(u), dP = d/du
    for (int m = 0; m <= L; ++m) {
      if (m > 0) dfact *= 2 * m - 1;
      P[m * W + m] = zero + dfact;
      dP[m * W + m] = zero;
      if (m + 1 <= L) {
        P[(m + 1) * W + m] = ((2 * m + 1) * dfact) * u;
        dP[(m + 1) * W + m] = zero + (2 * m + 1) * dfact;
      }
      for (int l = m + 2; l <= L; ++l) {
        const double a = ca[l * W + m], b = cb[l * W + m];
        P[l * W + m] = a * u * P[(l - 1) * W + m] - b * P[(l - 2) * W + m];
        dP[l * W + m] = a * (P[(l - 1) * W + m] + u * dP[(l - 1) * W + m]) - b * dP[(l - 2) * W + m];
      }
    }
    rp[0] = one;                                          // solid r^k, spherical r^-k
    for (int k = 1; k <= L; ++k) rp[k] = rp[k - 1] * (solid ? r : ir);
    for (int l = 0; l <= L; ++l)
      for (int m = -l; m <= l; ++m) {
        const int am = std::abs(m), i = l * l + l + m, p = solid ? l - am : am;
        const T ang = m >= 0 ? A[am] : B[am];
        T dax = zero, day = zero;
        if (am) {
          dax = double(am) * (m > 0 ? A[am - 1] : B[am - 1]);
          day = m > 0 ? -double(am) * B[am - 1] : double(am) * A[am - 1];
        }
        const T c = norm[i] * rp[p], Pc = c * P[l * W + am], dPc = c * dP[l * W + am] * ang;
        const T Yi = Pc * ang, g = double(solid ? p : -p) * ir2 * Yi;   // d(r^+-p) = +-p r^(+-p-2) v
        Y[i] = Yi;
        dY[i] = dPc * du[0] + Pc * dax + g * x;
        dY[nY + i] = dPc * du[1] + Pc * day + g * y;
        dY[2 * nY + i] = dPc * du[2] + g * z;
      }
  }
};

// radial.spline_eval and d/dr on one species pair's table, zero from its cutoff on.
void table(const double *coef, int64_t F, double x0, double h, int64_t n, double r, double rc,
           double *v, double *dv) {
  if (r >= rc) {
    std::memset(v, 0, F * sizeof(double));
    std::memset(dv, 0, F * sizeof(double));
    return;
  }
  const double k = (r - x0) / h + 1.0;
  int64_t ix = static_cast<int64_t>(std::floor(k));
  ix = ix < 1 ? 1 : (ix > n - 1 ? n - 1 : ix);
  const double t = k - ix, s = 1.0 - t;
  const double w[4] = {s * s * s / 6.0, (4.0 - 6.0 * t * t + 3.0 * t * t * t) / 6.0,
                       (4.0 - 6.0 * s * s + 3.0 * s * s * s) / 6.0, t * t * t / 6.0};
  const double dw[4] = {-0.5 * s * s / h, (1.5 * t * t - 2.0 * t) / h, (2.0 * s - 1.5 * s * s) / h,
                        0.5 * t * t / h};
  const double *c0 = coef + (ix - 1) * F, *c1 = c0 + F, *c2 = c1 + F, *c3 = c2 + F;
  for (int64_t c = 0; c < F; ++c) {
    v[c] = w[0] * c0[c] + w[1] * c1[c] + w[2] * c2[c] + w[3] * c3[c];
    dv[c] = dw[0] * c0[c] + dw[1] * c1[c] + dw[2] * c2[c] + dw[3] * c3[c];
  }
}

// Sites in batches of kLanes of one centre species, a site per SIMD lane: A, its adjoint, the DAG values and
// their adjoints are [entry][lane], so a batch reads each coefficient and index of the model (readout, DAG
// factors and groups: ~1 MB at o4d20) once for kLanes sites.  Per site that was 12 instructions a leaf, and
// on Zen 2 ~10 cycles (the dE/dA update's load waits for the previous store); a batch of 4 halves the
// kernel's time on an i9-14900K and an EPYC 7742 (docs/dev/accuracy-cost-results.md).  8 lanes need AVX-512.
#ifndef ACEJAX_FFI_LANES
#define ACEJAX_FFI_LANES 4
#endif
constexpr int kLanes = ACEJAX_FFI_LANES;
// The DAG's floats: double, or (prototype, -DACEJAX_FFI_DAG_FLOAT with -DACEJAX_FFI_LANES=8) float, twice the
// sites a register; the radials, harmonics, A and the force contraction stay double.
#ifdef ACEJAX_FFI_DAG_FLOAT
typedef float dreal;
#else
typedef double dreal;
#endif
#ifndef ACEJAX_FFI_E
#define ACEJAX_FFI_E 1                                      // the float DAG's energy: 0 float leaf terms, 1 a group's sum in double, 2 every term
#endif

// One batch's energies and dE/dv for each site's live neighbour vectors; model buffers and scratch.
struct Site {
  static constexpr int V = kLanes;
  const double *tab, *rc, *ct, *E0;
  const int32_t *run, *grp, *fac;
  int64_t nz, ncoef, F, n_rnl, n_runs, n_groups, n_nodes, N, n_a, nY, n, n_leaf, g_leaf, K;
  double x0, h;
  Harmonics harm;
  std::vector<double> hat, A, wl, tR, tdR;
  std::vector<dreal> wA, wB, L, w, ctd;                   // the DAG's own, and the readout in its type
  std::vector<v4> R, dR, Y, dY, at, q, dq;               // [lane][block of 4 neighbours][feature]
  int64_t Kb;
  std::vector<double> vin, G;                             // per lane: neighbour vectors in, dE/dv out
  std::vector<int32_t> zin, idx;                          // per lane: neighbour species, and atom or slot
  double e[V];
  int64_t nl[V], nzc;

  Site(ffi::Buffer<ffi::F64> rtab, ffi::Buffer<ffi::F64> rcut, ffi::Buffer<ffi::S32> runs,
       ffi::Buffer<ffi::S32> groups, ffi::Buffer<ffi::S32> factor, ffi::Buffer<ffi::F64> ctz,
       ffi::Buffer<ffi::F64> e0, double x0_, double h_, int64_t n_, int64_t n_rnl_, int64_t lmax,
       int64_t solid, int64_t n_leaf_, int64_t K_)
      : tab(rtab.typed_data()), rc(rcut.typed_data()), ct(ctz.typed_data()), E0(e0.typed_data()),
        run(runs.typed_data()), grp(groups.typed_data()), fac(factor.typed_data()),
        nz(rtab.dimensions()[0]), ncoef(rtab.dimensions()[2]), F(rtab.dimensions()[3]), n_rnl(n_rnl_),
        n_runs(runs.dimensions()[0]), n_groups(groups.dimensions()[0]), n_nodes(factor.element_count()),
        N(ctz.dimensions()[1]), n_a(N - n_nodes), nY((lmax + 1) * (lmax + 1)), n(n_), n_leaf(n_leaf_),
        g_leaf(0), K(K_), x0(x0_), h(h_), harm(static_cast<int>(lmax), solid != 0), hat(V * K_ * 3), A(n_a * V), wA(n_a * V), wB(n_a * V),
        wl(n_a), tR(4 * F), tdR(4 * F), L((N - n_leaf) * V), w((N - n_leaf) * V),
        ctd(ct, ct + ctz.element_count()),
        R(V * ((K_ + 3) / 4) * F), dR(R.size()), Y(V * ((K_ + 3) / 4) * nY), dY(3 * Y.size()), at(n_a), q(nY), dq(nY),
        Kb((K_ + 3) / 4), vin(V * K * 3), G(V * K * 3),
        zin(V * K), idx(V * K), nzc(e0.element_count()) {
    while (g_leaf < n_groups && grp[2 * g_leaf + 1] <= n_nodes - n_leaf) ++g_leaf;   // the first leaf group
  }

  // Lane v's radials, harmonics and A column for its nl neighbour vectors V3 (species zj); e[v] = E0 + pair.
  void neighbours(int v, int64_t zc, int64_t nl, const double *V3, const int32_t *zj) {
    double ei = E0[zc];
    v4 *Rv = &R[v * Kb * F], *dRv = &dR[v * Kb * F], *Yv = &Y[v * Kb * nY], *dYv = &dY[v * Kb * 3 * nY];
    double *hv = &hat[v * K * 3], *t = tR.data(), *td = tdR.data();
    const int64_t nb = (nl + 3) / 4;
    std::fill(at.begin(), at.end(), v4{});
    v4 *a4 = at.data();
    for (int64_t b = 0; b < nb; ++b) {                    // four neighbours: a padding one has R = 0
      v4 x = {0.6, 0.6, 0.6, 0.6}, y = {0.48, 0.48, 0.48, 0.48}, z = {0.64, 0.64, 0.64, 0.64};
      std::fill(t, t + 4 * F, 0.0);
      std::fill(td, td + 4 * F, 0.0);
      for (int j = 0; j < 4 && 4 * b + j < nl; ++j) {
        const int64_t s = 4 * b + j;
        const double *u = V3 + 3 * s;
        const double r = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        const int64_t pz = zc * nz + zj[s];
        table(tab + pz * ncoef * F, F, x0, h, n, r, rc[pz], t + j * F, td + j * F);
        for (int64_t c = n_rnl; c < F; ++c) ei += t[j * F + c];   // the folded pair radial
        for (int d = 0; d < 3; ++d) hv[s * 3 + d] = u[d] / r;
        x[j] = u[0]; y[j] = u[1]; z[j] = u[2];
      }
      v4 *Rb = Rv + b * F, *dRb = dRv + b * F, *Yb = Yv + b * nY;
      for (int64_t c = 0; c < F; ++c) {
        Rb[c] = v4{t[c], t[F + c], t[2 * F + c], t[3 * F + c]};
        dRb[c] = v4{td[c], td[F + c], td[2 * F + c], td[3 * F + c]};
      }
      harm.eval<v4>(x, y, z, Yb, dYv + b * 3 * nY);
      for (int64_t u = 0; u < n_runs; ++u) {
        const int32_t a0 = run[4 * u], rr = run[4 * u + 1], y0 = run[4 * u + 2], len = run[4 * u + 3];
        const v4 Rr = Rb[rr];
        for (int32_t m = 0; m < len; ++m) a4[a0 + m] += Rr * Yb[y0 + m];
      }
    }
    e[v] = ei;
    for (int64_t a = 0; a < n_a; ++a) A[a * V + v] = (a4[a][0] + a4[a][1]) + (a4[a][2] + a4[a][3]);
  }

  // The product basis of every lane (centre species zc): energies into e, dE/dA into wA.
  typedef dreal vd __attribute__((vector_size(sizeof(dreal) * kLanes), aligned(sizeof(dreal)), may_alias));
  static inline vd ld(const dreal *p) { return *(const vd *)p; }
  static inline void st(dreal *p, vd x) { *(vd *)p = x; }
  typedef double vdd __attribute__((vector_size(8 * kLanes)));   // the energy's sums are double
  static inline vdd up(vd x) { return __builtin_convertvector(x, vdd); }
  void dag(int64_t zc) {
    const int64_t n_in = n_nodes - n_leaf, M = n_a + n_in;   // leaves: the last level, never stored
    const int32_t *gp = grp, *fa = fac;
    dreal *Lv = L.data(), *wv = w.data(), *wa = wA.data(), *wb = wB.data();
    const dreal *cz = ctd.data() + zc * N, *cn = cz + n_a, *Av = Lv;   // A in the DAG's type
    const double *czd = ct + zc * N, *cnd = czd + n_a;
    (void)cnd;
    std::copy(A.data(), A.data() + n_a * V, Lv);

    int64_t k = 0;
    for (int64_t g = 0; g < g_leaf; ++g) {                // a group: one parent's children, together
      const vd lp = ld(Lv + gp[2 * g] * V);
      for (const int64_t end = gp[2 * g + 1]; k < end; ++k) st(Lv + (n_a + k) * V, lp * ld(Av + fa[k] * V));
    }
    vdd s0 = {}, s1 = {}, s2 = {}, s3 = {};                // four sums a lane: no add chain over M
    int64_t j = 0;
    for (; ACEJAX_FFI_E != 3 && j + 3 < M; j += 4) {
      const vdd c0 = czd[j] + vdd{}, c1 = czd[j + 1] + vdd{}, c2 = czd[j + 2] + vdd{}, c3 = czd[j + 3] + vdd{};
      s0 += c0 * up(ld(Lv + j * V)); s1 += c1 * up(ld(Lv + (j + 1) * V));
      s2 += c2 * up(ld(Lv + (j + 2) * V)); s3 += c3 * up(ld(Lv + (j + 3) * V));
    }
    for (; ACEJAX_FFI_E != 3 && j < M; ++j) {
      const vdd c = czd[j] + vdd{};
      s0 += c * up(ld(Lv + j * V));
    }
    std::fill(wa, wa + n_a * V, dreal(0));
    std::fill(wb, wb + n_a * V, dreal(0));
    std::fill(wv, wv + M * V, dreal(0));                       // adjoints less their seeds (the readout, added on read)
    vdd el = {};
    for (int64_t g = g_leaf; g < n_groups; ++g) {         // leaves: energy and both adjoints in one pass
      const int32_t p = gp[2 * g];
      const vd lp = ld(Lv + p * V);
      vd acc = {};
      const int64_t end = gp[2 * g + 1];
      vd ac2 = {};
#if ACEJAX_FFI_E == 2
      vdd ad = {};
#endif
      for (; k + 1 < end; k += 2) {                       // two dE/dA accumulators (wa, wb), a code site each:
        // on Zen 2 one accumulator's load waits for the previous leaf's store (1.16x, EPYC 7742)
        const vd c = cn[k] + vd{}, c2 = cn[k + 1] + vd{};
        const int64_t f = (int64_t)fa[k] * V, f2 = (int64_t)fa[k + 1] * V;
        acc += c * ld(Av + f);
        ac2 += c2 * ld(Av + f2);
#if ACEJAX_FFI_E == 2
        ad += cnd[k] * up(ld(Av + f)) + cnd[k + 1] * up(ld(Av + f2));
#endif
        st(wa + f, ld(wa + f) + c * lp);
        st(wb + f2, ld(wb + f2) + c2 * lp);
      }
      if (k < end) {
        const vd c = cn[k] + vd{};
        const int64_t f = (int64_t)fa[k] * V;
        acc += c * ld(Av + f);
#if ACEJAX_FFI_E == 2
        ad += cnd[k] * up(ld(Av + f));
#endif
        st(wa + f, ld(wa + f) + c * lp);
        ++k;
      }
      acc += ac2;
      st(wv + p * V, ld(wv + p * V) + acc);
#if ACEJAX_FFI_E == 2
      el += ad * up(lp);
#elif ACEJAX_FFI_E == 1
      el += up(acc) * up(lp);
#elif ACEJAX_FFI_E == 3
#else
      el += up(acc * lp);
#endif
    }
    const vdd tot = ((s0 + s1) + (s2 + s3)) + el;
    for (int v = 0; v < V; ++v) e[v] += tot[v];
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
#if ACEJAX_FFI_E == 3
    for (int64_t a = 0; a < n_a; ++a)                     // E = sum_a A_a w_a, w_a the adjoint of A_a as a node
      for (int v = 0; v < V; ++v) e[v] += A[a * V + v] * (static_cast<double>(wv[a * V + v]) + czd[a]);
#endif
    for (int64_t a = 0; a < n_a; ++a) st(wa + a * V, (ld(wa + a * V) + ld(wb + a * V)) + ld(wv + a * V) + cz[a]);
  }

  // Lane v's dE/dv for its nl neighbours, from wA (after dag).
  void grad(int v, int64_t nl, double *G) {
    const v4 *Rv = &R[v * Kb * F], *dRv = &dR[v * Kb * F], *Yv = &Y[v * Kb * nY], *dYv = &dY[v * Kb * 3 * nY];
    const double *hv = &hat[v * K * 3];
    for (int64_t a = 0; a < n_a; ++a) wl[a] = wA[a * V + v];
    v4 *q4 = q.data(), *dq4 = dq.data();
    for (int64_t b = 0; 4 * b < nl; ++b) {
      const v4 *Rb = Rv + b * F, *dRb = dRv + b * F, *Yb = Yv + b * nY, *dYb = dYv + b * 3 * nY;
      std::fill(q4, q4 + nY, v4{});                      // q_y = sum_a wA R_r(a), dq_y with R'
      std::fill(dq4, dq4 + nY, v4{});
      for (int64_t u = 0; u < n_runs; ++u) {
        const int32_t a0 = run[4 * u], rr = run[4 * u + 1], y0 = run[4 * u + 2], len = run[4 * u + 3];
        const v4 Rr = Rb[rr], dRr = dRb[rr];
        for (int32_t m = 0; m < len; ++m) {
          const double wm = wl[a0 + m];
          q4[y0 + m] += wm * Rr;
          dq4[y0 + m] += wm * dRr;
        }
      }
      v4 sc = {}, gx = {}, gy = {}, gz = {};
      for (int64_t c = n_rnl; c < F; ++c) sc += dRb[c];
      for (int64_t yy = 0; yy < nY; ++yy) {
        sc += dq4[yy] * Yb[yy];
        gx += q4[yy] * dYb[yy];
        gy += q4[yy] * dYb[nY + yy];
        gz += q4[yy] * dYb[2 * nY + yy];
      }
      for (int j = 0; j < 4 && 4 * b + j < nl; ++j) {
        const int64_t s = 4 * b + j;
        G[3 * s] = sc[j] * hv[s * 3] + gx[j];
        G[3 * s + 1] = sc[j] * hv[s * 3 + 1] + gy[j];
        G[3 * s + 2] = sc[j] * hv[s * 3 + 2] + gz[j];
      }
    }
  }
};

// Rows [i0, i1) in batches of kLanes of one centre species z[i].  gather(i, vec, zs, ix) writes row i's live
// neighbour vectors, species and atom (or slot) indices and returns their count; emit(i, e, nl, G, ix) takes
// the row's energy and dE/dv.  A batch runs when it fills, and the remainders at the end with empty lanes
// (no neighbours, results discarded).
template <class Gather, class Emit>
void batched(Site &site, int64_t i0, int64_t i1, const int32_t *z, Gather gather, Emit emit) {
  constexpr int V = kLanes;
  const int64_t K = site.K;
  std::vector<std::vector<int64_t>> pend(site.nzc);
  auto run = [&](int64_t zc, std::vector<int64_t> &rows) {
    const int nr = static_cast<int>(rows.size());
    for (int v = 0; v < V; ++v) {
      double *vec = &site.vin[v * K * 3];
      int32_t *zs = &site.zin[v * K], *ix = &site.idx[v * K];
      site.nl[v] = v < nr ? gather(rows[v], vec, zs, ix) : 0;
      site.neighbours(v, zc, site.nl[v], vec, zs);
    }
    site.dag(zc);
    for (int v = 0; v < nr; ++v) {
      double *g = &site.G[v * K * 3];
      site.grad(v, site.nl[v], g);
      emit(rows[v], site.e[v], site.nl[v], g, &site.idx[v * K]);
    }
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
                         ffi::Buffer<ffi::S32> factor, ffi::Buffer<ffi::F64> ct, ffi::Buffer<ffi::F64> e0,
                         double x0, double h, int64_t n, int64_t n_rnl, int64_t lmax, int64_t solid,
                         int64_t n_leaf, double rc2, ffi::ResultBuffer<ffi::F64> e,
                         ffi::ResultBuffer<ffi::F64> f) {
  const int64_t atoms = pos.dimensions()[0], K = nbr.dimensions()[0], rows = nbr.dimensions()[1];
  const double *x = pos.typed_data(), *wt = weight.typed_data();
  const int32_t *z = species.typed_data(), *nb = nbr.typed_data(), *cnt = num.typed_data();
  double *eo = e->typed_data(), *fo = f->typed_data();
  Site site(rtab, rcut, runs, groups, factor, ct, e0, x0, h, n, n_rnl, lmax, solid, n_leaf, K);
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
  batched(site, 0, std::min(rows, atoms), z, gather, emit);
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
                        ffi::Buffer<ffi::S32> factor, ffi::Buffer<ffi::F64> ct, ffi::Buffer<ffi::F64> e0,
                        double x0, double h, int64_t n, int64_t n_rnl, int64_t lmax, int64_t solid,
                        int64_t n_leaf, int64_t max_threads, ffi::ResultBuffer<ffi::F64> e,
                        ffi::ResultBuffer<ffi::F64> g) {
  const int64_t rows = rij.dimensions()[0], K = rij.dimensions()[1];
  const double *r = rij.typed_data();
  const int32_t *z = zi.typed_data(), *zn = zj.typed_data();
  const bool *live = mask.typed_data();
  double *eo = e->typed_data(), *go = g->typed_data();
  parallel_rows(pool, rows, max_threads, [&](int64_t i0, int64_t i1) {
    Site site(rtab, rcut, runs, groups, factor, ct, e0, x0, h, n, n_rnl, lmax, solid, n_leaf, K);
    auto gather = [&](int64_t i, double *V, int32_t *zs, int32_t *slot) {
      int64_t nl = 0;
      for (int64_t k = 0; k < K; ++k)
        if (live[i * K + k]) {
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
    batched(site, i0, i1, z, gather, emit);
  });
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
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG groups (n_groups, 2): parent, end
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG A factor per node
                                  .Arg<ffi::Buffer<ffi::F64>>()     // readout (NZ, n_A + nodes)
                                  .Arg<ffi::Buffer<ffi::F64>>()     // E0 (NZ,)
                                  .Attr<double>("x0")
                                  .Attr<double>("h")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("n_rnl")
                                  .Attr<int64_t>("lmax")
                                  .Attr<int64_t>("solid")
                                  .Attr<int64_t>("n_leaf")
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
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG groups (n_groups, 2): parent, end
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG A factor per node
                                  .Arg<ffi::Buffer<ffi::F64>>()     // readout (NZ, n_A + nodes)
                                  .Arg<ffi::Buffer<ffi::F64>>()     // E0 (NZ,)
                                  .Attr<double>("x0")
                                  .Attr<double>("h")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("n_rnl")
                                  .Attr<int64_t>("lmax")
                                  .Attr<int64_t>("solid")
                                  .Attr<int64_t>("n_leaf")
                                  .Attr<double>("rc2")
                                  .Ret<ffi::Buffer<ffi::F64>>()     // atom energies (atoms,)
                                  .Ret<ffi::Buffer<ffi::F64>>());   // -sum_i w_i grad e_i (atoms, 3): forces at w = 1
