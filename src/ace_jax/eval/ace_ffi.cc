// Lean linear ACE energies and forces on the CPU, one XLA FFI call on lammps-jax's neighbour matrix.
// Per row, pairs within the cutoff only: radial table, harmonics, A, product-basis DAG and its adjoint.
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

// ace_jax.eval.harmonics with gradients: Y[l * l + l + m], dY[d * nY + i].
struct Harmonics {
  int L;
  bool solid;
  std::vector<double> norm, ca, cb, A, B, P, dP, rp;

  Harmonics(int L_, bool solid_)
      : L(L_), solid(solid_), norm((L_ + 1) * (L_ + 1)), ca((L_ + 1) * (L_ + 1)), cb((L_ + 1) * (L_ + 1)),
        A(L_ + 1), B(L_ + 1), P((L_ + 1) * (L_ + 1)), dP((L_ + 1) * (L_ + 1)), rp(L_ + 1) {
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

  void eval(const double *v, double *Y, double *dY) {
    const double x = v[0], y = v[1], z = v[2], r2 = x * x + y * y + z * z;
    const double r = std::sqrt(r2), ir = 1.0 / r, ir2 = ir * ir, u = z * ir;
    const double du[3] = {-z * x * ir2 * ir, -z * y * ir2 * ir, (x * x + y * y) * ir2 * ir};
    const int W = L + 1, nY = W * W;
    A[0] = 1.0;
    B[0] = 0.0;
    for (int m = 1; m <= L; ++m) {                       // A_m + i B_m = (x + i y)^m
      A[m] = x * A[m - 1] - y * B[m - 1];
      B[m] = x * B[m - 1] + y * A[m - 1];
    }
    double dfact = 1.0;                                   // P(l, m) = Rbar_l^m(u), dP = d/du
    for (int m = 0; m <= L; ++m) {
      if (m > 0) dfact *= 2 * m - 1;
      P[m * W + m] = dfact;
      dP[m * W + m] = 0.0;
      if (m + 1 <= L) {
        P[(m + 1) * W + m] = (2 * m + 1) * u * dfact;
        dP[(m + 1) * W + m] = (2 * m + 1) * dfact;
      }
      for (int l = m + 2; l <= L; ++l) {
        const double a = ca[l * W + m], b = cb[l * W + m];
        P[l * W + m] = a * u * P[(l - 1) * W + m] - b * P[(l - 2) * W + m];
        dP[l * W + m] = a * (P[(l - 1) * W + m] + u * dP[(l - 1) * W + m]) - b * dP[(l - 2) * W + m];
      }
    }
    rp[0] = 1.0;                                          // solid r^k, spherical r^-k
    for (int k = 1; k <= L; ++k) rp[k] = rp[k - 1] * (solid ? r : ir);
    for (int l = 0; l <= L; ++l)
      for (int m = -l; m <= l; ++m) {
        const int am = std::abs(m), i = l * l + l + m, p = solid ? l - am : am;
        const double ang = m >= 0 ? A[am] : B[am];
        const double dax = am == 0 ? 0.0 : am * (m > 0 ? A[am - 1] : B[am - 1]);
        const double day = am == 0 ? 0.0 : am * (m > 0 ? -B[am - 1] : A[am - 1]);
        const double c = norm[i] * rp[p], Pc = c * P[l * W + am], dPc = c * dP[l * W + am] * ang;
        const double Yi = Pc * ang, g = (solid ? p : -p) * ir2 * Yi;   // d(r^+-p) = +-p r^(+-p-2) v
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

// One site's energy and dE/dv for its live neighbour vectors; model buffers and scratch.
struct Site {
  const double *tab, *rc, *ct, *E0;
  const int32_t *run, *grp, *fac;
  int64_t nz, ncoef, F, n_rnl, n_runs, n_groups, n_nodes, N, n_a, nY, n, n_leaf, g_leaf;
  double x0, h;
  Harmonics harm;
  std::vector<double> R, dR, Y, dY, hat, A, wA, L, w, q, dq;

  Site(ffi::Buffer<ffi::F64> rtab, ffi::Buffer<ffi::F64> rcut, ffi::Buffer<ffi::S32> runs,
       ffi::Buffer<ffi::S32> groups, ffi::Buffer<ffi::S32> factor, ffi::Buffer<ffi::F64> ctz,
       ffi::Buffer<ffi::F64> e0, double x0_, double h_, int64_t n_, int64_t n_rnl_, int64_t lmax,
       int64_t solid, int64_t n_leaf_, int64_t K)
      : tab(rtab.typed_data()), rc(rcut.typed_data()), ct(ctz.typed_data()), E0(e0.typed_data()),
        run(runs.typed_data()), grp(groups.typed_data()), fac(factor.typed_data()),
        nz(rtab.dimensions()[0]), ncoef(rtab.dimensions()[2]), F(rtab.dimensions()[3]), n_rnl(n_rnl_),
        n_runs(runs.dimensions()[0]), n_groups(groups.dimensions()[0]), n_nodes(factor.element_count()),
        N(ctz.dimensions()[1]), n_a(N - n_nodes), nY((lmax + 1) * (lmax + 1)), n(n_), n_leaf(n_leaf_),
        g_leaf(0), x0(x0_), h(h_), harm(static_cast<int>(lmax), solid != 0), R(K * F), dR(K * F),
        Y(K * nY), dY(K * 3 * nY), hat(K * 3), A(n_a), wA(n_a), L(N - n_leaf), w(N - n_leaf), q(nY), dq(nY) {
    while (g_leaf < n_groups && grp[2 * g_leaf + 1] <= n_nodes - n_leaf) ++g_leaf;   // the first leaf group
  }

  double eval(int64_t zc, int64_t nl, const double *V, const int32_t *zj, double *G) {
    double ei = E0[zc];
    for (int64_t s = 0; s < nl; ++s) {
      const double *v = V + 3 * s;
      const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      const int64_t pz = zc * nz + zj[s];
      table(tab + pz * ncoef * F, F, x0, h, n, r, rc[pz], &R[s * F], &dR[s * F]);
      for (int64_t c = n_rnl; c < F; ++c) ei += R[s * F + c];   // the folded pair radial
      harm.eval(v, &Y[s * nY], &dY[s * 3 * nY]);
      for (int d = 0; d < 3; ++d) hat[s * 3 + d] = v[d] / r;
    }
    if (nl == 0) return ei;
    std::fill(A.begin(), A.end(), 0.0);
    for (int64_t s = 0; s < nl; ++s) {
      const double *Rs = &R[s * F], *Ys = &Y[s * nY];
      for (int64_t u = 0; u < n_runs; ++u) {
        const int32_t a0 = run[4 * u], rr = run[4 * u + 1], y0 = run[4 * u + 2], len = run[4 * u + 3];
        const double Rr = Rs[rr];
        for (int32_t m = 0; m < len; ++m) A[a0 + m] += Rr * Ys[y0 + m];
      }
    }
    const int64_t n_in = n_nodes - n_leaf, M = n_a + n_in;   // leaves: the last level, never stored
    const int32_t *gp = grp, *fa = fac;                   // locals: no member reloads in the loops
    const double *cz = ct + zc * N, *cn = cz + n_a, *Av = A.data();
    double *Lv = L.data(), *wv = w.data(), *wa = wA.data();
    std::copy(Av, Av + n_a, Lv);
    int64_t k = 0;
    for (int64_t g = 0; g < g_leaf; ++g)                  // a group: one parent's children, together
      for (const double Lp = Lv[gp[2 * g]]; k < gp[2 * g + 1]; ++k) Lv[n_a + k] = Lp * Av[fa[k]];
    double sum[8] = {};                                   // eight sums: no add chain over N
    int64_t j = 0;
    for (; j + 7 < M; j += 8)
      for (int u = 0; u < 8; ++u) {
        sum[u] += cz[j + u] * Lv[j + u];
        wv[j + u] = cz[j + u];                            // adjoint seeds: the readout
      }
    for (; j < M; ++j) {
      sum[0] += cz[j] * Lv[j];
      wv[j] = cz[j];
    }
    std::fill(wa, wa + n_a, 0.0);
    double el = 0.0;
    for (int64_t g = g_leaf; g < n_groups; ++g) {         // leaves: energy and both adjoints in one pass
      const int32_t p = gp[2 * g];
      const double Lp = Lv[p];
      double acc = 0.0;
      for (const int64_t e = gp[2 * g + 1]; k < e; ++k) {
        acc += cn[k] * Av[fa[k]];
        wa[fa[k]] += cn[k] * Lp;
      }
      wv[p] += acc;
      el += acc * Lp;
    }
    ei += ((sum[0] + sum[1]) + (sum[2] + sum[3])) + ((sum[4] + sum[5]) + (sum[6] + sum[7])) + el;
    for (int64_t g = g_leaf - 1; g >= 0; --g) {           // children before their parents
      const int32_t p = gp[2 * g];
      const double Lp = Lv[p];
      double acc = 0.0;
      for (k = g ? gp[2 * g - 1] : 0; k < gp[2 * g + 1]; ++k) {
        acc += wv[n_a + k] * Av[fa[k]];
        wa[fa[k]] += wv[n_a + k] * Lp;
      }
      wv[p] += acc;
    }
    for (int64_t a = 0; a < n_a; ++a) wa[a] += wv[a];
    for (int64_t s = 0; s < nl; ++s) {
      const double *Rs = &R[s * F], *dRs = &dR[s * F], *Ys = &Y[s * nY], *dYs = &dY[s * 3 * nY];
      std::fill(q.begin(), q.end(), 0.0);                // q_y = sum_a wA R_r(a), dq_y with R'
      std::fill(dq.begin(), dq.end(), 0.0);
      for (int64_t u = 0; u < n_runs; ++u) {
        const int32_t a0 = run[4 * u], rr = run[4 * u + 1], y0 = run[4 * u + 2], len = run[4 * u + 3];
        const double Rr = Rs[rr], dRr = dRs[rr];
        for (int32_t m = 0; m < len; ++m) {
          q[y0 + m] += wA[a0 + m] * Rr;
          dq[y0 + m] += wA[a0 + m] * dRr;
        }
      }
      double sc = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
      for (int64_t c = n_rnl; c < F; ++c) sc += dRs[c];
      for (int64_t yy = 0; yy < nY; ++yy) {
        sc += dq[yy] * Ys[yy];
        gx += q[yy] * dYs[yy];
        gy += q[yy] * dYs[nY + yy];
        gz += q[yy] * dYs[2 * nY + yy];
      }
      G[3 * s] = sc * hat[s * 3] + gx;
      G[3 * s + 1] = sc * hat[s * 3 + 1] + gy;
      G[3 * s + 2] = sc * hat[s * 3 + 2] + gz;
    }
    return ei;
  }
};

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
  std::vector<double> V(K * 3), G(K * 3);
  std::vector<int32_t> zs(K), js(K);
  std::memset(eo, 0, atoms * sizeof(double));
  std::memset(fo, 0, atoms * 3 * sizeof(double));
  bool broken = false;
  for (int64_t i = 0; i < rows && i < atoms; ++i) {
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
    eo[i] = site.eval(z[i], nl, V.data(), zs.data(), G.data());
    const double wi = wt[i];                              // G is d e_i / dv alone: -sum_i w_i grad e_i
    for (int64_t s = 0; s < nl; ++s)
      for (int c = 0; c < 3; ++c) {
        fo[3 * i + c] += wi * G[3 * s + c];
        fo[3 * js[s] + c] -= wi * G[3 * s + c];
      }
  }
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
// so they run in parallel on XLA's intra-op pool (`max_threads` > 0 caps it).
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
    std::vector<double> V(K * 3), G(K * 3);
    std::vector<int32_t> zs(K), slot(K);
    for (int64_t i = i0; i < i1; ++i) {
      int64_t nl = 0;
      for (int64_t k = 0; k < K; ++k)
        if (live[i * K + k]) {
          std::memcpy(&V[3 * nl], r + 3 * (i * K + k), 3 * sizeof(double));
          zs[nl] = zn[i * K + k];
          slot[nl++] = static_cast<int32_t>(k);
        }
      eo[i] = site.eval(z[i], nl, V.data(), zs.data(), G.data());
      double *gi = go + 3 * i * K;
      std::memset(gi, 0, 3 * K * sizeof(double));
      for (int64_t s = 0; s < nl; ++s) std::memcpy(gi + 3 * slot[s], &G[3 * s], 3 * sizeof(double));
    }
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
