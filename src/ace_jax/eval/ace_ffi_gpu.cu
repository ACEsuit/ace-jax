// Lean linear ACE energies and forces on a GPU, one XLA FFI call, in three passes: each atom's A, its product
// basis, the forces. float64 or float32, one source for CUDA and ROCm. Two entry points: AceDirectGpu on
// lammps-jax's neighbour matrix (positions in, atom energies and forces out), and AceEdgesGpu on the dense edge
// input of EdgeSiteModel.energy_forces_virial_dense (rij, species, slot mask in, site energies and dE_i/drij
// out), which is the same passes with E = true.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>

// Shuffles and ballots work in 32-lane chunks, half a wave on AMD; every lane of the warp calls them.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
#define GPU_SM_COUNT hipDeviceAttributeMultiprocessorCount
#define GPU_WARP_SIZE hipDeviceAttributeWarpSize
#define GPU_SHARED_OPTIN hipDeviceAttributeSharedMemPerBlockOptin
#define GPU_DYNAMIC_SHARED hipFuncAttributeMaxDynamicSharedMemorySize
#define SHFL_XOR(v, o) __shfl_xor(v, o, kChunk)
#define SHFL_UP(v, o) __shfl_up(v, o, kChunk)
#define SHFL_DOWN(v, o) __shfl_down(v, o, kChunk)
#define BALLOT(p) static_cast<unsigned>(__ballot(p) >> (__lane_id() & 32))
using gpuStream_t = hipStream_t;
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
#define GPU_SM_COUNT cudaDevAttrMultiProcessorCount
#define GPU_WARP_SIZE cudaDevAttrWarpSize
#define GPU_SHARED_OPTIN cudaDevAttrMaxSharedMemoryPerBlockOptin
#define GPU_DYNAMIC_SHARED cudaFuncAttributeMaxDynamicSharedMemorySize
#define SHFL_XOR(v, o) __shfl_xor_sync(0xffffffffu, v, o)
#define SHFL_UP(v, o) __shfl_up_sync(0xffffffffu, v, o)
#define SHFL_DOWN(v, o) __shfl_down_sync(0xffffffffu, v, o)
#define BALLOT(p) __ballot_sync(0xffffffffu, p)
using gpuStream_t = cudaStream_t;
#endif

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kThreads = 256;                             // per block of the DAG pass, at most
constexpr int kChunk = 32;                                // lanes per shuffle
constexpr int kAhead = 8;                                 // list steps the DAG pass loads at a time (float32)
constexpr int kShared = 48 << 10;                         // pair-pass shared memory without opt-in
constexpr double kPi = 3.14159265358979323846;

template <typename R>
struct Model {                                            // 32-bit sizes save registers
  const R *tab, *rc, *ct, *E0, *lc;
  const int32_t *run, *grp, *fac, *lev, *pull;
  const int32_t *ai;                                      // the analytic radial (ffi.analytic_plan), nq > 0
  const R *ar;
  int nq, npq;
  int nz, ncoef, F, n_rnl, n_runs, N, n_a, n_lev, n_st, st_lev, n_lc, n, L, solid;
  int local;                                              // the table in radial.spline_local's form
  R x0, ih, rc2;                                          // ih = 1 / h
};

// The dense edge input (E = true): rij (rows, K, 3), neighbour species and live slots (rows, K), and the
// output dE_i / drij (rows, K, 3), zero on dead slots.
template <typename R>
struct Edges {
  const R *rij;
  const int32_t *zj;
  const bool *mask;
  R *g;
};

// radial.spline_eval and, if grad, d/dr, on species pair pz's table; zero past its cutoff.
template <bool grad, typename R>
__device__ void table(const Model<R> &m, int pz, R r, R *v, R *dv, int lane, int lanes) {
  const int F = m.F;
  if (r >= m.rc[pz]) {
    for (int c = lane; c < F; c += lanes) {
      v[c] = 0;
      if (grad) dv[c] = 0;
    }
    return;
  }
  const R k = (r - m.x0) * m.ih + 1, sixth = R(1) / 6;
  int ix = static_cast<int>(floor(k));
  ix = ix < 1 ? 1 : (ix > m.n - 1 ? m.n - 1 : ix);
  const R t = k - ix, s = 1 - t;                          // local: c1 plus the other rows' differences
  const R w[4] = {s * s * s * sixth, m.local ? R(1) : (4 - 6 * t * t + 3 * t * t * t) * sixth,
                  (4 - 6 * s * s + 3 * s * s * s) * sixth, t * t * t * sixth};
  const R dw[4] = {R(-0.5) * s * s * m.ih, m.local ? R(0) : (R(1.5) * t * t - 2 * t) * m.ih,
                   (2 * s - R(1.5) * s * s) * m.ih, R(0.5) * t * t * m.ih};
  const R *c0 = m.tab + (static_cast<int64_t>(pz) * m.ncoef + (m.local ? 4 : 1) * (ix - 1)) * F;
  const R *c1 = c0 + F, *c2 = c1 + F, *c3 = c2 + F;
  for (int c = lane; c < F; c += lanes) {
    v[c] = w[0] * c0[c] + w[1] * c1[c] + w[2] * c2[c] + w[3] * c3[c];
    if (grad) dv[c] = dw[0] * c0[c] + dw[1] * c1[c] + dw[2] * c2[c] + dw[3] * c3[c];
  }
}

// x^p for the exponents of the analytic radial: by products when p is a small integer.
template <typename R>
__device__ __forceinline__ R powi(R x, R p) {
  const int n = static_cast<int>(p);
  if (R(n) != p || n < 0 || n > 16) return pow(x, p);
  R y = 1, b = x;
  for (int k = n; k; k >>= 1, b *= b)
    if (k & 1) y *= b;
  return y;
}

// ace_jax.eval.harmonics of v, |v| = r = 1 / ir, for one m >= 0 and every l from m.
template <typename R>
__device__ void harmonics(int L, bool solid, int m, const R *norm, const R *ca, const R *cb, const R *v, R r,
                          R ir, R *Y, int st = 1) {
  const int W = L + 1;
  const R x = v[0], y = v[1], u = v[2] * ir;
  R A = 1, B = 0, dfact = 1, rad = 1;                     // A + i B = (x + i y)^m
  for (int t = 1; t <= m; ++t) {
    const R A1 = A;
    A = x * A1 - y * B;
    B = x * B + y * A1;
    dfact *= 2 * t - 1;
  }
  if (!solid)
    for (int t = 0; t < m; ++t) rad *= ir;
  R P1 = dfact, P2 = 0;
  for (int l = m; l <= L; ++l) {
    const R P = l == m ? P1 : l == m + 1 ? (2 * m + 1) * u * dfact
                                         : ca[l * W + m] * u * P1 - cb[l * W + m] * P2;
    Y[(l * l + l + m) * st] = norm[l * l + l + m] * P * (rad * A);   // this order stays in float32's range
    if (m) Y[(l * l + l - m) * st] = norm[l * l + l - m] * P * (rad * B);
    if (l > m) P2 = P1;
    P1 = P;
    if (solid) rad *= r;
  }
}

// The harmonics' norms and recurrence ratios and, if gz, their gradients' constants: for
// C_lm = r^l P_lm e^(i m phi), d/dz C_lm = (l + m) C_l-1,m, (d/dx + i d/dy) C_lm = -C_l-1,m+1,
// (d/dx - i d/dy) C_lm = (l + m)(l + m - 1) C_l-1,m-1. Every thread of the block calls it.
template <typename R>
__device__ void constants(int L, R *norm, R *ca, R *cb, R *gz, R *gp, R *gm) {
  const int W = L + 1;
  for (int u = threadIdx.x; u < W * W; u += blockDim.x) {
    const int l = static_cast<int>(sqrt(u + 0.5)), am = abs(u - l * l - l), lw = u / W, mw = u % W;
    const double c = exp(0.5 * (log(2.0 * l + 1) - log(4.0 * kPi) + lgamma(l - am + 1.0) -
                                lgamma(l + am + 1.0)));
    norm[u] = am ? c * sqrt(2.0) : c;
    ca[u] = lw >= mw + 2 ? (2.0 * lw - 1) / (lw - mw) : 0.0;
    cb[u] = lw >= mw + 2 ? (lw + mw - 1.0) / (lw - mw) : 0.0;
  }
  __syncthreads();
  for (int u = threadIdx.x; gz && u < W * W; u += blockDim.x) {
    const int l = u / W, m = u % W, j = (l - 1) * l;      // j: (l - 1, 0)
    const double c = l >= m && l > 0 ? norm[l * l + l + m] : 0.0;
    gz[u] = m < l ? (l + m) * c / norm[j + m] : 0.0;
    gp[u] = m + 1 < l ? (m ? -0.5 : -1.0) * c / norm[j + m + 1] : 0.0;   // m = 0: both conjugates
    gm[u] = m >= 1 && m <= l ? 0.5 * (l + m) * (l + m - 1) * c / norm[j + m - 1] : 0.0;
  }
}

// Pass 1: each row's A, its live pairs for pass 3, and E0 plus the pair energy. A block per row, Team lanes
// per pair, P pairs at a time. E: the pairs are the edge input's live slots.
template <typename R, int T, int Team, bool E>
__global__ void __launch_bounds__(T)
    ace_density(Model<R> m, const R *x, const int32_t *z, const int32_t *nb, const int32_t *cnt, Edges<R> ed,
                int atoms, int K, int rows, R *Ag, int32_t *lists, R *eo, int *broken) {
  constexpr int P = T / Team;
  extern __shared__ double sh[];
  const int na = m.n_a, nY = (m.L + 1) * (m.L + 1), F = m.F, t = threadIdx.x, lane = t % Team;
  const int team = t / Team;
  R *norm = reinterpret_cast<R *>(sh), *ca = norm + nY, *cb = ca + nY, *red = cb + nY, *As = red + T / kChunk;
  R *Rb = As + na, *Yb = Rb + P * F;
  int *ar = reinterpret_cast<int *>(Yb + P * nY), *ay = ar + na, *flag = ay + na, *live = flag + K;
  constants<R>(m.L, norm, ca, cb, nullptr, nullptr, nullptr);
  for (int u = t; u < m.n_runs; u += T)                   // each A entry's radial and harmonic
    for (int q = 0; q < m.run[4 * u + 3]; ++q) {
      ar[m.run[4 * u] + q] = m.run[4 * u + 1];
      ay[m.run[4 * u] + q] = m.run[4 * u + 2] + q;
    }
  for (int i = blockIdx.x; i < rows && i < atoms; i += gridDim.x) {
    for (int k = t; k < K; k += T) {
      int ok = 0;
      if constexpr (E) {
        const int64_t q = static_cast<int64_t>(i) * K + k;
        if (ed.mask[q]) {
          R r2 = 0;
          for (int c = 0; c < 3; ++c) r2 += ed.rij[3 * q + c] * ed.rij[3 * q + c];
          if (isnan(r2)) atomicOr(broken, 1);
          ok = 1;
        }
      } else if (k < cnt[i]) {
        const int j = nb[static_cast<int64_t>(k) * rows + i];
        if (j >= 0 && j < atoms) {
          R r2 = 0;
          for (int c = 0; c < 3; ++c) r2 += (x[3 * j + c] - x[3 * i + c]) * (x[3 * j + c] - x[3 * i + c]);
          if (isnan(r2)) atomicOr(broken, 1);
          ok = r2 <= m.rc2;                               // not a skin pair
        }
      }
      flag[k] = ok;
    }
    for (int a = t; a < na; a += T) As[a] = 0;
    __syncthreads();
    if (t < kChunk) {                                     // list order: sums do not vary
      int nl = 0;
      for (int k0 = 0; k0 < K; k0 += kChunk) {
        const int ok = k0 + t < K && flag[k0 + t];
        const unsigned b = BALLOT(ok);
        if (ok) live[nl + __popc(b & ((1u << t) - 1))] = k0 + t;
        nl += __popc(b);
      }
      if (t == 0) live[K] = nl;
    }
    __syncthreads();
    const int nl = live[K], zc = z[i];
    int32_t *list = lists + static_cast<int64_t>(i) * (K + 1);
    for (int k = t; k < nl; k += T) list[k] = live[k];
    if (t == 0) list[K] = nl;
    R e = 0;
    for (int s0 = 0; s0 < nl; s0 += P) {
      const int s = s0 + team;
      if (s < nl) {
        R v[3];
        int zn;
        if constexpr (E) {
          const int64_t q = static_cast<int64_t>(i) * K + live[s];
          for (int c = 0; c < 3; ++c) v[c] = ed.rij[3 * q + c];
          zn = ed.zj[q];
        } else {
          const int j = nb[static_cast<int64_t>(live[s]) * rows + i];
          for (int c = 0; c < 3; ++c) v[c] = x[3 * j + c] - x[3 * i + c];
          zn = z[j];
        }
        const R r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), ir = 1 / r;
        const int pz = zc * m.nz + zn;
        R *Rs = Rb + team * F;
        table<false, R>(m, pz, r, Rs, nullptr, lane, Team);
        for (int c = lane; c < F; c += Team) e += c >= m.n_rnl ? Rs[c] : 0;   // pair radial, own columns
        for (int mm = lane; mm <= m.L; mm += Team)
          harmonics(m.L, m.solid, mm, norm, ca, cb, v, r, ir, Yb + team * nY);
      }
      __syncthreads();
      for (int a = t; a < na; a += T) {
        R acc = As[a];
        for (int q = 0; q < P && s0 + q < nl; ++q) acc += Rb[q * F + ar[a]] * Yb[q * nY + ay[a]];
        As[a] = acc;
      }
      __syncthreads();
    }
    for (int a = t; a < na; a += T) Ag[static_cast<int64_t>(i) * na + a] = As[a];
    for (int o = kChunk / 2; o > 0; o /= 2) e += SHFL_XOR(e, o);   // a fixed order: energies do not vary
    if (t % kChunk == 0) red[t / kChunk] = e;
    __syncthreads();
    if (t == 0) {
      for (int c = 1; c < T / kChunk; ++c) e += red[c];
      eo[i] = m.E0[zc] + e;
    }
  }
}

// Pass 3 for an analytic tensor radial (ffi.analytic_plan: R_c = env(x) sum_q W_cq P_q(x), radial.py's Agnesi
// transform, polynomial recursion and envelope), a lane per neighbour and a warp per block: nothing is read
// per pair but the pair radial's table rows.  The lane puts its harmonics in its column of Ys and, per degree
// l, runs the recursion once: each P_q times the dE/dA of the radial columns that read it (the plan lists
// them by pair, degree and q; dE/dA is read by all lanes at one address) goes into the degree's 2l + 1 sums
// dE/dY and dE/dY', which stay in registers (`an_degree`, a function per l), then times Y and grad Y.  The
// same sums as ace_forces, in another order.
template <typename R>
struct AnLane {                                           // one neighbour's state
  const R *wA, *Ys, *gz, *gp, *gm, *pA, *pB, *pC, *wv, *qA, *qc;   // qA: the pair radial's recursion, qc its sums
  const int32_t *qp, *ent, *dqp, *dent;                   // dqp, dent, dw: pass 1's entries, per centre species
  const R *dw;
  int zn, nz;
  R x, dx, env, denv, ir, hat[3];
  int nq, npq, L, st, solid;
};

// x(r) and dx/dr of radial.agnesi_normalized and the envelope env_poly2sx(x) with its d/dx (zero unless `in`),
// from the pair's 12 parameters.
template <typename R>
__device__ __forceinline__ void an_radial(const R *par, R r, bool in, AnLane<R> &a) {
  const R p = par[0], q = par[1], aa = par[2], rin = par[3], r0 = par[4], yin = par[5], ycut = par[6];
  const R x1 = par[7], x2 = par[8], p1 = par[9], p2 = par[10], sc = par[11];
  R y = 1;
  a.dx = 0; a.env = 0; a.denv = 0;
  if (r > rin) {
    const R is = 1 / (r0 - rin), s1 = (r - rin) * is, sq = powi(s1, q), sd = powi(s1, q - p);
    const R den = 1 / (1 + sd), g = aa * sq * den, dg = aa * (q * sq - (q - p) * sq * sd * den) * den / s1;
    y = 1 / (1 + g);
    a.dx = -2 * dg * y * y * is / (ycut - yin);
  }
  a.x = -1 + 2 * (y - yin) / (ycut - yin);
  if (a.x <= -1 || a.x >= 1) {
    a.x = a.x <= -1 ? R(-1) : R(1);
    a.dx = 0;
  }
  if (in && a.x > x1 && a.x < x2) {
    const R ea = powi(a.x - x1, p1 - 1), eb = powi(x2 - a.x, p2 - 1);
    a.env = sc * ea * (a.x - x1) * eb * (x2 - a.x);
    a.denv = sc * (p1 * ea * eb * (x2 - a.x) - p2 * ea * (a.x - x1) * eb);
  }
}

// The plan's reals: per pair the tensor radial's 12 parameters, then the pair radial's 12 (its transform's 7,
// the envelope's cutoff, r0 and power, and its kind: 0 env_ace1_poly1sr, 1 env_poly1sr), the two recursions'
// A, B, C, the pair radial's coefficients per pair, the W entries.
template <typename R>
__device__ __forceinline__ void an_setup(const Model<R> &m, AnLane<R> &a) {
  const int np = m.nz * m.nz;
  a.pA = m.ar + 24 * np; a.pB = a.pA + m.nq; a.pC = a.pB + m.nq;
  a.qA = a.pC + m.nq; a.qc = a.qA + 3 * m.npq; a.wv = a.qc + np * m.npq;
  a.ent = m.ai + np * (m.L + 1) * (m.nq + 1);
  const int tot = a.ent[-1];                              // pass 3's entries; then pass 1's lists and weights
  a.dqp = a.ent + tot; a.dent = a.dqp + m.nz * (m.L + 1) * (m.nq + 1); a.dw = a.wv + tot; a.nz = m.nz;
  a.nq = m.nq; a.npq = m.npq; a.L = m.L; a.solid = m.solid;
}

// The folded pair radial V(r) = env(r) sum_q c_q P_q(x(r)) of pair pz and dV/dr, as radial.py (the model's
// pair energy per neighbour): never from the table, whose error the tensor radial's would no longer cancel.
template <typename R>
__device__ __forceinline__ void an_pair(const Model<R> &m, const AnLane<R> &a, int pz, R r, bool has, R &V, R &dV) {
  const R *par = m.ar + 12 * (m.nz * m.nz + pz);
  const R p = par[0], q = par[1], aa = par[2], rin = par[3], r0 = par[4], yin = par[5], ycut = par[6];
  const R rc = par[7], e0 = par[8], ep = par[9];
  V = dV = 0;
  if (!has || !(r < rc)) return;
  R y = 1, dx = 0;
  if (r > rin) {
    const R is = 1 / (r0 - rin), s1 = (r - rin) * is, sq = powi(s1, q), sd = powi(s1, q - p);
    const R den = 1 / (1 + sd), g = aa * sq * den, dg = aa * (q * sq - (q - p) * sq * sd * den) * den / s1;
    y = 1 / (1 + g);
    dx = -2 * dg * y * y * is / (ycut - yin);
  }
  R x = -1 + 2 * (y - yin) / (ycut - yin);
  if (x <= -1 || x >= 1) {
    x = x <= -1 ? R(-1) : R(1);
    dx = 0;
  }
  R env, denv;                                            // in r
  if (par[10] == 0) {                                     // s^-p - sc^-p + p sc^(-p-1) (s - sc), s = r / r0
    const R s = r / e0, sc = rc / e0, sp = 1 / powi(s, ep), scp = 1 / powi(sc, ep);
    env = sp - scp + ep * scp / sc * (s - sc);
    denv = ep * (scp / sc - sp / s) / e0;
  } else {                                                // (s^-p - 1)(1 - s), s = r / rc
    const R s = r / rc, sp = 1 / powi(s, ep);
    env = (sp - 1) * (1 - s);
    denv = (-ep * sp / s * (1 - s) - (sp - 1)) / rc;
  }
  const R *A = a.qA, *B = A + a.npq, *C = B + a.npq, *c = a.qc + pz * a.npq;
  R S = 0, dS = 0, P1 = 0, P2 = 0, D1 = 0, D2 = 0;
  for (int k = 0; k < a.npq; ++k) {
    const R t = A[k] * x + B[k];
    const R P = k == 0 ? A[0] : k == 1 ? t : t * P1 + C[k] * P2;
    const R D = k == 0 ? R(0) : k == 1 ? A[1] : A[k] * P1 + t * D1 + C[k] * D2;
    S += c[k] * P;
    dS += c[k] * D;
    P2 = P1; P1 = P; D2 = D1; D1 = D;
  }
  V = env * S;
  dV = denv * S + env * dS * dx;
}

// Pass 1 for an analytic tensor radial, a lane per neighbour and a warp per block (as ace_forces_lane): per
// degree the recursion once, each env P_q W times the lane's harmonics summed over the warp into the A of the
// radial columns that read it.  The entries are the centre species' (dqp, the same for every lane: the warp sums
// together), their W the lane's neighbour species' (zero where its radial does not read P_q).  The live lists,
// E0 and the pair energy as ace_density.
template <int LL, typename R>
__device__ __forceinline__ void an_density(const AnLane<R> &a, R *As) {
  constexpr int K = 2 * LL + 1;
  const int32_t *qp = a.dqp + LL * (a.nq + 1);
  R P1 = 0, P2 = 0;
  for (int k = 0; k < a.nq && qp[k] < qp[a.nq]; ++k) {
    const R t = a.pA[k] * a.x + a.pB[k];
    const R P = k == 0 ? a.pA[0] : k == 1 ? t : t * P1 + a.pC[k] * P2;
    for (int e = qp[k]; e < qp[k + 1]; ++e) {
      const R wP = a.dw[e * a.nz + a.zn] * a.env * P;
      R s[K];
      for (int m = 0; m < K; ++m) s[m] = wP * a.Ys[(LL * LL + m) * a.st];
      for (int o = kChunk / 2; o > 0; o /= 2)
        for (int m = 0; m < K; ++m) s[m] += SHFL_XOR(s[m], o);
      if (threadIdx.x < K) As[a.dent[e] + threadIdx.x] += s[threadIdx.x];   // lane m adds the sum of m
    }
    P2 = P1; P1 = P;
  }
}

template <typename R, bool E, int LM>
__global__ void __launch_bounds__(kChunk)
    ace_density_lane(Model<R> m, const R *x, const int32_t *z, const int32_t *nb, const int32_t *cnt, Edges<R> ed,
                     int atoms, int K, int rows, R *Ag, int32_t *lists, R *eo, int *broken) {
  constexpr int T = kChunk;
  extern __shared__ double sh[];
  const int na = m.n_a, nY = (m.L + 1) * (m.L + 1), F = m.F, t = threadIdx.x, L = m.L, nq = m.nq;
  const int np = m.nz * m.nz;
  R *norm = reinterpret_cast<R *>(sh), *ca = norm + nY, *cb = ca + nY, *As = cb + nY, *Ys = As + na + t;
  int *flag = reinterpret_cast<int *>(As + na + nY * T), *live = flag + K;
  AnLane<R> a;
  a.Ys = Ys;
  a.st = T;
  an_setup(m, a);
  constants<R>(L, norm, ca, cb, nullptr, nullptr, nullptr);
  for (int i = blockIdx.x; i < rows && i < atoms; i += gridDim.x) {
    for (int k = t; k < K; k += T) {
      int ok = 0;
      if constexpr (E) {
        const int64_t q = static_cast<int64_t>(i) * K + k;
        if (ed.mask[q]) {
          R r2 = 0;
          for (int c = 0; c < 3; ++c) r2 += ed.rij[3 * q + c] * ed.rij[3 * q + c];
          if (isnan(r2)) atomicOr(broken, 1);
          ok = 1;
        }
      } else if (k < cnt[i]) {
        const int j = nb[static_cast<int64_t>(k) * rows + i];
        if (j >= 0 && j < atoms) {
          R r2 = 0;
          for (int c = 0; c < 3; ++c) r2 += (x[3 * j + c] - x[3 * i + c]) * (x[3 * j + c] - x[3 * i + c]);
          if (isnan(r2)) atomicOr(broken, 1);
          ok = r2 <= m.rc2;                               // not a skin pair
        }
      }
      flag[k] = ok;
    }
    for (int q = t; q < na; q += T) As[q] = 0;
    __syncthreads();
    int nl = 0;                                           // list order: sums do not vary
    for (int k0 = 0; k0 < K; k0 += T) {
      const int ok = k0 + t < K && flag[k0 + t];
      const unsigned b = BALLOT(ok);
      if (ok) live[nl + __popc(b & ((1u << t) - 1))] = k0 + t;
      nl += __popc(b);
    }
    __syncthreads();
    const int zc = z[i];
    int32_t *list = lists + static_cast<int64_t>(i) * (K + 1);
    for (int k = t; k < nl; k += T) list[k] = live[k];
    if (t == 0) list[K] = nl;
    R e = 0;
    const int32_t *dq0 = a.dqp;
    a.dqp = dq0 + zc * (L + 1) * (nq + 1);
    for (int s0 = 0; s0 < nl; s0 += T) {
      const int s = s0 + t;
      const bool has = s < nl;
      R v[3] = {1, 0, 0};
      int zn = 0;
      if (has) {
        if constexpr (E) {
          const int64_t q = static_cast<int64_t>(i) * K + live[s];
          for (int c = 0; c < 3; ++c) v[c] = ed.rij[3 * q + c];
          zn = ed.zj[q];
        } else {
          const int j = nb[static_cast<int64_t>(live[s]) * rows + i];
          for (int c = 0; c < 3; ++c) v[c] = x[3 * j + c] - x[3 * i + c];
          zn = z[j];
        }
      }
      const R r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), ir = 1 / r;
      const int pz = zc * m.nz + zn;
      const bool in = has && r < m.rc[pz];
      a.zn = zn;
      an_radial(m.ar + 12 * pz, r, in, a);
      R V, dV;
      an_pair(m, a, pz, r, has, V, dV);
      e += V;
      for (int mm = 0; mm <= L; ++mm) harmonics(L, m.solid, mm, norm, ca, cb, v, r, ir, Ys, T);
      an_density<0, R>(a, As);
      if constexpr (LM >= 1) an_density<1, R>(a, As);
      if constexpr (LM >= 2) an_density<2, R>(a, As);
      if constexpr (LM >= 3) an_density<3, R>(a, As);
      if constexpr (LM >= 4) an_density<4, R>(a, As);
      if constexpr (LM >= 5) an_density<5, R>(a, As);
      if constexpr (LM >= 6) an_density<6, R>(a, As);
      if constexpr (LM >= 7) an_density<7, R>(a, As);
      if constexpr (LM >= 8) an_density<8, R>(a, As);
    }
    a.dqp = dq0;
    __syncthreads();
    for (int q = t; q < na; q += T) Ag[static_cast<int64_t>(i) * na + q] = As[q];
    for (int o = kChunk / 2; o > 0; o /= 2) e += SHFL_XOR(e, o);   // a fixed order: energies do not vary
    if (t == 0) eo[i] = m.E0[zc] + e;
    __syncthreads();
  }
}

template <int LL, typename R>
__device__ __forceinline__ void an_degree(const AnLane<R> &a, R &sc, R *g) {
  constexpr int K = 2 * LL + 1;
  R U[K], V[K];
  for (int m = 0; m < K; ++m) U[m] = V[m] = 0;
  const int32_t *qp = a.qp + LL * (a.nq + 1);
  R P1 = 0, P2 = 0, D1 = 0, D2 = 0;                       // P_q-1, P_q-2 and their d/dx
  for (int k = 0; k < a.nq && qp[k] < qp[a.nq]; ++k) {    // up to the degree's last polynomial
    const R t = a.pA[k] * a.x + a.pB[k];
    const R P = k == 0 ? a.pA[0] : k == 1 ? t : t * P1 + a.pC[k] * P2;
    const R D = k == 0 ? R(0) : k == 1 ? a.pA[1] : a.pA[k] * P1 + t * D1 + a.pC[k] * D2;
    for (int e = qp[k]; e < qp[k + 1]; ++e) {
      const R *w = a.wA + a.ent[e];
      const R wP = a.wv[e] * P, wD = a.wv[e] * D;
      for (int m = 0; m < K; ++m) {
        U[m] += w[m] * wP;
        V[m] += w[m] * wD;
      }
    }
    P2 = P1; P1 = P; D2 = D1; D1 = D;
  }
  const int st = a.st, W = a.L + 1, b = (LL - 1) * LL;    // grad Y from degree l - 1, b: (l - 1, 0)
  for (int m = 0; m < K; ++m) {
    const int y = LL * LL + m;
    const R u = a.env * U[m];                             // R = env S, dR/dr = x' (env' S + env S')
    sc += a.dx * (a.denv * U[m] + a.env * V[m]) * a.Ys[y * st];
    if (LL == 0) continue;
    const int sg = m < LL ? -1 : 1, am = sg * (m - LL), cu = LL * W + am;
    const bool up = am + 1 < LL, dn = am > 0;
    const R sp = up ? a.Ys[(b + sg * (am + 1)) * st] : 0, op = up ? a.Ys[(b - sg * (am + 1)) * st] : 0;
    const R sm = dn && (sg > 0 || am > 1) ? a.Ys[(b + sg * (am - 1)) * st] : 0;   // no sine at m = 0
    const R om = dn && (sg < 0 || am > 1) ? a.Ys[(b - sg * (am - 1)) * st] : 0;
    const R gy[3] = {a.gp[cu] * sp + a.gm[cu] * sm, sg * (a.gp[cu] * op - a.gm[cu] * om),
                     am < LL ? a.gz[cu] * a.Ys[(b + sg * am) * st] : 0};
    for (int c = 0; c < 3; ++c)
      g[c] += u * (a.solid ? gy[c] : (gy[c] - LL * a.hat[c] * a.Ys[y * st]) * a.ir);
  }
}

constexpr int kAnL = 8;                                   // the degrees an_degree is made for
constexpr int kAnDensity = 64;                            // A entries up to which pass 1 takes the analytic radial

template <typename R, bool E, int LM>                   // LM: the model's lmax (its registers are the degree's sums)
__global__ void __launch_bounds__(kChunk)
    ace_forces_lane(Model<R> m, const R *x, const int32_t *z, const int32_t *nb, const R *wt, Edges<R> ed,
                    int atoms, int K, int rows, const int32_t *lists, const R *wAg, const int *broken, R *eo,
                    R *fo) {
  constexpr int T = kChunk;
  const int64_t all = static_cast<int64_t>(gridDim.x) * T;
  if (*broken) {
    const int64_t nf = E ? 3ll * K * rows : 3ll * atoms;
    for (int64_t u = blockIdx.x * static_cast<int64_t>(T) + threadIdx.x; u < atoms + nf; u += all)
      (u < atoms ? eo[u] : (E ? ed.g : fo)[u - atoms]) = NAN;
    return;
  }
  extern __shared__ double sh[];
  const int na = m.n_a, nY = (m.L + 1) * (m.L + 1), F = m.F, lane = threadIdx.x, L = m.L, nq = m.nq;
  const int np = m.nz * m.nz;
  R *norm = reinterpret_cast<R *>(sh), *ca = norm + nY, *cb = ca + nY, *gz = cb + nY, *gp = gz + nY;
  R *gm = gp + nY, *wA = gm + nY, *Ys = wA + na + lane;   // Ys: the lane's column
  AnLane<R> a;
  a.wA = wA; a.Ys = Ys; a.gz = gz; a.gp = gp; a.gm = gm;
  a.st = T;
  an_setup(m, a);
  constants(L, norm, ca, cb, gz, gp, gm);
  for (int i = blockIdx.x; i < rows && i < atoms; i += gridDim.x) {
    const int32_t *list = lists + static_cast<int64_t>(i) * (K + 1);
    const int nl = list[K], zc = z[i];
    const R wi = E ? R(1) : wt[i];
    for (int q = lane; q < na; q += T) wA[q] = wAg[static_cast<int64_t>(i) * na + q];
    __syncthreads();
    R fi[3] = {0, 0, 0};
    for (int s0 = 0; s0 < nl; s0 += T) {
      const int s = s0 + lane;
      const bool has = s < nl;
      R v[3] = {1, 0, 0};
      int j = 0, zn = 0, slot = 0;
      if (has) {
        slot = list[s];
        if constexpr (E) {
          const int64_t q = static_cast<int64_t>(i) * K + slot;
          for (int c = 0; c < 3; ++c) v[c] = ed.rij[3 * q + c];
          zn = ed.zj[q];
        } else {
          j = nb[static_cast<int64_t>(slot) * rows + i];
          for (int c = 0; c < 3; ++c) v[c] = x[3 * j + c] - x[3 * i + c];
          zn = z[j];
        }
      }
      const R r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), ir = 1 / r;
      const int pz = zc * m.nz + zn;
      const bool in = has && r < m.rc[pz];
      a.ir = ir;
      for (int c = 0; c < 3; ++c) a.hat[c] = v[c] * ir;
      a.qp = m.ai + pz * (L + 1) * (nq + 1);
      an_radial(m.ar + 12 * pz, r, in, a);
      for (int mm = 0; mm <= L; ++mm) harmonics(L, m.solid, mm, norm, ca, cb, v, r, ir, Ys, T);
      R sc = 0, g[3] = {0, 0, 0}, V;
      an_pair(m, a, pz, r, has, V, sc);
      an_degree<0, R>(a, sc, g);
      if constexpr (LM >= 1) an_degree<1, R>(a, sc, g);
      if constexpr (LM >= 2) an_degree<2, R>(a, sc, g);
      if constexpr (LM >= 3) an_degree<3, R>(a, sc, g);
      if constexpr (LM >= 4) an_degree<4, R>(a, sc, g);
      if constexpr (LM >= 5) an_degree<5, R>(a, sc, g);
      if constexpr (LM >= 6) an_degree<6, R>(a, sc, g);
      if constexpr (LM >= 7) an_degree<7, R>(a, sc, g);
      if constexpr (LM >= 8) an_degree<8, R>(a, sc, g);
      if (has)
        for (int c = 0; c < 3; ++c) {
          const R fc = wi * (sc * a.hat[c] + g[c]);
          if constexpr (E) {
            ed.g[3 * (static_cast<int64_t>(i) * K + slot) + c] = fc;
          } else {
            fi[c] += fc;
            atomicAdd(&fo[3 * j + c], -fc);
          }
        }
    }
    if constexpr (!E) {
      for (int o = kChunk / 2; o > 0; o /= 2)
        for (int c = 0; c < 3; ++c) fi[c] += SHFL_XOR(fi[c], o);
      if (lane < 3 && nl > 0) atomicAdd(&fo[3 * i + lane], fi[lane]);
    }
    __syncthreads();                                      // wA is the next row's
  }
}

// Rows in batches of GA of one centre species for pass 2: perm (batches, GA), -1 where a species' last batch
// is short. cnt (nz,): each species' rows, then its next slot.
__global__ void ace_count(const int32_t *z, int n_rows, int *cnt) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows; i += gridDim.x * blockDim.x)
    atomicAdd(&cnt[z[i]], 1);
}

__global__ void ace_base(int nz, int GA, int *cnt) {
  for (int zc = 0, b = 0; zc < nz; ++zc) {
    const int c = cnt[zc];
    cnt[zc] = b;
    b += (c + GA - 1) / GA * GA;
  }
}

__global__ void ace_fill(const int32_t *z, int n_rows, int *cnt, int32_t *perm) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows; i += gridDim.x * blockDim.x)
    perm[atomicAdd(&cnt[z[i]], 1)] = i;
}

// Pass 2: the energy and dE/dA from the rows' A, a block per batch of GA rows of one species: a thread reads
// each term of ffi.pull_plan's lists once and applies it to the batch's rows, whose values lie side by side
// (index GA + row). Every sum is by the thread that owns its target (no atomic adds, so deterministic).
// Forward, the kept levels' values Lv (A, the kept nodes, then 1), by Mode, a template parameter (a pointer
// that may be shared or global makes every load a generic one): 0, with at most one kept level, none but A
// and the 1, a kept node's value being its two A factors, which halves the shared memory a row takes, so
// more rows are in flight; 1 in shared memory; 2 in scratch. Then the adjoints by parent,
// deepest level first: w_p = c_p + sum of w_k A[f_k] over p's children, the leaves' w their coefficients;
// the energy is sum_a A_a w_a. And dE/dA_f = w_f + sum of w_k L_parent(k) over the nodes with factor f. The
// leaves' parents go in tiles, whose adjoints wms are in shared memory: the leaves into wms, then (the mids)
// wms into the kept adjoints wv, in scratch. Every node's dE/dA term goes to a piece slot in pw: per list
// (a level, or a tile's mids) a thread takes an equal range of the nodes sorted by factor and writes each
// run's sum. The last inner level's values, the mids, are not kept: Lv[x] Lv[y]
// where a leaf needs one.
//
// R is the pass's own type and S the rows' (A, dE/dA, the energies): the same, or (mixed precision, the
// readout passed in float32) R float and S double. Mixed, the energy is sum_a A_a w_a in double from the
// rows' A and the float adjoints: good to ~1e-6 eV/atom, with noise from the float node values.
template <typename R, typename S, int GA, int Mode>
__global__ void __launch_bounds__(kThreads)
    ace_dag(Model<R> m, const int32_t *z, const int32_t *perm, int n_batches, int M, const S *Ag, S *wAg,
            R *scratch, int64_t per_batch, S *eo) {
  extern __shared__ double sh[];
  const int T = blockDim.x, W = T + 1, C = T / kChunk;   // ffi.pull_threads: the plan is for T
  constexpr int kAhead = sizeof(R) == 8 ? ::kAhead / 2 : ::kAhead;   // A100: float64 is slower at 8
  const int na = m.n_a, nL = m.n_a + m.n_st + 1, mids = m.n_lev > 1, t = threadIdx.x;
  const int nV = Mode ? nL : na + 1;                      // values kept
  const int32_t *P = m.pull, *tiles = P + P[3], *lcut = P + P[4], *litems = P + P[5], *ib = P + P[6];
  const int32_t *ist = P + P[7], *pst = P + P[8], *fst = P + P[9], *items = P + P[10];
  const int n_tiles = P[0], n_slots = P[1], n_lists = P[2];
  S *red = reinterpret_cast<S *>(sh);
  R *mine = reinterpret_cast<R *>(red + C * GA), *off = scratch + blockIdx.x * per_batch;
  R *Lv = Mode < 2 ? mine : off, *wms = mine + (Mode < 2 ? nV * GA : 0);
  R *wv = off + (Mode < 2 ? 0 : nL * GA), *pw = wv + nL * GA, *A = Lv;
  for (int b = blockIdx.x; b < n_batches; b += gridDim.x) {
    int i[GA];
    for (int g = 0; g < GA; ++g) i[g] = perm[b * GA + g];
    if (i[0] < 0) continue;                               // past the last batch
    const R *cz = m.ct + z[i[0]] * m.N, *lcz = m.lc + z[i[0]] * m.n_lc;
    for (int a = t; a < na; a += T)
      for (int g = 0; g < GA; ++g) A[a * GA + g] = i[g] >= 0 ? Ag[static_cast<int64_t>(i[g]) * na + a] : 0;
    if (t < GA) Lv[(nV - 1) * GA + t] = 1;
    __syncthreads();
    for (int l = 0; Mode && l < m.st_lev; ++l) {          // the kept levels
      for (int q = m.lev[l] + t; q < m.lev[l + 1]; q += T) {
        R Lp[GA];
        for (int g = 0; g < GA; ++g) Lp[g] = Lv[m.grp[2 * q] * GA + g];
        for (int k = q ? m.grp[2 * q - 1] : 0; k < m.grp[2 * q + 1]; ++k) {
          const int f = m.fac[k];
          for (int g = 0; g < GA; ++g) Lv[(na + k) * GA + g] = Lp[g] * A[f * GA + g];
        }
      }
      __syncthreads();
    }
    for (int j = t; j < nL - 1; j += T)
      for (int g = 0; g < GA; ++g) wv[j * GA + g] = cz[j];
    for (int ti = 0; ti < n_tiles; ++ti) {
      const int32_t *tl = tiles + 5 * ti, *lc = lcut + ti * W;
      const int u0 = tl[0], nt = tl[1];
      const R *c0 = cz + (mids ? na + m.n_st : 0) + u0;
      for (int j = t; j < nt; j += T)
        for (int g = 0; g < GA; ++g) wms[j * GA + g] = c0[j];
      __syncthreads();
      {                                                   // the leaves into their parents
        const int32_t *it = litems + tl[2];               // an item: factor, parent << 15, the top bit on
        const R *c = lcz + tl[2];                         // the parent's last
        R acc[GA] = {};
        for (int o = t, end = t + T * (lc[t + 1] - lc[t]); o < end; o += kAhead * T) {
          int vs[kAhead];                                 // the list is read kAhead steps at a time: the
          R cs[kAhead];                                   // pass waits on these loads
          for (int u = 0; u < kAhead; ++u) {
            const bool on = o + u * T < end;
            vs[u] = on ? it[o + u * T] : 0;
            cs[u] = on ? c[o + u * T] : 0;
          }
          for (int u = 0; u < kAhead && o + u * T < end; ++u) {
            const int v = vs[u], f = v & 32767;
            for (int g = 0; g < GA; ++g) acc[g] += cs[u] * A[f * GA + g];
            if (v < 0) {
              const int j = (v >> 15) & 65535;
              for (int g = 0; g < GA; ++g) {
                wms[j * GA + g] += acc[g];
                acc[g] = 0;
              }
            }
          }
        }
      }
      __syncthreads();
      if (mids) {
        for (int q = tl[3] + t; q < tl[4]; q += T) {      // the mids into theirs
          R acc[GA] = {};
          for (int k = q ? m.grp[2 * q - 1] : 0; k < m.grp[2 * q + 1]; ++k) {
            const int f = m.fac[k], j = k - m.n_st - u0;
            for (int g = 0; g < GA; ++g) acc[g] += wms[j * GA + g] * A[f * GA + g];
          }
          for (int g = 0; g < GA; ++g) wv[m.grp[2 * q] * GA + g] += acc[g];
        }
        const int L = m.n_lev + ti;                       // and into dE/dA's pieces
        const int32_t *it = items + ib[L];
        int pc = pst[L * W + t];
        R acc[GA] = {};
        const int end = t + T * (ist[L * W + t + 1] - ist[L * W + t]);
        if constexpr (Mode == 0) {                        // an item: 10 bits for each A factor, then the mid
          for (int o = t; o < end; o += kAhead * T) {
            int vs[kAhead];
            for (int u = 0; u < kAhead; ++u) vs[u] = o + u * T < end ? it[o + u * T] : 0;
            for (int u = 0; u < kAhead && o + u * T < end; ++u) {
              const int v = vs[u], y0 = v & 1023, y1 = (v >> 10) & 1023, x = (v >> 20) & 2047;
              for (int g = 0; g < GA; ++g) acc[g] += wms[x * GA + g] * (A[y0 * GA + g] * A[y1 * GA + g]);
              if (v < 0) {
                for (int g = 0; g < GA; ++g) {
                  pw[pc * GA + g] = acc[g];
                  acc[g] = 0;
                }
                ++pc;
              }
            }
          }
        } else {
          for (int o = t; o < end; o += T) {
            const int x = it[2 * o], y = it[2 * o + 1], yy = y < 0 ? ~y : y;
            for (int g = 0; g < GA; ++g) acc[g] += wms[x * GA + g] * Lv[yy * GA + g];
            if (y < 0) {
              for (int g = 0; g < GA; ++g) {
                pw[pc * GA + g] = acc[g];
                acc[g] = 0;
              }
              ++pc;
            }
          }
        }
      } else {
        for (int j = t; j < nt; j += T)
          for (int g = 0; g < GA; ++g) wv[(u0 + j) * GA + g] = wms[j * GA + g];
      }
      __syncthreads();
    }
    for (int l = m.st_lev - 1; l >= 0; --l) {             // the kept levels into theirs
      for (int q = m.lev[l] + t; q < m.lev[l + 1]; q += T) {
        R acc[GA] = {};
        for (int k = q ? m.grp[2 * q - 1] : 0; k < m.grp[2 * q + 1]; ++k) {
          const int f = m.fac[k];
          for (int g = 0; g < GA; ++g) acc[g] += wv[(na + k) * GA + g] * A[f * GA + g];
        }
        for (int g = 0; g < GA; ++g) wv[m.grp[2 * q] * GA + g] += acc[g];
      }
      __syncthreads();
    }
    S e[GA] = {};
    for (int a = t; a < na; a += T)
      for (int g = 0; g < GA; ++g) {
        if constexpr (sizeof(S) == sizeof(R)) e[g] += A[a * GA + g] * wv[a * GA + g];
        else if (i[g] >= 0) e[g] += Ag[static_cast<int64_t>(i[g]) * na + a] * static_cast<S>(wv[a * GA + g]);
      }
    for (int g = 0; g < GA; ++g)
      for (int o = kChunk / 2; o > 0; o /= 2) e[g] += SHFL_XOR(e[g], o);
    for (int l = 0; l < m.n_lev; ++l) {                   // dE/dA: the pieces
      const bool leaf = l == m.n_lev - 1;
      const R *w = lcz + n_slots;
      const int32_t *it = items + ib[l];
      int pc = pst[l * W + t];
      R acc[GA] = {};
      const int end = t + T * (ist[l * W + t + 1] - ist[l * W + t]);
      if constexpr (Mode == 0) {                          // an item: a leaf's three A factors, 10 bits each,
        for (int o = t; o < end; o += kAhead * T) {       // or a kept node's parent, then the node
          int vs[kAhead];
          R cs[kAhead];
          for (int u = 0; u < kAhead; ++u) {
            const bool on = o + u * T < end;
            vs[u] = on ? it[o + u * T] : 0;
            cs[u] = on && leaf ? w[o + u * T] : 0;
          }
          for (int u = 0; u < kAhead && o + u * T < end; ++u) {
            const int v = vs[u];
            if (leaf) {
              const int x0 = v & 1023, x1 = (v >> 10) & 1023, y = (v >> 20) & 1023;
              for (int g = 0; g < GA; ++g) acc[g] += cs[u] * (A[x0 * GA + g] * A[x1 * GA + g] * A[y * GA + g]);
            } else {
              const int y = v & 1023, x = (v >> 10) & 0x1fffff;
              for (int g = 0; g < GA; ++g) acc[g] += wv[x * GA + g] * A[y * GA + g];
            }
            if (v < 0) {
              for (int g = 0; g < GA; ++g) {
                pw[pc * GA + g] = acc[g];
                acc[g] = 0;
              }
              ++pc;
            }
          }
        }
      } else {
        for (int o = t; o < end; o += T) {
          const int x = it[2 * o], y = it[2 * o + 1], yy = y < 0 ? ~y : y;
          if (leaf) {
            const R co = w[o];
            for (int g = 0; g < GA; ++g) acc[g] += co * (Lv[x * GA + g] * Lv[yy * GA + g]);
          } else {
            for (int g = 0; g < GA; ++g) acc[g] += wv[x * GA + g] * Lv[yy * GA + g];
          }
          if (y < 0) {
            for (int g = 0; g < GA; ++g) {
              pw[pc * GA + g] = acc[g];
              acc[g] = 0;
            }
            ++pc;
          }
        }
      }
    }
    __syncthreads();
    for (int a = t; a < na; a += T) {
      R w[GA];
      for (int g = 0; g < GA; ++g) w[g] = wv[a * GA + g];
      for (int l = 0; l < n_lists; ++l)
        for (int pc = fst[l * (na + 1) + a]; pc < fst[l * (na + 1) + a + 1]; ++pc)
          for (int g = 0; g < GA; ++g) w[g] += pw[pc * GA + g];
      for (int g = 0; g < GA; ++g)
        if (i[g] >= 0) wAg[static_cast<int64_t>(i[g]) * na + a] = w[g];
    }
    if (t % kChunk == 0)
      for (int g = 0; g < GA; ++g) red[t / kChunk * GA + g] = e[g];
    __syncthreads();
    if (t < GA && i[t] >= 0) {                            // thread g sums row g's
      S s = 0;
      for (int c = 0; c < C; ++c) s += red[c * GA + t];
      eo[i[t]] += s;
    }
    __syncthreads();
  }
}

// Pass 2 in coupled form (eval/coupled.py; ffi.coupled_gpu_plan's streams P and their coefficients cw, per
// centre species): z = [A; X] and g = W z of a batch of GA rows of one species, a row's values side by side
// (index GA + row).  A thread runs its own list of each stream, a step an item, kAhead steps loaded before
// they are used (the lists' load latency); every target has one owner (no atomic adds, so deterministic).
//   X:      acc += c z_a z_b                       -> z_dst
//   rows:   acc_m += w z_(v + m), m < NC           -> g_(dst + m), and the energy acc_m z_(u + m) / 2
//   dE/dA:  acc += c g_x z_a                       -> g_dst
// A long task is in pieces: the later ones write piece slots, which cp_fix adds to their target.  Shared
// memory holds z, g of the A part and the pieces; g of the X part goes to the block's scratch and, after the
// rows, over z's X part, which dE/dA then reads.
template <typename R, int GA>
struct CpMem {
  R *Z, *gA, *pc, *gX;                                    // z; g of A; piece slots; g of X (scratch)
  int na, ntot;
  __device__ __forceinline__ R *target(int dst) const {
    return dst < na ? gA + dst * GA : dst < ntot ? gX + (dst - na) * GA : pc + (dst - ntot) * GA;
  }
};

template <typename R, int GA, bool X>
__device__ __forceinline__ void cp_bilinear(const int32_t *P, const int32_t *st, const R *cz, int T, int t,
                                           const CpMem<R, GA> &m) {
  constexpr int AH = sizeof(R) == 8 ? ::kAhead / 2 : ::kAhead;
  const int32_t *it = P + st[2] + t, *hd = P + st[3] + t;
  const R *co = cz + st[4] + t, *Z = m.Z;
  R acc[GA];
  for (int g = 0; g < GA; ++g) acc[g] = 0;
  for (int j = 0, steps = st[0]; j < steps; j += AH) {    // steps: a multiple of 8
    uint32_t v[AH];
    R c[AH];
    for (int a = 0; a < AH; ++a) {
      v[a] = static_cast<uint32_t>(it[(j + a) * T]);
      c[a] = co[(j + a) * T];
    }
    for (int a = 0; a < AH; ++a) {
      const int lo = (v[a] & 0x7fff) * GA, hi = ((v[a] >> 15) & 0xffff) * GA;
      for (int g = 0; g < GA; ++g) acc[g] += c[a] * (Z[lo + g] * Z[hi + g]);
      if (v[a] >> 31) {
        const int dst = static_cast<uint32_t>(*hd) & 0xffff;
        hd += T;
        R *d = X ? m.Z + dst * GA : m.target(dst);
        for (int g = 0; g < GA; ++g) {
          d[g] = X || dst >= m.ntot ? acc[g] : d[g] + acc[g];
          acc[g] = 0;
        }
      }
    }
  }
}

template <typename R, typename S, int GA, int NC>
__device__ __forceinline__ void cp_rows(const int32_t *P, const int32_t *st, const R *cz, int T, int t,
                                       const CpMem<R, GA> &m, S *e) {
  constexpr int AH = sizeof(R) == 8 ? ::kAhead / 2 : ::kAhead;
  const int32_t *it = P + st[2] + t, *hd = P + st[3] + t;
  const R *co = cz + st[4] + t, *Z = m.Z;
  R acc[NC * GA];
  for (int q = 0; q < NC * GA; ++q) acc[q] = 0;
  for (int j = 0, steps = st[0]; j < steps; j += AH) {
    uint32_t v[AH];
    R w[AH];
    for (int a = 0; a < AH; ++a) {
      v[a] = static_cast<uint32_t>(it[(j + a) * T]);
      w[a] = co[(j + a) * T];
    }
    for (int a = 0; a < AH; ++a) {
      const R *zv = Z + (v[a] & 0x7fff) * GA;
      for (int q = 0; q < NC * GA; ++q) acc[q] += w[a] * zv[q];
      if (v[a] >> 31) {
        const uint32_t h = static_cast<uint32_t>(*hd);
        hd += T;
        R *gd = m.target(h & 0xffff);
        const R *zu = Z + (h >> 16) * GA;
        for (int q = 0; q < NC * GA; ++q) {
          gd[q] = acc[q];
          e[q % GA] += static_cast<S>(acc[q]) * static_cast<S>(zu[q]);
          acc[q] = 0;
        }
      }
    }
  }
}

template <typename R, int GA>
__device__ __forceinline__ void cp_fix(const int32_t *P, const int32_t *fx, int T, int t, const CpMem<R, GA> &m) {
  for (int k = t; k < fx[0]; k += T) {
    const uint32_t w0 = static_cast<uint32_t>(P[fx[1] + 2 * k]), w1 = static_cast<uint32_t>(P[fx[1] + 2 * k + 1]);
    const int cnt = w0 >> 16, stride = w1 >> 16;
    R *d = m.target(w0 & 0xffff);
    const R *p = m.target(w1 & 0xffff);
    for (int g = 0; g < GA; ++g) {
      R s = d[g];
      for (int q = 0; q < cnt; ++q) s += p[q * stride * GA + g];
      d[g] = s;
    }
  }
}

template <typename R, typename S, int GA>
__global__ void __launch_bounds__(kThreads)
    ace_coupled(const int32_t *P, const R *cw, int n_real, const int32_t *z, const int32_t *perm, int n_batches,
                const S *Ag, S *wAg, R *scratch, int64_t per_block, S *eo) {
  extern __shared__ double sh[];
  const int T = blockDim.x, t = threadIdx.x, C = T / kChunk;
  const int na = P[1], nx = P[2], ntot = na + nx, ns = P[4];
  const int32_t *st = P + 5, *fx = st + 5 * ns;
  S *red = reinterpret_cast<S *>(sh);
  CpMem<R, GA> m;
  m.Z = reinterpret_cast<R *>(red + C * GA);
  m.gA = m.Z + ntot * GA;
  m.pc = m.gA + na * GA;
  m.gX = scratch + blockIdx.x * per_block;
  m.na = na;
  m.ntot = ntot;
  for (int b = blockIdx.x; b < n_batches; b += gridDim.x) {
    int i[GA];
    for (int g = 0; g < GA; ++g) i[g] = perm[b * GA + g];
    if (i[0] < 0) continue;                               // past the last batch
    const R *cz = cw + static_cast<int64_t>(z[i[0]]) * n_real;
    for (int a = t; a < na; a += T)
      for (int g = 0; g < GA; ++g) {
        m.Z[a * GA + g] = i[g] >= 0 ? Ag[static_cast<int64_t>(i[g]) * na + a] : 0;
        m.gA[a * GA + g] = 0;                             // an A block without a weight has no row
      }
    __syncthreads();
    cp_bilinear<R, GA, true>(P, st, cz, T, t, m);
    __syncthreads();
    S e[GA];
    for (int g = 0; g < GA; ++g) e[g] = 0;
    cp_rows<R, S, GA, 1>(P, st + 5, cz, T, t, m, e);
    cp_rows<R, S, GA, 2>(P, st + 10, cz, T, t, m, e);
    cp_rows<R, S, GA, 3>(P, st + 15, cz, T, t, m, e);
    cp_rows<R, S, GA, 4>(P, st + 20, cz, T, t, m, e);
    __syncthreads();
    cp_fix<R, GA>(P, fx, T, t, m);
    __syncthreads();
    for (int k = t; k < nx * GA; k += T) m.Z[na * GA + k] = m.gX[k];   // g of X over X
    __syncthreads();
    cp_bilinear<R, GA, false>(P, st + 25, cz, T, t, m);
    __syncthreads();
    cp_fix<R, GA>(P, fx + 2, T, t, m);
    __syncthreads();
    for (int g = 0; g < GA; ++g) e[g] *= S(0.5);
    for (int a = t; a < na; a += T)
      for (int g = 0; g < GA; ++g)
        if (i[g] >= 0) {
          const S la = cz[a], Aa = Ag[static_cast<int64_t>(i[g]) * na + a];
          wAg[static_cast<int64_t>(i[g]) * na + a] = static_cast<S>(m.gA[a * GA + g]) + la;
          e[g] += la * Aa;
        }
    for (int o = kChunk / 2; o > 0; o /= 2)
      for (int g = 0; g < GA; ++g) e[g] += SHFL_XOR(e[g], o);
    if (t % kChunk == 0)
      for (int g = 0; g < GA; ++g) red[t / kChunk * GA + g] = e[g];
    __syncthreads();
    if (t < GA && i[t] >= 0) {                            // thread g sums row g's
      S s = 0;
      for (int c = 0; c < C; ++c) s += red[c * GA + t];
      eo[i[t]] += s;
    }
    __syncthreads();
  }
}

// Pass 3: the forces, -sum_i w_i grad e_i, or NaN everywhere after a NaN position. A block per row, Team
// lanes per pair; a lane per harmonic takes dE/dY over its A entries, times grad Y. E: dE_i / drij
// per live slot instead, written once each (no atomics, so deterministic), no weights.
template <typename R, int T, int Team, bool E>
__global__ void __launch_bounds__(T)
    ace_forces(Model<R> m, const R *x, const int32_t *z, const int32_t *nb, const R *wt, Edges<R> ed,
               int atoms, int K, int rows, const int32_t *lists, const R *wAg, const int *broken, R *eo,
               R *fo) {
  constexpr int P = T / Team;
  const int64_t all = static_cast<int64_t>(gridDim.x) * T;
  if (*broken) {
    const int64_t nf = E ? 3ll * K * rows : 3ll * atoms;
    for (int64_t u = blockIdx.x * static_cast<int64_t>(T) + threadIdx.x; u < atoms + nf; u += all)
      (u < atoms ? eo[u] : (E ? ed.g : fo)[u - atoms]) = NAN;
    return;
  }
  extern __shared__ double sh[];
  const int na = m.n_a, nY = (m.L + 1) * (m.L + 1), F = m.F, t = threadIdx.x, lane = t % Team;
  const int team = t / Team, per_team = 2 * F + nY + 4;
  R *norm = reinterpret_cast<R *>(sh), *ca = norm + nY, *cb = ca + nY, *gz = cb + nY, *gp = gz + nY;
  R *gm = gp + nY, *red = gm + nY, *wA = red + 3 * (T / kChunk), *Rs = wA + na + team * per_team;
  R *dRs = Rs + F, *Ys = dRs + F, *hat = Ys + nY;         // hat: rhat, then 1 / r
  int *ys = reinterpret_cast<int *>(wA + na + P * per_team), *yr = ys + nY + 1, *ya = yr + na;
  int *live = ya + na;
  constants(m.L, norm, ca, cb, gz, gp, gm);
  // The A entries by harmonic: y's are ys[y] to ys[y + 1], radial yr and entry ya, in the runs' order.
  for (int y = t; y < nY; y += T) {
    int c = 0;
    for (int u = 0; u < m.n_runs; ++u) c += y >= m.run[4 * u + 2] && y < m.run[4 * u + 2] + m.run[4 * u + 3];
    ys[y + 1] = c;
  }
  __syncthreads();
  if (t == 0) {
    ys[0] = 0;
    for (int y = 0; y < nY; ++y) ys[y + 1] += ys[y];
  }
  __syncthreads();
  for (int y = t; y < nY; y += T)
    for (int u = 0, k = ys[y]; u < m.n_runs; ++u)
      if (y >= m.run[4 * u + 2] && y < m.run[4 * u + 2] + m.run[4 * u + 3]) {
        yr[k] = m.run[4 * u + 1];
        ya[k++] = m.run[4 * u] + y - m.run[4 * u + 2];
      }
  __syncthreads();
  for (int i = blockIdx.x; i < rows && i < atoms; i += gridDim.x) {
    const int32_t *list = lists + static_cast<int64_t>(i) * (K + 1);
    const int nl = list[K], zc = z[i];
    const R wi = E ? R(1) : wt[i];
    for (int k = t; k < nl; k += T) live[k] = list[k];
    for (int a = t; a < na; a += T) wA[a] = wAg[static_cast<int64_t>(i) * na + ya[a]];   // by harmonic
    __syncthreads();
    R fi = 0;                                             // lanes 0 to 2: the force on i
    for (int s0 = 0; s0 < nl; s0 += P) {
      const int s = s0 + team, has = s < nl;
      int j = 0;
      if (has) {
        R v[3];
        int zn;
        if constexpr (E) {
          const int64_t q = static_cast<int64_t>(i) * K + live[s];
          for (int c = 0; c < 3; ++c) v[c] = ed.rij[3 * q + c];
          zn = ed.zj[q];
        } else {
          j = nb[static_cast<int64_t>(live[s]) * rows + i];
          for (int c = 0; c < 3; ++c) v[c] = x[3 * j + c] - x[3 * i + c];
          zn = z[j];
        }
        const R r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), ir = 1 / r;
        const int pz = zc * m.nz + zn;
        table<true>(m, pz, r, Rs, dRs, lane, Team);
        for (int mm = lane; mm <= m.L; mm += Team) harmonics(m.L, m.solid, mm, norm, ca, cb, v, r, ir, Ys);
        if (lane < 4) hat[lane] = lane == 3 ? ir : (lane == 0 ? v[0] : lane == 1 ? v[1] : v[2]) * ir;
      }
      __syncthreads();
      R sc = 0, g[3] = {0, 0, 0};
      for (int c = m.n_rnl + lane; has && c < F; c += Team) sc += dRs[c];   // the folded pair radial
      for (int y = lane; has && y < nY; y += Team) {
        const int l = static_cast<int>(sqrtf(y + 0.5f)), sg = y < l * l + l ? -1 : 1;
        const int am = sg * (y - l * l - l);
        R u = 0, ud = 0;                                  // dE/dY and dE/dY'
        for (int k = ys[y]; k < ys[y + 1]; ++k) {
          u += Rs[yr[k]] * wA[k];
          ud += dRs[yr[k]] * wA[k];
        }
        sc += ud * Ys[y];
        if (l == 0) continue;
        const int cu = l * (m.L + 1) + am, b = (l - 1) * l;   // grad Y from degree l - 1, b: (l - 1, 0)
        const bool up = am + 1 < l, dn = am > 0;
        const R sp = up ? Ys[b + sg * (am + 1)] : 0, op = up ? Ys[b - sg * (am + 1)] : 0;
        const R sm = dn && (sg > 0 || am > 1) ? Ys[b + sg * (am - 1)] : 0;   // no sine at m = 0
        const R om = dn && (sg < 0 || am > 1) ? Ys[b - sg * (am - 1)] : 0;
        const R gy[3] = {gp[cu] * sp + gm[cu] * sm, sg * (gp[cu] * op - gm[cu] * om),
                         am < l ? gz[cu] * Ys[b + sg * am] : 0};
        for (int c = 0; c < 3; ++c) g[c] += u * (m.solid ? gy[c] : (gy[c] - l * hat[c] * Ys[y]) * hat[3]);
      }
      for (int o = Team / 2; o > 0; o /= 2) {             // the team's sums
        sc += SHFL_XOR(sc, o);
        for (int c = 0; c < 3; ++c) g[c] += SHFL_XOR(g[c], o);
      }
      if (has && lane < 3) {
        const R fc = wi * (sc * hat[lane] + (lane == 0 ? g[0] : lane == 1 ? g[1] : g[2]));
        if constexpr (E) {
          ed.g[3 * (static_cast<int64_t>(i) * K + live[s]) + lane] = fc;
        } else {
          fi += fc;
          atomicAdd(&fo[3 * j + lane], -fc);
        }
      }
      __syncthreads();
    }
    if constexpr (!E) {
      for (int o = Team; o < kChunk; o *= 2) fi += SHFL_XOR(fi, o);   // over the chunk's teams, then chunks
      if (t % kChunk < 3) red[3 * (t / kChunk) + t % kChunk] = fi;
      __syncthreads();
      if (t < 3 && nl > 0) {
        for (int c = 1; c < T / kChunk; ++c) fi += red[3 * c + t];
        atomicAdd(&fo[3 * i + t], fi);
      }
    }
  }
}

// The model's arrays and attributes, as both entry points take them.
struct Consts {
  ffi::AnyBuffer rtab, rcut;
  ffi::Buffer<ffi::S32> runs, groups, factor;
  ffi::AnyBuffer ct, e0;
  ffi::Buffer<ffi::S32> levels, pull;
  ffi::AnyBuffer lc;
  ffi::Buffer<ffi::S32> ai;
  ffi::AnyBuffer ar;
  double x0, h;
  int64_t n, n_rnl, lmax, solid, n_leaf, n_st, tile, lv, threads, form, nq, npq;
};

// Both entry points: x the positions (atoms, 3), or (E) rij (rows, K, 3); out the forces (atoms, 3), or (E)
// dE_i / drij (rows, K, 3). `types` are the float buffers besides x, which must share its type.
template <bool E>
ffi::Error Launch(gpuStream_t stream, ffi::ScratchAllocator &scratch, ffi::AnyBuffer x_in, const int32_t *zs,
                  const int32_t *nb, const int32_t *cnt, const void *wt, const int32_t *zj, const bool *mask,
                  int64_t atoms, int64_t K, int64_t rows, const Consts &c, double rc2,
                  std::initializer_list<ffi::DataType> types, ffi::Result<ffi::AnyBuffer> &e,
                  ffi::Result<ffi::AnyBuffer> &f) {
  const int64_t lmax = c.lmax;
  if (lmax > 31) return ffi::Error::InvalidArgument("the GPU kernel takes lmax up to 31: a lane per m");
  const ffi::DataType type = x_in.element_type(), tdag = c.ct.element_type();
  for (const ffi::DataType t : types)
    if (t != type || (t != ffi::DataType::F64 && t != ffi::DataType::F32))
      return ffi::Error::InvalidArgument("the GPU kernel's floats are all float64 or all float32");
  if (c.ar.element_type() != type) return ffi::Error::InvalidArgument("the analytic radial has the floats' type");
  if (c.lc.element_type() != tdag || (tdag != type && !(type == ffi::DataType::F64 && tdag == ffi::DataType::F32)))
    return ffi::Error::InvalidArgument("the GPU kernel's readout has the floats' type, or float32 with "
                                       "float64 (mixed precision)");
  const int64_t n_out = static_cast<int64_t>(f->element_count());
  for (const int64_t size : {3 * atoms, rows, n_out, static_cast<int64_t>(c.rtab.element_count()),
                             static_cast<int64_t>(c.ct.element_count()),
                             static_cast<int64_t>(c.lc.element_count())})
    if (size > INT_MAX) return ffi::Error::InvalidArgument("the GPU kernel indexes in 32 bits");
  const auto i32 = [](int64_t v) { return static_cast<int>(v); };
  const int N = i32(c.ct.dimensions()[1]), n_lev = i32(c.levels.element_count()) - 1;
  const int F = i32(c.rtab.dimensions().back()), n_a = N - i32(c.factor.element_count());
  const int nz = i32(c.rtab.dimensions()[0]), ncoef = i32(c.rtab.element_count()) / (nz * nz * F);
  const int local = c.rtab.dimensions().size() == 5;      // radial.spline_local's (NZ, NZ, n - 1, 4, F)
  const int64_t n_rows = std::min(rows, atoms), nY = (lmax + 1) * (lmax + 1), n_runs = c.runs.dimensions()[0];
  const size_t sz = ffi::ByteWidth(type), szd = ffi::ByteWidth(tdag);   // the rows' floats, the DAG's
  int dev = 0, sms = 1, warp = 32, optin = 48 << 10;      // errors surface at GetLastError
  (void)GPU(GetDevice)(&dev);
  (void)GPU(DeviceGetAttribute)(&sms, GPU_SM_COUNT, dev);
  (void)GPU(DeviceGetAttribute)(&warp, GPU_WARP_SIZE, dev);
  (void)GPU(DeviceGetAttribute)(&optin, GPU_SHARED_OPTIN, dev);
  const bool four = lmax < 4;                             // lanes per pair: a lane per m
  const int team = four ? 4 : 8;
  const auto pair_threads = [&](int64_t per_pair, int64_t fixed) {   // up to 128, fitting kShared
    int T = 128;
    while (T > warp && fixed + T / team * per_pair > kShared) T /= 2;
    return T;
  };
  const auto density_shared = [&](int T) {
    return (3 * nY + T / kChunk + n_a + T / team * (F + nY)) * sz +
           (2 * n_a + 2 * K + 1) * sizeof(int);
  };
  const auto forces_shared = [&](int T) {
    return (6 * nY + 3 * T / kChunk + n_a + T / team * (2 * F + nY + 4)) * sz +
           (nY + 1 + 2 * n_a + K) * sizeof(int);
  };
  const int T1 = pair_threads((F + nY) * sz, density_shared(0));
  const int T3 = pair_threads((2 * F + nY + 4) * sz, forces_shared(0));
  // DAG pass: batches of GA rows, as many as leave two blocks per SM; a tile of M leaf parents.
  const int64_t kept = n_lev > 1 ? c.n_st : 0, nL = n_a + kept + 1;
  const int64_t n_mid = n_lev > 1 ? N - n_a - kept - c.n_leaf : 0;
  const int64_t M = c.tile, TD = c.threads;
  if (TD < kChunk || TD > kThreads || TD % kChunk)
    return ffi::Error::InvalidArgument("the GPU kernel's DAG pass takes 32 to 256 threads, in 32s");
  const auto dag_shared = [&](int64_t GA, int mode) {
    return TD / kChunk * GA * sz + (M + (mode == 0 ? n_a + 1 : mode == 1 ? nL : 0)) * GA * szd;
  };
  const size_t room = static_cast<size_t>(optin) / 2;
  const int mode = c.lv == 0 ? 0 : c.lv == 1 && dag_shared(1, 1) <= static_cast<size_t>(optin) ? 1 : 2;
  int GA = szd == 8 ? 2 : 4;                              // A100: float64 fastest at 2, float32 at 4
  while (GA > 1 && dag_shared(GA, mode) > room) GA /= 2;
  // The coupled form (form = 1): `pull` is ffi.coupled_gpu_plan's streams, `lc` their coefficients, n_st the
  // X values and tile the piece slots; z, g and the pieces of a batch are in shared memory.
  const bool cpl = c.form != 0;
  const auto cpl_shared = [&](int64_t ga) {
    return TD / kChunk * ga * sz + (2 * n_a + c.n_st + c.tile) * ga * szd;
  };
  if (cpl) {
    const char *ga_env = std::getenv("ACEJAX_CPL_GA");   // rows a block (a development knob)
    GA = ga_env ? std::atoi(ga_env) : szd == 8 ? 2 : 4;
    if (GA != 1 && GA != 2 && GA != 4) return ffi::Error::InvalidArgument("ACEJAX_CPL_GA is 1, 2 or 4");
    while (GA > 1 && cpl_shared(GA) > (ga_env ? static_cast<size_t>(optin) : room)) GA /= 2;
  }
  const auto blocks = [&](auto kernel, int threads, size_t shared, int64_t work) {   // fill the device
    int resident = 1;
    if (shared > (48 << 10))
      (void)GPU(FuncSetAttribute)(reinterpret_cast<const void *>(kernel), GPU_DYNAMIC_SHARED,
                                  static_cast<int>(shared));
    (void)GPU(OccupancyMaxActiveBlocksPerMultiprocessor)(&resident, kernel, threads, shared);
    return std::max<int64_t>(1, std::min<int64_t>(work, std::max(resident, 1) * static_cast<int64_t>(sms)));
  };
  const bool lanes = c.nq > 0;                            // an analytic radial: the forces a lane per neighbour
  if (lanes && lmax > kAnL) return ffi::Error::InvalidArgument("the analytic radial's force pass takes lmax up to 8");
  const size_t s3l = (6 * nY + n_a + nY * kChunk) * sz;
  const size_t s1l = (3 * nY + n_a + nY * kChunk) * sz + (2 * K + 1) * sizeof(int);
  // Pass 1 a lane per neighbour only for few A (its sums over the warp: A100 float64, table then analytic,
  // ms at 32,768 atoms: n_a 25 0.35, 0.23; 166 0.77, 1.07; 323 0.98, 2.54) and in float64 (in float32 the
  // transform loses the pair term's digits: o3d8 2e-3 eV/atom).  Its energy then is the analytic radial's, as
  // the forces; with the table it is the table's, which the forces differ from by the table's error (~1e-8).
  const bool dlanes = lanes && sz == 8 && n_a <= kAnDensity;
  const size_t s1 = dlanes ? s1l : density_shared(T1), s2 = cpl ? cpl_shared(GA) : dag_shared(GA, mode);
  const size_t s3 = lanes ? s3l : forces_shared(T3);
  if (std::max({s1, s2, s3}) > static_cast<size_t>(optin))
    return ffi::Error::InvalidArgument("the model needs more shared memory per block than this GPU has");
  const int64_t lists = n_lev + (n_lev > 1 ? 2 * n_mid / M + 2 : 0);   // pw: a piece per list, factor and
  const int64_t per_batch = cpl ? GA * c.n_st : GA * ((1 + (mode == 2)) * nL + lists * (n_a + TD));   // range at most
  const int64_t n_A = n_rows * n_a, n_list = n_rows * (K + 1);
  const int64_t n_batches = (n_rows + nz * (GA - 1) + GA - 1) / GA;
  const auto run = [&](auto zero, auto zero_dag) {        // the rest in the floats' type R, the DAG pass in Rd
    using R = decltype(zero);
    using Rd = decltype(zero_dag);
    const Model<R> m{c.rtab.typed_data<R>(), c.rcut.typed_data<R>(), nullptr, c.e0.typed_data<R>(),
                     nullptr, c.runs.typed_data(), c.groups.typed_data(), c.factor.typed_data(),
                     c.levels.typed_data(), c.pull.typed_data(), c.ai.typed_data(), c.ar.typed_data<R>(),
                     i32(c.nq), i32(c.npq), nz, ncoef, F, i32(c.n_rnl), i32(n_runs), N,
                     n_a, n_lev, i32(kept), std::max(n_lev - 2, 0), i32(c.lc.dimensions()[1]),
                     i32(c.n), i32(lmax), i32(c.solid), local, R(c.x0), R(1.0 / c.h), R(rc2)};
    const Model<Rd> md{nullptr, nullptr, c.ct.typed_data<Rd>(), nullptr,   // the readout: all the DAG reads
                       c.lc.typed_data<Rd>(), m.run, m.grp, m.fac, m.lev, m.pull, nullptr, nullptr, 0, 0, nz, ncoef, F,
                       m.n_rnl, m.n_runs, N,
                       n_a, n_lev, m.n_st, m.st_lev, m.n_lc, m.n, m.L, m.solid, local, Rd(c.x0), Rd(1.0 / c.h),
                       Rd(rc2)};
    const auto density = T1 == 32 ? (four ? ace_density<R, 32, 4, E> : ace_density<R, 32, 8, E>)
                         : T1 == 64 ? (four ? ace_density<R, 64, 4, E> : ace_density<R, 64, 8, E>)
                                    : (four ? ace_density<R, 128, 4, E> : ace_density<R, 128, 8, E>);
    const auto forces = T3 == 32 ? (four ? ace_forces<R, 32, 4, E> : ace_forces<R, 32, 8, E>)
                        : T3 == 64 ? (four ? ace_forces<R, 64, 4, E> : ace_forces<R, 64, 8, E>)
                                   : (four ? ace_forces<R, 128, 4, E> : ace_forces<R, 128, 8, E>);
#define ACE_DAG(Mode) \
  (GA == 1 ? ace_dag<Rd, R, 1, Mode> : GA == 2 ? ace_dag<Rd, R, 2, Mode> : ace_dag<Rd, R, 4, Mode>)
    const auto dag = mode == 0 ? ACE_DAG(0) : mode == 1 ? ACE_DAG(1) : ACE_DAG(2);
#undef ACE_DAG
    decltype(&ace_forces_lane<R, E, 0>) const lane_k[kAnL + 1] = {
        ace_forces_lane<R, E, 0>, ace_forces_lane<R, E, 1>, ace_forces_lane<R, E, 2>, ace_forces_lane<R, E, 3>,
        ace_forces_lane<R, E, 4>, ace_forces_lane<R, E, 5>, ace_forces_lane<R, E, 6>, ace_forces_lane<R, E, 7>,
        ace_forces_lane<R, E, 8>};
    const auto forces_l = lane_k[lanes ? lmax : 0];
    decltype(&ace_density_lane<R, E, 0>) const dens_k[kAnL + 1] = {
        ace_density_lane<R, E, 0>, ace_density_lane<R, E, 1>, ace_density_lane<R, E, 2>, ace_density_lane<R, E, 3>,
        ace_density_lane<R, E, 4>, ace_density_lane<R, E, 5>, ace_density_lane<R, E, 6>, ace_density_lane<R, E, 7>,
        ace_density_lane<R, E, 8>};
    const auto density_l = dens_k[dlanes ? lmax : 0];
    const int64_t b1 = dlanes ? blocks(density_l, kChunk, s1, n_rows) : blocks(density, T1, s1, n_rows);
    const int64_t b3 = lanes ? blocks(forces_l, kChunk, s3, n_rows) : blocks(forces, T3, s3, n_rows);
    const auto coupled = GA == 1 ? ace_coupled<Rd, R, 1> : GA == 2 ? ace_coupled<Rd, R, 2> : ace_coupled<Rd, R, 4>;
    const int64_t b2 = cpl ? blocks(coupled, i32(TD), s2, n_batches) : blocks(dag, i32(TD), s2, n_batches);
    // Scratch: A and dE/dA per row, per block the DAG's adjoints and unshared values, then the live lists,
    // the species' counts and the batches.
    const int64_t bytes = 2 * n_A * sz + b2 * per_batch * szd + (n_list + 1 + nz + n_batches * GA) * sizeof(int);
    const auto mem = scratch.Allocate(bytes, 16);         // XLA aligns to 16 at most
    if (!mem) return ffi::Error(ffi::ErrorCode::kResourceExhausted, "no device scratch for the ACE kernel");
    R *Ag = static_cast<R *>(*mem), *wAg = Ag + n_A;
    Rd *D = reinterpret_cast<Rd *>(wAg + n_A);
    R *x = x_in.typed_data<R>(), *eo = e->typed_data<R>(), *fo = f->typed_data<R>();
    const Edges<R> ed{E ? x : nullptr, zj, mask, E ? fo : nullptr};
    int32_t *lists = reinterpret_cast<int32_t *>(D + b2 * per_batch), *broken = lists + n_list;
    int32_t *per_z = broken + 1, *perm = per_z + nz;
    (void)GPU(MemsetAsync)(eo, 0, atoms * sz, stream);
    (void)GPU(MemsetAsync)(fo, 0, n_out * sz, stream);
    (void)GPU(MemsetAsync)(broken, 0, (1 + nz) * sizeof(int), stream);
    (void)GPU(MemsetAsync)(perm, 0xff, n_batches * GA * sizeof(int), stream);
    if (n_rows > 0) {
      if (dlanes)
        density_l<<<b1, kChunk, s1, stream>>>(m, x, zs, nb, cnt, ed, i32(atoms), i32(K), i32(rows), Ag, lists, eo,
                                              broken);
      else
        density<<<b1, T1, s1, stream>>>(m, x, zs, nb, cnt, ed, i32(atoms), i32(K), i32(rows), Ag, lists, eo, broken);
      const int bc = i32(std::min<int64_t>((n_rows + 255) / 256, 4 * sms));
      ace_count<<<bc, 256, 0, stream>>>(zs, i32(n_rows), per_z);
      ace_base<<<1, 1, 0, stream>>>(nz, GA, per_z);
      ace_fill<<<bc, 256, 0, stream>>>(zs, i32(n_rows), per_z, perm);
      if (cpl)
        coupled<<<b2, i32(TD), s2, stream>>>(c.pull.typed_data(), c.lc.typed_data<Rd>(), i32(c.lc.dimensions()[1]),
                                             zs, perm, i32(n_batches), Ag, wAg, D, per_batch, eo);
      else
        dag<<<b2, i32(TD), s2, stream>>>(md, zs, perm, i32(n_batches), i32(M), Ag, wAg, D, per_batch, eo);
      if (lanes)
        forces_l<<<b3, kChunk, s3, stream>>>(m, x, zs, nb, static_cast<const R *>(wt), ed, i32(atoms), i32(K),
                                             i32(rows), lists, wAg, broken, eo, fo);
      else
        forces<<<b3, T3, s3, stream>>>(m, x, zs, nb, static_cast<const R *>(wt), ed, i32(atoms), i32(K), i32(rows),
                                       lists, wAg, broken, eo, fo);
    }
    const auto err = GPU(GetLastError)();
    if (err != GPU(Success))
      return ffi::Error::Internal(std::string("ACE kernel launch: ") + GPU(GetErrorString)(err));
    return ffi::Error::Success();
  };
  return type == ffi::DataType::F32 ? run(0.0f, 0.0f) : tdag == ffi::DataType::F32 ? run(0.0, 0.0f) : run(0.0, 0.0);
}

ffi::Error AceDirectGpuImpl(gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer pos,
                            ffi::Buffer<ffi::S32> species, ffi::Buffer<ffi::S32> nbr,
                            ffi::Buffer<ffi::S32> num, ffi::AnyBuffer weight, ffi::AnyBuffer rtab,
                            ffi::AnyBuffer rcut, ffi::Buffer<ffi::S32> runs, ffi::Buffer<ffi::S32> groups,
                            ffi::Buffer<ffi::S32> factor, ffi::AnyBuffer ct, ffi::AnyBuffer e0,
                            ffi::Buffer<ffi::S32> levels,
                            ffi::Buffer<ffi::S32> pull, ffi::AnyBuffer lc, ffi::Buffer<ffi::S32> ai,
                            ffi::AnyBuffer ar, double x0, double h,
                            int64_t n, int64_t n_rnl, int64_t lmax, int64_t solid, int64_t n_leaf,
                            int64_t n_st, int64_t tile, int64_t lv, int64_t threads, int64_t form, int64_t nq, int64_t npq, double rc2, ffi::Result<ffi::AnyBuffer> e,
                            ffi::Result<ffi::AnyBuffer> f) {
  const Consts c{rtab, rcut, runs, groups, factor, ct, e0, levels, pull, lc, ai, ar,
                 x0, h, n, n_rnl, lmax, solid, n_leaf, n_st, tile, lv, threads, form, nq, npq};
  return Launch<false>(stream, scratch, pos, species.typed_data(), nbr.typed_data(), num.typed_data(),
                       weight.untyped_data(), nullptr, nullptr, pos.dimensions()[0], nbr.dimensions()[0],
                       nbr.dimensions()[1], c, rc2,
                       {weight.element_type(), rtab.element_type(), rcut.element_type(),
                        e0.element_type(), e->element_type(), f->element_type()},
                       e, f);
}

// The ASE path (calc.ACECalculator(ffi=...) on a GPU): rows are the atoms, every live slot a pair.
ffi::Error AceEdgesGpuImpl(gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer rij,
                           ffi::Buffer<ffi::S32> zi, ffi::Buffer<ffi::S32> zj, ffi::Buffer<ffi::PRED> mask,
                           ffi::AnyBuffer rtab, ffi::AnyBuffer rcut, ffi::Buffer<ffi::S32> runs,
                           ffi::Buffer<ffi::S32> groups, ffi::Buffer<ffi::S32> factor, ffi::AnyBuffer ct,
                           ffi::AnyBuffer e0, ffi::Buffer<ffi::S32> levels,
                           ffi::Buffer<ffi::S32> pull, ffi::AnyBuffer lc, ffi::Buffer<ffi::S32> ai,
                           ffi::AnyBuffer ar, double x0, double h, int64_t n,
                           int64_t n_rnl, int64_t lmax, int64_t solid, int64_t n_leaf, int64_t n_st,
                           int64_t tile, int64_t lv, int64_t threads, int64_t form, int64_t nq, int64_t npq, ffi::Result<ffi::AnyBuffer> e,
                           ffi::Result<ffi::AnyBuffer> g) {
  const Consts c{rtab, rcut, runs, groups, factor, ct, e0, levels, pull, lc, ai, ar,
                 x0, h, n, n_rnl, lmax, solid, n_leaf, n_st, tile, lv, threads, form, nq, npq};
  const int64_t rows = rij.dimensions()[0], K = rij.dimensions()[1];
  return Launch<true>(stream, scratch, rij, zi.typed_data(), nullptr, nullptr, nullptr, zj.typed_data(),
                      mask.typed_data(), rows, K, rows, c, 0.0,
                      {rtab.element_type(), rcut.element_type(), e0.element_type(),
                       e->element_type(), g->element_type()},
                      e, g);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(AceDirectGpu, AceDirectGpuImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()            // positions (atoms, 3)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // model species (atoms,)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // neighbour matrix (K, rows)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // neighbour counts (rows,)
                                  .Arg<ffi::AnyBuffer>()            // site weights w (atoms,)
                                  .Arg<ffi::AnyBuffer>()            // radial table (NZ, NZ, ncoef, F)
                                  .Arg<ffi::AnyBuffer>()            // pair cutoffs (NZ, NZ)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // A runs (n_runs, 4): a0, r, y0, len
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG groups (n_groups, 2): parent, end
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG A factor per node
                                  .Arg<ffi::AnyBuffer>()            // readout (NZ, n_A + nodes)
                                  .Arg<ffi::AnyBuffer>()            // E0 (NZ,)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG levels' first groups (n_lev + 1,)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // ffi.pull_plan's lists
                                  .Arg<ffi::AnyBuffer>()            // leaf readout in its two orders (NZ, slots)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // ffi.analytic_plan's ints (or one zero)
                                  .Arg<ffi::AnyBuffer>()            // and reals: the analytic radial, nq > 0
                                  .Attr<double>("x0")
                                  .Attr<double>("h")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("n_rnl")
                                  .Attr<int64_t>("lmax")
                                  .Attr<int64_t>("solid")
                                  .Attr<int64_t>("n_leaf")
                                  .Attr<int64_t>("n_st")
                                  .Attr<int64_t>("tile")
                                  .Attr<int64_t>("lv")
                                  .Attr<int64_t>("threads")
                                  .Attr<int64_t>("form")            // 0: the DAG; 1: the coupled form
                                  .Attr<int64_t>("nq")              // the analytic radial's polynomials; 0: the table
                                  .Attr<int64_t>("npq")             // and its pair radial's
                                  .Attr<double>("rc2")
                                  .Ret<ffi::AnyBuffer>()            // atom energies (atoms,)
                                  .Ret<ffi::AnyBuffer>());          // -sum_i w_i grad e_i (atoms, 3)

XLA_FFI_DEFINE_HANDLER_SYMBOL(AceEdgesGpu, AceEdgesGpuImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()            // rij (rows, K, 3)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // centre species (rows,)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // neighbour species (rows, K)
                                  .Arg<ffi::Buffer<ffi::PRED>>()    // live slots (rows, K)
                                  .Arg<ffi::AnyBuffer>()            // radial table (NZ, NZ, ncoef, F)
                                  .Arg<ffi::AnyBuffer>()            // pair cutoffs (NZ, NZ)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // A runs (n_runs, 4): a0, r, y0, len
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG groups (n_groups, 2): parent, end
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG A factor per node
                                  .Arg<ffi::AnyBuffer>()            // readout (NZ, n_A + nodes)
                                  .Arg<ffi::AnyBuffer>()            // E0 (NZ,)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // DAG levels' first groups (n_lev + 1,)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // ffi.pull_plan's lists
                                  .Arg<ffi::AnyBuffer>()            // leaf readout in its two orders (NZ, slots)
                                  .Arg<ffi::Buffer<ffi::S32>>()     // ffi.analytic_plan's ints (or one zero)
                                  .Arg<ffi::AnyBuffer>()            // and reals: the analytic radial, nq > 0
                                  .Attr<double>("x0")
                                  .Attr<double>("h")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("n_rnl")
                                  .Attr<int64_t>("lmax")
                                  .Attr<int64_t>("solid")
                                  .Attr<int64_t>("n_leaf")
                                  .Attr<int64_t>("n_st")
                                  .Attr<int64_t>("tile")
                                  .Attr<int64_t>("lv")
                                  .Attr<int64_t>("threads")
                                  .Attr<int64_t>("form")            // 0: the DAG; 1: the coupled form
                                  .Attr<int64_t>("nq")              // the analytic radial's polynomials; 0: the table
                                  .Attr<int64_t>("npq")             // and its pair radial's
                                  .Ret<ffi::AnyBuffer>()            // site energies (rows,)
                                  .Ret<ffi::AnyBuffer>());          // dE_i / drij (rows, K, 3)
