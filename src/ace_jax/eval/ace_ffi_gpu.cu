// Lean linear ACE energies and forces on a GPU, one XLA FFI call, in three passes: each atom's A, its product
// basis, the forces. float64 or float32, one source for CUDA and ROCm. Two entry points: AceDirectGpu on
// lammps-jax's neighbour matrix (positions in, atom energies and forces out), and AceEdgesGpu on the dense edge
// input of EdgeSiteModel.energy_forces_virial_dense (rij, species, slot mask in, site energies and dE_i/drij
// out), which is the same passes with E = true.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
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

constexpr int kThreads = 256;                             // per block of the DAG pass
constexpr int kChunk = 32;                                // lanes per shuffle
constexpr int kShared = 48 << 10;                         // pair-pass shared memory without opt-in
constexpr double kPi = 3.14159265358979323846;

template <typename R>
struct Model {                                            // 32-bit sizes save registers
  const R *tab, *rc, *ct, *E0, *lc;
  const int32_t *run, *grp, *fac, *lev, *par, *lone, *chk, *lf;
  int nz, ncoef, F, n_rnl, n_runs, N, n_a, n_lev, n_groups, n_st, st_lev, n_lone, n_chunks, n_lf, n, L, solid;
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

// A value other threads' atomics changed, read past L1.
template <typename R>
__device__ R fresh(const R *p) { return *reinterpret_cast<const volatile R *>(p); }

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

// ace_jax.eval.harmonics of v, |v| = r = 1 / ir, for one m >= 0 and every l from m.
template <typename R>
__device__ void harmonics(int L, bool solid, int m, const R *norm, const R *ca, const R *cb, const R *v, R r,
                          R ir, R *Y) {
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
    Y[l * l + l + m] = norm[l * l + l + m] * P * (rad * A);   // this order stays in float32's range
    if (m) Y[l * l + l - m] = norm[l * l + l - m] * P * (rad * B);
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

// v added to w[key] for a chunk's lanes, nothing for key < 0; neighbouring lanes with one key sum first.
template <typename R>
__device__ void chunk_add(R *w, int key, R v) {
  const int lane = threadIdx.x % kChunk, prev = SHFL_UP(key, 1);
  const unsigned heads = BALLOT(lane == 0 || key != prev || key < 0), after = heads & ~((2u << lane) - 1u);
  const int end = after ? __ffs(after) - 1 : kChunk;      // the next head
  int o = 1;
  for (unsigned r = ~heads; r; r &= r >> o, o *= 2) {     // r: lanes starting o non-heads in a row
    const R u = SHFL_DOWN(v, o);
    if (lane + o < end) v += u;
  }
  if (key >= 0 && (heads >> lane & 1u)) atomicAdd(&w[key], v);
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

// Pass 2: the energy and dE/dA from each row's A, a group of T threads per row. Node values Lv, A and the
// kept nodes then 1, are in shared memory if lv_sh; adjoints wv and dE/dA wg in scratch. The last inner
// level's nodes, the mids, are recomputed with their leaves.
template <typename R, int T>
__global__ void __launch_bounds__(kThreads)
    ace_dag(Model<R> m, const int32_t *z, int n_rows, int lv_sh, const R *Ag, R *wAg, R *scratch,
            int64_t per_atom, R *eo) {
  extern __shared__ double sh[];
  const int na = m.n_a, nL = m.n_a + m.n_st + 1, mids = m.n_lev > 1;
  constexpr int G = kThreads / T;
  const int gi = threadIdx.x / T, t = threadIdx.x % T, lane = t % kChunk;
  R *red = reinterpret_cast<R *>(sh), *off = scratch + (blockIdx.x * G + gi) * per_atom;
  R *Lv = lv_sh ? red + kThreads + gi * nL : off, *wv = off + (lv_sh ? 0 : nL), *wg = wv + nL, *A = Lv;
  for (int i0 = blockIdx.x * G; i0 < n_rows; i0 += gridDim.x * G) {
    const int i = i0 + gi;
    const bool on = i < n_rows;                           // off groups meet the barriers too
    const int zc = on ? z[i] : 0;
    for (int a = t; a < na; a += T) A[a] = on ? Ag[static_cast<int64_t>(i) * na + a] : 0;
    if (t == 0) Lv[nL - 1] = 1;
    __syncthreads();
    const R *cz = m.ct + zc * m.N, *lcz = m.lc + zc * m.n_lf;
    R e = 0;
    for (int l = 0; l < m.st_lev; ++l) {                  // the kept levels
      for (int g = m.lev[l] + t; on && g < m.lev[l + 1]; g += T) {
        const R Lp = Lv[m.grp[2 * g]];
        for (int k = g ? m.grp[2 * g - 1] : 0; k < m.grp[2 * g + 1]; ++k) Lv[na + k] = Lp * A[m.fac[k]];
      }
      __syncthreads();
    }
    for (int j = t; on && j < nL; j += T) {
      const R c = j < nL - 1 ? cz[j] : 0;
      e += c * Lv[j];
      wv[j] = c;
      if (j < na) wg[j] = 0;
    }
    __syncthreads();
    for (int u0 = t - lane; on && u0 < m.n_lone; u0 += T) {       // mids without leaves, a lane each
      const int u = u0 + lane, ok = u < m.n_lone, k = ok ? m.lone[u] : 0, p = ok ? m.par[k] : 0;
      const int f = ok ? m.fac[k] : -1;
      const R c = ok ? cz[na + k] : 0, Lp = ok ? Lv[p] : 0, At = ok ? A[f] : 0;
      e += c * Lp * At;
      chunk_add(wg, f, c * Lp);
      chunk_add(wv, ok ? p : -1, c * At);
    }
    // The leaves, a lane per group, 32 groups at a time from the rank-by-lane copy; their parent q is a mid
    // or an A entry.
    for (int c = t / kChunk; on && c < m.n_chunks; c += T / kChunk) {
      const int g = m.lev[m.n_lev - 1] + kChunk * c + lane, ok = g < m.n_groups, q = ok ? m.grp[2 * g] : 0;
      const int p = mids && ok ? m.par[q - na] : q, f = mids && ok ? m.fac[q - na] : -1;
      const R Lp = ok ? Lv[p] : 0, At = f >= 0 ? A[f] : 1, Lt = Lp * At;
      R s = 0;
      for (int o = m.chk[2 * c] + lane, end = o + kChunk * m.chk[2 * c + 1]; o < end; o += kChunk) {
        const int fk = m.lf[o];                           // plan puts equal factors in neighbouring lanes
        s += fk >= 0 ? lcz[o] * A[fk] : 0;
        chunk_add(wg, fk, lcz[o] * Lt);
      }
      const R w = s + (f >= 0 ? cz[q] : 0);               // with the mid's readout
      e += w * Lt;
      chunk_add(wg, f, w * Lp);
      chunk_add(wv, ok ? p : -1, w * At);
    }
    __syncthreads();
    for (int l = m.st_lev - 1; l >= 0; --l) {             // the kept levels backwards
      for (int g = m.lev[l] + t; on && g < m.lev[l + 1]; g += T) {
        const int p = m.grp[2 * g];
        const R Lp = Lv[p];
        R acc = 0;
        for (int k = g ? m.grp[2 * g - 1] : 0; k < m.grp[2 * g + 1]; ++k) {
          const R w = fresh(&wv[na + k]);
          acc += w * A[m.fac[k]];
          atomicAdd(&wg[m.fac[k]], w * Lp);
        }
        wv[p] += acc;
      }
      __syncthreads();
    }
    for (int o = kChunk / 2; o > 0; o /= 2) e += SHFL_XOR(e, o);   // before the loop below: ptxas 12.8 -O2
    for (int a = t; on && a < na; a += T)                          // got it wrong after
      wAg[static_cast<int64_t>(i) * na + a] = fresh(&wv[a]) + fresh(&wg[a]);
    if (t % kChunk == 0) red[threadIdx.x / kChunk] = e;
    __syncthreads();
    if (on && t == 0) {
      for (int c = 1; c < T / kChunk; ++c) e += red[threadIdx.x / kChunk + c];
      eo[i] += e;
    }
    __syncthreads();
  }
}

// Pass 3: the forces, -sum_i w_i grad e_i, or NaN everywhere after a NaN position. A block per row, Team
// lanes per pair; a lane per harmonic takes dE/dY over its radial channels, times grad Y. E: dE_i / drij
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
  int *run = reinterpret_cast<int *>(wA + na + P * per_team), *lr = run + 4 * m.n_runs;
  int *live = lr + m.L + 2 + m.n_runs;
  constants(m.L, norm, ca, cb, gz, gp, gm);
  for (int u = t; u < 4 * m.n_runs; u += T) run[u] = m.run[u];
  if (t == 0)                                             // runs by degree l: lr[l] to lr[l + 1], after L + 2
    for (int l = 0, k = 0; l <= m.L; ++l) {
      lr[l] = k;
      for (int u = 0; u < m.n_runs; ++u)
        if (static_cast<int>(sqrt(m.run[4 * u + 2] + 0.5)) == l) lr[m.L + 2 + k++] = u;
      lr[l + 1] = k;
    }
  for (int i = blockIdx.x; i < rows && i < atoms; i += gridDim.x) {
    const int32_t *list = lists + static_cast<int64_t>(i) * (K + 1);
    const int nl = list[K], zc = z[i];
    const R wi = E ? R(1) : wt[i];
    for (int k = t; k < nl; k += T) live[k] = list[k];
    for (int a = t; a < na; a += T) wA[a] = wAg[static_cast<int64_t>(i) * na + a];
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
        for (int k = lr[l]; k < lr[l + 1]; ++k) {
          const int *q = run + 4 * lr[m.L + 2 + k];       // a0, r, y0, len
          if (y >= q[2] && y < q[2] + q[3]) {
            const R w = wA[q[0] + y - q[2]];
            u += Rs[q[1]] * w;
            ud += dRs[q[1]] * w;
          }
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
  ffi::Buffer<ffi::S32> levels, parent, lone, chunks, lf;
  ffi::AnyBuffer lc;
  double x0, h;
  int64_t n, n_rnl, lmax, solid, n_leaf, n_st;
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
  const ffi::DataType type = x_in.element_type();
  for (const ffi::DataType t : types)
    if (t != type || (t != ffi::DataType::F64 && t != ffi::DataType::F32))
      return ffi::Error::InvalidArgument("the GPU kernel's floats are all float64 or all float32");
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
  const size_t sz = ffi::ByteWidth(type);
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
           (5 * n_runs + lmax + 2 + K) * sizeof(int);
  };
  const int T1 = pair_threads((F + nY) * sz, density_shared(0));
  const int T3 = pair_threads((2 * F + nY + 4) * sz, forces_shared(0));
  int T2 = warp;                                          // DAG pass: about a thread per A entry
  while (T2 < n_a && T2 < kThreads) T2 *= 2;
  const int64_t G = kThreads / T2, kept = n_lev > 1 ? c.n_st : 0, nL = n_a + kept + 1;
  const size_t base = kThreads * sz, room = static_cast<size_t>(optin) / 2;   // two blocks per SM
  const int lv_sh = base + nL * G * sz <= room;
  const auto blocks = [&](auto kernel, int threads, size_t shared, int64_t work) {   // fill the device
    int resident = 1;
    if (shared > (48 << 10))
      (void)GPU(FuncSetAttribute)(reinterpret_cast<const void *>(kernel), GPU_DYNAMIC_SHARED,
                                  static_cast<int>(shared));
    (void)GPU(OccupancyMaxActiveBlocksPerMultiprocessor)(&resident, kernel, threads, shared);
    return std::max<int64_t>(1, std::min<int64_t>(work, std::max(resident, 1) * static_cast<int64_t>(sms)));
  };
  const size_t s1 = density_shared(T1), s2 = base + lv_sh * nL * G * sz, s3 = forces_shared(T3);
  if (std::max(s1, s3) > static_cast<size_t>(optin))
    return ffi::Error::InvalidArgument("the model needs more shared memory per block than this GPU has");
  const int64_t per_atom = (2 - lv_sh) * nL + n_a, n_A = n_rows * n_a, n_list = n_rows * (K + 1);
  const auto run = [&](auto zero) {                       // the rest in the floats' type R
    using R = decltype(zero);
    const Model<R> m{c.rtab.typed_data<R>(), c.rcut.typed_data<R>(), c.ct.typed_data<R>(), c.e0.typed_data<R>(),
                     c.lc.typed_data<R>(), c.runs.typed_data(), c.groups.typed_data(), c.factor.typed_data(),
                     c.levels.typed_data(), c.parent.typed_data(), c.lone.typed_data(), c.chunks.typed_data(),
                     c.lf.typed_data(), nz, ncoef, F, i32(c.n_rnl), i32(n_runs), N, n_a, n_lev,
                     i32(c.groups.dimensions()[0]), i32(kept), std::max(n_lev - 2, 0),
                     i32(c.lone.element_count()), i32(c.chunks.dimensions()[0]), i32(c.lf.element_count()),
                     i32(c.n), i32(lmax), i32(c.solid), local, R(c.x0), R(1.0 / c.h), R(rc2)};
    const auto density = T1 == 32 ? (four ? ace_density<R, 32, 4, E> : ace_density<R, 32, 8, E>)
                         : T1 == 64 ? (four ? ace_density<R, 64, 4, E> : ace_density<R, 64, 8, E>)
                                    : (four ? ace_density<R, 128, 4, E> : ace_density<R, 128, 8, E>);
    const auto forces = T3 == 32 ? (four ? ace_forces<R, 32, 4, E> : ace_forces<R, 32, 8, E>)
                        : T3 == 64 ? (four ? ace_forces<R, 64, 4, E> : ace_forces<R, 64, 8, E>)
                                   : (four ? ace_forces<R, 128, 4, E> : ace_forces<R, 128, 8, E>);
    const auto dag = T2 == 32 ? ace_dag<R, 32> : T2 == 64 ? ace_dag<R, 64>
                     : T2 == 128 ? ace_dag<R, 128> : ace_dag<R, 256>;
    const int64_t b1 = blocks(density, T1, s1, n_rows), b3 = blocks(forces, T3, s3, n_rows);
    const int64_t b2 = blocks(dag, kThreads, s2, (n_rows + G - 1) / G);
    // Scratch: A and dE/dA per row, per group the DAG's adjoints and unshared values, then the live lists.
    const int64_t bytes = (2 * n_A + b2 * G * per_atom) * sz + (n_list + 1) * sizeof(int);
    const auto mem = scratch.Allocate(bytes, 16);         // XLA aligns to 16 at most
    if (!mem) return ffi::Error(ffi::ErrorCode::kResourceExhausted, "no device scratch for the ACE kernel");
    R *Ag = static_cast<R *>(*mem), *wAg = Ag + n_A, *D = wAg + n_A;
    R *x = x_in.typed_data<R>(), *eo = e->typed_data<R>(), *fo = f->typed_data<R>();
    const Edges<R> ed{E ? x : nullptr, zj, mask, E ? fo : nullptr};
    int32_t *lists = reinterpret_cast<int32_t *>(D + b2 * G * per_atom), *broken = lists + n_list;
    (void)GPU(MemsetAsync)(eo, 0, atoms * sz, stream);
    (void)GPU(MemsetAsync)(fo, 0, n_out * sz, stream);
    (void)GPU(MemsetAsync)(broken, 0, sizeof(int), stream);
    if (n_rows > 0) {
      density<<<b1, T1, s1, stream>>>(m, x, zs, nb, cnt, ed, i32(atoms), i32(K), i32(rows), Ag, lists, eo, broken);
      dag<<<b2, kThreads, s2, stream>>>(m, zs, i32(n_rows), lv_sh, Ag, wAg, D, per_atom, eo);
      forces<<<b3, T3, s3, stream>>>(m, x, zs, nb, static_cast<const R *>(wt), ed, i32(atoms), i32(K), i32(rows),
                                     lists, wAg, broken, eo, fo);
    }
    const auto err = GPU(GetLastError)();
    if (err != GPU(Success))
      return ffi::Error::Internal(std::string("ACE kernel launch: ") + GPU(GetErrorString)(err));
    return ffi::Error::Success();
  };
  return type == ffi::DataType::F32 ? run(0.0f) : run(0.0);
}

ffi::Error AceDirectGpuImpl(gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer pos,
                            ffi::Buffer<ffi::S32> species, ffi::Buffer<ffi::S32> nbr,
                            ffi::Buffer<ffi::S32> num, ffi::AnyBuffer weight, ffi::AnyBuffer rtab,
                            ffi::AnyBuffer rcut, ffi::Buffer<ffi::S32> runs, ffi::Buffer<ffi::S32> groups,
                            ffi::Buffer<ffi::S32> factor, ffi::AnyBuffer ct, ffi::AnyBuffer e0,
                            ffi::Buffer<ffi::S32> levels, ffi::Buffer<ffi::S32> parent,
                            ffi::Buffer<ffi::S32> lone, ffi::Buffer<ffi::S32> chunks,
                            ffi::Buffer<ffi::S32> lf, ffi::AnyBuffer lc, double x0, double h,
                            int64_t n, int64_t n_rnl, int64_t lmax, int64_t solid, int64_t n_leaf,
                            int64_t n_st, double rc2, ffi::Result<ffi::AnyBuffer> e,
                            ffi::Result<ffi::AnyBuffer> f) {
  const Consts c{rtab, rcut, runs, groups, factor, ct, e0, levels, parent, lone, chunks, lf, lc,
                 x0, h, n, n_rnl, lmax, solid, n_leaf, n_st};
  return Launch<false>(stream, scratch, pos, species.typed_data(), nbr.typed_data(), num.typed_data(),
                       weight.untyped_data(), nullptr, nullptr, pos.dimensions()[0], nbr.dimensions()[0],
                       nbr.dimensions()[1], c, rc2,
                       {weight.element_type(), rtab.element_type(), rcut.element_type(), ct.element_type(),
                        e0.element_type(), lc.element_type(), e->element_type(), f->element_type()},
                       e, f);
}

// The ASE path (calc.ACECalculator(ffi=...) on a GPU): rows are the atoms, every live slot a pair.
ffi::Error AceEdgesGpuImpl(gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer rij,
                           ffi::Buffer<ffi::S32> zi, ffi::Buffer<ffi::S32> zj, ffi::Buffer<ffi::PRED> mask,
                           ffi::AnyBuffer rtab, ffi::AnyBuffer rcut, ffi::Buffer<ffi::S32> runs,
                           ffi::Buffer<ffi::S32> groups, ffi::Buffer<ffi::S32> factor, ffi::AnyBuffer ct,
                           ffi::AnyBuffer e0, ffi::Buffer<ffi::S32> levels, ffi::Buffer<ffi::S32> parent,
                           ffi::Buffer<ffi::S32> lone, ffi::Buffer<ffi::S32> chunks,
                           ffi::Buffer<ffi::S32> lf, ffi::AnyBuffer lc, double x0, double h, int64_t n,
                           int64_t n_rnl, int64_t lmax, int64_t solid, int64_t n_leaf, int64_t n_st,
                           ffi::Result<ffi::AnyBuffer> e, ffi::Result<ffi::AnyBuffer> g) {
  const Consts c{rtab, rcut, runs, groups, factor, ct, e0, levels, parent, lone, chunks, lf, lc,
                 x0, h, n, n_rnl, lmax, solid, n_leaf, n_st};
  const int64_t rows = rij.dimensions()[0], K = rij.dimensions()[1];
  return Launch<true>(stream, scratch, rij, zi.typed_data(), nullptr, nullptr, nullptr, zj.typed_data(),
                      mask.typed_data(), rows, K, rows, c, 0.0,
                      {rtab.element_type(), rcut.element_type(), ct.element_type(), e0.element_type(),
                       lc.element_type(), e->element_type(), g->element_type()},
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
                                  .Arg<ffi::Buffer<ffi::S32>>()     // parent per node
                                  .Arg<ffi::Buffer<ffi::S32>>()     // mids without leaves
                                  .Arg<ffi::Buffer<ffi::S32>>()     // leaf chunks (n_chunks, 2): slot, ranks
                                  .Arg<ffi::Buffer<ffi::S32>>()     // leaf A factors rank by lane, -1 padding
                                  .Arg<ffi::AnyBuffer>()            // leaf readout rank by lane (NZ, slots)
                                  .Attr<double>("x0")
                                  .Attr<double>("h")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("n_rnl")
                                  .Attr<int64_t>("lmax")
                                  .Attr<int64_t>("solid")
                                  .Attr<int64_t>("n_leaf")
                                  .Attr<int64_t>("n_st")
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
                                  .Arg<ffi::Buffer<ffi::S32>>()     // parent per node
                                  .Arg<ffi::Buffer<ffi::S32>>()     // mids without leaves
                                  .Arg<ffi::Buffer<ffi::S32>>()     // leaf chunks (n_chunks, 2): slot, ranks
                                  .Arg<ffi::Buffer<ffi::S32>>()     // leaf A factors rank by lane, -1 padding
                                  .Arg<ffi::AnyBuffer>()            // leaf readout rank by lane (NZ, slots)
                                  .Attr<double>("x0")
                                  .Attr<double>("h")
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("n_rnl")
                                  .Attr<int64_t>("lmax")
                                  .Attr<int64_t>("solid")
                                  .Attr<int64_t>("n_leaf")
                                  .Attr<int64_t>("n_st")
                                  .Ret<ffi::AnyBuffer>()            // site energies (rows,)
                                  .Ret<ffi::AnyBuffer>());          // dE_i / drij (rows, K, 3)
