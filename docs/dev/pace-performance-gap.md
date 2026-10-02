# PACE performance gap: ace-jax vs ML-PACE on the GPU

Branch `perf/pace-gap-investigation` (on top of `feat/bench-scaling`). Everything
here was measured on Modal A100-80GB, unless a line says it is an estimate. The
code is in `bench/perf/`. The raw results are in `bench/perf/results/*.json`. §11
lists the commands that reproduce every number.

## TL;DR

The headline case is **Cantor medium, float64, 8192 atoms**. Here ML-PACE
(`pace/kk product`) runs at 2.1–2.3 M atom-steps/s, which is 3.6–3.8 ms per
force call.

1. **Most of the end-to-end gap is the calculator, not the model.** The ASE
   `ACECalculator` spends 12–19 ms per call before the model runs, against a
   6.7–7.5 ms model call:
   - rebuilding the neighbour matrix: 6–10 ms;
   - about 10 eager JAX ops that build the arguments: 3.5 ms;
   - a Python loop that maps species: 1.7–2.1 ms.

   LAMMPS reuses its list across steps (Verlet skin), so ML-PACE pays nothing
   for this in the timed segment. A skin-reusing calculator (prototype) is
   **2.5× faster end to end on the same model** (369k → 907k atom-steps/s).
2. **Model call alone, the gap is 1.9×** (7.2 ms vs 3.7 ms). ace-jax moves about
   1.4× the bytes ML-PACE does and computes about 3.5× the flops (both
   estimated), and both codes are memory- and latency-bound. The time goes to:
   - **per-edge scalar math: 40%.** Of this, the reverse pass through the
     SBessel radial basis is one fused kernel costing 1.15 ms. It is f64
     transcendental-bound, with 9 `sin`, 9 `div` and their adjoints per edge.
   - **the dense A assembly: 31%.** A one-hot species expansion to
     (n, K, 5·26) and a 1170-wide outer product per slot, of which 180 entries
     are used.
   - **the product-basis adjoint: 19%.** Scatter-adds of about 4150 row updates
     into A.
   - **the force scatter-add: 7.5%.**

   Reverse-mode forces cost 2.8× the energy-only forward pass. ML-PACE's
   analytic adjoint costs about 0.6× its forward pass.
3. **Prototypes, all at force parity ≤ 5e-15 eV/Å with the unmodified model.**
   - Four model changes: the recurrence SBessel, "pool first" A assembly, a
     feature-major product basis, and reverse-edge force gathers. Together they
     halve the model call: **7.50 → 3.93 ms**, the same as ML-PACE's 3.65–3.8 ms.
   - With the skin calculator, ace-jax reaches **1.37 M atom-steps/s** at 8k
     atoms (3.7× the benchmarked calculator) and **1.73 M** at 65k.
   - In LAMMPS, the owned-rows, cutoff-sized bundle plus the model changes gives
     **1.22 M** at 8k, against 468k for the stock bundle (2.6×). At 32k it gives
     1.47 M, where the stock bundle runs out of memory. It runs at 131k atoms,
     four times the old out-of-memory point.
4. **What remains** is the product basis (about 38% of the new GPU time) and the
   A assembly (about 27%). XLA can't keep a node's A and dA on chip. Every XLA
   formulation of the product-basis adjoint I tried costs about 1 ms at 8k
   atoms (§8.3). That is the case for a Pallas/Triton kernel, which I estimate
   at 0.2–0.3 ms.

**The one-GPU sweep** (§2) covers all 6 models, 1k–65k atoms, f64:
- **End to end,** the benchmarked calculator trails ML-PACE by:
  - 5.5–22× at 1k atoms;
  - 2.5–4.3× at 65k atoms.
- **The model call alone** trails by 1.2–3.1×. The gap widens with N for the
  medium and large models, as the baseline's throughput decays above 16k atoms.
- **The skin calculator plus the model variant** trails by only 1.3–2.1× at
  16k–65k atoms, where the benchmarked calculator trails by 2.5–8.3×. On some
  large-model cases it has 2× the throughput of the unmodified model
  (Cantor large at 65k: 247k → 537k).

---

## 1. What was measured, and where

- **Hardware.** Modal A100-80GB. Most containers were SXM4 (2.04 TB/s,
  9.7 TFLOP/s f64 without tensor cores, 19.5 with DMMA). The one-GPU sweep in
  §2 landed on a PCIe card (1.94 TB/s), and so did one profiling run; each
  table names its card. **Comparisons are only made within a container.**
  Across containers, the same baseline varied by up to ±8% (6.7–7.5 ms).
- **Software.** JAX 0.11.2 (CUDA 12). LAMMPS `patch_10Sep2025` with KOKKOS
  sm_80 and ML-PACE (`/opt/lmp.sh`). LAMMPS develop with lammps-jax
  (`/opt/lmp-jax.sh`). matscipy-neighbours with CUDA. The image is
  `bench/scaling/modal_app.py::base_image`; `bench/perf/modal_profile.py` adds
  kokkos-tools on top.
- **Models.** `bench/scaling/models/pace_{SiGe,Cantor}_{small,medium,large}.yace`.
  Cantor medium has:
  - NZ = 5 elements;
  - nradbase = 8, nradmax = 6, lmax = 2, ndensity = 2;
  - 496 functions and 633 (half-basis) m-combinations per element;
  - rcut = 5.0 Å, so 42 neighbours per atom (78 within rcut + 1 Å skin).

  In ace-jax this becomes:
  - 36 local A entries, so n_A = 5·36 = 180 per atom;
  - 1486 real AA products (the union over central elements; each element uses 770);
  - 4150 gather positions over all orders.
- **The benchmark rows** (`bench/scaling/results/*.jsonl`) were read directly
  (the aggregation is in §11).
  - A100 (`modal-a100.jsonl`), Cantor medium f64:
    - ML-PACE plateaus at 2.2–2.3 M from 4k atoms up.
    - ace-jax standalone end to end peaks at 646k (32k atoms). Its model call
      alone peaks at 1.29 M (16k) and falls to 754k at 262k.
    - ace-jax in LAMMPS peaks at 477k (8k).
  - `moriarty-gpu.jsonl`: the standalone ace-jax PACE rows have
    `force_s` ≈ 0.2–0.8 s per call. Model-only throughput is 1k at 256 atoms
    and at most 171k (Cantor small f32, 131k atoms). This matches an older,
    un-jitted calculator path, so these rows are not comparable with the A100
    rows. Its LAMMPS rows are comparable: ML-PACE reaches 1.0 M for Cantor
    medium; ace-jax f64 reaches 21–34k and runs out of memory from 2k atoms.
  - `moriarty-cpu.jsonl`: this file holds only parity rows, and the ace-jax CPU
    gate is `unsupported`. **I could not verify the 9k-vs-250k CPU figure from
    the results files**, and I did not run the CPU (moriarty was off-limits);
    see §10.

## 2. The gap, characterised (one A100, both codes in the same container)

`bench/perf/modal_profile.py::bigsweep` ran every case in one container, one
after another. The columns are:
- **ML-PACE:** LAMMPS `pace/kk product`, 100 timed steps (30 above 16k atoms).
- **ASE calc e2e:** `ACECalculator.calculate` exactly as the benchmark runs it.
- **model call only:** its `last_timing["model_s"]`.
- **skin calc:** the prototype calculator of §8.2, with the unmodified model.
- **skin + variant:** the same calculator with the `rec+pool+fm` model.

Gap = ML-PACE ÷ ace-jax. All values are atom-steps/s, float64 unless the dtype
column says 32.

GPU: NVIDIA A100 80GB PCIe

| model | N | dtype | ML-PACE (LAMMPS) | ace-jax ASE calc e2e | ace-jax model call only | gap e2e | gap model | skin calc | skin calc + variant | gap (skin+variant) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cantor_small | 1024 | 64 | 1,900k | 86k | 837k | 22.2x | 2.3x | 250k | 290k | 6.6x |
| Cantor_small | 4096 | 64 | 5,084k | 261k | 2,139k | 19.5x | 2.4x | 773k | 912k | 5.6x |
| Cantor_small | 16384 | 64 | 5,756k | 697k | 3,009k | 8.3x | 1.9x | 1,763k | 2,035k | 2.8x |
| Cantor_small | 65536 | 64 | 5,917k | 1,392k | 3,701k | 4.3x | 1.6x | 2,304k | 2,873k | 2.1x |
| Cantor_medium | 8192 | 32 | - | 406k | 2,190k | - | - | 1,035k | 1,138k | - |
| Cantor_medium | 65536 | 32 | - | 808k | 1,476k | - | - | 1,156k | 2,140k | - |
| Cantor_medium | 1024 | 64 | 1,166k | 80k | 619k | 14.6x | 1.9x | 226k | 248k | 4.7x |
| Cantor_medium | 4096 | 64 | 2,178k | 227k | 1,181k | 9.6x | 1.8x | 572k | 748k | 2.9x |
| Cantor_medium | 8192 | 64 | 2,223k | 374k | 1,202k | 5.9x | 1.8x | 749k | 1,106k | 2.0x |
| Cantor_medium | 16384 | 64 | 2,281k | 500k | 1,280k | 4.6x | 1.8x | 940k | 1,418k | 1.6x |
| Cantor_medium | 65536 | 64 | 2,319k | 641k | 914k | 3.6x | 2.5x | 807k | 1,703k | 1.4x |
| Cantor_large | 1024 | 64 | 521k | 81k | 369k | 6.5x | 1.4x | 198k | 176k | 3.0x |
| Cantor_large | 4096 | 64 | 702k | 204k | 528k | 3.4x | 1.3x | 393k | 432k | 1.6x |
| Cantor_large | 16384 | 64 | 715k | 235k | 341k | 3.0x | 2.1x | 304k | 569k | 1.3x |
| Cantor_large | 65536 | 64 | 728k | 227k | 256k | 3.2x | 2.8x | 247k | 537k | 1.4x |
| SiGe_small | 1024 | 64 | 1,504k | 95k | 858k | 15.9x | 1.8x | 260k | 246k | 6.1x |
| SiGe_small | 4096 | 64 | 3,409k | 291k | 2,037k | 11.7x | 1.7x | 852k | 853k | 4.0x |
| SiGe_small | 16384 | 64 | 3,784k | 666k | 2,326k | 5.7x | 1.6x | 1,402k | 1,801k | 2.1x |
| SiGe_small | 65536 | 64 | 3,621k | 1,028k | 2,434k | 3.5x | 1.5x | 1,650k | 2,698k | 1.3x |
| SiGe_medium | 1024 | 64 | 961k | 92k | 614k | 10.5x | 1.6x | 202k | 229k | 4.2x |
| SiGe_medium | 4096 | 64 | 1,620k | 246k | 1,098k | 6.6x | 1.5x | 584k | 576k | 2.8x |
| SiGe_medium | 8192 | 64 | 1,681k | 343k | 1,069k | 4.9x | 1.6x | 733k | 788k | 2.1x |
| SiGe_medium | 16384 | 64 | 1,702k | 528k | 1,229k | 3.2x | 1.4x | 890k | 840k | 2.0x |
| SiGe_medium | 65536 | 64 | 1,737k | 523k | 729k | 3.3x | 2.4x | 647k | 1,013k | 1.7x |
| SiGe_large | 1024 | 64 | 392k | 71k | 270k | 5.5x | 1.5x | 151k | 153k | 2.6x |
| SiGe_large | 4096 | 64 | 480k | 171k | 402k | 2.8x | 1.2x | 281k | 272k | 1.8x |
| SiGe_large | 16384 | 64 | 483k | 191k | 252k | 2.5x | 1.9x | 232k | 317k | 1.5x |
| SiGe_large | 65536 | 64 | 485k | 144k | 156k | 3.4x | 3.1x | 152k | 324k | 1.5x |

ASE calculator split (ms per call): call = neighbour list + transfers + model + rest
| model | N | call | nlist_s (list + layout + H2D) | model call | skin rebuild |
|---|---:|---:|---:|---:|---:|
| Cantor_large | 1024 | 12.7 | 8.1 | 2.8 | 10.8 |
| Cantor_large | 4096 | 20.1 | 9.6 | 7.8 | 11.6 |
| Cantor_large | 16384 | 69.8 | 15.8 | 48.0 | 20.2 |
| Cantor_large | 65536 | 289.2 | 25.0 | 255.6 | 14.7 |
| Cantor_medium | 1024 | 12.8 | 9.1 | 1.7 | 11.6 |
| Cantor_medium | 4096 | 18.1 | 11.6 | 3.5 | 12.4 |
| Cantor_medium | 8192 | 21.9 | 12.1 | 6.8 | 13.4 |
| Cantor_medium | 16384 | 32.8 | 15.6 | 12.8 | 13.1 |
| Cantor_medium | 65536 | 102.3 | 22.3 | 71.7 | 15.6 |
| Cantor_small | 1024 | 12.0 | 8.7 | 1.2 | 11.5 |
| Cantor_small | 4096 | 15.7 | 10.7 | 1.9 | 12.7 |
| Cantor_small | 16384 | 23.5 | 13.5 | 5.4 | 13.2 |
| Cantor_small | 65536 | 47.1 | 22.6 | 17.7 | 16.9 |
| SiGe_large | 1024 | 14.4 | 8.2 | 3.8 | 10.9 |
| SiGe_large | 4096 | 24.0 | 11.3 | 10.2 | 12.6 |
| SiGe_large | 16384 | 85.9 | 14.9 | 65.0 | 14.8 |
| SiGe_large | 65536 | 455.7 | 25.3 | 418.9 | 19.3 |
| SiGe_medium | 1024 | 11.2 | 7.6 | 1.7 | 10.6 |
| SiGe_medium | 4096 | 16.6 | 10.2 | 3.7 | 11.8 |
| SiGe_medium | 8192 | 23.9 | 12.6 | 7.7 | 15.3 |
| SiGe_medium | 16384 | 31.0 | 13.6 | 13.3 | 13.5 |
| SiGe_medium | 65536 | 125.3 | 23.5 | 89.9 | 16.0 |
| SiGe_small | 1024 | 10.8 | 7.8 | 1.2 | 11.2 |
| SiGe_small | 4096 | 14.1 | 9.6 | 2.0 | 12.0 |
| SiGe_small | 16384 | 24.6 | 13.2 | 7.0 | 12.8 |
| SiGe_small | 65536 | 63.8 | 28.7 | 26.9 | 18.7 |

Skin calculator vs ASE calculator, same model: max dF ≤ 3.5e-13 eV/Å in f64, and ≤ 1.5e-5 in f32.

This sweep ran the first version of the skin calculator (an eqx.filter_jit call plus separate D→H copies). The streamlined version in §8.2 is 1.2× faster at 8k (749k → 907k).

What this shows:

- **The end-to-end gap shrinks with N because the calculator's overhead is
  about 8–25 ms per call, roughly fixed.** It is 11–16× at 1k atoms and
  2.5–3.5× at 65k.
- **The model-only gap is 1.2–2.4×.** It is smallest for the large models,
  where both codes are dominated by the product basis, and largest for the
  medium models at 65k atoms.
- **The model-only throughput of the baseline falls above about 16k atoms.**
  For example, SiGe medium drops from 1.23 M to 729k. Its gather and scatter
  working sets (A at n × 258 × 8 B, the AA rows, the force scatter) stop
  fitting in the 40 MB L2. The prototype variant degrades far less (SiGe
  medium at 65k: 1.01 M).

### 2.1 Where an ASE-calculator call goes (Cantor medium, f64, SXM4)

From `bench/perf/results/acejax_Cantor_medium_8192_float64_dense_baseline*.json`
(two containers, so each cell gives a range):

| part of `calculate()` | 8192 atoms |
|---|---:|
| total call | 18.4–31.1 ms |
| `nlist_s` (list + layout + H→D), of which: | 9.4–18.9 ms |
| — `dense_graph` (neighbour_matrix on the GPU, via DLPack) | 6.2–10.0 ms |
| — building the arguments (≈10 eager JAX ops: `asarray`, `node_z[idx]`, mask, broadcast) | 3.5 ms |
| — `_species_index` (a Python dict lookup per atom) | 1.7–2.1 ms |
| compiled model call (`model_s`) | 6.6–7.6 ms |
| D→H of E, F, V and the rest | ≈ 2–4 ms |

ML-PACE's timed segment has `Neighbor list builds = 0`: at dt = 0.1 fs no atom
moves the 0.5 Å half-skin in 100 steps. So the fair end-to-end comparison is
against a calculator that also reuses its list (§8.2).

## 3. ace-jax on the GPU: profile of the compiled force call

`jax.profiler` traces (CUPTI kernel events) joined to the optimised HLO
(`--xla_dump_to`). Each kernel is mapped to its fusion, its JAX `op_name`
(forward vs `transpose(jvp(...))`), and its operand and result bytes. The tools
are `bench/perf/hlo_trace.py` and `bench/perf/hlo_fusion.py`. The case is
Cantor medium, f64, 8192 atoms, dense layout, n × K = 8192 × 44 slots
(344,064 real edges). One call runs 53 kernels, with 6.38 ms of GPU time and a
6.7–7.5 ms wall time.

| stage (the kernels I assigned to it) | GPU time | share | what dominates |
|---|---:|---:|---|
| per-edge scalar math: r, r̂, SBessel g_k, inner cutoff, Y_lm, core repulsion, R_nl = crad[zi,zj]·g, forward + reverse | 2.57 ms | 40% | **one fused reverse kernel of the whole per-edge chain: 1.15 ms** (≈680 HLO ops, 40 per-edge inputs, ~100 GB/s, i.e. f64 transcendental- and latency-bound); forward 0.49 ms; the `crad[zi,zj]` gather + contraction 0.26 + 0.21 ms |
| A assembly (dense): one-hot channel expansion to (n,K,5,26), `einsum nkr,nky->nry` (n,130,9), select 180 of 1170, forward + reverse | 1.99 ms | 31% | expansion 0.39 ms; forward GEMM 0.33 ms; reverse GEMMs 0.35 + 0.35 ms; one-hot reduction of dR 0.34 ms; select scatter 0.13 ms. The GEMM kernels reach 1.3–1.46 TB/s, i.e. bandwidth-bound |
| product basis: AA = ∏ A[s] (1486 rows), ρ = AA·c̃, and its reverse | 1.22 ms | 19% | **reverse scatter-adds into A: 0.89 ms**; forward gather-products 0.31 ms |
| force assembly `F.at[idx].add(-g)` | 0.48 ms | 7.5% | one scatter of 360k×3 into 8192×3 (atomics) |
| virial strain matmuls, embedding, misc | 0.14 ms | 2% | |

**Energy-only vs forces.**
- Energy only: 1.81–2.23 ms.
- Energy + forces + virial: 6.96–7.50 ms.
- So the reverse pass costs about 2.8× the forward pass. ML-PACE's analytic
  adjoint (Weights + Derivative + Force) costs about 0.6× its forward pass (§4).

**Achieved rates.** XLA's cost analysis of the compiled force call gives
3.78 GFLOP and 4.93 GB, which is 461 kflop and 602 kB per atom. In 6.96 ms that
is:
- **0.54 TFLOP/s**, 5.6% of the f64 non-tensor peak;
- **0.71 TB/s**, 35% of HBM peak.

Per kernel, the picture is mixed:
- the GEMM and loop kernels are bandwidth-bound (1.2–1.5 TB/s);
- the scatter kernels run at 0.3–0.7 TB/s (atomic-bound);
- the big per-edge reverse fusion runs at about 0.1 TB/s, and is instead bound
  by f64 `sin`/`cos`/`div` (software sequences on the GPU), by register
  pressure, or by both.

Nothing is near the FLOP peak, so this function is **memory- and latency-bound,
not compute-bound**. The one exception is the f64 transcendental fusion.

**Is it float64-bound?** Only the transcendental fusion is.
- The recurrence SBessel (§8.1) removes 1.4 ms, about 20%, in f64, and that
  change only replaces `sin`/`div` with FMAs.
- The rest scales with bytes. float32 halves the bytes but is only
  1.8× faster (model call, 8k atoms) and 1.6× (65k) in the Cantor medium rows of the §2 table.

## 4. ML-PACE on the GPU

**Kernel timer.** I used the kokkos-tools `simple-kernel-timer` (with fences)
on Cantor medium at 8192 atoms, `pace/kk product`. The default
`chunksize 4096` means two passes of the whole kernel sequence per force call.
Times are per force call (2 chunks):

| kernel | parallelisation | ms / call | share |
|---|---|---:|---:|
| ComputeWeights: dE/dA via the product rule, atomic adds | (atom, ms-comb) thread | 0.81 | 22% |
| ComputeAi: Y_lm (m ≥ 0), A += R_nl Y_lm, atomics | team of 32 atoms × neighbour | 0.62 | 17% |
| ComputeRho: forward/backward prefix products, ρ += c̃·B (atomic) | (atom, ms-comb) thread | 0.60 | 16% |
| ComputeRadial: cubic-spline g_k, R_nl, and derivatives | atom × neighbour | 0.47 | 13% |
| Kokkos::ViewFill: zeroing weights, A, ρ | – | 0.33 | 9% |
| ComputeDerivative: recompute Y_lm and dY_lm; f_ij = Σ w·∇(R Y) | atom × neighbour | 0.26 | 7% |
| ConjugateAi: fill −m from +m | atom | 0.23 | 6% |
| ComputeNeigh: filter the 78-entry skin list to the 42 within rcut | team per atom | 0.16 | 4% |
| ComputeForce: Σ f_ij, atomic scatter to atoms | atom | 0.13 | 4% |
| ComputeFS: embedding | atom | 0.05 | 1% |
| **total** | | **3.65** | |

The LAMMPS breakdown for the same run was Pair 93.8%, Comm 3.3%, Neigh 0% and
Modify 0.9%, with 2.14 M atom-steps/s (1.96 M with the timer's fences).

**Algorithm** (read from `src/KOKKOS/pair_pace_kokkos.cpp`, patch_10Sep2025):
- **`product`, not `recursive`.** The recursive evaluator is refused on the GPU.
  The C-tilde basis is complex and stores only half of the m-combinations,
  using A_{l,−m} = (−1)^m A*_{l,m}.
- **Radials are spline tables.** Each (μi, μj) pair has cubic-Hermite tables at
  0.001 Å bins, for g_k and for R_nl directly, giving value and derivative from
  one lookup. No transcendental functions run per step, and no crad
  contraction.
- **A is the full block.** For each (n, l, m ≥ 0), A[i, μj, lm, n] is
  accumulated with `atomic_add`. That is 36 complex numbers per neighbour here.
  It does not select the used entries, and it has no one-hot channel: the
  channel μj is an index.
- **Forces use the analytic adjoint, not a reverse pass.**
  - ComputeRho stores dB (the product of all A except factor t, from
    forward/backward prefix products) per (atom, ms-comb).
  - ComputeWeights forms w = Σ θ·dB with θ = Σ_p F'(ρ_p) c̃_p. These are the
    weights dE/dA.
  - ComputeDerivative loops over neighbours again, recomputes Y_lm and dY_lm,
    and forms f_ij = Σ_{nlm} w·(dR Y r̂ + R ∇Y), reusing the stored R and dR.

  Nothing per edge except R, dR, g, dg is kept between forward and backward.
- **Memory is chunked.** Arrays are sized by `chunksize × maxneigh` and
  `chunksize × n_ms_combs`, so memory is bounded at any N.
- **ML-PACE has headroom of its own.**
  - With `chunksize 16384` it reaches 2.63 M (8k) and 2.55 M (65k), against
    2.33 M / 2.40 M at the default 4096 (`mlpace_chunk_Cantor_medium.json`).
  - Its product-basis kernels are atomic-heavy (Weights is its costliest
    kernel).
  - It is not at the bandwidth roof either (§5).

## 5. FLOP and byte accounting: Cantor medium, 8192 atoms

- **ace-jax** figures are XLA `cost_analysis()` of the compiled programs, a
  measurement of the compiled HLO. XLA counts a transcendental as about one
  flop, so it understates the f64 `sin`/`div` cost.
- **ML-PACE** figures are my **estimate** from its source, using the model's
  sizes: 42 neighbours in cutoff, 78 in the skin list, 633 ms-combs per element
  with mean rank 2.52, and 36 (n, l, m ≥ 0) entries per neighbour. Its "bytes"
  count atomics as L2 read-modify-writes, not DRAM traffic.

| | kflop / atom | kB / atom | time @ 8192 | achieved |
|---|---:|---:|---:|---|
| ML-PACE, radial (27 splines × ~10 flop, 0.9 kB table reads + 0.4 kB writes per neighbour) | 11 | 55 | 0.47 ms | |
| ML-PACE, A (Ai + Conj: 80 atomics per neighbour) | 5 | 70 | 0.85 ms | |
| ML-PACE, Rho + Weights (633 ms-combs × (55 + 30) flop, ~430 B) | 54 | 272 | 1.41 ms | |
| ML-PACE, Derivative + Force (42 × ~1.4 kflop) | 59 | 40 | 0.39 ms | |
| **ML-PACE, total** | **≈ 130** | **≈ 440** | 3.65 ms | ≈ 0.29 TFLOP/s, ≈ 1.0 TB/s (L2 incl. atomics) |
| **ace-jax baseline** (efv) | **461** | **602** | 6.96 ms | 0.54 TFLOP/s, 0.71 TB/s |
| ace-jax baseline, energy only | 137 | 179 | 1.81 ms | |
| **ace-jax prototype** `rec+pool+rev+fm` | **247** | **301** | 3.93 ms | 0.51 TFLOP/s, 0.63 TB/s |

Where ace-jax's extra work comes from (baseline):
- **The dense A assembly computes the full outer product.** Per slot, the
  (5·26 radial columns) × (9 Y) product is 1170 products; 180 of them are used,
  and each neighbour contributes to only one of the 5 species channels. So the
  arithmetic is about 25× what is needed, and the materialised (n, K, 130)
  operand is 375 MB. ML-PACE does 36 complex multiply-adds per neighbour.
- **R_nl = crad[zi,zj]·g is formed per edge**, gathering a 144-entry slice of
  crad for every edge (0.47 ms). ML-PACE reads R_nl from a spline.
- **The SBessel basis evaluates K + 1 = 9 `sin` and 9 divisions per edge**, and
  the reverse pass doubles that. ML-PACE does a table lookup.
- **The product basis computes all 1486 products for every atom.** Only 770
  have a nonzero coefficient for any given central element.
- **Reverse-mode AD saves per-edge intermediates** (40 per-edge f64 arrays feed
  the big reverse fusion alone) **and uses scatter-add adjoints for every
  gather.** ML-PACE uses the analytic adjoint with recomputation.

## 6. Explanation of the gap

- **End to end (the "3–5×" and "7×" headlines):** the ASE calculator's fixed
  per-call overhead dominates, not the model. There are three parts:
  - a neighbour-list rebuild every call: 6–10 ms at 8k atoms, 25 ms at 65k;
  - eager argument construction: 3.5 ms;
  - a per-atom Python species loop: 2 ms.

  LAMMPS amortises its list over about 20+ steps and does the equivalent
  filtering in a 0.16 ms kernel. This is why the gap is 11–16× at 1k atoms and
  3–3.5× at 65k.
- **Model only: 1.2–2.4×.** The cause is the *formulation* XLA is given, not
  XLA itself:
  - it materialises per-slot tensors that ML-PACE never forms, namely the
    one-hot (n, K, 130) expansion and the per-edge R;
  - it does f64 transcendental math per edge where ML-PACE interpolates a
    table;
  - it takes forces through a generic reverse pass: about 2.8× the forward
    pass, with scatter adjoints and saved intermediates.

  Both codes sit in the same regime: a few hundred kB of traffic per atom at
  0.7–1 TB/s effective. So time tracks bytes moved. The prototypes that remove
  bytes (pool-first: −41% bytes) and transcendental work (rec) close the
  model-only gap at 8k atoms.
- **In LAMMPS (400–500k),** the stock bundle evaluates the dense model on every
  row of the `max_atoms` capacity. That includes about 1.1 × (owned + ghost)
  rows, while ghost rows never have edges. It also uses
  k_dense = k(rcut + skin) + 8 = 86 slots, but lammps-jax already drops skin
  pairs, so only about 42 are ever live. That is 4.5× the slots of the
  standalone layout at 8k atoms and 2.8× at 262k
  (`bench/perf/lammps_capacity.py`). It explains both the lower throughput and
  the early out-of-memory (§9).
- **float32 is only 1–1.5× faster end to end (1.6–1.8× on the model call alone)** because little of the work is
  compute-bound. The f64 transcendental fusion is the exception, and it is 18%
  of the time; halving bytes does not halve latency- or atomic-bound kernels.

## 7. Recommendations, ranked by expected gain ÷ effort

Gains are for Cantor medium f64 on the A100 unless stated. "Measured" means a
prototype on this branch, at parity; "estimated" means reasoned from the
profiles.

| # | change | where | gain | effort | status |
|---|---|---|---|---|---|
| 1 | **Reuse the neighbour list with a Verlet skin; one jitted step that forms rij, compacts to the cutoff slots and evaluates. Vectorise the species map. One H→D and one D→H per call** | `calc/point.py` (new calculator path; prototype `bench/perf/fast_calc.py`) | **2.5× end to end at 8k (369k → 907k)**, 1.27× at 65k, 4.3× at 1k; amortised rebuild about 10 ms every ~20–50 MD steps | S–M | measured |
| 2 | **LAMMPS bundle: evaluate only owned rows (n_rows = ⌈1.1·n_owned⌉) with slots sized for rcut, not rcut + skin** | `export/lammps.py::make_energy_fn` + `bench/scaling/run_lammps.py::capacity` (prototype `bench/perf/lammps_variant.py`) | **1.7× at 8k (468k → 793k)**; runs at 32k (stock runs out of memory; 938k there); about 3–4× less memory | S | measured, PE identical |
| 3 | **Recurrence SBessel**: sin(kx) from one sin/cos by Chebyshev recurrence; one reciprocal | `eval/pace_radial.py::_sbessel` | **−1.4 ms of 7.0 (−20%) on the model call** | XS | measured, dF ≤ 3e-15 |
| 4 | **"Pool first" dense A:** pool g_k ⊗ Y_lm per (node, neighbour species), then apply crad[zi, μ] per node; no per-edge R_nl, one-hot over 8 columns not 26 | `EdgeSiteModel.pool_a_dense` / `PACEModel.site_energies_dense` (prototype `FastPACE.pool_first_dense`) | −0.5 ms alone; with #3 and #5, bytes −41% and peak memory −54% (937 → 432 MB) | M | measured, dF ≤ 1e-15 |
| 5 | **Feature-major product basis:** produce A as (n_A, n) so AA gathers read contiguous rows and their adjoints add whole rows | `PACEModel._node_energies` | −0.3 to −0.6 ms (with #4); forward AA 0.49 → 0.18 ms | S | measured |
| 6 | **Reverse-edge force gather:** F_i = Σ_k g[i,k] − Σ_k g[idx[i,k], rev[i,k]] instead of `.at[idx].add` | `EdgeSiteModel.energy_forces_virial_dense`; `rev` from the neighbour builder (matscipy-neighbours) | −0.25 to −0.5 ms (the 0.48 ms scatter becomes a gather) | S (needs `rev` from the list) | measured, dF ≤ 3e-15 |
| 7 | **Pallas/Triton kernel for the product basis, forward and adjoint:** one program per block of nodes keeps A and dA on chip and loops over the static `aa_specs`; this is ML-PACE's Rho + Weights without atomics to global memory | new `eval/pace_pallas.py` | **estimated −0.9 to −1.0 ms** of the remaining 3.9 ms: the 0.9 ms of scatters become about 0.2 ms; 34 M multiply-adds at 8k atoms is under 0.1 ms of f64 compute; A + dA traffic is 24 MB | L | estimated; §8.3 shows no XLA formulation beats about 1 ms |
| 8 | **Fused per-edge kernel (Pallas):** g, Y, the pooling into Ag and, in the adjoint, ∂E/∂r_ij in registers; ML-PACE's Radial + Ai + Derivative structure | same | estimated −0.5 to −0.7 ms of the remaining about 0.9 ms of A assembly | L | estimated |
| 9 | **Chunk the atoms** (`lax.map` + `jax.checkpoint` over node blocks of about 16k), as ML-PACE's `chunksize` does | `EdgeSiteModel.energy_forces_virial*` | bounds peak memory at any N; stops the large-N throughput decay (§2); about 0% speed cost above 16k/chunk | M | estimated |
| 10 | **Evaluate the product basis per central element** (sort atoms by species; per-element `aa_specs` hold 770 of 1486) | `pace_build.py` + `_node_energies` | about 2× less product-basis work (−0.4 to −0.6 ms after #7) | M–L | estimated |
| 11 | **Spline radials, as ML-PACE does** (tables of g_k and R_nl per pair) | `pace_radial.py` | replaces the remaining SBessel math and the crad contraction; small after #3 and #4 (−0.1 to −0.2 ms); risks the 5e-14 parity unless the tables match ML-PACE's | M | estimated, low priority |
| 12 | **Analytic `custom_vjp` of the A → B → E chain** | `PACEModel` | little, **on its own**. The measured cost is in the scatter adjoints and per-slot materialisation, which a custom VJP written in jnp does not remove. It pays only inside #7 and #8 | M | not recommended as a standalone |
| 13 | **Mixed precision** (f32 A/B, f64 accumulate) | – | ≤ 1.5× even for pure f32 (§2), and it breaks parity with ML-PACE (5e-14/atom) | M | not recommended |
| – | `edge_a_kind` gather vs matmul | – | not applicable to the dense layout (it selects via `A_full[:, sel]`). In the sparse layout (the calculator's fallback above the dense budget) it is still a calibrated choice | – | – |

Why the rest follows #1–#6: they are simple, they compose, and together they
bring the model call to ML-PACE's time. After them, the remaining GPU time
(3.17 ms at 8k) is:
- product basis: about 1.2 ms (38%);
- A assembly: about 0.9 ms (27%);
- per-edge scalar math: about 0.7 ms;
- the rest: small.

#7 then #8 are the only levers large enough to go clearly past ML-PACE. Each
removes work that XLA cannot fuse: dynamic-index accumulation into per-node
arrays held on chip.

## 8. Prototypes and measurements

All prototypes are in `bench/perf/variants.py` (`FastPACE`, a `PACEModel`
subclass with flags) and `bench/perf/fast_calc.py`. Parity is checked against
the unmodified `PACEModel` on the same inputs (`profile_acejax.py` →
`parity_vs_baseline`), with these results:

| model and N | dE per atom | max dF (eV/Å) | max dV |
|---|---:|---:|---:|
| SiGe small/medium, Cantor medium/large, 256 atoms (CPU) | ≤ 3e-16 | ≤ 5e-15 | ≤ 1.3e-12 |
| Cantor medium, 8192 atoms (A100) | ≤ 6e-17 | ≤ 3.1e-15 | ≤ 2.3e-12 |

These are round-off level, far inside the 5e-14/atom and 4e-10 ML-PACE parity
of the benchmark gate.

### 8.1 Model-call variants (Cantor medium, f64, 8192 atoms, A100-SXM4; one container per block)

| variant | efv (ms) | energy only (ms) | XLA GFLOP / GB | vs baseline |
|---|---:|---:|---:|---:|
| baseline | 6.96 | 1.81 | 3.78 / 4.93 | 1.00 |
| rec | 5.56 | 2.01 | 3.75 / 4.84 | 1.25× |
| pool | 6.46 | 1.84 | 2.08 / 2.94 | 1.08× |
| rev (efv_rev) | 6.39 | – | 3.78 / 4.98 | 1.09× |
| rec+pool+rev | 5.31 | 1.59 | 2.03 / 2.85 | 1.31× |
| *second container:* baseline | 7.50 | 2.23 | 3.78 / 4.93 | 1.00 |
| pool+fm | 6.41 | 1.79 | 2.08 / 2.55 | 1.17× |
| rec+pool+fm | 4.50 | 1.39 | 2.03 / 2.41 | 1.67× |
| **rec+pool+rev+fm** | **3.93** | 1.58 | 2.03 / 2.46 | **1.91×** |

ML-PACE on the same kind of card: 3.65 ms per force call (kernel sum), 3.7–3.8 ms per step.

The trace of `rec+pool+rev+fm` has 3.17 ms of GPU time in 56 kernels:
- product-basis adjoint scatters: 0.95 ms;
- forward AA: 0.18 ms;
- A assembly: about 0.87 ms;
- per-edge scalar math: about 0.67 ms.

`fm` needs `pool` to take effect. Given A as (n, 180), XLA's layout assignment
kept it node-major however it was transposed. `pool_first_dense(...,
transposed=True)` produces it as (180, n) by construction.

### 8.2 End to end: the skin-reusing calculator (Cantor medium, f64, A100-SXM4)

`SkinDenseCalculator` (`bench/perf/fast_calc.py`) works as follows:
- It builds the (n, K_skin) neighbour matrix for rcut + 1 Å, and rebuilds only
  when an atom has moved more than 0.5 Å.
- Each call copies positions to the device once and makes one `jax.jit` call.
  Inside it, it forms rij = x[j] − x[i] + shift, compacts to the K = 44 slots
  within the cutoff (a cumsum and a search, gathers only), and evaluates E, F, V.
- Each call copies one packed array back.

Every timed call displaces all atoms by fresh 1e-3 Å noise, as consecutive MD
steps would. Parity between the skin calculator and the ASE calculator is
dF ≤ 4e-14.

| N | ASE calc (as benchmarked) | skin calc, same model | skin calc + `rec+pool+fm` | ML-PACE |
|---:|---:|---:|---:|---:|
| 1024 | 99k | 429k | 501k | 1.0–1.17 M |
| 8192 | 369k | 907k | **1.37 M** | 2.1–2.3 M |
| 65536 | 639k | 809k | **1.73 M** | 2.29–2.40 M |

Details:
- A steady-state rebuild costs 9.4–10.7 ms at 1k–8k atoms (16–19 ms at 65k), or
  about 0.2–0.5 ms per step if it happens every 20–50 steps.
- What remains per call at 8k is prep 0.63 ms (host displacement check + H→D)
  plus dispatch 0.40 ms. At 1k atoms this overhead is still larger than the
  0.93 ms of device time. Moving the displacement check onto the device, and
  bypassing ASE's `atoms.copy()`, are the next steps for small systems.

### 8.3 Why the product-basis adjoint needs a custom kernel

`bench/perf/micro_scatter.py` sums the 4150 per-position adjoint rows Q (P, n)
into dA (180, n) on the A100 at 8192 atoms, where Q is 272 MB:

| formulation | ms |
|---|---:|
| scatter-add, node-major (what XLA emits for the baseline) | 1.26 |
| scatter-add of whole rows, feature-major | 0.99 |
| static pairwise tree reduction (sorted rows, 9 gather + add levels, no atomics) | 1.07 |
| one-hot matmul Sᵀ·Q on f64 tensor cores | 1.17 |
| `segment_sum(indices_are_sorted=True)` | the kernel failed to launch (`CUDA_ERROR_OUT_OF_MEMORY` / invalid argument), reproducibly |

All of them are about 1 ms, because each materialises and re-reads the
P × n × 8 B adjoint. A kernel that keeps a node's 180 A values and 180 dA
accumulators on chip reads and writes about 3 kB per atom instead (§7, #7).

## 9. The lammps-jax path

The capacities come from `bench/perf/lammps_capacity.py`. Throughput comes from
`bench/perf/lammps_variant.py` running real LAMMPS runs through `/opt/lmp-jax.sh`,
with `newton on neigh half` and Kokkos on one GPU. All rows are Cantor medium,
f64.

| N | bundle | rows × slots | atom-steps/s | PE (step 0) |
|---:|---|---|---:|---|
| 8192 | stock dense (as benchmarked) | 18,673 × 86 | 468k | 1639.909472802445 |
| 8192 | owned rows, k(rcut) + 4 | 9,012 × 46 | 793k | 1639.909472802445 |
| 8192 | owned + `rec+pool+fm` | 9,012 × 46 | **1.22 M** | 1639.9094728024452 |
| 32768 | stock dense | 58,182 × 86 | **OOM** (9.1 GiB alloc) | – |
| 32768 | owned | 36,045 × 46 | 938k | 6580.574888366931 |
| 32768 | owned + variant | 36,045 × 46 | **1.47 M** | 6580.574888366932 |
| 131072 | owned | 144,180 × 46 | OOM | – |
| 131072 | owned + variant | 144,180 × 46 | 1.12 M | 26388.46379465847 |
| 262144 | owned + variant | 288,359 × 46 | OOM | – |

ML-PACE on the same models: 2.2–2.3 M.

LAMMPS-specific recommendations:
1. **Size the rows for owned atoms and the slots for rcut** (§7, #2). This is a
   change to ace-jax's own `make_energy_fn`. The static `n_rows` cap is safe:
   - senders are always owned atoms, which LAMMPS numbers first;
   - an overflow of rows or slots returns NaN, never a truncation.
2. **Fix `bench/scaling/run_lammps.capacity`.** It sizes `k_dense` and
   `max_edges` from the rcut + skin list. lammps-jax already drops skin pairs
   before packing (`PackNeighborFunctor`, `r² > cutsq`), so both can be sized
   for rcut. That halves the sparse fallback's edge buffer too.
3. **Consider lammps-jax's own layouts.**
   - The `max_neighbors` neighbour-matrix layout with `max_owned`: owned rows
     by construction, no in-graph argsort and scatter regrouping. It carries
     the skin list, so the model would compact per row, as in `fast_calc._step`.
   - The `force_output="edge-force"` layout: the model returns per-edge
     ∂E/∂r_ij, so LAMMPS does the force scatter, and the reverse pass drops
     the `positions[r] − positions[s]` gather adjoint.
4. **Chunk over atoms (§7, #9) to fix the remaining out-of-memory at
   131k–262k.** The bundle's peak memory grows with max_atoms × k × (widths).
   ML-PACE's `chunksize` does exactly this.

## 10. CPU (not measured)

Moriarty was off-limits, and the CPU results file holds no timing rows (§1). So
this section is reasoning only, from the source and the GPU profile:
- **ML-PACE on the CPU** runs the `recursive` evaluator by default
  (`pair_pace.cpp:267`), per atom, across 16 MPI ranks. A (about 4 kB per atom)
  and the weights stay in L1/L2, and there are no atomics.
- **ace-jax on the CPU** executes the same whole-system HLO as on the GPU from
  one process. That means about 600 kB of intermediates per atom streamed
  through DRAM, serial scatter adjoints (XLA:CPU scatters are single-threaded
  loops), and f64 transcendentals.

A 27× gap is consistent with these three factors multiplied together:
- DRAM-streaming vs cache-resident per-atom work;
- the scatter serialisation (XLA:CPU emits scatters as sequential loops, as
  far as I know);
- 16 ranks vs one XLA thread pool, which parallelises poorly across fusions.

On the CPU, the pool-first and recurrence changes should help at least as much
as on the GPU. Chunking (#9) is essential there, to keep a chunk's
intermediates cache-resident. This is not verified.

## 11. Reproducing

Setup, from the worktree root:

```bash
# copy the git-ignored models from the main checkout:
cp /Users/u1470235/gits/ace-jax/.worktrees/bench-scaling/bench/scaling/models/pace_*.yace bench/scaling/models/
MP="uv run --with modal modal run bench/perf/modal_profile.py"
```

A100 profiles:

```bash
$MP::probe_main                                        # tools and GPU in the image
$MP::acejax --model Cantor_medium --n 8192             # §2.1 split, §3 stages, cost analysis
$MP::acejax --model Cantor_medium --n 8192 --no-stages --tag _trace   # §3 kernel table
$MP::mlpace --model Cantor_medium --n 8192             # §4 Kokkos kernel timer
$MP::mlpace_chunk                                      # §4 ML-PACE chunksize headroom
```

Prototypes (§8, §9):

```bash
$MP::variants --model Cantor_medium --n 8192           # §8.1 first block
$MP::variants --model Cantor_medium --n 8192 --names baseline,pool+fm,rec+pool+fm,rec+pool+rev+fm \
    --trace rec+pool+rev+fm --tag _fm2                 # §8.1 second block + trace
$MP::e2e --model Cantor_medium --ns 1024,8192 --tag _v2  # §8.2
$MP::e2e --model Cantor_medium --ns 8192,65536         # §8.2 (65k row)
$MP::micro --models Cantor_medium,SiGe_medium --ns 8192,32768   # §8.3
$MP::lammps_variants                                   # §9 (8k, 32k)
$MP::lammps_variants --ns 131072,262144 --modes owned,owned:rec+pool+fm --tag _large
```

One-GPU sweep, both codes (§2; about 60 min):

```bash
$MP::bigsweep
python bench/perf/tables.py bench/perf/results/bigsweep.json
```

Local checks:

```bash
PYTHONPATH=bench:src uv run python bench/perf/profile_acejax.py \
    bench/scaling/models/pace_Cantor_medium.yace Cantor 256 --no-stages --variant rec+pool+rev+fm   # parity (CPU)
PYTHONPATH=bench:src uv run python bench/perf/lammps_capacity.py pace_Cantor_medium Cantor 8192 32768 65536 262144
uv run python bench/perf/show_profile.py bench/perf/results/<file>.json          # kernel tables
uv run python bench/perf/show_profile.py --brief bench/perf/results/variants_*.json
```

The benchmark rows of §1 were read with a one-off pivot over
`bench/scaling/results/modal-a100.jsonl`. It keys on
(code, mode, model, dtype, layout) and computes n / call_s (end to end) and
n / force_s (model only); nothing in `bench/scaling/results` was modified.

## 12. Caveats

- **ML-PACE's FLOP and byte counts are estimates** from source reading (§5).
  XLA's counts treat a transcendental as one flop.
- **The kernel timer adds fences.** ML-PACE runs 8–10% slower with it than
  without, so the per-kernel times sum slightly above the untimed step time.
- **Modal assigns SXM4 or PCIe A100s per container**, which I could not choose.
  The same baseline varied by up to 8% across containers. Every ratio in this
  report is taken within one container.
- **The prototypes are research code under `bench/perf/`, not library
  changes:**
  - `rev` needs reverse-slot indices, computed on the host here (§7, #6);
  - `fm` supports only the `distance` inner cutoff (not `zbl`);
  - the skin calculator uses the dense layout only.
- **nsys was not usable** (only the target binary bundled with Nsight Compute
  is present). ncu is present but was not needed: the JAX trace plus the HLO
  dump gave per-kernel times and attribution, and kokkos-tools did the same
  for ML-PACE.
