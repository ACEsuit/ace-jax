# Where ace-jax's CPU gap comes from

Branch `perf/cpu-gap-profile` (from `origin/main` at `7ae532e`, ace-jax 0.1.2). An
investigation only: `src/` is unchanged. Every number below was measured on
moriarty for this report. The scripts are in `bench/perf/` (`cpu_*.py`), the raw
rows in `bench/perf/results/cpu_gap/`, and section 8 gives the commands.

## Summary

The published CPU gap mixes two parallel models: ace-jax is one process with
XLA's thread pool, while ML-PACE and the ACEpotentials.jl trim library run one
MPI rank per core. Measured separately on moriarty (medium models, 2,048 atoms,
float64, energy + forces + virial per MD step):

**The 11–13× gap to ML-PACE at 16 cores = 3.4–4.0× per core × 3.0–3.3×
parallel scaling.**

| PACE model vs ML-PACE (µs per atom-step) | SiGe | Cantor |
|---|--:|--:|
| ace-jax, 1 core | 221.8 | 127.6 |
| ML-PACE, 1 rank | 56.0 | 38.0 |
| **per-core gap** | **3.96×** | **3.36×** |
| ace-jax speed-up, 1 → 16 cores (XLA threads) | 3.48× | 3.13× |
| ML-PACE speed-up, 1 → 16 ranks | 10.5–11.5× | 10.1–10.3× |
| **parallel-scaling gap** | **3.0–3.3×** | **3.2–3.3×** |
| **gap at 16 cores** (ace-jax 63.7 / 40.7 µs vs ML-PACE 4.88–5.32 / 3.68–3.76 µs) | **12.0–13.0×** | **10.8–11.1×** |

The linear ACE model against the trim library: **17.9× (SiGe) = 4.1× per
core × 4.4× scaling, and 11.1× (Cantor) = 2.7× × 4.0×.** The trim library
scales better than ML-PACE (11.4–13.2× on 16 ranks).

Against direct ACEpotentials.jl, ace-jax is faster per core: 119 vs 149 µs on
SiGe, 93 vs 290 on Cantor. At 16 threads it is 1.3× slower on SiGe and 2×
faster on Cantor. (Julia rebuilds its neighbour list every call and scales
4.1–5.2× on threads.)

**The parallel factor (3.0–3.3×) is XLA:CPU's intra-op threading, not the
hardware.**
- XLA threads give ace-jax 2.6–3.5× on 16 cores.
- The same work as 16 single-thread processes, each on a 1/16 periodic cell,
  gets 8.0–10.4× (§3.3). ML-PACE's ranks get 10.1–11.5×.
- In the 16-thread traces, a third to a half of the time is in thunks that do
  not speed up at all: serial scatters, small batched dots and small fusions.
  The rest gets 2–3×, at ~15 GB/s against a 67 GB/s memory roof.
- An in-process `shard_map` over host devices (rows split, exact) gets
  1.4–2.6× over threads (§3.3).

**The per-core factor (2.7–4.1×) is the formulation, not XLA's per-op code
quality.**
- The force call streams 0.5–1.5 MB per atom (XLA's cost analysis: 1.1–3.1
  GB per 2,048-atom call, a 233–594 MB live set). ML-PACE keeps a few kB per
  atom in L1/L2.
- One core runs it at 0.6–2.9 GFLOP/s and 5–7 GB/s: a few percent of its f64
  peak, and a third of the 20 GB/s a plain copy reaches. Most kernels are
  scalar.
- Two stages hold most of the time (single-thread traces, §4):
  - **The product basis AA and its adjoint:** 60% of ACE SiGe (146 of
    244 ms), 32% of PACE SiGe and 26% of PACE Cantor. For PACE SiGe that is
    71 µs per atom, more than ML-PACE's whole 56 µs. `jnp.prod` over
    gathered factors compiles to layout transposes and pad/slice adjoints
    over (order·n_AA, n) arrays, which run scalar at 2–6 GB/s. On the order ≥ 3
    models the VJP costs 14–18× the stage's forward.
  - **The A assembly:** 46% of PACE SiGe, 33% of PACE Cantor and 39% of ACE
    Cantor. It is made of batched dots with a contraction length of 1–9 (the
    per-l outer products and their adjoints), one-hot species expansions
    that do NZ× the needed work, and transposing copies.
- The rest is per-edge transcendental work that ML-PACE replaces with spline
  tables (the radial stage: 12–37%), and Y_lm (2–9%).
- Neighbour handling is not the problem on the CPU. The skin list is reused
  (no rebuild in steady state), and the step's compaction and force gather
  add 5–8%.

**Rank parallelism alone hits a new ceiling.** 16 ace-jax processes move
48–55 GB/s, against the 67 GB/s that 16 concurrent copies reach. Once the
threading loss is removed, the per-atom bytes of §4 set the 16-core number.

**XLA flags do not help.** Fast math, 512-bit vectors, XNNPACK off, oneDNN,
the thunk-runtime switch and the concurrency scheduler are all within ±3% or
slower. Among the shipped knobs, the defaults (dense, skin 1.0, lean) are
already the fastest. float32 gives 1.35–1.86× but breaks parity.

**Ranked opportunities (§7):**
1. **Rank-parallel CPU evaluation:** 1.5–2.6× at 16 cores measured
   in-process; 2.3–4.0× emulated as separate ranks. **This comes from
   lammps-jax's CPU support (MPI domain decomposition), not from ace-jax:**
   an in-process `shard_map` would duplicate it. ace-jax's own work is
   therefore the per-core items 2–4, which matter more once ranks remove the
   threading loss (16 ranks run near the DRAM roof, §3.3).
2. **A CPU form of the product basis:** feature-major explicit products,
   measured at 1.65× per core on ACE SiGe and 1.13× on PACE SiGe.
3. **No one-hot species expansion in A, and spline tables for the per-edge
   transcendentals:** 1.1–1.4× each, estimated. (A hand-written product-basis
   adjoint was considered and dropped: #2's forward rewrite removes most of
   the reverse-mode cost while keeping autodiff, see §7.)
4. **A fused per-atom kernel with an analytic adjoint:** needed to close the
   last ~2× per core, as ML-PACE does.

## 1. Setup

- **Host:** moriarty, Intel Xeon Silver 4216 (Cascade Lake, AVX-512, 1 FMA
  port per core), 16 cores / 32 hardware threads, 22 MB L3, 1 MB L2 per core,
  62 GB, Rocky 9 (kernel 5.14). Physical cores are CPUs 0–15, their hyperthread
  siblings 16–31 (`lscpu -e`). `grep -c ^processor /proc/cpuinfo` = 32. The
  frequency is 3.2 GHz single-core turbo, lower all-core (not pinned or
  measured, and it is part of every scaling number here, for every code).
- **Pinning:** every case runs under `taskset`. One core is CPU 0; T cores are
  CPUs 0..T-1 (physical cores, no hyperthreads); "32" is 0–31. MPI runs use
  `mpirun --bind-to core --cpu-set 0-(R-1)`. `/proc/loadavg` is recorded before
  and after every case (in the rows).
- **XLA threading (jaxlib 0.11.2):** `intra_op_parallelism_threads` is *not* an
  XLA flag. With the leading `--` it is a fatal unknown flag. Without it XLA
  silently ignores it, so `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
  intra_op_parallelism_threads=1"` only sets the first. The PjRt CPU client
  sizes its pool from the process's CPU affinity, so the thread count is the
  `taskset` set: 10 OS threads at 2 CPUs, 66 at 16, 124 at 32. Single-thread
  runs add `--xla_cpu_multi_thread_eigen=false`, which changes nothing measurable.
- **Software:** ace-jax 0.1.2 (`7ae532e`), jax/jaxlib 0.11.2, Python 3.14.4,
  equinox 0.13.8, numpy 2.5.3. The venv is `/storage/eng/essswb/venvs/ace-jax-cpuprof`
  (no matscipy-neighbours: irrelevant here, since the steady state reuses the
  skin list). LAMMPS `patch_10Sep2025` (`9792f6a`; ML-PACE from
  lammps-user-pace v.2023.11.25.fix2, `pair_style pace` = the recursive
  evaluator) and `develop` `ec02ed0` (for the PR 309 `pair_style ace` plugin).
  OpenMPI 4.1.5 / GCC 12.3. Julia 1.12.6, ACEpotentials.jl 0.10.2,
  EquivariantTensors 0.4.3, JuliaC 0.3.10.
- **Models and structures:** `bench/scaling/models/` (`models.py`), the
  medium size. PACE: 499 (SiGe) and 496 (Cantor) functions per element.
  Linear ACE: 469 and 447. The structures are `structures.supercell`:
  2,048 atoms, SiGe diamond with K = 28 neighbours per atom (fill 1.00),
  Cantor fcc with K = 44 slots (fill 0.95).
- **Trim libraries:** the lestrade builds target the i9-14900K, so I rebuilt
  both medium libraries on moriarty (`cpu_gap_julia.py build-trim`, into
  `/storage/eng/essswb/cpu-gap/trim/`). Gates passed: ETACE vs exact
  ≤ 1.3e-14 eV/Å, exported vs ETACE ≤ 6.4e-15 eV/Å. moriarty has no
  `lmp-ace.sh`, so they run through lestrade's wrapper, which execs
  moriarty's `lammps-dev` binary, and the lestrade-built `aceplugin.so`
  (plain `-O3`, no `-march`).
- **What is timed:**
  - ace-jax: the ASE calculator as `run_standalone.py` runs it: an MD-like
    N(0, 1e-3 Å) displacement before each call, the skin list reused. The
    median over 20 calls, after the compile call and 2 warm-ups. `call_s` and
    the compiled step alone (`step_s`) agree within 1%.
  - LAMMPS: the timed segment after a 50-step warm-up (`run_lammps.lammps_input`:
    200 steps, `neighbor 1.0 bin`). No neighbour rebuild happens in the segment.
  - ACEpotentials.jl direct: `run_standalone.jl`. Its neighbour list is
    rebuilt every call; the others' is not.
- **Noise:** repeated single-thread runs of the same case (the `single`,
  `xla` and probe baselines) agree within ±3% on SiGe (ACE SiGe: 118.8, 120.3,
  122.0 µs) and within ~6% on ACE Cantor (93–99 µs). From 18:24 another session ran single-thread jobs on this
  host, one at a time, on varying CPUs. The 16-core series that overlapped them
  were repeated with each case gated on a 1-minute load below 1.3, i.e. at
  most that one job; §3 gives both runs where they differ. 16-rank LAMMPS
  repeats differ by up to 8%, and 16-process PACE runs by up to 15%. 1-core
  series on CPU 0 were not affected, and the XLA-thread series (§3.1) ran
  before 18:24.

## 2. Per-core cost (1 core, 1 thread, 1 rank)

µs per atom-step, 2,048 atoms (`cpu_gap_sweep.py single`, `cpu_gap_lammps.py`,
`cpu_gap_julia.py direct`):

| evaluator | SiGe | Cantor |
|---|--:|--:|
| ace-jax PACE (`pace_*.yace`) | 221.8 | 127.6 |
| ML-PACE in LAMMPS, 1 rank | 56.0 | 38.0 |
| **ace-jax PACE / ML-PACE** | **3.96×** | **3.36×** |
| ace-jax linear ACE (`ace_*.npz`) | 118.8 | 93.0 |
| ACEpotentials.jl trim library in LAMMPS, 1 rank | 29.0 | 33.9 |
| **ace-jax ACE / trim** | **4.10×** | **2.74×** |
| ACEpotentials.jl direct, 1 thread (neighbour list every call; GC 6–7%) | 149.1 | 290.2 |
| **ace-jax ACE / direct** | **0.80×** | **0.32×** |

LAMMPS spends 99.9% of the 1-rank loop in `Pair`, so these are the
evaluators' costs. ace-jax's per-atom cost rises with system size, from 100 µs
at 256 atoms to 148 µs at 8,192 for ACE SiGe (`nsize` series). This is the
cache signature of its whole-system intermediates. ML-PACE's working set is
per atom.

| ace-jax, 1 core, µs/atom-step | 256 | 512 | 1,024 | 2,048 | 4,096 | 8,192 |
|---|--:|--:|--:|--:|--:|--:|
| ACE SiGe | 100.3 | 99.5 | 113.1 | 118.8 | 129.6 | 148.0 |
| ACE Cantor | 99.1 | 102.7 | 98.9 | 93.0 | 109.7 | 118.1 |
| PACE SiGe | 178.6 | 194.4 | 201.4 | 221.8 | 228.5 | 226.8 |
| PACE Cantor | 128.1 | 130.7 | 134.3 | 127.6 | 140.8 | 143.8 |

## 3. Parallel scaling, 1 → 16 cores

### 3.1 ace-jax with XLA threads vs LAMMPS ranks

µs per atom-step (`threads` series; LAMMPS ranks bound one per core):

| | 1 | 2 | 4 | 8 | 16 | 32 (HT) | speed-up at 16 |
|---|--:|--:|--:|--:|--:|--:|--:|
| ace-jax ACE SiGe | 118.8 | 76.2 | 68.6 | 60.4 | 45.7 | 45.8 | 2.60× |
| ace-jax ACE Cantor | 93.0 | 55.7 | 40.4 | 30.9 | 28.4 | 32.8 | 3.27× |
| ace-jax PACE SiGe | 221.8 | 122.7 | 99.8 | 76.5 | 63.7 | 67.3 | 3.48× |
| ace-jax PACE Cantor | 127.6 | 75.4 | 60.4 | 45.4 | 40.7 | 45.2 | 3.13× |
| ML-PACE SiGe | 56.0 | 29.6 | 15.3 | 8.35 | 4.88–5.32 | | 10.5–11.5× |
| ML-PACE Cantor | 38.0 | 19.2 | 10.2 | 5.78 | 3.68–3.76 | | 10.1–10.3× |
| trim SiGe | 29.0 | 15.0 | 8.00 | 4.21 | 2.55 | | 11.4× |
| trim Cantor | 33.9 | 17.2 | 9.27 | 4.64 | 2.56 | | 13.2× |
| ACEpotentials.jl direct SiGe (threads) | 149.1 | | | | 33.8–36.1 | | 4.1–4.4× |
| ACEpotentials.jl direct Cantor (threads) | 290.2 | | | | 55.3–55.7 | | 5.2× |

- The 16-core rows reproduce the published moriarty-cpu numbers. ace-jax ACE
  SiGe: 21.9k atom-steps/s here vs 22.4k published (which used 32 threads).
  ML-PACE: 188–205k / 266–272k vs 208k / 275k.
- Hyperthreads do not help ace-jax (32 threads ≤ 16).
- ML-PACE's own loss at 16 ranks is mostly `Comm` (7–16% of the loop at
  2,048 atoms: 128 owned atoms per rank against a ghost shell several
  times that), plus the all-core clock.
- 16-rank LAMMPS rows were run twice (§8); the range is the two runs. Unbound
  ranks (`mpirun --bind-to none`) are 1.1–1.7× slower than bound ones, with
  Comm at 18–41% of the loop (two runs each).

### 3.2 Where XLA's threading loses it

`cpu_thread_scaling.py` matches every thunk of the force program between the
1-thread and 16-thread traces (the same HLO). It buckets them by speed-up:

| ACE SiGe, efv program | thunks | ms at 1 thread | ms at 16 | share at 16 |
|---|--:|--:|--:|--:|
| speed-up < 1.5× (serial) | 153 | 29.0 | 30.6 | 32% |
| 1.5–4× | 69 | 171.3 | 59.9 | 63% |
| 4–8× | 4 | 19.6 | 4.3 | 5% |
| ≥ 8× | 19 | 23.9 | 0.5 | 0% |
| **all** (wall 238 → 84 ms: 2.8×) | 245 | 243.8 | 95.3 | |

| PACE Cantor, efv program | thunks | ms at 1 | ms at 16 | share at 16 |
|---|--:|--:|--:|--:|
| speed-up < 1.5× (serial) | 205 | 47.1 | 52.4 | 47% |
| 1.5–4× | 65 | 144.3 | 49.9 | 44% |
| 4–8× | 7 | 46.3 | 9.4 | 8% |
| ≥ 8× | 2 | 12.2 | 0.6 | 1% |
| **all** (wall 250 → 80 ms: 3.1×) | 279 | 249.9 | 112.2 | |

By kind, at 16 threads:

| kind | ACE SiGe: speed-up | ACE SiGe: share at 16 | PACE Cantor: speed-up | PACE Cantor: share at 16 |
|---|--:|--:|--:|--:|
| elementwise / reduce fusions | 3.1× | 50% | 2.0× | 41% |
| library dots (XNNPACK/YNN, Eigen) | 1.9× | 21% | 2.9× | 22% |
| scatters (`wrapped_scatter*`, the gather adjoints) | 0.9× | 11% | 1.0× | 6% |
| copies / transposes | 3.3× | 10% | 2.1× | 20% |

- **Scatters are serial.** XLA:CPU runs `scatter-add` as one sequential loop.
- **Small dots are serial.** The `dot.*`/`dot_general.*` thunks with tiny
  per-batch matrices (the spline's (1×4)@(4×F) per edge, contraction 3–9 in
  A) run 0.8–0.9×: no faster on 16 threads.
- **Fused loops get only 2–3×,** though they are memory-streaming kernels.
  The 16-thread call moves 1.28 GB (ACE SiGe, cost analysis) in 84 ms:
  15 GB/s, against 67 GB/s for 16 concurrent copies (`cpu_stream.py`).
  So the limit is not DRAM bandwidth. It is the partitioning of each of the
  ~280 thunks per call, with a fork/join at every thunk boundary. (The thunk
  executor overlaps independent thunks: thunk time 95 ms > wall 84 ms.)

### 3.3 Rank-style parallelism, emulated

**Separate processes** (`cpu_gap_sweep.py procs`): P concurrent 1-thread
processes, process p on CPU p, each on a periodic 2048/P-atom cell. That is
the owned-rows work of a domain-decomposed rank without the ghost exchange.
Shared L3 and DRAM contention and the all-core clock are included.

| µs per atom-step | P = 2 | 4 | 8 | 16 | speed-up at 16 vs 1 core | vs 16 XLA threads |
|---|--:|--:|--:|--:|--:|--:|
| ACE SiGe | 60.1 | 31.0 | 18.9 | 11.5 | 10.4× | **4.0×** |
| ACE Cantor | 50.9 | 27.2 | 15.5 | 11.1 | 8.4× | **2.6×** |
| PACE SiGe | 103.6 | 53.7 | 38.4 | 27.2 | 8.2× | **2.3×** |
| PACE Cantor | 69.4 | 35.7 | 23.4 | 15.9 | 8.0× | **2.6×** |

The P = 16 rows ran twice; the table shows the later run. In atom-steps/s:
- ACE SiGe 88.5k then 87.3k; ACE Cantor 89.8k then 90.1k;
- PACE SiGe 33.9k then 36.8k; PACE Cantor 73.8k then 62.9k.

The first PACE runs had a load of 3.5 from another session's jobs, the later ones
1.1–1.3. So read the PACE 16-process numbers as ±15%: 8.0–9.4× over one core
for PACE Cantor, and 2.6–3.0× over threads.

- **Rank-style parallelism closes 2.3–4.0× of the 16-core gap.** That is most
  of the 3.0–4.4× parallel factor of §3.1. 16 processes reach 8.0–10.4× over one
  core, against ML-PACE's 10.1–11.5× and the trim library's 11.4–13.2×.
- **At 16 processes ace-jax reaches the DRAM roof.** 16 concurrent copies
  reach 67 GB/s in total (`cpu_stream.py`). 16 processes of ace-jax move
  48–55 GB/s by XLA's byte count: ACE SiGe 87k atom-steps/s × 625 kB, PACE
  SiGe 37k × 1.5 MB, PACE Cantor 63k × 858 kB. ML-PACE's working set is
  per atom, so it has no such ceiling. **Once the threading loss is gone,
  per-atom bytes are the limit, so the per-core work of §4 sets the 16-core
  number too.**

A 256-atom cell costs 0.81–1.07× per atom what 2,048 atoms do (§2 table;
128 atoms was not measured), so part of the 16-process gain on SiGe is cache,
not parallelism. Rank parallelism gives ace-jax 50–65% efficiency at 16
(8.0–10.4×), against ML-PACE's 63–72%; the DRAM roof above is the
difference.

**In-process `shard_map` over host devices** (`cpu_shard_probe.py`):
`--xla_force_host_platform_device_count=D`, the dense rows split over D CPU
devices, `energy_forces_virial_dense(return_edge_grad=True)` per shard,
E and V `psum`-ed. dE/drij matches the unsharded call exactly (max diff
0–1e-16). Compared with the same call on 1 device with the D cores as XLA
threads:

| efv call, µs per atom | 8 cores: XLA threads | 8: sharded | speed-up | 16 cores: XLA threads | 16: sharded | speed-up |
|---|--:|--:|--:|--:|--:|--:|
| ACE SiGe | 56.1 | 23.1 | **2.42×** | 41.7 | 16.0 | **2.60×** |
| ACE Cantor | 29.3 | 21.5 | 1.37× | 27.0 | 17.4 | 1.55× |
| PACE SiGe | 75.7 | 41.4 | 1.83× | 62.6 | 39.4 | 1.59× |
| PACE Cantor | 43.6 | 28.8 | 1.51× | 41.3 | 21.5 | 1.92× |

- This is a working in-process implementation of the model call, exact to
  0–3e-16 in dE/drij. It gets 1.4–2.6× over XLA threads. That is less than
  the separate processes, because the devices share one runtime and one XLA
  thread pool.
- Most of these runs had another session's single-thread job on the box (load
  before 0.9–2.2, recorded in `shard.jsonl`), so take them as lower bounds
  (the 16-device PACE Cantor case had 2.2).
- An earlier run of ACE SiGe / PACE Cantor at 8 devices gave 2.33× / 2.08×.

## 4. Per-stage breakdown on one core

Two complementary measurements, both on CPU 0 at 2,048 atoms:
- isolated stages (`cpu_stages.py`): each stage jitted alone, forward and
  forward + VJP, with XLA's cost analysis;
- a `jax.profiler` trace of the whole E / dE/drij / V program, joined to its
  optimised HLO (`cpu_trace.py`). XLA:CPU emits one trace event per thunk. Each
  thunk is attributed to stages by the source frames of its fused
  instructions, and to forward or backward by `transpose(...)` in its op_name.
  Its object file (one per kernel with `--xla_dump_to`) is disassembled to
  count SIMD vs scalar FP instructions.

The model evaluated is the calculator's `eval_model`. For ACE that is `lean`:
pruned, pair folded, l-blocked and species-compact (`blk_compact`), with
spline R_nl. For PACE it is the pool-first `PACEModel`.

### 4.1 Whole-program numbers

| | ACE SiGe | ACE Cantor | PACE SiGe | PACE Cantor |
|---|--:|--:|--:|--:|
| energy only (ms / 2,048 atoms) | 47.9 | 61.1 | 126.2 | 190.9¹ |
| E + dE/drij + V (ms) | 239.2 | 181.9 | 452.1 | 250.3 |
| ratio (forces / energy) | 5.0 | 3.0 | 3.6 | 1.3¹ |
| calculator step (adds compaction + force gather) | 251.6 | 196.5 | 466.0 | 268.5 |
| kflop / atom (efv, XLA cost analysis) | 72 | 94 | 628 | 292 |
| kB / atom moved (efv) | 625 | 529 | 1,501 | 858 |
| achieved GFLOP/s | 0.62 | 1.06 | 2.85 | 2.39 |
| achieved GB/s | 5.4 | 6.0 | 6.8 | 7.0 |
| live buffer set (XLA buffer assignment) | 278 MB | 233 MB | 594 MB | 324 MB |
| thunks per call (efv) | 280 | 266 | 487 | 408 |
| thunk time in scalar-dominant / vector-dominant / library-or-no-FP kernels (ms) | 154 / 17 / 73 | 56 / 33 / 100 | 130 / 107 / 215 | 28 / 90 / 132 |

¹ The forward-only PACE Cantor program is fused worse than the forward part of
the force program: its SBessel recurrence fusion is duplicated into its
consumers. The 1.3× ratio is an artefact of that.

For scale: one core does a 256 MB `b[:] = a` copy at 19.9 GB/s (read + write;
`cpu_stream.py`). Its f64 FMA peak is about 16 flop/cycle × 3.2 GHz
≈ 51 GFLOP/s (nominal, not measured). ace-jax runs at 1–6% of that peak, and
at 27–35% of that copy bandwidth. The whole-call intermediates do not fit in
cache (233–594 MB live, against a 22 MB L3), so most of the kernels below
stream from DRAM.

### 4.2 By stage (trace of the force program, ms per call, 2,048 atoms)

| stage | ACE SiGe fwd / bwd | ACE Cantor fwd / bwd | PACE SiGe fwd / bwd | PACE Cantor fwd / bwd |
|---|--:|--:|--:|--:|
| product basis AA (+ rho for PACE) | 60.5 / 85.8 (**60%**) | 11.6 / 14.5 (14%) | 35.9 / 108.6 (**32%**) | 17.8 / 47.4 (**26%**) |
| A assembly | 10.1 / 22.0 (13%) | 19.0 / 54.5 (**39%**) | 110.5 / 96.2 (**46%**) | 44.6 / 39.0 (**33%**) |
| radial (transform, envelope, spline / SBessel) | 33.2 / 12.7 (19%) | 51.6 / 17.5 (**37%**) | 36.2 / 17.9 (12%) | 65.3 / 17.0 (**33%**) |
| Y_lm | 2.4 / 3.6 (2%) | 3.8 / 5.3 (5%) | 7.9 / 31.6 (9%) | 5.3 / 5.7 (4%) |
| readout / embedding / core / pair | 5.8 / 4.7 (4%) | 5.0 / 1.4 (3%) | 1.2 / 1.2 (1%) | 1.1 / 2.0 (1%) |
| strain trick, force mask, misc | 1.6 / 1.4 (1%) | 2.1 / 2.1 (2%) | 2.7 / 1.3 (1%) | 2.6 / 2.0 (2%) |
| **total** | 113.6 / 130.2 = 244 | 93.2 / 95.2 = 188 | 194.3 / 256.9 = 451 | 136.8 / 113.1 = 250 |

The isolated-stage timings agree in ranking. Their forward sums match the
energy-only programs (46.5 vs 47.9 ms ACE SiGe; 65.4 vs 61.1 ACE Cantor;
122.8 vs 126.2 PACE SiGe). The VJP / forward ratio per stage is where reverse
mode costs most: AA 17.6× (ACE SiGe), 14.1× (PACE SiGe), 17.5× (PACE Cantor);
A 3.3–3.9× (ACE). The full tables are in `results/cpu_gap/stages_*.json` and
are printed by `cpu_gap_tables.py stages`.

### 4.3 The hot kernels and what they compute

ACE SiGe (order 3, 739 + 234 + 18 AA entries over 84 A entries):

| thunk | ms | what (from the HLO) | MB moved | GB/s | FP instructions zmm / ymm / scalar |
|---|--:|---|--:|--:|---|
| `multiply_bitcast_fusion.1` | 40.5 | forward order-3 product: the gathered (739·3, n) factors transposed to (3, n, 739) and multiplied, part of `jnp.prod`'s JVP | 72.6 | 1.8 | 0 / 3 / 15 |
| `bitcast_copy_fusion.2` | 25.4 | its adjoint: pads, adds, and a transpose back to (2217, n) | 145.3 | 5.7 | 0 / 2 / 8 |
| `multiply_pad_fusion{,.1}` | 23.9 + 23.4 | the "product of the other factors" for the adjoint, padded to (2, n, 739) | 101 + 101 | 4.3 | 0 / 2 / 6 |
| `ynn_fusion.2` | 15.1 | adjoint of the per-l outer product `nkr,nky->ryn` (batched dots, contraction 1–5) | 29.8 | 2.0 | library |
| `slice_pad_fusion` | 11.9 | forward: a pad of a (1, n, 739) slice to (2, n, 739) | 60.5 | 5.1 | none |
| `wrapped_scatter` | 7.3 | the gather adjoint: scatter-add of (2217, n) into A (84, n); serial | 39.1 | 5.4 | – |

`jnp.prod(A[:, g], axis=-1)` is the cost. XLA lowers prod's reverse mode
through a JVP of products of slices with pads, and lays out the gathered
factors so that every step transposes. The kernels do no arithmetic to speak of
(3.5 MFLOP for the whole forward AA), run scalar at 2–6 GB/s, and move
~25 kB per atom per pass.

ACE Cantor (order 2: AA is small): `ynn_fusion.2`, 37.6 ms, 1.6 GB/s. This is
the adjoint of the l-blocked A assembly. A batched dot (2048 batch) of (44×3) @
(3×20) per l is fused with the one-hot z_j reduction over 5 species
(`(90112, 5, 7)` · broadcast → reduce over the species axis). It is a
contraction of length 3 per atom, so XNNPACK has nothing to block on, and the
one-hot multiplies the work by NZ = 5.

PACE SiGe (lmax 4, nradbase 13, 2,941 AA entries up to order 5):
`copy_bitcast_fusion.1` (66.5 ms, the adjoint of the pool-first W[z] contraction,
scalar, 1.7 GB/s), two `transpose_copy_fusion` (34 + 26 ms) feeding the A
einsums, then the AA adjoint copies/pads (24 + 12 + 12 ms).

PACE Cantor: `bitcast_concatenate_fusion.3` (20.2 ms, 0.64 GB/s, 238 ymm
instructions) is the SBessel rotation recurrence (sin/cos/reciprocal per edge),
compute-bound in f64 transcendentals. Then the A-assembly adjoint
(`ynn_fusion.2`, 19.6 ms) and the pool-first transposes.

### 4.4 Does XLA:CPU vectorise?

Partly. Of the thunk time, kernels whose FP instructions are mostly packed
(ymm/zmm) take 17 of 244 ms (ACE SiGe), 33 of 188 (ACE Cantor), 107 of 451
(PACE SiGe) and 90 of 250 (PACE Cantor). Scalar-dominant kernels take
154 / 56 / 130 / 28 ms. The rest is library dots (XNNPACK "YNN" fusions,
Eigen), gathers, copies and scatters.

The vectorised kernels are the elementwise per-edge maths. The
transpose/pad/gather kernels of the product basis and A assembly are scalar,
because their access is strided. Hardly any code uses zmm: XLA prefers
256-bit vectors on this CPU, and `--xla_cpu_prefer_vector_width=512` makes it
0–11% slower (§6).

## 5. Mapping to ML-PACE's C++ (`ace_recursive.cpp`, `ace_radial.cpp`)

ML-PACE's CPU evaluator (`pair_style pace`, recursive) loops over atoms. Per
atom:
1. One pass over neighbours. Each neighbour does:
   - a cubic spline lookup of g_k, R_nl and the core term, with value and
     derivative from one table read;
   - complex Y_lm and dY_lm for m ≥ 0;
   - accumulation into A[μ_j][n][l][m], with R, dR, Y, dY cached per
     neighbour.
2. The conjugate −m half.
3. The recursive DAG for the product basis and ρ, then its backward pass for
   the weights dE/dA.
4. A second neighbour pass: f_ij = Σ w · ∇(R Y) from the cached values.

The working set is one atom's A and weights (a few kB) plus its ~30–45
neighbours' caches. It stays in L1/L2, there are no atomics, and no array
spans the system.

Where ace-jax does more work, measured or counted:

| ace-jax | ML-PACE | cost on the CPU (§4) |
|---|---|---|
| **Whole-system, materialised intermediates:** every stage writes an (n, …) or (n·K, …) array. 233–594 MB live, 0.5–1.5 MB moved per atom per call; per-atom cost rises 1.1–1.5× from 256 to 8,192 atoms | per-atom, cache-resident | the 5–7 GB/s streaming regime of §4.1, at every stage |
| **Product basis by `jnp.prod` over gathered factors**, reverse-mode: every factor gathered into an (order·n_AA, n) array, transposed, multiplied, then the adjoint scatter-adds into A. VJP / fwd 14–18× | DAG recursion that reuses partial products, and an analytic backward over the same DAG (Weights) | ACE SiGe 146 ms (60%), PACE 65–145 ms (26–32%) |
| **One-hot species channel:** ACE's species-compact A and PACE's pool-first both expand each edge to NZ channels (`one_hot(zj)` ⊗ features: 5× on Cantor, 2× on SiGe) before contracting | the channel μ_j is an index: `A(mu_j, n) += …` | part of the A stage, 33–46% for PACE and ACE Cantor |
| **A as batched tiny dots** (per-l outer products with contraction 1–9; spline as a (1×4)@(4×F) dot per edge) | scalar loops accumulating into A | serial dot thunks (§3.2); `ynn_fusion.2` 15–38 ms |
| **Per-edge transcendentals:** PACE SBessel (sin/cos/reciprocal recurrence); the ACE Agnesi transform `s**q`, `s**(q-p)` (pow) for the many-body and pair radials | spline tables of g_k, R_nl and the core term, per element pair | radial 12–37% |
| **Generic reverse mode:** saved per-edge intermediates, a scatter or gather adjoint for every gather; forces cost 3–5× energy | the analytic adjoint reuses cached R, dR, Y, dY; derivatives are computed in the forward pass | the bwd columns of §4.2 |
| Dense padding: SiGe fill 1.00, Cantor 0.95 (K = 44 slots) | the per-atom neighbour list, filtered to r < rcut | ≤ 5% |
| Skin compaction and the reverse-slot force gather in the step | LAMMPS's half/full list and the force reduction | step − efv = 12–18 ms (5–8%) |

Two ACE-specific notes:
- The trim library is 2× faster per core than ML-PACE on SiGe (29.0 vs 56.0 µs),
  on a model of similar size.
- ace-jax's lean ACE beats direct ACEpotentials.jl per core (119 vs 149 µs SiGe,
  93 vs 290 µs Cantor). Julia's direct path rebuilds its neighbour list every
  call and allocates (GC 6–7% at 1 thread, 21–26% at 16).

## 6. What-if probes

### 6.1 Shipped knobs and XLA flags (1 core, µs/atom-step; ratio > 1 is faster than the default)

| knob | ACE SiGe | ACE Cantor | PACE SiGe | PACE Cantor |
|---|--:|--:|--:|--:|
| default (dense, skin 1.0, lean, CHUNK_NODES 16384) | 118.8 | 93.0 | 221.8 | 127.6 |
| float32 | 88.2 (1.35×) | 68.9 (1.35×) | 119.1 (1.86×) | 83.2 (1.53×) |
| skin = 0 (list rebuilt every call) | 183.2 (0.65×) | 214.2 (0.43×) | 291.9 (0.76×) | 258.4 (0.49×) |
| layout sparse (edge_a auto) | 221.1 (0.54×) | 401.3 (0.23×) | 360.9 (0.61×) | 304.4 (0.42×) |
| sparse, edge_a matmul / gather | 0.53× / 0.42× | 0.23× / 0.15× | – | – |
| lean = False | 176.9 (0.67×) | 225.6 (0.41×) | – | – |
| CHUNK_NODES 32 / 128 / 256 / 1024 | **1.42×** / 1.36× / 1.20× / 0.95× | 0.72× / 0.77× / 0.80× / 0.78× | 1.10× / 1.06× / 1.03× / 0.88× | 0.56× / 0.65× / 0.60× / 0.57× |
| `--xla_cpu_enable_fast_math=true` | 0.97× | 0.93× | 0.99× | 0.98× |
| `--xla_cpu_prefer_vector_width=512` | 0.97× | 0.90× | 0.92× | 0.89× |
| `--xla_cpu_use_xnnpack=false` | 0.97× | 0.90× | 0.96× | 0.98× |
| `--xla_cpu_use_thunk_runtime=false` | 0.97× | 0.90× | 0.94× | 0.98× |
| `--xla_cpu_use_onednn=true` | 0.97× | 0.98× | 0.94× | 0.98× |
| 16 cores + `--xla_cpu_enable_concurrency_optimized_scheduler=true` (vs 16-core default) | 45.2 (1.01×) | 29.7 (0.96×) | 70.5 (0.90×) | 40.8 (1.00×) |

- **XLA flags:** none helps.
  - All the flag rows ran in one later series. Its unflagged level is a few
    percent above the `single` baseline: the 1-core repeats spread ±3% on SiGe
    and up to ~6% on ACE Cantor (93–99 µs).
  - So 0.9–1.0× reads as "no effect". `prefer_vector_width=512` is the one
    consistently slower flag on PACE (0.89–0.92×).
  - `use_thunk_runtime=false` is accepted and changes nothing measurable.
- **`CHUNK_NODES`:**
  - Small blocks make ACE SiGe's product-basis transposes cache-resident
    (1.42× at 32 rows), despite `jax.checkpoint` recomputing the forward.
  - Everywhere else the recompute and the `lax.map` loop cost more than the
    cache saves.
  - At 2,048 atoms the default is one block. Chunking matters only above
    16,384 atoms, where it bounds memory.
- **The defaults are right.** The skin reuse, the dense layout and `lean` are
  each worth 1.3–4× on the CPU.

### 6.2 Formulation probes (patched in-process; step time, 1 core; forces checked against the unpatched step)

`cpu_probe.py`:

| model | product basis | spline | CHUNK_NODES | speed-up | max \|ΔF\| (eV/Å) |
|---|---|---|---|--:|--:|
| ACE SiGe | explicit chain of multiplies, node-major | | | 1.27× | 4e-14 |
| ACE SiGe | **feature-major explicit** (At = A.T; At[g0]·At[g1]·…) | | | **1.65×** | 4e-14 |
| ACE SiGe | feature-major | elementwise 4-term sum | | 1.49× | 4e-14 |
| ACE SiGe | feature-major | | 32 / 128 | 1.58× / 1.51× | 4e-14 |
| ACE SiGe | (as shipped) | elementwise 4-term sum | | 0.94× | 4e-14 |
| ACE Cantor | explicit / feature-major | | | 0.97× / 0.94× | 6e-15 |
| PACE SiGe | explicit chain (already feature-major) | | | 1.13× | 4e-16 |
| PACE SiGe | explicit chain | | 32 | 1.20× | 5e-15 |
| PACE Cantor | explicit chain | | | 1.03× | 2e-16 |
| PACE | (as shipped) | elementwise sum | | 0.99–1.01× | 0 |

- **Feature-major explicit products** remove `jnp.prod`'s JVP machinery and
  its transposes. That gives 1.65× on the order-3 ACE SiGe model. They do
  nothing for Cantor's order-2 ACE, whose cost is in the A stage.
- They do not compound with chunking: both remove the same transposing
  traffic.
- `ace-vs-pace-gap.md` 4.1 chose the transposed handover for the GPU (the
  order-3/4 adjoint layout regressed there). So a CPU variant would have to be
  backend-selected.
- Rewriting the spline contraction as an elementwise sum does not help: XLA
  already fuses that dot well enough.

## 7. Ranked optimisation opportunities

Estimates are per 16-core moriarty throughput unless stated. "Measured" means a
probe on this branch at force parity. "Estimated" means reasoned from the
§3–§5 numbers.

| # | change | speed-up | evidence | effort | risk |
|---|---|---|---|---|---|
| 1 | **Rank-parallel CPU evaluation.** The calculator's step under `shard_map` over host devices: dense rows split across D = cores devices, 1 thread each, E/V `psum`, the force gather after an all-gather of dE/drij. Alternatively the lammps-jax bundle on CPU with MPI ranks. **Decision: left to lammps-jax**, which is gaining CPU support with MPI domain decomposition; ace-jax will not add its own `shard_map` path | **1.5–2.6× at 16 cores (measured in-process); 2.3–4.0× as separate ranks (emulated)** | §3.3: shard_map 1.4–2.6× over threads (exact); 16 processes 2.3–4.0× over 16 threads, at 8.0–10.4× of one core | S–M (calculator: device mesh, row padding to D, the gather) | low; exact. The XLA device count must be set before JAX starts, so it is an opt-in env/config, and it fights other in-process JAX use |
| 2 | **CPU product basis: feature-major explicit products** (no `jnp.prod`; At[g0]·At[g1]·…), backend-selected so the GPU keeps its layout | **1.65× per core** on ACE SiGe (order 3), 1.13× PACE SiGe, ~1× on order-2 Cantor | §6.2 measured; §4.3 AA kernels | S | low (exact to 4e-14); keep the GPU path |
| (3) | ~~Analytic adjoint of the product basis~~ (`custom_vjp`). **Dropped:** hand-written gradients are not wanted, and most of the reverse-mode cost was `jnp.prod`'s lowering (pad/slice JVP, transposing copies), which #2's forward rewrite removes while keeping autodiff. The remaining serial `wrapped_scatter` (3% of 1-core time) can be addressed by the gather's formulation (sorted segment-sum or a one-hot matmul) if it still shows after #2, and matters less under one-thread MPI ranks | – | §4.2, §4.3 | – | – |
| 4 | **No one-hot species expansion in A** (ACE `blk_compact`, PACE pool-first): contract per species with a static loop over NZ on masked or sorted-by-species slots, or pool per (node, z_j) by segment-sum into an (n, NZ, …) buffer | 1.2–1.4× per core on Cantor (NZ = 5), ~1.1× on SiGe | A stage 33–46% on PACE and ACE Cantor; the one-hot multiplies its flops and bytes by NZ (§4.3 `ynn_fusion.2`) | M | low–medium (fusion and layout changes again need GPU checks) |
| 5 | **Replace per-edge transcendentals with tables** (ML-PACE's approach): PACE g_k(r) splines per pair; ACE: spline the radial in r including the Agnesi transform and envelope, not only in x | 1.1–1.3× per core (radial is 12–37%; PACE Cantor's SBessel fusion alone is 20 ms, 8%) | §4.2, §4.3 | M | parity becomes spline-tolerance (as `lean`'s learned-radial splines already are, ~1e-9) |
| 6 | **Fewer, larger kernels for the per-edge chain and A** (avoid the tiny batched dots: write the per-l outer products as broadcast-multiply-reduce fusions, so XLA emits loops it partitions) | 1.1–1.3×, mostly through thread scaling: library dots are 21–22% of 16-thread time at a 1.9–2.9× speed-up, and the smallest are serial (0.8–0.9×) | §3.2 dot rows | S–M | low; must be checked against the GPU's GEMM path |
| 7 | **A fused per-atom kernel with an analytic adjoint** (an XLA FFI custom call in C++, or a Pallas-CPU/Triton-CPU kernel): ML-PACE's recursive evaluator structure behind the same `EdgeSiteModel` contract | up to the remaining 2–3× per core after #2–#6, i.e. ML-PACE/trim per-core parity; with #1 the whole gap | §5: the remaining difference is whole-system streaming vs per-atom cache residency | L | high: the lammps-jax bundle must resolve the FFI target (the reason `harmonics.py` avoids one), and two implementations need parity tests |
| 8 | Smaller `CHUNK_NODES` on the CPU | 1.4× ACE SiGe only; 0.6–0.8× elsewhere | §6.1 | XS | superseded by #2; not worth a default |
| 9 | float32 | 1.35–1.86× | §6.1 | – | breaks the float64 parity contract; not recommended |
| – | XLA flags (fast math, vector width, XNNPACK, oneDNN, scheduler) | none | §6.1 | – | – |

#1 and #2–#6 compose: one is parallel efficiency (to come from lammps-jax's CPU
MPI ranks), the others per-core work (ace-jax's part).
Taking the low ends (2× × 1.65 × 1.2 × 1.1, the 1.2 now from #4 rather than #3) gives about 4–5× on ACE SiGe
at 16 cores. That is about 100k atom-steps/s, against ML-PACE's 188–205k and the
trim library's 392k. Rank parallelism then runs into the DRAM roof (§3.3),
which #2–#6 lower by cutting bytes per atom. The rest needs #7.

## 8. Reproducing

All from the worktree root, with `.venv` → `/storage/eng/essswb/venvs/ace-jax-cpuprof`
and `bench/scaling/models` pointing at the benchmark models
(`~/bench-scaling/ace-jax/bench/scaling/models`).

```bash
export PYTHONPATH=bench:src
P=.venv/bin/python
$P bench/perf/cpu_gap_sweep.py single            # §2: 1 core, CPU 0
$P bench/perf/cpu_gap_sweep.py threads           # §3.1: 2..32 XLA threads
$P bench/perf/cpu_gap_sweep.py nsize             # §2: 256..8192 atoms
$P bench/perf/cpu_gap_sweep.py procs             # §3.3: P concurrent 1-thread processes
$P bench/perf/cpu_gap_sweep.py knobs; $P bench/perf/cpu_gap_sweep.py xla   # §6.1
$P bench/perf/cpu_gap_sweep.py stages --scratch /tmp/x   # §4: stages_*.json, trace_*.json
taskset -c 0-15 $P bench/perf/cpu_stages.py bench/scaling/models/ace_SiGe_medium.npz SiGe 2048 \
    --threads 16 --trace /tmp/x/tr16 --dump /tmp/x/d16              # §3.2
$P bench/perf/cpu_trace.py /tmp/x/tr16 /tmp/x/d16 5 --top 300 > trace16.json
$P bench/perf/cpu_thread_scaling.py trace1full.json trace16.json
taskset -c 0 $P bench/perf/cpu_probe.py bench/scaling/models/ace_SiGe_medium.npz SiGe 2048 --aa fm   # §6.2
taskset -c 0-7 $P bench/perf/cpu_shard_probe.py bench/scaling/models/ace_SiGe_medium.npz SiGe 2048 --devices 8
$P bench/perf/cpu_gap_lammps.py mlpace SiGe 2048 1 2 4 8 16
$P bench/perf/cpu_gap_julia.py build-trim SiGe        # juliac for this CPU
$P bench/perf/cpu_gap_lammps.py trim SiGe 2048 1 2 4 8 16 \
    --trim-lib /storage/eng/essswb/cpu-gap/trim/libace_SiGe_medium.so \
    --lmp /storage/eng/essswb/bench-scaling-lestrade/lmp-ace.sh
$P bench/perf/cpu_gap_julia.py direct SiGe 2048 1 16
$P bench/perf/cpu_stream.py 1 16
$P bench/perf/cpu_gap_tables.py all              # the tables above
```

## 9. Caveats

- **The published table's headline numbers are lestrade's** (i9-14900K,
  8 P-cores, 8 ranks). There, ML-PACE is 423k / 567k and the trim library
  ~1M. This study is moriarty's: ML-PACE 188–205k / 266–272k at 16 ranks, matching
  the published moriarty-cpu rows. Per-core ratios should carry over;
  absolute numbers and scaling do not. lestrade's P-cores are ~2× faster
  per core, and it has 8 cores, so XLA's poor threading costs relatively
  less there. lestrade was not measured.
- **ACEpotentials.jl direct rebuilds its neighbour list every call;**
  ace-jax and LAMMPS do not. So the direct line is not a pure evaluator
  comparison.
- **The trim library compiles the exact (unsplined) radials;** ace-jax and
  direct Julia use the npz's splines. The models are the same basis and
  weights but not bit-identical evaluators (spline error 1.5e-4 eV/Å in
  force, from the build's gates).
- **The rank emulation (§3.3)** has no ghost exchange or communication, and
  its per-process cells (128 atoms at P = 16) are likely cheaper per atom
  than 2,048 atoms (cache; 256 atoms costs 0.81–1.07× per atom, and 128 was
  not measured). It is an upper bound for a real rank-parallel ace-jax.
  The shard_map probe is a real in-process implementation of the model call,
  but without the step's compaction and force gather.
- **Frequency:** turbo was not pinned. Every 16-core number, for every code,
  includes the all-core clock drop.
- **Noise:** ±3–6% between repeated 1-core runs. 16-core runs: up to 8% (LAMMPS)
  and 15% (16 ace-jax processes) between repeats. Another session's
  single-thread jobs shared the host from 18:24 (§1); the load is recorded in
  every row.
- **Environment:** this venv (Python 3.14, no matscipy-neighbours) is not the
  benchmark venv (Python 3.12). Its 16-core numbers reproduce the published
  moriarty-cpu rows (ace-jax ACE SiGe 21.9k vs 22.4k; ML-PACE 188–205k vs 208k),
  and the steady state never builds a neighbour list.
- **Not measured:**
  - ML-PACE's own per-kernel CPU profile. The §5 mapping is from its source,
    not a profiler.
  - Hardware counters (perf/VTune), so "cache-resident" vs "streaming" is
    inferred from the bytes, the live set and the n-dependence, not counted.
  - lammps-jax on the CPU: it ships only the KOKKOS/CUDA `pair_style jax/kk`.
  - lestrade.
