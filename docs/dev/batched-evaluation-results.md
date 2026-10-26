# Batched evaluation: results (phase 1)

`ace_jax.batch.BatchCalculator` evaluates many structures in one compiled call
(spec: `docs/dev/specs/2026-10-23-batched-evaluation-design.md`). This page
records what it gains over `ACECalculator` one structure per call, and the
measurements behind its defaults.

Setup: Si o4d12 linear ACE (`runs/accuracy_cost/ace/o4d12/model.npz`, the
accuracy-cost ladder), rattled (0.05 Å) diamond cells of 2 (primitive), 8,
64 and 216 atoms. Each point is the median of the whole `calculate` call:
packing on the host, the neighbour lists, the model, the copy back. The
loop column is `ACECalculator` over 32 cells, one per call, with its own
defaults (skin list, matscipy-neighbours). µs per atom throughout. Driver:
`bench/batch/run.py`; tables: `bench/batch/table.py`. Raw results:
`/storage/eng/essswb/projects/ace-jax/runs/batch/`.

## GPU (lestrade, RTX 4000 Ada, JAX 0.11.2)

Builders: `jax` (all pairs over periodic images, on the device) and `msn`
(matscipy-neighbours' batched XLA FFI build, libAtoms/matscipy-neighbours#11,
branch `batched-neighbour-matrix` ea16b2c). Final code (evaluation on the
occupied slots, linear-time compaction, power-of-two small buckets), idle GPU,
2026-10-24 (`lestrade-gpu-final.jsonl`; CPU tests ran on the host meanwhile).
The earlier runs, before those changes, are kept as `lestrade-gpu-v2.jsonl`
(5-7x at 64 atoms, 28-54x at 2-8 atoms).

| dtype | atoms | builder | loop | B=1 | B=4 | B=16 | B=64 | B=256 | B=1024 | best speed-up |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| float32 | 2 | jax | 229.2 | 747 | 192 | 46.8 | 13.7 | 5.24 | 3.25 | 71x |
| float32 | 2 | msn | 229.2 | 714 | 190 | 49.7 | 14.6 | 5.55 | 3.19 | 72x |
| float32 | 8 | jax | 58.1 | 207 | 51.9 | 14.9 | 4.66 | 1.86 | 1.39 | 42x |
| float32 | 8 | msn | 58.1 | 192 | 45.9 | 13.0 | 3.85 | 1.72 | 1.28 | 45x |
| float32 | 64 | jax | 7.8 | 28.0 | 7.78 | 2.20 | 0.98 | 0.74 | 0.72 | 11x |
| float32 | 64 | msn | 7.8 | 23.2 | 6.31 | 1.93 | 0.97 | 0.73 | 0.70 | 11x |
| float32 | 216 | jax | 2.5 | 9.44 | 2.49 | 1.57 | 1.03 | 0.97 | 0.90 | 2.8x |
| float32 | 216 | msn | 2.5 | 13.3 | 3.04 | 1.30 | 1.19 | 0.87 | 0.80 | 3.2x |
| float64 | 2 | jax | 370.7 | 925 | 214 | 59.0 | 18.0 | 8.94 | 6.16 | 60x |
| float64 | 2 | msn | 370.7 | 917 | 250 | 76.2 | 21.3 | 7.44 | 5.21 | 71x |
| float64 | 8 | jax | 101.7 | 262 | 57.3 | 17.9 | 7.04 | 3.70 | 3.55 | 29x |
| float64 | 8 | msn | 101.7 | 196 | 59.6 | 16.0 | 5.90 | 2.90 | 3.59 | 35x |
| float64 | 64 | jax | 13.9 | 32.9 | 8.36 | 4.19 | 2.35 | 2.98 | 2.69 | 5.9x |
| float64 | 64 | msn | 13.9 | 33.6 | 11.2 | 3.43 | 2.24 | 2.79 | 2.61 | 6.2x |
| float64 | 216 | jax | 4.9 | 12.9 | 5.49 | 3.34 | 3.93 | 3.72 | 3.64 | 1.5x |
| float64 | 216 | msn | 4.9 | 11.5 | 4.46 | 2.47 | 3.31 | 3.07 | 2.99 | 2.0x |

- The gain is largest where one structure per call is all overhead: 29-72x
  for 2- and 8-atom cells. At 64 atoms it is 6-11x, at 216 atoms 1.5-3.2x (a
  216-atom cell already keeps this card fairly busy).
- float32 64-atom cells reach 0.70 µs/atom for the whole call, about the
  0.66 µs/atom of the model alone on prebuilt lists in the spec's evidence.
- The two builders are within run-to-run noise up to 64 atoms. At 216
  atoms `msn` is 10-20% faster (its image search is pruned by plane distance; the
  `jax` builder visits every image of the range).
- Batch size: most of the gain is in by approximately 4k-16k atoms per call; float32 gains another
  1.1-1.2x to 2^18 (see "Batch size and device memory" below; `MAX_BATCH_ATOMS` is now 2^18).
- B = 1 is 2-3x slower than the loop: a batch of one pays the batch path's
  fixed cost without amortising it. `BatchCalculator` is for many structures.

## GPU (Sulis, A100-PCIE-40GB, JAX 0.11.2)

Same benchmark and code (rev 8826703), Sulis job 2261539 on gpu003, one A100 to itself
(`bench/batch/sulis-gpu.sbatch`; results in `runs/batch/sulis/sulis-a100-{points,md}.jsonl`).
Single points, µs per atom:

| dtype | atoms | builder | loop | B=1 | B=4 | B=16 | B=64 | B=256 | B=1024 | best speed-up |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| float32 | 2 | jax | 483.4 | 1307 | 343 | 94.7 | 29.0 | 12.1 | 8.11 | 60x |
| float32 | 2 | msn | 483.4 | 1376 | 354 | 93.4 | 29.0 | 12.2 | 7.78 | 62x |
| float32 | 8 | jax | 122.2 | 342 | 85.8 | 24.0 | 7.43 | 3.35 | 2.56 | 48x |
| float32 | 8 | msn | 122.2 | 350 | 90.3 | 23.7 | 7.42 | 3.37 | 2.54 | 48x |
| float32 | 64 | jax | 16.5 | 46.1 | 12.3 | 3.60 | 1.41 | 1.10 | 0.99 | 17x |
| float32 | 64 | msn | 16.5 | 42.2 | 11.2 | 3.20 | 1.34 | 1.02 | 0.78 | 21x |
| float32 | 216 | jax | 5.5 | 14.0 | 4.53 | 1.83 | 1.45 | 1.07 | 0.84 | 6.6x |
| float32 | 216 | msn | 5.5 | 13.5 | 3.80 | 1.29 | 1.08 | 0.92 | 0.62 | 8.9x |
| float64 | 2 | jax | 610.8 | 1471 | 367 | 98.3 | 29.0 | 12.0 | 7.94 | 77x |
| float64 | 2 | msn | 610.8 | 1424 | 383 | 95.3 | 28.6 | 12.7 | 7.76 | 79x |
| float64 | 8 | jax | 151.0 | 360 | 91.3 | 24.0 | 7.74 | 3.43 | 2.68 | 56x |
| float64 | 8 | msn | 151.0 | 352 | 87.4 | 23.2 | 7.43 | 3.40 | 2.68 | 56x |
| float64 | 64 | jax | 20.2 | 45.0 | 12.7 | 3.88 | 1.65 | 1.35 | 1.16 | 17x |
| float64 | 64 | msn | 20.2 | 44.0 | 11.7 | 3.31 | 1.29 | 1.26 | 1.11 | 18x |
| float64 | 216 | jax | 6.7 | 15.0 | 4.86 | 2.21 | 1.87 | 1.37 | 1.10 | 6.1x |
| float64 | 216 | msn | 6.7 | 14.0 | 4.00 | 1.42 | 0.95 | 0.80 | 0.79 | 8.5x |

- The one-structure loop is about 2x slower than on lestrade at every size, and so is the batched
  floor for 2- and 8-atom cells (2.5 against 1.3 µs/atom, float32): these are per-call overhead, which
  is the host's (gpu003's CPU and PCIe), not the card's. The speed-up is the same or larger (48-79x).
- float64 costs little more than float32 on the A100 (fp64 at half the fp32 rate; the Ada card is at
  1/64): at 216 atoms 0.79 µs/atom against 2.99 on lestrade, so the 216-atom speed-up rises from 2.0x
  to 8.5x and the 64-atom one from 6.2x to 18x.
- `msn` beats `jax` by 20-40% at 64-216 atoms and B = 1024 (0.78 against 0.99, 0.79 against 1.10);
  below that they tie.
- Still improving at B = 1024 for small cells (2 x 1024 atoms is far from saturating 40 GB). Larger
  cells keep gaining to approximately 256k atoms per call (next section).

Batched MD (`bench/batch/md.py`: JAX-MD NVE under vmap, the JAX model, float64, compacted K and
batch-sized row blocks, 20 steps per jitted chunk, lists fixed), µs per atom per step. Rerun with the
single-species matmul radial (rev 31116ac, Sulis job 2264940 on gpu012; the einsum-era results of job
2261539 are in `runs/batch/sulis/einsum-a9d2877/`):

| atoms | cells=1 | 16 | 64 | 256 | 1024 |
|--:|--:|--:|--:|--:|--:|
| 8 | 44.3 | 3.59 | 1.20 | 0.561 | 0.391 |
| 64 | 6.29 | 0.732 | 0.481 | 0.341 | 0.388 |
| 216 | 2.41 | 0.555 | 0.414 | 0.406 | 0.473 |

- Against the einsum: 0.88-1.17x per point (median 1.07x); the single points of the same job are
  unchanged (median 1.00x in every dtype/builder group), so the table above them stands.  The radial
  change is a compile-time fix, not a speed-up.
- Best 0.34 µs/atom/step (64-atom cells x 256, 16k atoms), 18x one 64-atom cell; 8-atom cells reach
  0.39 at 1024 cells, 113x one cell.
  Below the single-point floor (1.1 µs/atom) because the MD step builds no list and evaluates
  compacted slots.
- Past approximately 16k-55k atoms per call it gets slightly worse (64 x 1024, 216 x 1024).

## Batch size and device memory

`bench/batch/memory.py`: one batched call per point, the padded atoms per call doubled from 4096 until the
device runs out (it never did) or 65,536 cells; each point in its own process, peak from
`peak_bytes_in_use` with `XLA_PYTHON_CLIENT_PREALLOCATE=false` (it includes compile-time autotuning
scratch, which dominates below ~32k atoms). Builder `msn`, o4d12. Sulis A100-PCIE-40GB (jobs
2261695/6, 2261710/1; JAX may use 31.8 GB of it) and lestrade RTX 4000 Ada (20 GB; idle);
`runs/batch/sulis/sulis-a100-memory.jsonl`, `runs/batch/lestrade-gpu-memory.jsonl`.

Single points (`calculate`), peak GB and µs per atom, 64- and 216-atom cells:

| padded atoms | A100 f32 64 | A100 f32 216 | A100 f64 64 | A100 f64 216 | Ada f32 64 | Ada f64 64 |
|--:|--:|--:|--:|--:|--:|--:|
| 16k | 0.16, 1.01 | 0.16, 1.07 | 0.17, 1.26 | 0.17, 1.36 | 0.14, 0.80 | 0.31, 2.83 |
| 64k | 0.22, 0.90 | 0.22, 0.91 | 0.42, 0.97 | 0.42, 1.17 | 0.29, 0.71 | 0.42, 2.60 |
| 256k | 0.89, 0.67 | 0.60, 0.61 | 1.07, 0.79 | 0.88, 0.78 | 0.88, 0.66 | 1.04, 2.51 |
| 1M | 2.32, 0.65 | 2.32, 0.61 | 3.63, 0.78 | 3.63, 0.77 | 2.32, 0.63 | 3.56, 2.50 |
| 4M | 9.0, 0.70 | 9.0, 0.63 | 14.1, 0.82 | 14.1, 0.78 | | |

- Memory is not the limit. Above ~256k atoms the peak grows linearly: 2.2 KB per padded atom in
  float32, 3.4 KB in float64 (row blocks bound the model's temporaries; lists, rij and forces remain).
  The A100 would hold approximately 9M float64 atoms per call, the Ada card approximately 4M.
- Speed: on the A100, 64-216 atom cells run 1.6-1.7x faster at 256k atoms per call than at 16k, flat
  to 1M, slightly slower at 4M. On the Ada card float32 gains 1.1-1.2x, float64 (fp64-bound there) 5%
  past 32k.
- 2- and 8-atom cells gain little past 16k atoms on either card (A100: 7 and 2.0-2.2 µs/atom, Ada
  float32 8-atom 1.24 to 1.10).
- Hence `MAX_BATCH_ATOMS` = 2^18 on a GPU (~1 GB in float64 for o4d12). The CPU keeps 2^14
  (`MAX_BATCH_ATOMS_CPU`, not measured here). With a larger B, a bucket's last batch takes the power
  of two its structures need (`pack`), so a short tail is not padded to the full B.

Batched MD (JAX-MD NVE under vmap, float64, 10 steps per jitted chunk; the cell count is the
caller's: `neighbour_lists` puts every structure in one batch, `MAX_BATCH_ATOMS` does not apply),
peak GB and µs per atom per step:

| padded atoms | A100 8 | A100 64 | A100 216 | Ada 64 | Ada 216 |
|--:|--:|--:|--:|--:|--:|
| 16k | 0.78, 0.41 | 0.31, 0.54 | 0.31, 0.57 | 0.20, 2.14 | 0.20, 2.53 |
| 64k | | 0.46, 0.41 | 0.46, 0.43 | 0.57, 2.25 | 0.57, 2.57 |
| 256k | | 1.99, 0.42 | 1.76, 0.49 | 1.98, 1.96 | 1.98, 2.72 |
| 1M | | | 7.36, 0.50 | 7.36, 2.62 | 7.36, 2.34 |

- About 7 KB per atom at 1M (state, lists at cutoff + skin, forces): the A100 holds ~4M atoms of
  batched MD, the Ada card ~2M.
- The A100 is best at 32k-128k atoms (0.37-0.45 µs/atom/step); beyond, slightly slower.
- **Slow A100 compiles.** Every shape took ~10 minutes to compile on Sulis with the persistent compile
  cache off (with it on, only about one in three did: 2-atom 4096 and 8192, 8-atom 4096 and 16384, 64-atom
  8192 and 32768, 216-atom 8192; the rest were cache hits), all of it in one `ptxas` call (~580 s, 8.5 GB RSS,
  plus one of ~68 s) while compiling the vmapped JAX-MD `init`.  Cause: XLA's fusion autotuner.  The radial
  contraction `einsum("eq,enq->en", P, Wnlq[zi, zj])` in `ACEModel._radial_one` (a per-edge gathered
  weight) becomes a reduce fusion, and the autotuner compiles candidate kernels for it: shuffle-heavy,
  256-thread PTX, 49,232 lines on the A100 (6,905 on lestrade's RTX 4000 Ada, where ptxas takes 1.8 s).
  The kernel XLA keeps is small.  On the Ada the candidates go with
  `--xla_gpu_experimental_enable_fusion_autotuner=false` or `--xla_gpu_autotune_level=0` (step time
  unchanged); capping the unroll factor (`--xla_gpu_experimental_max_unroll_factor` 1 or 2) does not help
  on the A100 (init 588 s, 587 s).  A deviceless A100 compile (`jax.experimental.topologies` with XLA's
  `a100_sxm_40.txtpb`) does not reproduce it, since it skips autotuning.  Fix: for one species the
  contraction is a plain `P @ Wnlq[0, 0].T` (a GEMM fusion; no reduce, no such candidates; step 3.5% faster
  on the Ada, forces equal to 2e-13).  Multi-species models keep the einsum: set the flag above if they
  compile slowly.  On the A100 (Sulis job 2264925; 64-atom cells x 128, NVE, compile cache off):

  | radial form | init compile (s) | step (s, 10 steps) |
  |---|--:|--:|
  | einsum (before) | 578.0 | 0.0505 |
  | einsum, `--xla_gpu_experimental_enable_fusion_autotuner=false` | 4.1 | 0.0511 |
  | elementwise sum over q | 5.5 | 0.0342 |
  | matmul (now, one species) | 5.1 | 0.0312 |

  The 1.6x step at this shape does not carry over: the benchmark grid gives 0.88-1.17x (MD table
  above).  Multi-species (Cantor, 5 species, `bench/batch/cantor_model.py` o3d10, `Wnlq` (5, 5, 107, 75)
  after `lean`; fcc 32-atom cells x 128; Sulis job 2264941) does not hit the slow compile:

  | radial form | A100 init (s) | A100 step (s) | Ada init (s) | Ada step (s) |
  |---|--:|--:|--:|--:|
  | einsum (now) | 9.3 | 0.2603 | 3.6 | 1.285 |
  | einsum, fusion autotuner off | 9.1 | 0.2598 | | |
  | elementwise sum over q | 18.9 | 0.2214 | 7.4 | 0.890 |
  | one GEMM over all 25 pairs, gather | | | 6.4 | 5.469 |

  The sum over q steps 1.18x faster on the A100 (1.44x on the Ada) for twice the compile; forces equal
  to 6e-16.  Float32 MD on lestrade first failed on a
  float64 momentum (`initial_momenta` now takes `dtype`).

## CPU (lestrade, 32 cores, float64)

Not an idle machine: another user's jobs kept the load average at 11-15
throughout. Treat these numbers as indicative.

**The neighbour build decides the CPU path.** Build and evaluation timed
separately (µs per atom; `_build` and `_efv_slots` jitted, warm):

| cells | builder | build | evaluation, C++ kernel | evaluation, JAX |
|---|---|--:|--:|--:|
| 64 x 64 atoms | jax (all pairs) | 11-13 | 1.0 | 41 |
| 16 x 216 atoms | jax (all pairs) | 22-38 | 0.73 | 26 |
| 64 x 64 atoms | msn (FFI, 16 threads) | 0.39 | 1.2 | |
| 16 x 216 atoms | msn (FFI, 16 threads) | 0.41 | 0.73 | |

- The all-pairs JAX builder costs 10-40 µs per atom on a CPU from 64
  atoms: XLA runs it on few cores, and the work is O(N^2 S). With a top_k
  compaction it was worse again (sorting N S candidates per row); a running
  count and a scatter made it linear.
- The batched FFI build costs 0.4 µs per atom. With it, the C++ kernel path
  is about 1.1-1.6 µs per atom against 5-12 for the kernel one cell per call.
  That is the target once matscipy-neighbours#10 and #11 are released.
- Until then `builder="auto"` on a CPU picks per batch: all pairs for at most
  `CPU_ALLPAIRS_MAX` (512) candidates per atom (N S, padded), else
  `host_neighbours` (matscipy-neighbours' cell list per structure on the
  host, about 70 µs per structure).

The kernel, `builder="auto"` (`lestrade-cpu.jsonl`):

| atoms | loop | B=1 | B=16 | B=64 | B=256 | B=1024 | best speed-up |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 260.9 | 794 | 240 | 20.9 | 6.76 | 5.05 | 52x |
| 8 | 79.0 | 202 | 18.8 | 9.88 | 4.35 | 4.15 | 19x |
| 64 | 12.2 | 31.9 | 5.19 | 5.86 | 5.68 | 5.58 | 2.4x |
| 216 | 5.3 | 12.4 | 4.90 | 3.64 | 4.20 | | 1.5x |

The JAX model on a CPU, with the host cell list (`lestrade-cpu-host.jsonl`):

| atoms | loop | B=1 | B=16 | B=64 | B=256 | B=1024 | best speed-up |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 330.0 | 992 | 102 | 71.9 | 87.4 | 79.4 | 4.6x |
| 8 | 76.6 | 247 | 54.4 | 51.6 | 50.1 | 55.4 | 1.5x |
| 64 | 35.5 | 68.7 | 45.6 | 48.0 | 49.0 | 65.1 | 0.8x |
| 216 | 40.2 | 59.6 | 55.2 | 55.0 | 74.2 | | 0.7x |

- The JAX model gains from batching on a CPU only for the smallest cells.
  XLA's CPU backend does not spread the rows of a block over cores, so a
  larger block only adds memory traffic (and, above `CHUNK_NODES`, the
  recomputation of checkpointed blocks). Hence `N_DEVICE_CPU_JAX` (16): with
  the JAX model on a CPU, larger structures go through `ACECalculator` one at
  a time.
- The kernel spreads rows over XLA's thread pool, so it gains at every size.

## Batched MD on a CPU: the C++ kernel under vmap

JAX-MD NVE under `jax.vmap` (`bench/batch/md.py`, `BatchCalculator.energy_fn`), 20 steps per jitted
chunk, lists fixed; lestrade CPU, float64, load average ~6. µs per atom per step
(`lestrade-md-cpu.jsonl`):

| atoms | cells | JAX model | C++ kernel | ratio |
|--:|--:|--:|--:|--:|
| 8 | 1 | 48.0 | 8.28 | 5.8x |
| 8 | 16 | 70.9 | 3.52 | 20x |
| 8 | 64 | 59.1 | 1.59 | 37x |
| 64 | 1 | 65.8 | 2.60 | 25x |
| 64 | 16 | 57.1 | 1.29 | 44x |
| 64 | 64 | 56.6 | 1.18 | 48x |

- The kernel vmaps by flattening: `eval.ffi._edge_kernel` gives its FFI call a `custom_vmap` rule that
  reshapes (B, rows, K, 3) to (B rows, K, 3), so a vmapped batch is ONE kernel call threaded over
  every row of every structure. Exact (bitwise against per-structure calls, `tests/test_ffi.py`).
- The JAX model does not gain from the batch on a CPU (as for single points above); the kernel does,
  because its rows are spread over XLA's thread pool.

## CPU with matscipy-neighbours' batched build (the branch of #11)

The C++ kernel, `builder="auto"` now picking the batched FFI build (matscipy-neighbours
`batched-neighbour-matrix`, ea16b2c); lestrade CPU, load average ~2 (`lestrade-cpu-msn.jsonl`); µs per
atom of the whole `calculate`:

| atoms | loop | B=64 | B=256 | B=1024 | best speed-up |
|--:|--:|--:|--:|--:|--:|
| 2 | 150.6 | 15.4 | 7.05 | 4.30 | 35x |
| 8 | 49.9 | 4.33 | 2.80 | 1.99 | 25x |
| 64 | 7.2 | 2.14 | 2.08 | 2.09 | 3.5x |
| 216 | 4.7 | 3.91 | 3.71 | | 1.3x |

(The quieter machine made the loop column faster than in the table above, so the ratios are not
directly comparable.)  Against the host cell list the 64-atom batch went from 5.6 to 2.1 µs/atom. 216-atom
cells cost more per atom than 64-atom ones in the batch; not yet explained (the brute-force build is
O(N) per atom, but measured at 0.4 µs/atom it does not account for it).

**In-loop rebuilds** (`BatchNeighbours.update_fn`: the skin test every step and the batched rebuild under
`lax.cond`, inside the jitted loop): vmapped kernel NVE, 64 cells, µs per atom per step, lists fixed /
checked every step: 8 atoms 1.58 / 1.73 (+9%), 64 atoms 1.19 / 1.20 (+1%).

## Does the loop stay on the GPU?

nsys (`/software/easybuild/software/CUDA/12.5.0/bin/nsys`, called by path: loading the CUDA module
breaks JAX's pip CUDA libraries), device-to-host copies per jitted call of 200 NVE steps over 5 small
cells (difference between 3 and 1 calls after the compile), lestrade RTX 4000 Ada shared with another job:

| loop | D2H copies per call | ms per call |
|---|--:|--:|
| `vmap(fori_loop(apply))`, lists fixed | 0 | 62 |
| batch-level loop, `update_fn` with `lax.cond`, msn / jax builder | 200 / 200 | 486 / 484 |
| same, `branchless=True`, msn / jax | 0 / 0 | 145 / 100 |
| `lax.cond` check every 10 steps (nested fori_loop) | 20 | 78 |

- The vmapped step and both list builders stay on the device. The copy is XLA's: the GPU
  backend reads a `lax.cond` predicate on the host (a small copy and a stream sync each time).
  `--xla_gpu_enable_command_buffer=...,CONDITIONAL,WHILE` did not remove it (JAX 0.11.2).
- A sync per step is expensive on a shared GPU (7.8x here), but not on an idle one. The same
  runs on the idle GPU (2026-10-24): fixed 30.2 ms, `lax.cond` every step 36.1 (msn) / 34.3
  (jax), branch-free 69.5 / 48.7, every 10 steps 34.8. So `update_fn` keeps `lax.cond` by
  default; `branchless=True` is for shared GPUs; checking every n steps is cheapest either way.
- These cells are tiny (46 atoms in all), so fixed costs dominate; with more atoms per step the
  build's share and the sync's share both fall.

## The kernel on the single-structure route (`jaxmd.potential`)

JAX-MD NVE with a JAX-MD Dense list, K = in-cutoff capacity + 4, 20 steps per jitted chunk; lestrade CPU,
load ~1: µs per atom per step, JAX model / C++ kernel (`ffi=True`): 216 atoms 33.4 / 3.44 (9.7x), 1728
atoms 29.9 / 1.68 (18x). The adapter and the batched path share the dense energy core (`eval/dense.py`),
so both take the kernel.

## Defaults these set

| setting | value | from |
|---|---|---|
| `MAX_BATCH_ATOMS` | 2^18 (262144) | GPU speed and memory, "Batch size and device memory" |
| `MAX_BATCH_ATOMS_CPU` | 2^14 (16384) | not measured above it |
| `N_DEVICE` | 256 | all-pairs cost O(N^2 S); 216-atom cells still gain on the GPU |
| `N_DEVICE_CPU_JAX` | 16 | JAX model on a CPU, above |
| `CPU_ALLPAIRS_MAX` | 512 | CPU build split, above |
| bucket sizes | 1, 2, 4, 8, then multiples of 8 to 128, then quarter octaves | 2-atom cells padded to 8 cost 2.5-4x on the GPU |

## Not yet measured

- `MAX_BATCH_ATOMS_CPU` above 2^14; the elementwise multi-species radial (1.18x A100, 1.44x Ada MD step).
- Mixed-size streams: compile counts per bucket in practice.
