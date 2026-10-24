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
- Batch size: the card saturates at approximately 4k-16k atoms per call.
  `MAX_BATCH_ATOMS` (16384) sits there.
- B = 1 is 2-3x slower than the loop: a batch of one pays the batch path's
  fixed cost without amortising it. `BatchCalculator` is for many structures.

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
| `MAX_BATCH_ATOMS` | 16384 | GPU saturation, above |
| `N_DEVICE` | 256 | all-pairs cost O(N^2 S); 216-atom cells still gain on the GPU |
| `N_DEVICE_CPU_JAX` | 16 | JAX model on a CPU, above |
| `CPU_ALLPAIRS_MAX` | 512 | CPU build split, above |
| bucket sizes | 1, 2, 4, 8, then multiples of 8 to 128, then quarter octaves | 2-atom cells padded to 8 cost 2.5-4x on the GPU |

## Not yet measured

- A Sulis A100 (more cells in flight).
- `MAX_BATCH_ATOMS` against device memory (`bench/gpu_memory`); the
  default is far below what either card holds for o4d12.
- Mixed-size streams: compile counts per bucket in practice.
