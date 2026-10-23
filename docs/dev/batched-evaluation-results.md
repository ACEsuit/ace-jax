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
built locally). The GPU was idle (`lestrade-gpu-v2.jsonl`; 2-atom rows from
`lestrade-gpu-2atom-pow2.jsonl`, after small buckets became powers of two).
These runs predate two later changes, which can only make them faster:
evaluation on the occupied neighbour slots only, and the linear-time
compaction in the `jax` builder. A rerun on an idle GPU is still to do.

| dtype | atoms | builder | loop | B=1 | B=4 | B=16 | B=64 | B=256 | B=1024 | best speed-up |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| float32 | 2 | jax | 275.6 | 973 | 140 | 58.8 | 16.8 | 9.41 | 6.47 | 43x |
| float32 | 2 | msn | 275.6 | 649 | 140 | 82.1 | 13.1 | 8.75 | 6.04 | 46x |
| float32 | 8 | jax | 86.2 | 135 | 74.0 | 13.0 | 5.32 | 1.94 | 1.68 | 51x |
| float32 | 8 | msn | 86.2 | 225 | 57.7 | 16.1 | 4.10 | 2.76 | 1.75 | 49x |
| float32 | 64 | jax | 8.3 | 19.3 | 6.50 | 2.16 | 1.79 | 1.79 | 1.44 | 5.8x |
| float32 | 64 | msn | 8.3 | 18.4 | 7.05 | 2.15 | 1.62 | 1.23 | 1.12 | 7.4x |
| float32 | 216 | jax | 2.7 | 9.93 | 3.70 | 3.30 | 2.04 | 1.89 | 1.88 | 1.4x |
| float32 | 216 | msn | 2.7 | 10.5 | 3.80 | 2.14 | 1.43 | 1.23 | 1.27 | 2.2x |
| float64 | 2 | jax | 469.0 | 941 | 514 | 77.5 | 26.0 | 11.2 | 9.06 | 52x |
| float64 | 2 | msn | 469.0 | 780 | 232 | 59.6 | 52.1 | 13.3 | 8.64 | 54x |
| float64 | 8 | jax | 128.9 | 216 | 52.9 | 12.7 | 6.69 | 4.92 | 6.31 | 26x |
| float64 | 8 | msn | 128.9 | 236 | 60.5 | 17.8 | 8.03 | 4.62 | 5.59 | 28x |
| float64 | 64 | jax | 18.5 | 39.0 | 9.58 | 4.63 | 3.93 | 4.85 | 4.82 | 4.7x |
| float64 | 64 | msn | 18.5 | 33.0 | 12.2 | 4.36 | 3.82 | 4.48 | 4.28 | 4.8x |
| float64 | 216 | jax | 5.9 | 20.7 | 7.46 | 5.42 | 6.46 | 6.36 | 6.35 | 1.1x |
| float64 | 216 | msn | 5.9 | 14.2 | 6.64 | 4.00 | 5.40 | 5.08 | 5.11 | 1.5x |

- The gain is largest where one structure per call is all overhead: 40-54x
  for 2- and 8-atom cells. At 64 atoms it is 5-7x, at 216 atoms 1.1-2.2x (a
  216-atom cell already keeps this card fairly busy).
- float32 64-atom cells reach 1.1 µs/atom, against 0.66 µs/atom for the
  model alone on prebuilt lists in the spec's evidence. The rest is the
  neighbour build and the copies.
- The two builders are within run-to-run noise up to 64 atoms. At 216
  atoms `msn` is faster (its image search is pruned by plane distance; the
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

## Defaults these set

| setting | value | from |
|---|---|---|
| `MAX_BATCH_ATOMS` | 16384 | GPU saturation, above |
| `N_DEVICE` | 256 | all-pairs cost O(N^2 S); 216-atom cells still gain on the GPU |
| `N_DEVICE_CPU_JAX` | 16 | JAX model on a CPU, above |
| `CPU_ALLPAIRS_MAX` | 512 | CPU build split, above |
| bucket sizes | 1, 2, 4, 8, then multiples of 8 to 128, then quarter octaves | 2-atom cells padded to 8 cost 2.5-4x on the GPU |

## Not yet measured

- An idle GPU with the final code, and a Sulis A100 (more cells in flight).
- `MAX_BATCH_ATOMS` against device memory (`bench/gpu_memory`); the
  default is far below what either card holds for o4d12.
- Mixed-size streams: compile counts per bucket in practice.
