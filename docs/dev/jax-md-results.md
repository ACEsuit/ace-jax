# JAX-MD vs LAMMPS (lammps-jax): throughput and memory (2026-10-19)

`ace_jax.jaxmd` (PR #104) runs the whole MD step -- neighbour-list update, model,
NVE update -- in one compiled program on the device. This page compares it with
the same models in LAMMPS through lammps-jax (`pair_style jax/kk`), on the linear
ACE ladder of the accuracy-cost benchmark (`docs/dev/accuracy-cost-results.md`).

**Protocol** (`bench/scaling/run_jaxmd.py`, `bench/accuracy_cost/timing.py --modes`).
- Rattled Si supercell, 32768 atoms (plus a 4096-atom plateau check), NVE from
  rest, 0.1 fs, skin 1 Å checked every step (JAX-MD `dr_threshold` = LAMMPS
  `neighbor 1.0 bin`, `neigh_modify every 1 delay 0 check yes`). Median of 3.
- Same models, same evaluation model (lean, splined at 1e-10), same slots: K is
  the lammps-jax bundle's `k_dense` (`neighbour_capacity`, margin 8). "skin"
  slots hold rc + skin pairs (never overflow between rebuilds); "tight" slots
  (`slots="cutoff"`) hold rc pairs, as `lammps-tight`.
- JAX-MD: JAX-MD's own Dense cell list, forces by autodiff
  (`potential(...).force`), one jitted 10-step chunk called repeatedly (a static
  trip count keeps the loop a CUDA graph). `jaxmd.potential(chunk="auto")`: rows
  in blocks sized by `export.lammps.block_rows_for`, as the bundles.
- Every JAX-MD row passed its gate (E/F against `ACECalculator`, three
  216-atom cells; `parity.TOL["acejax-splined"]`).
- ms/step; × = lammps / JAX-MD (above 1: JAX-MD faster).

Raw rows: `bench/accuracy_cost/results/sulis/sulis-gpu-jaxmd-*.jsonl` (lammps:
`-o3`, `-o4a`, `-o4d16`, `-o4d20-lammps`; lammps-tight: `-tight-*`; JAX-MD:
`-auto-*`, `-auto-tight-*`; the `-k54-*`, `-o4d20-jaxmd` and JAX-MD rows inside
`-o3`/`-o4a`/`-o4d16` are superseded runs with fixed 16384-row blocks, and K = 48
for the latter) and `results/lestrade-gpu-jaxmd*.jsonl`.

## Sulis A100 (40 GB), float64

| model | lammps | JAX-MD | × | lammps tight | JAX-MD tight | × | lammps 4096 | JAX-MD 4096 | × |
|---|---|---|---|---|---|---|---|---|---|
| o3d8 | 8.62 | 7.73 | 1.12 | 7.84 | 5.72 | 1.37 | 1.39 | 1.04 | 1.34 |
| o3d10 | 10.32 | 8.87 | 1.16 | 9.85 | 7.41 | 1.33 | 1.79 | 1.29 | 1.38 |
| o3d12 | 17.05 | 12.64 | 1.35 | 11.13 | 7.85 | 1.42 | 2.14 | 1.58 | 1.35 |
| o4d12 | 18.05 | 13.66 | 1.32 | 12.15 | 8.89 | 1.37 | 2.29 | 1.64 | 1.39 |
| o4d14 | 19.42 | 18.38 | 1.06 | 16.61 | 15.19 | 1.09 | 2.89 | 2.44 | 1.19 |
| o4d16 | 30.30 | 28.85 | 1.05 | 26.27 | 24.76 | 1.06 | 4.34 | 3.73 | 1.17 |
| o4d20 | 101.75 | 113.90 | 0.89 | 100.01 | 109.25 | 0.92 | 14.04 | 14.42 | 0.97 |

ML-PACE (KOKKOS, tabulated radials, `results/sulis/sulis-gpu-ace_tab3.jsonl`) on
the o4 models: 8.84, 14.39, 26.79 and 134.85 ms (o4d12 ... o4d20).

## lestrade RTX 4000 Ada (20 GB)

float64:


| model | lammps | JAX-MD | × | lammps tight | JAX-MD tight | × | lammps 4096 | JAX-MD 4096 | × |
|---|---|---|---|---|---|---|---|---|---|
| o3d8 | 39.27 | 37.00 | 1.06 | 29.36 | 25.43 | 1.15 | 3.37 | 2.75 | 1.23 |
| o3d10 | 49.61 | 48.48 | 1.02 | 35.05 | 30.64 | 1.14 | 6.15 | 5.98 | 1.03 |
| o3d12 | 61.32 | 57.78 | 1.06 | 43.54 | 41.54 | 1.05 | 8.00 | 7.12 | 1.12 |
| o4d12 | 63.51 | 59.64 | 1.07 | 45.50 | 43.93 | 1.04 | 8.28 | 7.42 | 1.12 |
| o4d14 | 90.91 | 79.76 | 1.14 | 67.58 | 60.38 | 1.12 | 11.28 | 9.91 | 1.14 |
| o4d16 | 121.35 | 109.54 | 1.11 | 95.61 | 85.39 | 1.12 | 15.40 | 13.49 | 1.14 |
| o4d20 | 314.53 | 307.93 | 1.02 | 280.83 | 275.88 | 1.02 | 39.51 | 38.68 | 1.02 |

float32:


| model | lammps | JAX-MD | × | lammps tight | JAX-MD tight | × | lammps 4096 | JAX-MD 4096 | × |
|---|---|---|---|---|---|---|---|---|---|
| o3d8 | 6.07 | 4.94 | 1.23 | 5.48 | 4.02 | 1.36 | 0.75 | 0.51 | 1.48 |
| o3d10 | 8.47 | 7.37 | 1.15 | 7.18 | 5.69 | 1.26 | 1.11 | 0.91 | 1.23 |
| o3d12 | 10.68 | 9.96 | 1.07 | 9.60 | 8.16 | 1.18 | 1.45 | 1.25 | 1.16 |
| o4d12 | 12.63 | 11.66 | 1.08 | 11.43 | 9.82 | 1.16 | 1.70 | 1.46 | 1.16 |
| o4d14 | 18.86 | 16.41 | 1.15 | 16.27 | 14.32 | 1.14 | 2.48 | 2.02 | 1.23 |
| o4d16 | 32.76 | 30.14 | 1.09 | 29.30 | 27.00 | 1.09 | 4.23 | 3.76 | 1.13 |
| o4d20 | 152.12 | 150.21 | 1.01 | 147.85 | 145.95 | 1.01 | 19.44 | 18.58 | 1.05 |

## Summary

- JAX-MD is 1.05-1.42x faster than lammps-jax on the A100 up to o4d16 (skin and
  tight slots), 1.17-1.39x at 4096 atoms, and ~10% slower at o4d20. On the RTX
  4000 Ada it is 1.01-1.23x faster everywhere (float64 and float32), 1.01-1.36x
  with tight slots.
- The gain is largest for small models and small systems and gone at o4d20,
  where the model evaluation dominates the step. Where the per-step time outside
  the model goes in LAMMPS (plugin call, neighbour-matrix copy, ghosts) was not
  profiled here.

## Row blocks

`jaxmd.potential` first used fixed 16384-row blocks (`CHUNK_NODES`): one block up
to 16384 atoms, no checkpoint. A/B with the matscipy-neighbours list (below) at
32768 atoms, lestrade (`results/lestrade-gpu-jaxmd-blocks-ab-*.jsonl`):

| | 16384-row blocks | auto (`block_rows_for`) |
|---|---|---|
| o4d20 float64 | 366 ms, 10.2 GB | 309 ms, 1.32 GB |
| o4d20 float32 | 160 ms, 2.92 GB | 148 ms, 0.77 GB |
| o3d10 float64 | 44.2 ms, 1.84 GB | 49.2 ms, 0.77 GB |
| o3d10 float32 | 8.65 ms, 0.77 GB | 7.09 ms, 0.50 GB |

The bundles' rule cuts memory 1.5-8x and is faster except for small float64
models, which pay the checkpoint recompute (o3d10 float64 at 16384 atoms: 15.2
-> 24.4 ms; it then matches lammps-jax, which uses the same rule). On the A100
ladder the change moved JAX-MD by at most ±12% per model. `chunk=<int>` trades
memory back for speed.

## Size and memory: JAX-MD's neighbour list

Process GPU memory (NVML peak, preallocation off), lestrade, skin slots. "matscipy
list" is the `jaxmd-shift` mode: an on-device matscipy-neighbours list built with
its XLA FFI (libAtoms/matscipy-neighbours#10, unreleased), on branch
`feat/jaxmd-shiftlist`, not in PR #104.


| atoms | model | dtype | lammps ms / GB | JAX-MD ms / GB | matscipy list ms / GB |
|---|---|---|---|---|---|
| 8192 | o3d10 | 64 | 12.2 / 0.50 | 11.9 / 2.51 | 12.2 / 0.50 |
| 8192 | o3d10 | 32 | 2.0 / 0.36 | 1.7 / 1.44 | 1.7 / 0.37 |
| 8192 | o4d14 | 64 | 23.0 / 0.50 | 19.8 / 2.51 | 19.8 / 0.50 |
| 8192 | o4d14 | 32 | 4.8 / 0.36 | 3.9 / 1.44 | 3.9 / 0.37 |
| 8192 | o4d20 | 64 | 78.9 / 1.05 | 76.8 / 2.52 | 76.8 / 1.05 |
| 8192 | o4d20 | 32 | 38.1 / 0.64 | 36.8 / 1.44 | 37.1 / 0.77 |
| 32768 | o3d10 | 64 | 49.7 / 0.79 | 48.6 / 4.66 | 49.4 / 0.77 |
| 32768 | o3d10 | 32 | 8.5 / 0.52 | 7.4 / 2.51 | 7.1 / 0.50 |
| 32768 | o4d14 | 64 | 90.9 / 0.79 | 79.8 / 4.66 | 79.9 / 0.77 |
| 32768 | o4d14 | 32 | 18.9 / 0.52 | 16.3 / 2.51 | 16.0 / 0.50 |
| 32768 | o4d20 | 64 | 314.5 / 1.34 | 309.3 / 4.67 | 308.9 / 1.32 |
| 32768 | o4d20 | 32 | 152.2 / 0.79 | 148.4 / 2.52 | 148.6 / 0.77 |
| 131072 | o3d10 | 64 | 201.1 / 2.55 | OOM | 201.4 / 2.38 |
| 131072 | o3d10 | 32 | 38.8 / 1.54 | OOM | 33.6 / 1.31 |
| 131072 | o4d14 | 64 | 352.5 / 2.55 | OOM | 323.2 / 2.38 |
| 131072 | o4d14 | 32 | 76.9 / 1.54 | OOM | 69.3 / 1.31 |
| 131072 | o4d20 | 64 | 1252.6 / 2.50 | OOM | 1236.5 / 2.39 |
| 131072 | o4d20 | 32 | 625.9 / 1.54 | OOM | 604.4 / 1.31 |

JAX-MD's own list costs 0.8-3.9 GB more than lammps-jax at these sizes (2-6x) and runs
out of memory at 131072 atoms (an 8.88 GiB allocation, float32 and float64
alike). The list itself is small (32768 x 57 int32, 7 MB); its build is not.
Measured at 32768 atoms (o3d10, float64; XLA `memory_analysis`):

- the cell list has a fixed capacity per cell (32 atoms here, against ~8 on
  average), and each atom tests every slot of its 27 neighbouring cells: 864
  candidates for ~57 neighbours, 28 M candidate pairs, each with a float64
  displacement, a squared distance, a mask and indices, then compacted;
- the compiled `nbrs.update` needs 1.39 GB of temporaries (the rebuild sits in a
  `lax.cond`, so they are reserved every step), against 0.29 GB for the model's
  forces; the eager first build (`allocate`) peaks at 2.25 GB, which the
  allocator keeps for the run (the A100's flat `peak_bytes_in_use` of 2.21 GB).

This grows as N x 27 x cell capacity: ~9 GB at 131072 atoms, the allocation that
fails. The matscipy list builds the compact list directly, O(N K), and matches
lammps-jax's memory (0.77 vs 0.79 GB at 32768 atoms) at the same speed. Until it
is released, JAX-MD runs with ace-jax need 2-6x the GPU memory of LAMMPS, and
the list sets the size limit (between 32768 and 131072 atoms on a 20 GB card).
