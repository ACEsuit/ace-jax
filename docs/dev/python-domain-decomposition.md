# Python-native domain decomposition: assessment (2026-10-23, deferred)

Should ace-jax run MD across several devices in one Python process, with a spatial
domain decomposition and no LAMMPS? The candidate building block is lammps-jax's
`lammps_jax.dist.parallel.ghost_exchange`
(<https://github.com/abhijeetgangan/lammps-jax/blob/main/python/lammps_jax/dist/parallel/ghost_exchange.py>,
reviewed at lammps-jax 0.2.1, `b66fb1e`). **Decision: not now.** The only case it would
serve is multi-GPU MD that stays in Python. That case is potentially worthwhile, because
JAX-MD on one A100 is up to 1.4x faster than LAMMPS + lammps-jax on the ACE ladder
(`jax-md-results.md`). Revisit when that use case is needed.

## What ghost_exchange provides

A LAMMPS-style spatial decomposition in JAX, run under `shard_map` with one device per
rank (`axis_name`):

- `create_config` / `default_grid`: a `(px, py, pz)` grid over fractional coordinates,
  with capacities from the mean density (`capacity_mult`).
- `pack_tiles`: global atoms into per-rank tile buffers.
- `ghost_exchange`: ghosts collected axis by axis, both ring directions per hop, with
  `lax.ppermute` on a halo of cutoff + skin. It records a plan.
- `exchange_apply`: replays the plan on new positions or features until an atom has moved
  skin / 2 (`needs_rebuild`). Autodiff of the exchange is the reverse communication, so
  ghost forces return to their owners with no extra code.
- `redistribute`: wraps positions and migrates atoms one tile per axis. Extra columns
  (e.g. velocities) and integer columns travel with their rows.
- `ghost_exchange_subgraph`: a tile-local cell list, giving edges into owned atoms over the
  owned + ghost node order (`subgraph_params` sizes it).
- Fixed capacities everywhere; overflow and misplacement are rank-local flags (reduce with
  `psum`).

Its users at 0.2.1 are `feature_exchange` (message passing across ranks),
`scripts/nequix_bench.py` (rebuild + a `shard_map`'d energy-and-gradient step, checked
against one device) and `tests/test_dist_parallel.py`. It is a force-evaluation harness:
no integrator, no velocity migration, no thermostat. The file has two commits (June and
September 2026), so the API should be treated as research code that may change.

## Fit with ace-jax

- **The model fits well.** ace-jax models map edge vectors to site energies.
  `ghost_exchange_subgraph` gives an edge list into the owned rows over owned + ghost
  nodes, so the sparse layout (or a dense compaction, as `jaxmd.potential` does for K)
  evaluates owned site energies directly. The forces come from autodiff through
  `exchange_apply`.
- **The C++ kernel fits too.** Its `AceEdges` entry point (`eval.ffi.KernelModel`, #121)
  takes dense per-owned-row edges, and its custom VJP composes with the exchange's
  autodiff. That would be the CPU form if a CPU path were ever wanted.
- **Not inside the JAX-MD adapter (`ace_jax.jaxmd`).** JAX-MD's space functions, neighbour
  lists and `simulate` integrators all assume one global `(N, 3)` position array. The
  decomposition replaces all three: per-rank tiles in fractional coordinates, its own
  cell-list edges, and atoms migrating between tiles at rebuilds. Integrator state
  (velocities, thermostat variables) must migrate with the atoms (`redistribute`'s extra
  columns can carry it, but JAX-MD's state handling cannot be reused). Only the energy
  function would be shared, so it would be a separate module (e.g. `ace_jax.dist`) that
  reuses the adapter's model plumbing: K-slot compaction and overflow as NaN.

## Where it would add value

| case | covered today by | value of in-process decomposition |
|---|---|---|
| CPU, one node | `ACECalculator(ffi=True)`: the C++ kernel on XLA's thread pool, 17x the JAX model at 1.7k-4.1k atoms on 32 cores (#121) | little: fake CPU devices would share the same cores |
| CPU, several nodes | LAMMPS + lammps-jax, MPI ranks with the C++ kernel (#112, #119) | little: JAX cross-process CPU collectives are less mature than MPI |
| one GPU | `ace_jax.jaxmd`, or LAMMPS + lammps-jax | none: o4d20 Si fits 4.4M atoms on one A100-40GB (`gpu-memory-results.md`) |
| **several GPUs, MD in Python** | nothing (LAMMPS + Kokkos does several GPUs, but outside Python) | **the gap**: up to ~N-GPU throughput or system size, for workflows LAMMPS cannot do (differentiating through a trajectory, custom integrators, in-loop analysis) |

The single-GPU JAX-MD advantage (`jax-md-results.md`, A100, float64, 32768 atoms) is
1.05-1.35x for o3d8-o4d16 (1.06-1.42x with tight slots) and 0.89x at o4d20. A multi-GPU
Python path would keep that advantage only if the exchange (one `ppermute` per stage per
step for positions, the same again in the backward pass) costs less than LAMMPS's MPI
halo exchange. That is not measured.

## Costs and risks

- **Effort for a full MD driver:** velocity-Verlet in the tile layout, velocities migrating
  with `redistribute`, rebuilds with `needs_rebuild`, overflow handling (a capacity
  increase changes shapes, so it recompiles), thermostats, global observables (`psum`),
  gathering positions for output. Plus multi-GPU testing on Sulis (CI has no GPUs).
- **API maturity:** research code in lammps-jax, used only by its benchmarks and tests.
- **A known XLA CPU crash:** lammps-jax's `pyproject.toml` excludes jax 0.11.2 ("jax 0.11.2
  segfaults compiling ghost_exchange.redistribute on CPU; 0.11.1 passes"). ace-jax's lock
  has jax 0.11.2 for Python >= 3.12, and CI tests on the CPU (conftest forces two host
  devices). So CI could not compile the migration step unless ace-jax also excluded
  0.11.2 or the bug is fixed upstream in XLA.

## If revisited

1. A force-only spike first: a `shard_map`'d energy-and-forces function on tiles for an
   ace-jax model (`ghost_exchange` rebuild + step, as `nequix_bench.py` does), checked
   against `ACECalculator`, and timed on a multi-GPU Sulis node at 1, 2 and 4 GPUs
   against single-GPU `ace_jax.jaxmd` and LAMMPS + lammps-jax on the same GPUs.
2. Build the MD driver only if that scaling justifies it. Check with the lammps-jax
   maintainer first whether `dist.parallel` is meant to become a stable API.
3. Re-check the jax 0.11.2 CPU crash on the jax ace-jax then locks.
