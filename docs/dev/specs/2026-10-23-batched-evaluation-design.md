# Batched evaluation of many structures (design)

Status: phase 1 implemented, 2026-10-23 (`ace_jax.batch`; results and the measured defaults in
`docs/dev/batched-evaluation-results.md`). Phases 2-3 are sketched so that phase 1's layout serves them.
Departures from the design below, from the measurements: on a CPU the neighbour lists are built per
structure on the host for all but the smallest cells (`host_neighbours`), since the all-pairs builder
costs 10-40 µs per atom there; the JAX model on a CPU batches only structures up to 16 atoms (it gains
nothing above); evaluation is one dense block of B x N rows rather than a vmap; and `ACECalculator`'s
option resolution is reused by building one inside `BatchCalculator`, not by refactoring it out.
Design 7 (JAX-MD under vmap) is implemented too, for a fixed cell: `BatchCalculator.energy_fn`,
`neighbour_lists` -> `BatchNeighbours` (`allocate`, `update`, `needs_rebuild`, `unpad`) and the padding
helpers `initial_momenta`, `zero_padding`, `temperature`. Checked: vmapped FIRE, NVE and Langevin equal
the per-structure runs; padding found two leaks (JAX-MD's init gives padding atoms momenta and removes
the centre of mass over every slot; Nose-Hoover counts padding as degrees of freedom). The C++ kernel
vmaps too (a flattening `custom_vmap` rule on its FFI call), 37-48x the JAX model for batched MD on a CPU.

## Goal and scope

Evaluate energies, forces and stresses of **many structures in one compiled call**, so that small cells fill a GPU
(and, with the C++ kernel, a CPU's cores) instead of paying one dispatch per structure. TorchSim does this in
PyTorch; neither lammps-jax nor JAX-MD does (below).

- **Phase 1 (this spec): batched single-point.** A list of structures of any sizes, cells and species in; per-
  structure energy, forces and stress out, in input order. Directly useful for screening, finite-displacement
  phonons, committee / active-learning pools and `aj eval`.
- **Phases 2-3 (sketched only): batched relaxation and MD with JAX-MD.** ace-jax does **not** host integrators or
  optimizers (user decision, 2026-10-23). It provides the batched energy function, its neighbour lists, and
  padding helpers that let JAX-MD's own `simulate` / `minimize` functions run over a batch with `vmap`.
- **Out of scope:** autobatching by out-of-memory probing (memory is estimated instead), multi-GPU, cell degrees of
  freedom (NPT, cell filters), and UQ outputs (`posterior=`) in the batched path.

Success: on a GPU, a batch of small cells costs approximately the per-atom cost of one large cell (below:
0.66 µs/atom for 64-atom o4d12 cells in float32 on an RTX 4000 Ada, against 12.3 µs/atom through `ACECalculator`
one cell at a time); results agree with `ACECalculator` structure by structure to roundoff; nothing changes for
existing users.

## Evidence (2026-10-23, lestrade)

**Batching pays on a GPU.** o4d12 Si (accuracy-cost ladder), B rattled 64-atom cells packed into one dense block
(per-cell neighbour lists with images, node offsets: the fitting `Dataset` layout), one jitted
`energy_forces_virial_dense` call, RTX 4000 Ada:

| cells per call | float32 µs/atom | float64 µs/atom |
|--:|--:|--:|
| 1 | 4.09 | 15.4 |
| 4 | 1.57 | 5.34 |
| 16 | 0.87 | 3.51 |
| 64 | 0.66 | 3.37 |
| 256 | 1.08 | 3.65 |
| 1024 | 1.04 | 3.64 |
| `ACECalculator`, one cell per call | 12.3 | 20.5 |

The card saturates at ~4k atoms per call; an A100 needs more cells in flight. (The float32 rise past 64 cells is
probably the GPU default dense block of 4096 rows, #115: to check per bucket.) TorchSim reports the same shape for
MACE: 102x over ASE at 16 atoms, 18x at 108, 3.9x at 864 (arXiv 2508.06628, Fig. 1).

**One neighbour-list call per structure would dominate, on either device.** Si, rc 5 Å, K 64, per cell:

| cell | matscipy-neighbours CPU (OpenMP), Python call | matscipy-neighbours CUDA, Python call (DLPack) | matscipy-neighbours XLA FFI under `jit(vmap)`, CUDA: regular / planned | ASE's list |
|---|--:|--:|--:|--:|
| 2-atom primitive | 69-79 µs | 1035 µs | 272 / 157 µs | 2.0 ms |
| 8-atom cubic | 71-112 µs | 1039 µs | 250 / 135 µs | 0.98 ms |
| 64 atoms | 226-256 µs | 1073 µs | 274 / 206 µs | 2.3 ms |
| 216 atoms | 646-671 µs | 1169 µs | -- | 6.6 ms |

(lestrade; the FFI column is B = 1024 cells in one compiled call, per cell; it does not fall with B.) The
cost is a floor per call, not the device: a CUDA call through Python pays ~1 ms of launch, hand-off and
synchronisation however small the cell, and the experimental XLA FFI (`matscipy_neighbours.xla`, libAtoms/
matscipy-neighbours#10; jittable, any-order periodic images, so cells smaller than the cutoff work) is declared
`vmap_method="sequential"`, so a vmapped batch is B consecutive builds: 135-210 µs per cell planned. Per atom that
is 3.2 µs at 64 atoms, 17 µs at 8 and 78 µs at 2 -- 5-120x the batched evaluation (0.66 µs/atom, float32).
For MD the list is rebuilt every few tens of steps, so 64-atom cells are already about acceptable
(~0.1 µs/atom per step); a single-point call rebuilds every time. Phase 1 therefore needs **one build for the whole
batch** (Design 3), now contributed upstream as libAtoms/matscipy-neighbours#11: 3.9 µs per 64-atom cell and
0.09 µs per 2-atom cell at B = 1024 on the same GPU.

**Why not JAX-MD or lammps-jax.**
- lammps-jax: one LAMMPS system per process; its multi-replica mode is separate processes, which share no GPU
  kernels.
- JAX-MD: functional enough to `vmap` a step over equally sized systems (TorchSim's table rates its batching
  "partial"; `ace_jax.jaxmd.potential` takes species per call, so it should vmap -- not tested). But its
  neighbour lists are minimum-image: every perpendicular cell width must exceed 2 (rc + skin)
  (`jaxmd.min_image_ok`), which rules out Si cells below the 64-atom cube at rc 5 Å + skin 1 Å -- the small cells
  where batching gains most. Phases 2-3 therefore pair JAX-MD's integrators with ace-jax's lists (Design 7).

## What exists to build on

- **The batched layout.** `fit.data.Dataset` packs configurations into dense per-atom slots with `node_cfg`, from
  per-configuration lists with periodic images (`nlist.dense_graph`): any cell size, the layout benchmarked above.
- **The skin update.** `calc.skin.step` forms `rij = rij0 + u[j] - u[i]` with the image inside `rij0`, so moving
  atoms stay correct in any cell between rebuilds (no minimum image), and compacts skin slots to K in-cutoff slots
  on the device (`skin.compact`).
- **The models.** `EdgeSiteModel.energy_forces_virial_dense` (ACE and PACE) and the C++ kernel's `AceEdges`
  (`eval.ffi.KernelModel`, rows independent and threaded on XLA's pool) both take the dense edge input.
- **Option resolution.** `ACECalculator._eval_form` / `_ffi_choice` turn (lean, spline_tol, radial_table, aa_dag,
  dtype, energy_reference, ffi) into an evaluation model; the batched path must give the same model.
- **Memory estimates.** `EdgeSiteModel.estimate_a_bytes`, `export.lammps.block_rows_for` and
  `calc.point.device_memory_bytes`, measured against the device in `bench/gpu_memory`.

## Design

### 1. Layout: padded per-structure buckets

Structures are grouped into **buckets** by atom count (rounded up to a bucket size: multiples of 8 up to 64, then
quarter-octave steps, as `splinify.BUCKETS` does for intervals). A bucket is a padded batch: positions
`(B, N, 3)`, `mask (B, N)`, species `(B, N)`, cells `(B, 3, 3)`, pbc `(B, 3)`. Padding atoms are masked out of
every neighbour list and every output.

Why padded `(B, N)` rather than one concatenated node axis (the fitting `Dataset`): it is the shape JAX-MD's
integrators take under `vmap` (phases 2-3); per-structure outputs fall out without segment sums; and each bucket has
static shapes, so a stream of mixed sizes compiles once per bucket. Bucketing bounds the padding waste (< 12.5%
above 64 atoms). Internally the evaluation may flatten `(B, N)` rows into one dense block when that is faster
(the benchmark above did); the public layout stays padded.

### 2. Batch sizes from the memory estimate

`pack` splits each bucket into batches of a **fixed B per bucket** (the last batch padded with empty structures, so
B does not recompile), B the largest power of two whose estimate fits the device budget: `estimate_a_bytes` for
B·N rows of K slots plus the neighbour builder's scratch (Design 3), against `DENSE_BUDGET_FRACTION` of
`device_memory_bytes()`. No out-of-memory probing. `max_batch_atoms=` overrides. Measured against the device in
`bench/gpu_memory` before relying on it; if the estimate is unreliable for some model family, fall back to a
conservative fixed atoms-per-batch.

### 3. Neighbour lists: one build per batch

The list of a batch must be built in one launch (or a few), not one per structure.

- **The batched entry point in matscipy-neighbours' XLA FFI**, contributed as part of this work
  (libAtoms/matscipy-neighbours#11, stacked on #10): `xla.neighbour_matrix_batched(positions (B, N, 3),
  cells (B, 3, 3), cutoff, pbc (3-tuple or (B, 3)), K, n_atoms (B,), types, cutoffs)` returns `idx (B, N, K)`,
  `shift (B, N, K, 3)`, `count (B, N)` (0 on padding rows) and `status (B, 2)` = (overflow, invalid) per
  structure, and `jax.vmap` of `neighbour_matrix` / `neighbour_matrix_planned` routes to it (`custom_vmap`).
  Brute force over the images across planes within the cutoff, so any cell (smaller than the cutoff
  included) and any per-structure periodicity; OpenMP on the host, three kernel launches with no sync on CUDA
  (graph-capturable). Measured (RTX 4000 Ada, rc 6 Å, B = 1024): 0.09 / 0.23 / 3.9 µs per 2- / 8- / 64-atom
  Si cell, against 135-210 µs for the sequential vmap; one CPU core 1.0 / 5.0 / 127 µs. ace-jax adds only the
  call (with `cutoff = rc + skin` for MD) and maps `status` onto its overflow retry and NaN rows.
- **Fallback while #10/#11 are unreleased: a brute-force builder in JAX for small cells**, which vmaps natively: all pairs over the
  periodic-image shifts. Shifts per structure from its cell alone (host): `n_k = ceil((rc + skin) / w_k)` images
  per periodic axis (`w_k` the perpendicular widths, as `jaxmd.min_image_ok` computes them), none on
  non-periodic axes, padded to the bucket's largest count S (a 2-atom primitive Si cell at rc 5 Å needs up to
  ~7^3 shifts, a 64-atom cube 27). On the device `rij[b, i, j, s] = x[b, j] - x[b, i] + shift[b, s]`, valid
  where both atoms are live, not (i, i, zero shift), and r < rc (+ skin for MD), compacted to K slots per atom
  with `skin.compact`'s search. Cost O(N^2 S) per structure (110k distances for a 64-atom cube), which bounds it
  to `N_DEVICE` atoms (set by benchmark; expect a few hundred).
- **Large cells** (above `N_DEVICE`): the per-structure FFI build under `jit`, or the existing list paths; large
  cells are where a per-structure build is cheap per atom and where batching gains least.
- Either way: the dense edge input (`rij (B, N, K, 3)`, neighbour index, mask) plus an overflow flag (more than K
  in-cutoff neighbours: the call is repeated with a larger K, which recompiles) -- never truncation.

### 4. Evaluation

One function per bucket, jitted: `vmap` over B of the single-structure dense evaluation
(`energy_forces_virial_dense`), giving `E (B,)`, `F (B, N, 3)`, `V (B, 3, 3)`. The evaluation model comes from the
**same option resolution as `ACECalculator`** (refactor `_eval_form` / `_ffi_choice` into a function both use), so
lean, splines, radial tables, `aa_dag`, dtype, `energy_reference` and `ffi` mean the same thing. With `ffi`, the
kernel takes the flattened `(B·N, K)` rows (its rows are independent; `KernelModel` gains leading-dimension
support by reshaping, or the batched path calls it on flattened rows) and threads them on the CPU. Row blocks per
bucket from `block_rows_for` (the float32 rise above is the check).

Stress: `-V / volume` per structure for fully periodic cells, NaN otherwise (as `aj eval` writes). Energies with
`energy_reference="E0"` also return the per-structure E0 offset, as `ACECalculator` does.

### 5. API

```python
from ace_jax.batch import BatchCalculator

bc = BatchCalculator("fit/model.npz", dtype=None, lean=True, spline_tol="auto", radial_table=None,
                     aa_dag="auto", ffi=False, energy_reference="absolute", max_batch_atoms=None)
res = bc.calculate(atoms_list)        # BatchResults, input order
res.energy                            # (n_structures,)
res.forces                            # list of (n_i, 3)
res.stress                            # (n_structures, 6), ASE Voigt order; NaN rows for non-periodic cells
res.timing                            # neighbour lists, evaluation, compiles, buckets used
res.to_atoms()                        # copies with SinglePointCalculator results attached
```

Lower level, for JAX users and phases 2-3 (names indicative):

```python
from ace_jax import batch
b = batch.pack(atoms_list, cutoff, skin=0.0)      # padded buckets: R, mask, species, cells, pbc, shifts
nl = batch.neighbours(b, R, K=None)               # device builder (Design 3)
E, F, V = batch.efv(model, b, nl)                 # padded outputs
batch.unpad(b, F)                                 # list of per-structure arrays, input order
```

`aj eval` uses `BatchCalculator` when no posterior is given (its loop over structures is exactly this case).

### 6. Validation

- **Parity** with `ACECalculator` per structure (energies to 1e-12 relative, forces to 1e-11 of max|F|, stress
  to roundoff), same options: 2-atom primitive and 8-atom cubic Si, a sheared triclinic cell, a slab (mixed pbc),
  a non-periodic cluster (stress NaN), a single periodic atom, SiGe and Cantor5 (species), a species-embedding
  model, a PACE `.yace`, mixed sizes in one call (several buckets), input order preserved.
- **Neighbour builder** (the FFI or the brute-force one) against `nlist.sparse_graph` (same pair sets and edge
  vectors) on the same cells, and its overflow retry.
- **Kernel path** (`ffi=True`) against the JAX path on the CPU.
- **Benchmarks** (results in `docs/dev/batched-evaluation-results.md`): µs/atom against batch size for 2-, 8-,
  64- and 216-atom Si, float32 / float64, on lestrade (RTX 4000 Ada) and a Sulis A100; the CPU kernel on 32 cores;
  each against an `ACECalculator` loop; neighbour-builder time against per-structure builds (CPU, CUDA, FFI); compile counts.

### 7. Phases 2-3 with JAX-MD (sketch; not built here)

JAX-MD's `simulate` (NVE, NVT Langevin, ...) and `minimize` (FIRE, ...) are pure functions of an energy (or force)
function and a `shift_fn`; they `vmap` over a leading batch axis, giving per-structure thermostat and FIRE state
(FIRE's global power and step size become per structure, as they must). ace-jax would provide:

- `batch.energy_fn(model, b)`: a per-structure `energy(R, nl)` over the padded positions, using ace-jax's lists
  (any cell size) rather than JAX-MD's, so integrators run with `space.free()` (positions unwrapped between list
  rebuilds; the image is in the list).
- `batch.pad` / `unpad` / `mask`: padding atoms carry zero force; helpers keep them inert in integrator state
  (zero velocity, excluded from kinetic energy and temperature) and strip them on output.
- `batch.needs_rebuild` / `rebuild`: the skin test per structure (an atom moved skin / 2) and a rebuild that wraps
  positions back into the cell; an in-flight relaxation driver replaces converged structures with new ones between
  rebuilds (TorchSim's in-flight batcher, as a recipe over JAX-MD's FIRE, not an ace-jax optimizer).

Open for phases 2-3: cell degrees of freedom (NPT, cell filters for relaxation) need the cell in the state and the
lists in fractional form; JAX-MD's own NPT uses `space.periodic_general`, whose minimum image again excludes small
cells.

## Files

- `src/ace_jax/batch.py` (or a `batch/` package: `pack.py`, `neighbours.py`, `calc.py`): Designs 1-5.
- `calc/point.py`: the option resolution moved into a shared function.
- `eval/ffi.py`: `KernelModel` on leading batch dimensions.
- `cli.py`: `aj eval` through `BatchCalculator`.
- `tests/test_batch.py`; `bench/batch/` (the benchmarks of Design 6).
- Docs: a user how-to, "Evaluate many structures"; `skills/ace-jax/SKILL.md`; CLAUDE.md.

## Risks and open questions

- **The batched list build:** the FFI entry point (matscipy-neighbours#11) depends on #10 and #11 merging and
  a release with built XLA handlers; until then the JAX brute-force fallback's O(N^2 S) cost bounds it
  (`N_DEVICE`, from the benchmark). Both are O(N^2) per structure, so `N_DEVICE` applies either way.
- **Recompiles:** one compile per (bucket, B, K, S); a wide mix of sizes or a K overflow recompiles. Bucketing and
  fixed B per bucket bound it; `res.timing` reports compiles.
- **Memory estimate accuracy** for the batched shapes (Design 2): checked against `bench/gpu_memory` first.
- **vmapped against flattened evaluation:** which is faster per bucket is measured, not assumed.
- **float32 absolute positions:** the builder forms `rij` from positions plus shifts; small cells keep |x| small,
  but large cells in float32 carry ~ulp(|x|) (as lammps-jax's float32 bundles; docs/dev/fp32-results.md).
