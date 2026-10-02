# Export a model to LAMMPS

ace-jax models run in LAMMPS through
[lammps-jax](https://github.com/abhijeetgangan/lammps-jax), which provides
`pair_style jax/kk`: a KOKKOS pair style that executes a compiled JAX
program on the GPU. ace-jax writes the program and its metadata into one JSON
*bundle*.

!!! note "lammps-jax is not on PyPI"
    Install it from a clone or a pinned commit, for example
    `pip install "lammps-jax @ git+https://github.com/abhijeetgangan/lammps-jax"`,
    and build its LAMMPS plugin following the lammps-jax instructions. ace-jax's
    CI tests the export against lammps-jax commit `4a7f4fb`. The pair style is
    GPU-only (KOKKOS with CUDA).

## Export

The bundle's buffers have fixed sizes, so they are sized from the structure
the run starts from:

```python
import jax
jax.config.update("jax_enable_x64", True)
from ase.build import bulk
import ace_jax as aj
from ace_jax.export.lammps import export_lammps, neighbour_capacity

model, meta, _ = aj.load("fit/model.npz")                 # or a .yace
atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(4)
cap = neighbour_capacity(atoms, meta["rcut"], skin=1.0)    # skin = the LAMMPS `neighbor` skin
export_lammps(model, meta, "si_bundle.json",
              max_atoms=cap["max_atoms"], max_edges=cap["max_edges"],
              k_dense=cap["k_dense"], max_neighbors=cap["max_neighbors"],
              max_owned=cap["max_owned"],
              type_elements=[14])                          # Z of LAMMPS types 1, 2, ...
```

`type_elements` maps LAMMPS atom types to elements: type 1 is the first
entry. The returned dictionary, also stored in the bundle, records what was
exported under its `ace_jax` key (layout, capacities, whether the lean form
and splining were applied, and the lammps-jax version).

In the LAMMPS input:

```text
pair_style jax/kk <path to the PJRT GPU plugin>    # see the lammps-jax documentation
pair_coeff * * si_bundle.json
```

Run LAMMPS with KOKKOS on the GPU, for example
`lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps`.

## Capacities

`neighbour_capacity(atoms, rcut, skin=1.0, slots="skin", margin=8, list_headroom=0.5)`
returns every buffer size for a structure:

| Key | Meaning |
|---|---|
| `max_atoms` | owned plus ghost atoms (the ghost shell is sized on the cell's face spacings, so triclinic cells are covered) |
| `max_owned` | owned atoms (with 10% headroom); rows past it are never evaluated |
| `k_dense` | model neighbour slots per atom |
| `max_neighbors` | LAMMPS neighbour-list slots per atom (with 50% headroom: the list grows fastest when a structure compresses) |
| `max_edges` | the packed edge buffer of the sparse and dense layouts |

A structure that outgrows a buffer never gives silently truncated forces: the
energy and forces become NaN, or LAMMPS aborts the run. Re-export with sizes
for the larger structure.

`slots="cutoff"` sizes the model slots for pairs within the cutoff only,
1.2 to 1.4 times faster on a five-component alloy, but it is only safe for
stable MD of a fitted model whose coordination stays close to the starting
structure's.

## Layouts

| Layout | What it does | When `"auto"` picks it |
|---|---|---|
| `matrix` | reads the LAMMPS neighbour list directly, copied only when LAMMPS rebuilds it; no per-step packing | `k_dense` and `max_neighbors` given, the installed lammps-jax supports it, and one block fits in memory |
| `dense` | packs the edge buffer into per-atom slots every step | `k_dense` given and `matrix` not chosen |
| `sparse` | an edge list | no `k_dense`, or one dense block does not fit in memory |

A LAMMPS plugin older than the lammps-jax Python package rejects a `matrix`
bundle. Rebuild the plugin at the Python package's version, or export with
`layout="dense"`.

## Learned radials and precision

- `lean=True` (the default) exports the lean evaluation form, as the ASE
  calculator does. With `spline_tol="auto"` a learned radial is splined at
  1e-10 first; see [Learn the radial basis](learned-radials.md#deployment-spline-speed).
- `dtype="float64"` is the default; `"float32"` is faster and less accurate.
