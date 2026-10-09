# Export a model to LAMMPS

ace-jax models run in LAMMPS with
[lammps-jax](https://github.com/abhijeetgangan/lammps-jax). lammps-jax gives
`pair_style jax/kk`, a KOKKOS pair style that runs a compiled JAX program on
the GPU. ace-jax writes the program and its metadata into one JSON
*bundle*.

!!! note "lammps-jax is not on PyPI"
    1. Install lammps-jax from a clone or a fixed commit, for example
       `pip install "lammps-jax @ git+https://github.com/abhijeetgangan/lammps-jax"`.
    2. Build its LAMMPS plugin. Use the lammps-jax instructions.

    The ace-jax CI tests the export with lammps-jax commit `4a7f4fb`. The pair
    style runs only on a GPU (KOKKOS with CUDA).

## Export

The buffers of the bundle have fixed sizes. Calculate these sizes from the
initial structure of the run:

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

`type_elements` maps LAMMPS atom types to elements. Type 1 is the first
entry. `export_lammps` returns a dictionary, and also writes it in the
bundle. Its `ace_jax` key records what was exported: the layout, the
capacities, the use of the lean form and of splines, and the lammps-jax
version.

In the LAMMPS input:

```text
pair_style jax/kk <path to the PJRT GPU plugin>    # see the lammps-jax documentation
pair_coeff * * si_bundle.json
```

Run LAMMPS with KOKKOS on the GPU. For example:
`lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps`.

## Capacities

`neighbour_capacity(atoms, rcut, skin=1.0, slots="skin", margin=8, list_headroom=0.5)`
returns all buffer sizes for a structure:

| Key | Meaning |
|---|---|
| `max_atoms` | owned plus ghost atoms (the ghost shell size comes from the face spacings of the cell, so it is also correct for triclinic cells) |
| `max_owned` | owned atoms (with 10% headroom); rows after it are never evaluated |
| `k_dense` | model neighbour slots per atom |
| `max_neighbors` | LAMMPS neighbour-list slots per atom (with 50% headroom: the list increases most when a structure is compressed) |
| `max_edges` | the packed edge buffer of the sparse and dense layouts |

If a structure becomes too large for a buffer, the forces are never
truncated without a message. The energy and forces become NaN, or LAMMPS
stops the run. If this occurs, export again with sizes for the larger
structure.

`slots="cutoff"` sizes the model slots only for pairs in the cutoff. On a
five-component alloy, this is 1.2 to 1.4 times faster.

!!! warning
    Use `slots="cutoff"` only for stable MD of a fitted model, where the
    coordination stays near the coordination of the initial structure.

## Layouts

| Layout | What it does | When `"auto"` picks it |
|---|---|---|
| `matrix` | reads the LAMMPS neighbour list directly, copied only when LAMMPS rebuilds it; no per-step packing | `k_dense` and `max_neighbors` given, the installed lammps-jax supports it, and a block fits in memory |
| `dense` | packs the edge buffer into per-atom slots every step | `k_dense` given and `matrix` not chosen |
| `sparse` | an edge list | no `k_dense`, or no dense block fits in memory |

## GPU memory

The `matrix` and `dense` layouts evaluate the atoms in blocks of rows. One
block sets the peak memory. `export_lammps` uses the largest block that fits
in half of the GPU memory. It starts at 32768 rows and halves the block down to
1024 rows. If no block fits, `layout="auto"` uses `sparse`.

By default, `export_lammps` reads the memory of the GPU that does the export.
If LAMMPS runs on a different GPU, give its memory:

```python
export_lammps(model, meta, "si_bundle.json", ...,
              device_memory=30e9)                          # bytes JAX can use on the LAMMPS GPU
```

`device_memory` is the memory that JAX can use, not the total memory of the
GPU. By default, JAX uses 75% of the GPU memory. For a 40 GB A100, use
approximately 30e9. `block_rows` sets the block size directly.

The bundle records the block size as `ace_jax.block_rows`. It also records
`ace_jax.device_memory` and `ace_jax.dense_budget`.

A LAMMPS plugin that is older than the lammps-jax Python package does not
accept a `matrix` bundle. To correct this, build the plugin again at the
version of the Python package, or export with `layout="dense"`.

## Learned radials and precision

- `lean=True` (the default) exports the lean evaluation form, as the ASE
  calculator does. With `spline_tol="auto"`, `export_lammps` first changes a
  learned radial to a spline at 1e-10; see [Learn the radial basis](learned-radials.md#deployment-spline-speed).
- `dtype="float64"` is the default. `"float32"` is faster and less accurate.
