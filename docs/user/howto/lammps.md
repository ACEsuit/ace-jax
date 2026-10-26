# Export a model to LAMMPS

ace-jax models run in LAMMPS with
[lammps-jax](https://github.com/abhijeetgangan/lammps-jax). lammps-jax gives
`pair_style jax/kk`, a KOKKOS pair style that runs a compiled JAX program on
the GPU. ace-jax writes the program and its metadata into one JSON
*bundle*.

!!! note "Install lammps-jax"
    1. Install ace-jax with the `lammps` extra: `pip install "ace-jax[lammps]"`.
       For a CPU run, also install `"lammps-jax[cpu]"`. This gives the XLA
       CPU PJRT plugin (Linux x86_64 only).
    2. Build its LAMMPS plugin at the same version. Use the lammps-jax instructions.

    The ace-jax CI tests the export with lammps-jax 0.2.1. The pair style runs
    on a GPU (KOKKOS with CUDA). It also runs on CPUs (KOKKOS with OpenMP) if
    you export with `platform="cpu"`. The plugin refuses a bundle that was
    exported for the other platform.

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
entry.

The `aj export` command does the same steps. It reads the initial structure
from the first configuration of an extxyz file:

```bash
aj export --model fit/model.npz --out si_bundle.json --structure start.xyz \
    --skin 1.0 --type-elements Si
```

For the CPU backend, add `--platform cpu`. Add `--procs px,py,pz` to size the
buffers for one MPI rank. Use the same grid in LAMMPS (`processors px py pz`).
For a linear ACE model on CPUs, the export uses a C++ kernel in place of the
XLA program. See [CPU kernel](#cpu-kernel). On a CUDA GPU, add `--ffi` to use
a CUDA kernel. See [GPU kernel](#gpu-kernel).

 `export_lammps` returns a dictionary, and also writes it in the
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

## CPU kernel

On CPUs, the export uses a C++ kernel for linear ACE models. The kernel
replaces the compiled JAX program. It runs one MPI rank per core, with one
thread per rank. The export compiles the kernel the first time, and keeps it
in a cache.

The export uses the kernel if all these conditions are true:

- The model is a linear ACE model, in float64.
- The bundle can use the `matrix` layout (`--layout` `auto` or `matrix`).
- lammps-jax is version 0.2.1 or later.
- A C++17 compiler is available (`$CXX`, `c++`, `g++` or `clang++`).

If a condition is false, the export uses the XLA program. `aj export` prints
the reason.

!!! warning "Environment for LAMMPS"
    A bundle with the kernel needs the variable `LAMMPS_JAX_FFI_HANDLERS` in
    LAMMPS. Without it, LAMMPS stops with an error that gives the variable.
    Export with `--no-ffi` to get a bundle that needs no variable.

!!! warning "lammps-jax version"
    The kernel needs the LAMMPS plugin of lammps-jax 0.2.1 or later. Earlier
    plugins register custom calls for CUDA only, so LAMMPS cannot find the
    kernel on the CPU. Build the plugin at the version of the Python package.

1. Export with `--platform cpu`:

    ```bash
    aj export --model fit/model.npz --out si_bundle.json --structure start.xyz \
        --platform cpu --procs 8,4,4
    ```

    In Python, use `export_lammps(..., platform="cpu")`.

2. Set the variable that the command prints. It gives LAMMPS the kernel library:

    ```bash
    export LAMMPS_JAX_FFI_HANDLERS="ace_jax_direct=<library>:AceDirect"
    ```

3. Run LAMMPS with `newton on` and one MPI rank per core:

    ```bash
    mpirun -np 128 lmp -k on t 1 -sf kk -pk kokkos newton on neigh half -in in.lammps
    ```

To get the variable on the machine that runs LAMMPS, use
`aj export --print-env si_bundle.json`. If the library is not on that machine,
the command compiles the kernel there.

| Option | Command line | Python |
|---|---|---|
| Use the kernel when possible (default) | (no option) | `ffi="auto"` |
| Always use the kernel; stop with an error if it is not possible | `--ffi` | `ffi=True` |
| Always use the XLA program | `--no-ffi` | `ffi=False` |

The kernel tabulates the radial basis in r (`radial_table=True`). Energies
agree with the XLA program to approximately 1e-12 relative. Forces agree to
1.5e-8 of the largest force. The bundle records the kernel in `ace_jax.ffi`:
the custom-call target, the library and the `LAMMPS_JAX_FFI_HANDLERS` value.
If the export used the XLA program on a CPU, `ace_jax.ffi_reason` gives the
reason.

The compiler flags are `-O3 -march=native` by default. They tune the kernel
for the CPU of the machine that exports.

!!! warning "Login nodes and compute nodes"
    A kernel compiled with `-march=native` can fail on a different CPU type.
    Export on a node of the same type as the LAMMPS run. Or give generic
    flags, for example `--ffi-flags="-O3 -march=x86-64-v3"`. To compile the
    kernel on a compute node before the export, use `aj native --build`.

- The cache is `~/.cache/ace-jax/ffi`. Set `ACEJAX_FFI_CACHE` to use a different directory.
- Set `ACEJAX_FFI_FLAGS` to change the default flags.
- Use the same jax version as the lammps-jax CPU plugin.

## GPU kernel

On a CUDA GPU, a CUDA kernel can replace the compiled JAX program. It is for
linear ACE models, as the [CPU kernel](#cpu-kernel) is. It supports categorical
models with one or more species, and species-embedded (TrACE) models. The
export compiles the kernel with `nvcc` the first time, and keeps it in the
same cache.

!!! note "CUDA compiler"
    The `cuda` extra installs `nvcc`: `pip install "ace-jax[cuda]"`. It uses
    the CUDA 13 packages, which need an NVIDIA driver 580 or later.
    With an older driver, install the `cuda12` extra, and set `NVCC` to
    the `nvcc` of a CUDA toolkit. The kernel also finds `nvcc` on `PATH`
    or in `$CUDA_HOME/bin`. The flags are `-O3 -arch=sm_XY`, for the GPU
    that JAX finds at the export. On a machine without a GPU, give the
    architecture, for example `--ffi-flags="-O3 -arch=sm_80"` for an A100.

1. Export with `--ffi`. The platform is `cuda` by default:

    ```bash
    aj export --model fit/model.npz --out si_bundle.json --structure start.xyz --ffi
    ```

    In Python, use `export_lammps(..., ffi=True)`.

2. Set the variable that the command prints:

    ```bash
    export LAMMPS_JAX_FFI_HANDLERS="ace_jax_direct_gpu=<library>:AceDirectGpu"
    ```

3. Run LAMMPS with `newton on`:

    ```bash
    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps
    ```

In float64, the requirements and the accuracy are the same as for the CPU kernel.

The GPU kernel also runs float32 bundles. To export one, add `--dtype float32`.
The kernel then calculates in float32, with the accuracy of the float32 XLA
program. On a GPU with low float64 throughput, it is 2 to 4 times faster than
in float64. On an A100, it is 1.2 to 1.3 times faster.
See [Learned radials and precision](#learned-radials-and-precision).

In Python, `ACECalculator(path, ffi=True)` uses the same kernel on a CUDA GPU.
Add `dtype="float32"` for float32. `BatchCalculator(..., ffi=True)` also uses it.
To compile the kernel before the first use, run `aj native --build`.

- Set `ACEJAX_FFI_GPU_FLAGS` to change the default flags.
- The source also compiles for AMD GPUs with `hipcc` (`eval.ffi.build_gpu`).
  It is not tested on AMD GPUs, and lammps-jax bundles do not run on AMD GPUs.

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

## cuequivariance on the GPU

!!! warning "Removed"
    The cuequivariance form (`aa_dag="cueq"`, the `cueq` extra) was removed.
    Use the [GPU kernel](#gpu-kernel). It is faster, and it supports
    all linear ACE models. `aa_dag="cueq"` now gives an error.

## Learned radials and precision

- `lean=True` (the default) exports the lean evaluation form, as the ASE
  calculator does. With `spline_tol="auto"`, `export_lammps` first changes a
  learned radial to a spline at 1e-10; see [Learn the radial basis](learned-radials.md#deployment-spline-speed).
- `dtype="float64"` is the default. `"float32"` is faster and less accurate.
  On a GPU with low float64 throughput, a float32 bundle is 2 to 4.5 times faster.
  On an A100, it is 1.3 to 1.5 times faster.
  See [Choose float64 or float32](precision.md).
- The float32 error depends on the model. Large coefficients that cancel make it larger.
- `precision_structure=atoms` compares float32 with float64 on `atoms`.
  The bundle records the result as `ace_jax.precision_check`.
- A warning occurs above 0.1 meV/atom of energy error or 1e-4 of relative force error.
  `aj export --dtype float32` does this check on `--structure`.
