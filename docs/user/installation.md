# Installation

ace-jax needs Python 3.11 or later. It is tested on Python 3.11 to 3.14. It
is a pure-Python package; JAX gives the compiled numerics.

```bash
--8<-- "install.txt"
```

Most users need only this command. With [uv](https://docs.astral.sh/uv/),
use `uv pip install ...`, or `uv add` with the same requirement in a
project.

## What gets installed

| Install | Adds | Needed for |
|---|---|---|
| `ace-jax` | the core: building a basis, fitting (`aj fit`, every arm), learned radials, model loading, evaluation, ASE calculators, `aj basis`, `aj eval` | everything below except the pathfinder rung |
| `ace-jax[gp]` | blackjax | the `pathfinder` rung of `--rungs` |
| `ace-jax[cuda]` | `jax[cuda12]` | running on an NVIDIA GPU |
| `ace-jax[fast-neighbours]` | matscipy-neighbours | faster neighbour lists (ASE's list is the fallback) |

You can combine extras, for example `ace-jax[gp,cuda]`.

### Building a basis: supported platforms

To build a new basis (`aj fit --order ...`, `aj basis`, or `build_basis` in
Python), ace-jax uses `ace-jax-coupling`. This compiled library calculates
the symmetry-adapted coupling coefficients. pip installs it automatically on
these platforms:

- Linux x86_64 and aarch64 (manylinux_2_28, so glibc 2.28 or newer)
- macOS arm64
- Windows x64

You do not need to set up anything else, and nothing is downloaded at run
time.

On other platforms, ace-jax installs without this library. All functions
work, except the build of a new basis:

- A basis file built on a different machine fits and evaluates normally
  (`aj fit --model basis.npz`).
- If you try to build a basis, ace-jax gives the error `BasisUnavailable`,
  with this instruction.

The first build of a new basis specification calculates its coupling
coefficients. This takes milliseconds. ace-jax keeps the result in
`~/.cache/ace-jax/coupling` (or `$ACEJAX_COUPLING_CACHE`). Later builds of
the same specification read the cache.

### `fast-neighbours`

matscipy-neighbours is a source distribution. Thus its installation compiles
C++, and needs CMake and a C++17 compiler. ace-jax uses matscipy-neighbours
when it can import it. If not, ace-jax uses the ASE neighbour list.

## GPU

1. Add `cuda` to the extras that you install.
2. Make sure that JAX finds the GPU:

    ```bash
    python -c "import jax; print(jax.devices())"    # the output must show a CUDA device
    ```

The `cuda` extra installs the JAX CUDA 12 wheels. For other CUDA versions or
platforms:

1. Install JAX with the
   [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
2. Install ace-jax.

To use the CPU on a GPU machine, set `JAX_PLATFORMS=cpu`.

## Precision (float64)

--8<-- "float64.md"

Evaluation works in float32 and float64. By default (`dtype=None`),
`ACECalculator` uses the JAX default precision. Thus, if float64 is not
enabled, it uses float32. Enable float64 also for evaluation, unless you
need the float32 speed and accept its error.

## LAMMPS

LAMMPS export needs [lammps-jax](https://github.com/abhijeetgangan/lammps-jax).
lammps-jax is not on PyPI.

1. Install lammps-jax from a clone (`pip install -e <lammps-jax>`).
2. Build its LAMMPS plugin.

See [LAMMPS export](howto/lammps.md).

## Development install

```bash
git clone https://github.com/ACEsuit/ace-jax
cd ace-jax
uv sync                    # core + the gp extra's packages + dev tools
uv run pytest              # the fast test suite
```

## Checking the install

```bash
aj --help
python -c "import ace_jax_coupling; print(ace_jax_coupling.build_info()['platform'])"   # basis building
```
