# Installation

ace-jax needs Python 3.11, 3.12 or 3.13. It is a pure-Python package; JAX
provides the compiled numerics.

```bash
--8<-- "install.txt"
```

That is what most users want. With [uv](https://docs.astral.sh/uv/), put
`uv` in front (`uv pip install ...`), or use `uv add` with the same
requirement in a project.

## What gets installed

| Install | Adds | Needed for |
|---|---|---|
| `ace-jax` | the core: building a basis, model loading, evaluation, ASE calculators, `aj basis`, `aj eval` | evaluating a model, building a basis |
| `ace-jax[gp]` | numpyro, optax, blackjax | `aj fit` (every arm, the linear one included) and learned radials |
| `ace-jax[cuda]` | `jax[cuda12]` | running on an NVIDIA GPU |
| `ace-jax[fast-neighbours]` | matscipy-neighbours | faster neighbour lists (ASE's list is the fallback) |

Extras combine, for example `ace-jax[gp,cuda]`.

### Building a basis: supported platforms

Building a new basis (`aj fit --order ...`, `aj basis`, or `build_basis` in
Python) uses `ace-jax-coupling`, a compiled library for the symmetry-adapted
coupling coefficients. It is installed automatically, with nothing else to set
up and nothing downloaded at run time, on:

- Linux x86_64 and aarch64 (manylinux_2_28, so glibc 2.28 or newer)
- macOS arm64
- Windows x64

On any other platform ace-jax installs without it. Everything except building
a new basis shape still works: a basis file built elsewhere fits and
evaluates normally (`aj fit --model basis.npz`), and building one raises
`BasisUnavailable` with that hint.

The first build of a new basis shape computes its coupling coefficients (it
takes milliseconds) and caches them in `~/.cache/ace-jax/coupling` (or
`$ACEJAX_COUPLING_CACHE`). Later builds of the same shape read the cache.

### `fast-neighbours`

matscipy-neighbours is published as a source distribution, so installing it
compiles C++ (it needs CMake and a C++17 compiler). ace-jax uses it when it
is importable and otherwise falls back to ASE's neighbour list.

## GPU

Add `cuda` to the extras you install, then check that JAX sees the GPU:

```bash
python -c "import jax; print(jax.devices())"    # should list a CUDA device
```

The `cuda` extra installs JAX's CUDA 12 wheels. For other CUDA versions or
platforms, install JAX first by following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then install ace-jax. To force the CPU on a GPU machine, set
`JAX_PLATFORMS=cpu`.

## Precision (float64)

The ace-jax library never changes JAX's precision setting, so the caller
chooses it. JAX defaults to float32.

- The `aj` command line enables float64 itself.
- In Python, fitting and radial learning require float64. Enable it before
  anything else imports JAX:

    ```python
    import jax
    jax.config.update("jax_enable_x64", True)
    ```

    or set `JAX_ENABLE_X64=1` in the environment.

- Evaluation works in either precision. `ACECalculator` (with its default
  `dtype=None`) computes in JAX's default dtype, so without float64 enabled
  it runs in float32. Enable float64 for evaluation too unless you want
  float32 speed and accept its error.

## LAMMPS

LAMMPS export needs [lammps-jax](https://github.com/abhijeetgangan/lammps-jax),
which is not on PyPI. Install it from a clone (`pip install -e <lammps-jax>`)
and build its LAMMPS plugin; see [LAMMPS export](howto/lammps.md).

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
