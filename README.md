# ace-jax

[![PyPI](https://img.shields.io/pypi/v/ace-jax)](https://pypi.org/project/ace-jax/)
[![Docs](https://img.shields.io/github/actions/workflow/status/ACEsuit/ace-jax/docs.yml?branch=main&label=docs)](https://acesuit.github.io/ace-jax/)
[![Tests](https://img.shields.io/github/actions/workflow/status/ACEsuit/ace-jax/test.yml?branch=main&label=tests)](https://github.com/ACEsuit/ace-jax/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ACEsuit/ace-jax/blob/main/LICENSE)

Build, fit and evaluate **Atomic Cluster Expansion (ACE)** interatomic
potentials in Python with [JAX](https://docs.jax.dev), on CPU or GPU, all from
one `pip install`.

**Documentation: <https://acesuit.github.io/ace-jax/>**: installation,
quickstart, tutorials, how-to guides, and the CLI and API reference.

## What it does

- **Build a basis and fit it in one command.** `aj fit --order 3 --max-degree 10
  --train train.xyz --out fit` builds the symmetry-adapted ACE basis and fits it.
  The `fit.yaml` it writes reproduces the run.
- **Bayesian fits.** Linear ACE is fitted by Bayesian linear regression, with the
  energy, force and virial noise levels and the prior scale chosen by maximising
  the evidence: no hand-tuned weights. A hybrid ACE + Gaussian-process arm adds
  a calibrated uncertainty ladder (MAP, Laplace, Pathfinder, VI, NUTS), and the
  linear model has POPS and ARD uncertainties. ARD serves conformally calibrated
  per-atom force uncertainty (`forces_std`, `forces_q`, a 3x3 `forces_cov`),
  recalibrated on new labelled cells with `aj calibrate`. The radial basis can be learned
  as part of the fit (`--learn-radial`).
- **Fast evaluation.** `ACECalculator` and `GPCalculator` are ASE calculators,
  fast enough for molecular dynamics, with predicted `energy_std` and
  `forces_std` for GP and ARD models. `export_lammps` deploys a model to LAMMPS
  through [lammps-jax](https://github.com/abhijeetgangan/lammps-jax).
- **PACE potentials.** pacemaker `.yace` files load, evaluate and write back.
- **Data in, data out.** Training data is extended XYZ or a list of `ase.Atoms`
  (stress labels are converted to virials). `aj eval` writes predictions as
  extended XYZ, with an RMSE table per config type.

## Install

```bash
pip install ace-jax
```

Extras: `"ace-jax[cuda]"` (CUDA 12 JAX), `"ace-jax[gp]"` (the Pathfinder rung)
and `"ace-jax[fast-neighbours]"` (a C++ neighbour list). Building new bases needs
the `ace-jax-coupling` wheel, a core dependency available for Linux x86_64 and
aarch64, macOS arm64 and Windows x64; elsewhere, fit and evaluate from an
existing basis file. See
[Installation](https://acesuit.github.io/ace-jax/installation/).

## Quickstart

Fit a linear ACE model to labelled data, check it on a test set, and evaluate it:

```bash
aj fit --order 3 --max-degree 10 --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs --out fit
aj eval --model fit/model.npz --data test.xyz --out predictions.xyz
```

Use the fitted model from Python:

```python
import jax
jax.config.update("jax_enable_x64", True)

from ase.build import bulk
from ace_jax import ACECalculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = ACECalculator("fit/model.npz")
print(atoms.get_potential_energy(), atoms.get_forces())
```

The [Quickstart](https://acesuit.github.io/ace-jax/quickstart/) runs this on a
small silicon data set, and the
[tutorials](https://acesuit.github.io/ace-jax/tutorials/) are notebooks that run
on a laptop CPU, including ones adapted from the
[MLIP school 2026](https://github.com/ACEsuit/MLIP-school-2026). For coding
agents, [`skills/ace-jax/SKILL.md`](https://github.com/ACEsuit/ace-jax/blob/main/skills/ace-jax/SKILL.md)
is a compact usage guide.

## Performance and validation

Fits reproduce ACEfit's design matrix and least-squares solve to 1e-8, and PACE
evaluation matches the ML-PACE C++ code and python-ace; CI checks both. The
throughput benchmarks against LAMMPS ML-PACE are in
[`docs/dev/benchmarks.md`](https://github.com/ACEsuit/ace-jax/blob/main/docs/dev/benchmarks.md).

## Contributing

See [CONTRIBUTING.md](https://github.com/ACEsuit/ace-jax/blob/main/CONTRIBUTING.md)
for the development setup, tests and the reference-parity jobs, and the
[changelog](https://github.com/ACEsuit/ace-jax/blob/main/CHANGELOG.md) for
releases. ace-jax is MIT-licensed and part of [ACEsuit](https://github.com/ACEsuit).
