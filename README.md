# ace-jax

[![PyPI](https://img.shields.io/pypi/v/ace-jax)](https://pypi.org/project/ace-jax/)
[![Docs](https://img.shields.io/github/actions/workflow/status/ACEsuit/ace-jax/docs.yml?branch=main&label=docs)](https://acesuit.github.io/ace-jax/)
[![Tests](https://img.shields.io/github/actions/workflow/status/ACEsuit/ace-jax/test.yml?branch=main&label=tests)](https://github.com/ACEsuit/ace-jax/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ACEsuit/ace-jax/blob/main/LICENSE)

Build, fit and evaluate **Atomic Cluster Expansion (ACE)** interatomic
potentials in Python with [JAX](https://docs.jax.dev), on CPU or GPU. One
`pip install` gives all of these functions.

**Documentation: <https://acesuit.github.io/ace-jax/>**. It contains the
installation, a quickstart, tutorials, how-to guides, and the CLI and API
reference.

## What it does

- **Build a basis and fit it with one command.** `aj fit --order 3
  --max-degree 10 --train train.xyz --out fit` builds the symmetry-adapted
  ACE basis and fits it. The `fit.yaml` file that it writes runs the fit
  again.
- **Bayesian fits.** Linear ACE uses Bayesian linear regression. The fit
  selects the energy, force and virial noise levels and the prior scale by
  maximising the evidence, so you do not set weights by hand.
    - A hybrid ACE + Gaussian-process model adds a calibrated set of
      uncertainty approximations (MAP, Laplace, Pathfinder, VI, NUTS).
    - The linear model has POPS and ARD uncertainties.
    - ARD gives a conformally calibrated per-atom force uncertainty
      (`forces_std`, `forces_q`, a 3x3 `forces_cov`). `aj calibrate`
      recalibrates it on new labelled cells.
    - The fit can also learn the radial basis (`--learn-radial`).
- **Fast evaluation.** `ACECalculator` and `GPCalculator` are ASE
  calculators, fast enough for molecular dynamics. GP and ARD models also
  predict `energy_std` and `forces_std`. `export_lammps` deploys a model to
  LAMMPS with [lammps-jax](https://github.com/abhijeetgangan/lammps-jax).
- **PACE potentials.** ace-jax loads, evaluates and writes pacemaker
  `.yace` files.
- **Data in, data out.** Training data is extended XYZ or a list of
  `ase.Atoms`. ace-jax converts stress labels to virials. `aj eval` writes
  predictions as extended XYZ, with an RMSE table for each configuration
  type.

## Install

```bash
pip install ace-jax
```

Extras: `"ace-jax[cuda]"` (CUDA 12 JAX), `"ace-jax[gp]"` (the Pathfinder rung)
and `"ace-jax[fast-neighbours]"` (a C++ neighbour list).

To build a new basis, ace-jax needs the `ace-jax-coupling` wheel. This core
dependency is available for Linux x86_64 and aarch64, macOS arm64 and
Windows x64. On other platforms, fit and evaluate from an existing basis
file. See
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
small silicon dataset. The
[tutorials](https://acesuit.github.io/ace-jax/tutorials/) are notebooks that run
on a laptop CPU. Some of them come from the
[MLIP School 2026](https://mlipschool.uk/). For coding
agents, [`skills/ace-jax/SKILL.md`](https://github.com/ACEsuit/ace-jax/blob/main/skills/ace-jax/SKILL.md)
is a compact usage guide.

## Performance and validation

Fits agree with the ACEfit design matrix and least-squares solution to 1e-8.
PACE evaluation agrees with the ML-PACE C++ code and python-ace. CI checks
both. The throughput benchmarks against LAMMPS ML-PACE are in
[`docs/dev/benchmarks.md`](https://github.com/ACEsuit/ace-jax/blob/main/docs/dev/benchmarks.md).

## Contributing

See [CONTRIBUTING.md](https://github.com/ACEsuit/ace-jax/blob/main/CONTRIBUTING.md)
for the development setup, tests and the reference-parity jobs. See the
[changelog](https://github.com/ACEsuit/ace-jax/blob/main/CHANGELOG.md) for
releases. ace-jax has the MIT licence and is part of [ACEsuit](https://github.com/ACEsuit).
