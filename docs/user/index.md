# ace-jax

ace-jax builds, fits and evaluates **Atomic Cluster Expansion (ACE)**
interatomic potentials in Python with [JAX](https://docs.jax.dev). One
`pip install` gives all of these functions. It also evaluates pacemaker **PACE `.yace`** potentials.

[![Docs](https://img.shields.io/github/actions/workflow/status/ACEsuit/ace-jax/docs.yml?branch=main&label=docs)](https://acesuit.github.io/ace-jax/)
[![Tests](https://img.shields.io/github/actions/workflow/status/ACEsuit/ace-jax/test.yml?branch=main&label=tests)](https://github.com/ACEsuit/ace-jax/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ACEsuit/ace-jax/blob/main/LICENSE)

---

- **Fit from data with one command.** `aj fit --order 3 --max-degree 10 ...`
  builds the ACE basis for the species in your data and fits it. The fit is
  a Bayesian linear regression: the evidence sets the weights and the
  regularisation. Each fit writes a `fit.yaml` file that runs it again.
- **Build a basis on its own** with `aj basis`. You can then save it, share
  it or fit it again.
- **Learn the radial basis** by variable projection. For deployment, the
  learned radials become splines, so they are as fast as a stock model.
- **Evaluate** with `aj eval`, or in Python with the ASE calculator
  `ACECalculator`.
- **Deploy** to LAMMPS with [lammps-jax](https://github.com/abhijeetgangan/lammps-jax).

```bash
--8<-- "install.txt"
aj fit --order 3 --max-degree 10 \
    --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs \
    --out fit
aj eval --model fit/model.npz --data test.xyz --out predictions.xyz
```

The [Quickstart](quickstart.md) runs these commands on a small silicon
dataset. The [tutorials](tutorials/index.md) are marimo notebooks. You can
run them on your computer or open them in molab.

## Where to go next

| If you want to | Read |
|---|---|
| install ace-jax and its extras | [Installation](installation.md) |
| run fit → eval once from the shell | [Quickstart](quickstart.md) |
| work through a fit in a notebook | [Tutorials](tutorials/index.md) |
| understand the basis, E0, the prior and the fit | [Concepts](concepts.md) |
| use a model in ASE or LAMMPS, or load a `.yace` | How-to guides: [ASE](howto/ase.md), [LAMMPS](howto/lammps.md), [PACE](howto/pace.md) |
| reproduce or vary a fit from its `fit.yaml` | [Run files](howto/fit-yaml.md) |
| learn the radial basis | [Learn the radial basis](howto/learned-radials.md) |
| put calibrated error bars on forces | [Per-atom force uncertainty](howto/force-uncertainty.md) |
| look up every CLI flag | [CLI reference](reference/cli.md) |
| look up a Python function | [Python API](reference/api.md) |
| correct a problem | [FAQ and troubleshooting](faq.md) |

ace-jax is part of the [ACEsuit](https://github.com/ACEsuit) organisation.
Its licence is MIT; see [Licence and citation](licence.md).
