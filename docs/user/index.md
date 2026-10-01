# ace-jax

ace-jax builds, fits and evaluates **Atomic Cluster Expansion (ACE)**
interatomic potentials in Python with [JAX](https://docs.jax.dev), all from
one `pip install`. It also evaluates pacemaker **PACE `.yace`** potentials.

[![Docs](https://img.shields.io/github/actions/workflow/status/ACEsuit/ace-jax/docs.yml?branch=main&label=docs)](https://acesuit.github.io/ace-jax/)
[![Tests](https://img.shields.io/github/actions/workflow/status/ACEsuit/ace-jax/test.yml?branch=main&label=tests)](https://github.com/ACEsuit/ace-jax/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ACEsuit/ace-jax/blob/main/LICENSE)

---

- **Fit from data in one command**: `aj fit --order 3 --max-degree 10 ...`
  builds the ACE basis for the species in your data and fits it, as Bayesian
  linear regression with evidence-chosen weights and regularisation. Every
  fit writes a `fit.yaml` that reproduces it.
- **Build a basis on its own** with `aj basis`, to save, share or refit it.
- **Learn the radial basis** by variable projection, then deploy it at
  spline speed.
- **Evaluate** with `aj eval`, or in Python through the ASE calculator
  `ACECalculator`.
- **Deploy** to LAMMPS through [lammps-jax](https://github.com/abhijeetgangan/lammps-jax).

```bash
--8<-- "install.txt"
aj fit --order 3 --max-degree 10 \
    --train train.xyz --test test.xyz --e0 lsq --m-per-species 0 --opt lbfgs --out fit
aj eval --model fit/model.npz --data test.xyz --forces
```

The [Quickstart](quickstart.md) runs these commands on a small silicon
dataset, and the [tutorials](tutorials/index.md) are marimo notebooks you can
run locally or open in molab.

## Where to go next

| If you want to | Read |
|---|---|
| install ace-jax and its extras | [Installation](installation.md) |
| run fit → eval once from the shell | [Quickstart](quickstart.md) |
| work through a fit in a notebook | [Tutorials](tutorials/index.md) |
| know what the basis, E0, the prior and the fit are | [Concepts](concepts.md) |
| use a model in ASE or LAMMPS, or load a `.yace` | How-to guides: [ASE](howto/ase.md), [LAMMPS](howto/lammps.md), [PACE](howto/pace.md) |
| reproduce or vary a fit from its `fit.yaml` | [Run files](howto/fit-yaml.md) |
| learn the radial basis | [Learn the radial basis](howto/learned-radials.md) |
| look up every CLI flag | [CLI reference](reference/cli.md) |
| look up a Python function | [Python API](reference/api.md) |
| fix something that went wrong | [FAQ and troubleshooting](faq.md) |

ace-jax is developed in the [ACEsuit](https://github.com/ACEsuit) organisation
and released under the MIT licence; see [Licence and citation](licence.md).
