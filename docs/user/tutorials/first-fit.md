# Tutorial 1: a first fit for silicon

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/first_fit_si.py)

Notebook: [`first_fit_si.py`](notebooks/first_fit_si.py) ·
[view on GitHub](https://github.com/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/first_fit_si.py)

```bash
uvx marimo edit --sandbox first_fit_si.py
```

Fit a linear ACE model to `si_tiny_train.xyz`, a 53-configuration silicon
dataset (DFT labels), and use it as an ASE calculator. It is the
[Quickstart](../quickstart.md) in notebook form, with the checks you would
make on a real fit.

## Goals

1. Build an ACE basis and know what the correlation order and the maximum
   degree control.
2. Fit it with the evidence-based linear fit, and read the test metrics.
3. Judge the fit with a parity plot, not only an RMSE.
4. Use the fitted model in ASE: an equation of state and NVE molecular
   dynamics.

## Steps

| Step | What happens | Checkpoint |
|---|---|---|
| 1. Data | fetch the dataset, drop the isolated atom, split 39 train / 13 test | |
| 2. Basis | `build_basis(BasisSpec(order=3, max_degree=10, radial_mode="onehot"))`: 120 functions | |
| 3. Fit | `FitConfig` + `fit`, the Python form of `aj fit` | test F RMSE below 0.2 eV/Å |
| 4. Parity plot | evaluate the test set with `ACECalculator` | the calculator reproduces the fit's RMSE |
| 5. Equation of state | a Birch–Murnaghan fit for diamond Si | a0 and the bulk modulus are physical |
| 6. Molecular dynamics | 200 NVE steps of a 64-atom cell at 600 K | total energy conserved |

On a CPU the whole notebook runs in about a minute, about 30 s of it the fit.
Typical results: test errors of 24 meV/atom and 0.10 eV/Å, a0 = 5.46 Å,
B = 88 GPa.

## Exercises

1. Change the maximum degree to 8 and 12: how do the basis size and the
   test errors move?
2. Switch the radial mode to `glorot_normal`, the default of `aj basis` and
   `aj fit`, and compare.
3. Keep the isolated atom in the training set: why does the energy error
   jump? (See [E0](../concepts.md#e0-the-reference-energy).)
4. Repeat the equation of state for β-tin.
