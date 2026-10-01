# Tutorial 2: learned radials

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/learned_radials_si.py)

Notebook: [`learned_radials_si.py`](notebooks/learned_radials_si.py) ·
[view on GitHub](https://github.com/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/learned_radials_si.py)

```bash
uvx marimo edit --sandbox learned_radials_si.py
```

Learn the radial basis of a silicon ACE model from data, by variable
projection with a held-out gate, and compare it with the frozen, seeded
radial basis that a basis is built with by default. Do
[Tutorial 1](first-fit.md) first.

## Goals

1. Fit a baseline with the frozen, seeded radials.
2. Learn the radial weights, with a held-out gate deciding whether to keep
   them.
3. Refit with the learned basis and compare the test errors.
4. Deploy the learned model: `ACECalculator` splines it automatically.

## Steps

| Step | What happens | Checkpoint |
|---|---|---|
| 1. Data | the Tutorial 1 split, plus a fit / validation split of the training set | |
| 2. Frozen baseline | a random-start basis (`radial_mode="glorot_normal"`), fitted and tested | |
| 3. Learn | `fit_radial`: 40 L-BFGS steps, then the gate | the learned radials win the gate |
| 4. Inspect | the radial functions before and after, over the pair-distance histogram | |
| 5. Refit and test | the learned basis, fitted exactly as the baseline | smaller test force error |
| 6. Deploy | `ACECalculator` splines the learned radials; compare with the exact evaluation | splined and exact agree |

On a CPU the notebook runs in about three minutes, two of them for learning.
Typical results on the 13-configuration test set:

| Basis | E RMSE (meV/atom) | F RMSE (eV/Å) |
|---|---|---|
| frozen seeded radials | 470 | 0.197 |
| learned radials | 28 | 0.149 |

The notebook starts from seeded random radials on purpose: kept frozen they
are a poor fit, and the learned ones recover most of the difference. The
default `onehot` basis of Tutorial 1 is already good on this small dataset
(24 meV/atom, 0.103 eV/Å); learning from it (exercise 2) lowers the energy
error to about 19 meV/atom and leaves the force error unchanged. On larger
MACE-labelled datasets (200 SiGe and 150 CrMnFeCoNi training configurations),
learning from the ACEpotentials default radials lowered the held-out gate
score by 21% and 48%.

The learning step uses the Python API (`ace_jax.fit.radial_learn`). On the
command line, `aj fit --learn-radial` runs the same steps in one fit; see
[Learn the radial basis](../howto/learned-radials.md).

## Exercises

1. Vary the L-BFGS step budget, including 0 (nothing learned: the gate keeps
   the initial radials).
2. Start from the `onehot` basis of Tutorial 1. Does learning still help, and
   what does the gate select?
3. Add a roughness penalty (`lam_grid=(0.0, 1e-2)`) and let the gate choose.
