# Quickstart

On this page, you fit a linear ACE model to labelled data. Then you evaluate
the model and use it from Python.

The data is a small silicon dataset with DFT labels. It contains diamond and
β-tin cells, liquid snapshots and an isolated atom. It is already divided
into 40 training configurations and 13 test configurations. The procedure
takes approximately 1 minute on a CPU.

```bash
--8<-- "install.txt"
```

## 1. Get the data

```bash
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/si/train.xyz
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/si/test.xyz
```

The two files are extended XYZ files. Their labels are `energy`, `forces`
and `virial`, the names that `aj` reads by default. If your data uses
different names, give them with `--energy-key`, `--force-key` and
`--virial-key`.

## 2. Fit

```bash
aj fit --order 3 --max-degree 10 \
    --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs \
    --out fit
```

- `--order 3 --max-degree 10` builds the basis in the fit, for the species
  in the data.
    - `--order` is the correlation order: the number of neighbours that a
      basis function couples. Order 3 gives four-body terms.
    - `--max-degree` sets the maximum polynomial degree. This sets the basis
      size.
- The radial basis is the radial polynomials (`--radial-mode onehot`, the
  default). The radial basis stays frozen, and the fit changes only the
  coefficients. To learn the radial basis also, see
  [learned radials](howto/learned-radials.md).
- `--e0 lsq` sets the reference energy $E_0$ of each species. The training
  set has an isolated atom, so $E_0$ is set to the energy of that atom. If
  there is no isolated atom, the fit calculates $E_0$ with the model (see
  [E0](concepts.md#e0-the-reference-energy)).
- `--m-per-species 0` selects the linear model (no Gaussian-process arm).
- `--opt lbfgs` maximises the evidence with L-BFGS. For the linear model, a
  small number of Newton steps then refine the result to a stationary point.
  `--opt lbfgs` is the default; this example gives it only to show it.

The fit logs the basis that it built. Then it prints a table of test errors
for each configuration type:

```text
basis Si order 3 max-degree 10 -> 110 B functions (elements from the data)
E0: isolated-atom energies for Z=14 -158.544968
r0 2.400 A (mean bond length of the basis; pass --r0 to override)
gamma missing from basis Si order 3 max-degree 10 -- rebuilt via basis.prior (algebraic smoothness prior)
...
RMSE, test (map)
-----------------------------------------------------------------
config type  configs  atoms  E (meV/atom)  F (eV/Å)  V (meV/atom)
-----------------------------------------------------------------
bt                 6     12         34.87    0.0937        203.63
dia                7     14         11.93    0.0995         98.91
-----------------------------------------------------------------
all               13     26         25.25    0.0969        156.23
fitted model: fit/model.npz
```

The `gamma missing` line is normal: the fit builds the smoothness prior
again from the basis. The last digits can be different on a different
platform.

The fit writes these files in `fit/`:

| File | Contents |
|---|---|
| `model.npz` | the fitted model: the basis and its coefficients, in one usual ACE model file |
| `fit.yaml` | all settings of the run; `aj fit --config fit/fit.yaml` runs it again ([Run files](howto/fit-yaml.md)) |
| `metrics.csv` | test RMSE, MAE and the uncertainty calibration columns, for each quantity ([Reading the metrics](concepts.md#reading-the-metrics)) |
| `theta_map.json` | the hyperparameters that the evidence selected (noise and prior scales) |
| `config.json` | the configuration of the run, as the fitting pipeline used it |

## 3. Evaluate

```bash
aj eval --model fit/model.npz --data test.xyz --out predictions.xyz
```

`aj eval` prints the same table for each labelled file. Then it writes the
structures again, with the predictions added:

```text
...
all               13     26         25.25    0.0969        156.23
wrote 13 configurations with predictions (ace_energy, ace_forces, ...) to predictions.xyz
```

`predictions.xyz` keeps all original labels and adds:

- `ace_energy` (eV);
- `ace_forces` (eV/Å, for each atom);
- `ace_stress` (eV/Å³), for periodic cells only.

Without `--out`, `aj eval` only prints the table.

To make parity plots, read the file in Python with the `cextxyz` format.
This format comes from the ase-extxyz plugin, which ace-jax installs. It
keeps each label under its own name:

```python
from ase.io import read

frames = read("predictions.xyz", ":", format="cextxyz")
dft = [a.info["energy"] / len(a) for a in frames]
ace = [a.info["ace_energy"] / len(a) for a in frames]
```

!!! note
    Do not use the built-in ASE `extxyz` reader for this file. It moves
    `energy` and `forces` into a calculator, and it does not accept 3×3
    values (for example `virial`) written as nested lists.

## 4. Use the model from Python

```python
import jax
jax.config.update("jax_enable_x64", True)
from ase.build import bulk
from ace_jax import ACECalculator

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = ACECalculator("fit/model.npz")
print(atoms.get_potential_energy(), atoms.get_forces().shape, atoms.get_stress())
```

## Saving a basis on its own

`aj fit --order ...` keeps the basis in `model.npz`. To use the same unfitted
basis for more than one fit, or to share it:

1. Build the basis with `aj basis`.
2. Give the basis file to `aj fit --model`.

```bash
aj basis --elements Si --order 3 --max-degree 10 --out si.npz
aj fit --model si.npz --r0 2.4 \
    --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs \
    --out fit_from_file
```

With `--model`, you must give `--r0`. `--r0` is the typical
nearest-neighbour distance in Å, and it is the centre of the hyperprior.
The two fits are identical.

## Next steps

- [Tutorial 1](tutorials/first-fit.md) does the same fit in a notebook. It
  adds a parity plot, an equation of state and a short MD run.
- [Concepts](concepts.md) explains `--order`, `--max-degree`, $E_0$ and the
  evidence fit.
- The [CLI reference](reference/cli.md) lists all flags.
