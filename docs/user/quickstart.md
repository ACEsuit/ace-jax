# Quickstart

This page runs the command-line workflow once: fit a linear ACE model
straight from labelled data, evaluate it, and use it from Python. The data
is a small silicon set (DFT labels; diamond and β-tin cells, a liquid
snapshot and an isolated atom), already split into 40 training and 13 test
configurations. It takes about a minute on a CPU.

```bash
--8<-- "install.txt"
```

## 1. Get the data

```bash
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/si/train.xyz
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/si/test.xyz
```

Both are extended XYZ files with the labels stored as `energy`, `forces` and
`virial`, the names `aj` reads by default. If your own data uses other names,
pass them with `--energy-key`, `--force-key` and `--virial-key`.

## 2. Fit

```bash
aj fit --order 3 --max-degree 10 \
    --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs \
    --out fit
```

- `--order 3 --max-degree 10` builds the basis inside the fit, for the
  species found in the data. `--order` is the correlation order (how many
  neighbours a basis function couples; 3 gives four-body terms) and
  `--max-degree` bounds the polynomial degree, which sets the basis size.
- The radial basis is the radial polynomials themselves (`--radial-mode
  onehot`, the default); it stays frozen, and only the readout is fitted
  (to learn it too, see [learned radials](howto/learned-radials.md)).
- `--e0 lsq` sets the per-species reference energy $E_0$. The isolated atom
  in the training set fixes it to that atom's energy; without one it would
  be fitted by least squares (see [E0](concepts.md#e0-the-reference-energy)).
- `--m-per-species 0` selects the linear model (no Gaussian-process arm).
- `--opt lbfgs` maximises the evidence with L-BFGS, much faster than the
  default Adam on small data.

The fit logs the basis it built, then a table of test errors per
configuration type:

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

The `gamma missing` line is expected: the smoothness prior is rebuilt from the
basis. The last digits may differ with the platform.

`fit/` holds:

| File | Contents |
|---|---|
| `model.npz` | the fitted model: the basis and its coefficients, one ordinary ACE model file |
| `fit.yaml` | the whole resolved run; `aj fit --config fit/fit.yaml` reproduces it ([Run files](howto/fit-yaml.md)) |
| `metrics.csv` | test RMSE, MAE and the uncertainty calibration columns, per quantity ([Reading the metrics](concepts.md#reading-the-metrics)) |
| `theta_map.json` | the hyperparameters chosen by the evidence (noise and prior scales) |
| `config.json` | the configuration of the run, as the fitting pipeline saw it |

## 3. Evaluate

```bash
aj eval --model fit/model.npz --data test.xyz --out predictions.xyz
```

`aj eval` prints the same table for any labelled file, then writes the
structures back with the predictions added:

```text
...
all               13     26         25.25    0.0969        156.23
wrote 13 configurations with predictions (ace_energy, ace_forces, ...) to predictions.xyz
```

`predictions.xyz` keeps every original label and adds `ace_energy` (eV),
`ace_forces` (eV/Å, per atom) and, for periodic cells, `ace_stress`
(eV/Å³). Open it with ASE or any extended-XYZ reader to make parity plots.
Without `--out`, `aj eval` only prints the table.

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

`aj fit --order ...` stores the basis inside `model.npz`. To keep an
unfitted basis, for example to fit it to several datasets or share it, build
it with `aj basis` and pass it to the fit with `--model`:

```bash
aj basis --elements Si --order 3 --max-degree 10 --out si.npz
aj fit --model si.npz --r0 2.4 \
    --train train.xyz --test test.xyz \
    --e0 lsq --m-per-species 0 --opt lbfgs \
    --out fit_from_file
```

With `--model`, `--r0` (the typical nearest-neighbour distance in Å, which
centres the hyperprior) is required. The two fits are identical.

## Next steps

- [Tutorial 1](tutorials/first-fit.md) does the same fit in a notebook, with
  a parity plot, an equation of state and a short MD run.
- [Concepts](concepts.md) explains what `--order`, `--max-degree`, $E_0$ and the
  evidence fit mean.
- The [CLI reference](reference/cli.md) lists every flag.
