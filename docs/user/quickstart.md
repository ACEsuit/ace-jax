# Quickstart

This page runs the command-line workflow once: fit a linear ACE model
straight from labelled data, evaluate it, and use it from Python. It uses
`si_tiny_train.xyz`, a 53-configuration silicon dataset from the ace-jax
test fixtures (DFT labels; diamond and β-tin cells, two liquid snapshots and
one isolated atom). It takes about two minutes on a CPU.

```bash
--8<-- "install.txt"
```

## 1. Get the data

```bash
curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/fixtures/si_tiny_train.xyz
```

The labels are stored under `dft_energy`, `dft_force` and `dft_virial`. The
`aj` commands default to `energy`, `forces` and `virial`, so every command
below passes the names explicitly:

```bash
K="--energy-key dft_energy --force-key dft_force --virial-key dft_virial"
```

Split the file into a training and a test set. The isolated atom is left out:
its energy is the reference energy E0 itself, and here E0 is fitted to the
bulk energies instead (`--e0 lsq` below; see [E0](concepts.md#e0-the-reference-energy)).

```bash
python - <<'EOF'
from ase.io import read, write
frames = [a for a in read("si_tiny_train.xyz", ":") if a.info["config_type"] != "isolated_atom"]
write("test.xyz", frames[::4])                                        # 13 configs
write("train.xyz", [a for i, a in enumerate(frames) if i % 4])       # 39 configs
EOF
```

## 2. Fit

```bash
aj fit --order 3 --max-degree 10 --radial-mode onehot \
    --train train.xyz --test test.xyz $K \
    --e0 lsq --m-per-species 0 --opt lbfgs --out fit
```

- `--order 3 --max-degree 10` builds the basis inside the fit, for the
  species found in the data. `--order` is the correlation order (how many
  neighbours a basis function couples; 3 gives four-body terms) and
  `--max-degree` bounds the polynomial degree, which sets the basis size.
- `--radial-mode onehot` uses the radial polynomials themselves as the
  radial basis. The default, `glorot_normal`, mixes them with seeded random
  weights: the usual start for [learning the radials](howto/learned-radials.md),
  but a noticeably worse frozen basis (see the [FAQ](faq.md#my-fit-is-much-worse-than-expected)).
- `--m-per-species 0` selects the linear model (no Gaussian-process arm).
- `--e0 lsq` fits the per-species reference energy by least squares on the
  training energies; a freshly built basis has E0 = 0.
- `--opt lbfgs` maximises the evidence with L-BFGS, much faster than the
  default Adam on small data.

The fit logs the basis it built and the hyperprior length scale it chose,
then the test metrics and the path of the fitted model:

```text
basis Si order 3 max-degree 10 -> 110 B functions (elements from the data)
r0 2.400 A (mean bond length of the basis; pass --r0 to override)
gamma missing from basis Si order 3 max-degree 10 -- rebuilt via basis.prior (algebraic smoothness prior)
L-BFGS: best of 1 start(s) = start 0, logpost -477.665
test map {'E': {'rmse': 24.2603, 'crps': 16.9333, 'coverage': 0.0769, 'rho': 0.4011, 'rms_z': 3.5221}, 'F': {'rmse': 0.1034, ...}, 'V': {'rmse': 0.2698, ...}}
fitted model: fit/model.npz
```

The `gamma missing` line is expected: the smoothness prior is rebuilt from the
basis. Energies are in meV/atom, forces in eV/Å and virials in eV; the last
digits may differ with the platform. The calibration columns (`coverage`,
`rms_z`) say the linear model's σ is too small here, which is typical of the
posterior σ of a small fit (see [Reading the metrics](concepts.md#reading-the-metrics)).

`fit/` holds:

| File | Contents |
|---|---|
| `model.npz` | the fitted model: the basis and its coefficients, one ordinary ACE model file |
| `fit.yaml` | the whole resolved run; `aj fit --config fit/fit.yaml` reproduces it ([Run files](howto/fit-yaml.md)) |
| `metrics.csv` | test RMSE, MAE and the calibration columns, per quantity |
| `theta_map.json` | the hyperparameters chosen by the evidence (noise and prior scales) |
| `config.json` | the configuration of the run, as the fitting pipeline saw it |

## 3. Evaluate

```bash
aj eval --model fit/model.npz --data test.xyz $K --forces --out pred.csv
```

```text
wrote 13 predictions to pred.csv
E RMSE 24.260 meV/atom  (13.0 configs)
F RMSE 0.1034 eV/A  (78.0 components)
```

`pred.csv` has one row per configuration (`energy`, `energy_per_atom`, and
`fmax`, the largest force component). Without `--out` the first rows are
printed instead. `aj eval` evaluates one configuration at a time without
compiling, which is simple but slow for large sets; the ASE calculator below
is the fast path.

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
aj basis --elements Si --order 3 --max-degree 10 --radial-mode onehot --out si.npz
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --e0 lsq --m-per-species 0 --opt lbfgs --r0 2.4 --out fit_from_file
```

With `--model`, `--r0` (the typical nearest-neighbour distance in Å, which
centres the hyperprior) is required. The two fits are identical.

## Next steps

- [Tutorial 1](tutorials/first-fit.md) does the same fit in a notebook, with
  a parity plot, an equation of state and a short MD run.
- [Concepts](concepts.md) explains what `--order`, `--max-degree`, E0 and the
  evidence fit mean.
- The [CLI reference](reference/cli.md) lists every flag.
