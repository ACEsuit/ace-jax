# Run files: `fit.yaml`

Each `aj fit` writes a `fit.yaml` file in its `--out` directory. The file
contains:

- all options of the run, with their final values;
- a `provenance` block: the ace-jax version, the coupling library, the
  command line and a timestamp.

`aj fit --config` reads a run file.

## Reproduce a fit

```bash
aj fit --config fit/fit.yaml --out fit_again
```

This command gives the same model and metrics as the run that wrote
`fit/fit.yaml`.

## Vary a fit

Flags on the command line override the values in the file:

```bash
aj fit --config fit/fit.yaml --max-degree 12 --out fit_deg12     # a bigger basis
aj fit --config fit/fit.yaml --train more.xyz --out fit_more     # different data
aj fit --config fit/fit.yaml --model other.npz --r0 2.4 --out f3 # a saved basis instead of `basis:`
```

The fit logs each value that a flag overrides.

## Write one by hand

A run file needs only the keys that you want to set. All other keys take
the `aj fit` default.

- The keys are the flag names, with underscores in place of dashes.
- The basis settings go in a `basis:` block, with the `aj basis` flag names.

```yaml
# fit_si.yaml
train: train.xyz
test: test.xyz
e0: lsq
m_per_species: 0
opt: lbfgs
basis:
  order: 3
  max_degree: 10
```

If your labels are not stored as `energy`, `forces` and `virial`, add
`energy_key:`, `force_key:` and `virial_key:` lines.

```bash
aj fit --config fit_si.yaml --out fit
```

- Relative paths in the file (`train`, `test`, `ood`, `data`, `model`, ...)
  are relative to the directory of the file. Thus a run file works from any
  directory. `out` is relative to the current directory.
- `model: basis.npz` (a saved basis) and a `basis:` block are alternatives.
  Give only one.
- `elements` is optional. Without it, the fit builds the basis for the
  species in the data.
- If a key is spelled incorrectly or a value is not valid, the error gives
  the key. If the key is nearly correct, the error also gives the correct
  key.

## From Python

The Python pipeline takes the same settings as a `FitConfig`; see the
[Python API](../reference/api.md#fitting). `FitConfig(model=...)` accepts
the path of a basis file, a `Basis` or a `BasisSpec`.
