# Run files: `fit.yaml`

Every `aj fit` writes `fit.yaml` into its `--out` directory: the whole run,
every option resolved, plus a `provenance` block (ace-jax version, coupling
library, the command line and a timestamp). `aj fit --config` reads a run
file back.

## Reproduce a fit

```bash
aj fit --config fit/fit.yaml --out fit_again
```

gives the same model and metrics as the run that wrote `fit/fit.yaml`.

## Vary a fit

Flags given on the command line override the file:

```bash
aj fit --config fit/fit.yaml --max-degree 12 --out fit_deg12     # a bigger basis
aj fit --config fit/fit.yaml --train more.xyz --out fit_more     # different data
aj fit --config fit/fit.yaml --model other.npz --r0 2.4 --out f3 # a saved basis instead of `basis:`
```

Each override of a value from the file is logged.

## Write one by hand

A run file needs only the keys you want to set; everything else takes the
`aj fit` default. Keys are the flag names with underscores instead of dashes,
and the basis settings go in a `basis:` block (the `aj basis` flag names):

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
  are relative to the file's directory, so a run file keeps working when it is
  run from somewhere else. `out` stays relative to the current directory.
- `model: basis.npz` (a saved basis) and a `basis:` block are alternatives;
  give one.
- `elements` is optional: without it the basis is built for the species in
  the data.
- A misspelt key or a bad value is an error that names the key and, for a
  near-miss, suggests the right one.

## From Python

The Python pipeline takes the same settings as a `FitConfig`; see the
[Python API](../reference/api.md#fitting). `FitConfig(model=...)` accepts a
basis file path, a `Basis` or a `BasisSpec`.
