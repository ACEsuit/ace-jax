# Command-line reference

`ace-jax` and its short alias `aj` are the same program. This page is
generated from the program's own `--help` when the documentation is built, so
it matches the installed version.

The commands, in the order of a typical workflow:

- [`aj fit`](#aj-fit) fits a model to labelled extxyz data. It builds the
  basis from `--order`/`--max-degree` or takes a saved one with `--model`, and
  writes the fitted model, metrics and a `fit.yaml` run file to `--out`.
- [`aj eval`](#aj-eval) evaluates a fitted model (ACE `.npz`, `gp_model.npz`
  or PACE `.yace`) on an extxyz file.
- [`aj calibrate`](#aj-calibrate) recalibrates the per-atom force
  uncertainty of a `--uq ard` fit on labelled target-regime data.
- [`aj basis`](#aj-basis) builds a basis and saves it on its own.

Every command exits with status 0 on success. Label keys default to
`energy`, `forces` and `virial`; pass `--energy-key`, `--force-key` and
`--virial-key` for other names.

<!-- aj-help -->

## aj fit

The options fall into groups:

- **data**: `--train`/`--test` files, or one `--data` file split by a seeded
  permutation (`--ntrain`, `--ntest`, `--test-start`); `--ood` adds an extra
  test set; `--weights` sets per-config-type weights;
- **basis**: the options under "basis" below build it inside the fit;
  `--model` takes a saved basis instead (then `--r0` is required);
- **model**: `--m-per-species 0` is linear ACE; a positive value adds a
  Gaussian-process arm on that many inducing sites per species (the default,
  500, is the GP arm);
- **hyperparameters**: `--opt`, `--map-steps`, `--map-restarts`, `--map-polish`,
  `--strict`, `--init`;
  `--rungs` beyond `map` adds hyperparameter-posterior approximations, at a
  much higher cost;
- **uncertainty**: `--uq blr` (default), `pops` or `ard` for the linear model;
  the `--ard-*` and `--force-shape` options configure `ard`
  ([Per-atom force uncertainty](../howto/force-uncertainty.md));
- **run file**: `--config fit.yaml`, with command-line flags taking
  precedence ([Run files](../howto/fit-yaml.md)).

<!-- aj-help fit -->

## aj eval

With `--posterior` (a `posterior.npz` from `aj fit --uq ard`), `--per-atom`
writes the per-atom force uncertainty and `--support` adds the support flag
([Per-atom force uncertainty](../howto/force-uncertainty.md)).

<!-- aj-help eval -->

## aj calibrate

Recomputes the per-group scales of an ARD posterior on labelled
configurations that were not in training, and writes a new posterior; the
model is unchanged. See
[Recalibrate on target data](../howto/force-uncertainty.md#recalibrate-on-target-data-aj-calibrate).

<!-- aj-help calibrate -->

## aj basis

<!-- aj-help basis -->
