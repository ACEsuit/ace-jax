# Command-line reference

`ace-jax` and its short alias `aj` are the same program. The documentation
build makes this page from the `--help` output of the program. Thus the page
agrees with the installed version.

The commands, in the order of a typical workflow:

- [`aj fit`](#aj-fit) fits a model to labelled extxyz data. It builds the
  basis from `--order`/`--max-degree`, or uses a saved basis from `--model`.
  It writes the fitted model, the metrics and a `fit.yaml` run file to
  `--out`.
- [`aj eval`](#aj-eval) evaluates a fitted model (ACE `.npz`, `gp_model.npz`
  or PACE `.yace`) on an extxyz file.
- [`aj calibrate`](#aj-calibrate) recalibrates the per-atom force
  uncertainty of a `--uq ard` fit on labelled target-regime data.
- [`aj basis`](#aj-basis) builds a basis and saves it on its own.

Each command exits with status 0 if it is successful. The default label
keys are `energy`, `forces` and `virial`. For other names, use
`--energy-key`, `--force-key` and `--virial-key`.

<!-- aj-help -->

## aj fit

The options fall into groups:

- **data**: `--train`/`--test` files, or one `--data` file divided by a
  seeded permutation (`--ntrain`, `--ntest`, `--test-start`). `--ood` adds a
  second test set. `--weights` sets weights for each configuration type.
- **basis**: the options under "basis" below build the basis in the fit.
  Alternatively, `--model` uses a saved basis; then `--r0` is necessary.
- **model**: `--m-per-species 0` is linear ACE. A positive value adds a
  Gaussian-process arm, with that number of inducing sites for each
  species. The default (500) is the GP arm.
- **hyperparameters**: `--opt`, `--map-steps`, `--map-restarts`, `--map-polish`,
  `--strict`, `--init`, `--noise`. `--noise shared` learns one noise scale
  for all weighted rows, so `--weights` sets the balance of E, F and V (as
  in ACEpotentials). `--rungs` other than `map` adds approximations of
  the hyperparameter posterior, at a much higher cost.
- **uncertainty**: `--uq blr` (default), `pops` or `ard` for the linear
  model. The `--ard-*` and `--force-shape` options configure `ard`
  ([Per-atom force uncertainty](../howto/force-uncertainty.md)).
- **run file**: `--config fit.yaml`. Command-line flags override the file
  ([Run files](../howto/fit-yaml.md)).

<!-- aj-help fit -->

## aj eval

With `--posterior` (a `posterior.npz` from `aj fit --uq ard`), `--per-atom`
writes the per-atom force uncertainty, and `--support` adds the support flag
([Per-atom force uncertainty](../howto/force-uncertainty.md)).

<!-- aj-help eval -->

## aj calibrate

Calculates the group scales of an ARD posterior again, on labelled
configurations that were not in the training set. It writes a new
posterior. The model does not change. See
[Recalibrate on target data](../howto/force-uncertainty.md#recalibrate-on-target-data-aj-calibrate).

<!-- aj-help calibrate -->

## aj basis

<!-- aj-help basis -->
