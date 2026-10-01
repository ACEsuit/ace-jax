# `aj fit --learn-radial`: learned radials in the fit pipeline

**Date:** 2026-10-01. **Status:** approved design. **Branch:** `feat/fit-learn-radial`.

## Goal

Learning the tensor radials currently takes two steps:

1. `bench/learn_radial/run.py` runs VarPro over a training split with a
   held-out gate and writes a learned `model.npz`.
2. `aj fit --model DIR/model.npz` then does the actual fit.

This change makes it **one `aj fit` run**, for the first PyPI release and its
tutorial. Learned radials work reliably (`docs/learn-radial-results.md`); the
CLI is what's missing.

**Success criteria:**

- One `aj fit --learn-radial …` run gives the same radials and readout as the
  driver followed by `aj fit --model learned.npz` on the same splits.
- The saved model is marked `radial_learned`, so `ACECalculator` and
  `export_lammps` spline it at deploy time.
- `out/fit.yaml` reproduces the run.

## Decisions (agreed)

- **Approach A:** a pipeline stage inside `fit()`, not CLI-only orchestration
  and not a separate subcommand. Python users get it through `FitConfig` too.
- **Gate split:** a seeded fraction (`radial_val_frac`, default 0.2) of the
  training configs, as `ard_val_frac` and `pops_val_frac` already do. `--test`
  stays an untouched final evaluation.
- **Minimal flag surface:** five options. The spectral and data-gap priors,
  `extra_train`, `init_radials`, `tol`, `reprofile_every` and
  `learn_sigma_e_mult` stay in the Python API (`fit_radial`).

## Interface

CLI (`aj fit`):

| flag | default | meaning |
|---|---|---|
| `--learn-radial` | off | learn the tensor radials before the fit |
| `--radial-n-q` | 12 | tensor-radial polynomial span after widening (`to_analytic`) |
| `--radial-steps` | 40 | L-BFGS steps per roughness weight |
| `--radial-lam-grid` | `0,1e-2` | relative roughness weights; the gate picks among them and the initial radials |
| `--radial-val-frac` | 0.2 | fraction of the training configs held out for the gate |

These are the driver's defaults. `FitConfig` gains the same fields:
`learn_radial: bool = False`, `radial_n_q: int = 12`, `radial_steps: int = 40`,
`radial_lam_grid: tuple = (0.0, 1e-2)` and `radial_val_frac: float = 0.2`.
`fit.yaml` gets the five keys automatically, because every `aj fit` dest is a
run-file key; `radial_lam_grid` is a comma list, like `rungs`.

The stage works with `--model` and with the inline basis
(`--order/--max-degree`). The **final** fit may use any arm or UQ: linear, POPS,
ARD or GP. Only the learning itself is linear (`require_linear`).

## Data flow

A new stage function, `learn_radials(cfg, data, log)` in
`fit/pipeline/radials.py`, runs first in `fit()`, before `build_problem`, when
`cfg.learn_radial` is set:

1. **Split.** A seeded permutation (`cfg.seed`) puts `radial_val_frac` of
   `data.train` into `val` and the rest into `fit`. Both sets are built with the
   pipeline's own batching and E0.
2. **Analytic radials.** `to_analytic(data.model, radial_n_q)` converts a spline
   radial or widens an analytic one. The conversion's maximum relative residual
   is logged and recorded.
3. **Learning problem.** A linear problem (M = 0), built as the driver builds
   it: `GPConfig` from the model meta, `select_inducing(..., 0, ...)`, the
   smoothness prior (`prior_diagonal`) and `default_prior(r0)`, where r0 is
   `cfg.r0`, falling back to `data.r0`.
4. **Gate.** `fit_radial(prob, ds_fit, ds_val, W0, lam_grid=radial_lam_grid,
   steps=radial_steps, rough_weights=1/(1 + rnl_degrees(meta))**2)` returns the
   selected radials and the readout fitted for them. The candidates are "init"
   plus one learned candidate per λ; ties go to init.
5. **Hand-off.** `patch_radial_npz` writes the selected radials and their
   readout into the source npz arrays **in memory** (BytesIO). The model is
   marked `radial_learned` unless the gate kept "init". `FitData` is then
   reloaded from those arrays (`model`, `meta`, `z`), the same in-memory
   hand-off #23 uses for a built basis, so the export path is unchanged.
6. **Final fit.** Everything downstream runs unchanged on the **full** training
   set, refitting the readout and any UQ.

`fit()` reports the stage as `on_stage("radial", RadialResult)`. `FitResult`
gains `radial: object = None`, appended last, so positional use is unaffected.

## Outputs

- **`out/model.npz` or `gp_model.npz`** carries the learned radials and
  `radial_learned`.
- **`out/radial_info.json`** holds `selected`, `scores` (per candidate),
  `n_q`, `lam_grid`, `val_frac`, `to_analytic_relres_max`, `n_fit`, `n_val`
  and `seconds`.
- **`out/fit.yaml`** includes the five radial keys.

## Errors

- **Options without the stage.** A `--radial-*` option given without
  `--learn-radial` is a CLI argument error (`_check_fit_args`). It names the
  option.
- **`radial_val_frac`** outside (0, 1), or a split that leaves `fit` or `val`
  empty, is a `ValueError` from `FitConfig.validate()` or the stage. It states
  the counts.
- **Unsupported radial types.** A model whose tensor radial `to_analytic`
  cannot handle, e.g. the factorised radial of `--basis-embedding` models,
  raises a clear `ValueError`; it is never a silent no-op. PACE `.yace` models
  are already refused for fitting.
- **Precision.** float64 is already enabled by the CLI. The Python API keeps
  `radial_learn`'s existing `require_x64` error.

## Testing

All tests run on the Si fixture (`fixtures/si_tiny_train.xyz`).

- **Parity.** `fit(FitConfig(learn_radial=True, ...))` gives the same radials,
  readout and test predictions as `fit_radial` called directly on the same
  splits, followed by a fit of the patched model.
- **Gate keeps init.** With `radial_steps=0` the gate keeps "init", and the
  saved model is not marked `radial_learned`.
- **Learned flag.** On a recoverable case the saved model is marked
  `radial_learned`, and `ACECalculator(lean=True)` splines it (`calc.splined`).
- **CLI and run file.** `aj fit --learn-radial ...` writes `radial_info.json`.
  `aj fit --config out/fit.yaml` reproduces the radials. A stray `--radial-*`
  option is a `SystemExit` that names the option.
- **Unsupported radial.** An embedding (factorised-radial) model is refused
  with the message.
- **Speed.** Fits use few configs, a low order and degree, and few steps, so
  the new tests stay within the suite's budget. A slow one gets
  `@pytest.mark.slow`.

## Docs

SKILL.md, README and the user docs gain the flags. The docs agent's
learned-radial tutorial switches from the Python API to the CLI route.

## Out of scope

- Exposing the spectral and data-gap priors, `extra_train` or `init_radials` on
  the CLI.
- Learning the pair radial.
- Learning radials inside the GP objective.
- Removing `bench/learn_radial/run.py`; it stays the research driver.
