# Accuracy-matched speed benchmark (Si): spec

Status: agreed 2026-10-04. The quick figure (Phase A) is ready to plan; Phase B
hardens it into a reproducible benchmark.

## Goal

Answer reviewer feedback on [benchmarks.md](benchmarks.md):

> I'd be interested in seeing a speed comparison between the smallest possible
> model of each type fit to identical datasets that achieves a target RMSE.
>
> Right now having a MACE medium model on there isn't super useful and the PACE
> model being the same size as the linear model maybe isn't realistic?

The existing page times random-weight ACE/PACE models on a fixed size ladder
next to foundation MACE models. That isolates the code path but says nothing
about cost at equal accuracy. This benchmark fits every model family to the
same Si training set, scores them all on the same held-out test set, and plots
test force RMSE against throughput. The "smallest model that meets the target"
is read off each family's curve.

The output is a Pareto plot plus a table, not a single threshold: one target
is fragile, while the curve shows the whole trade-off.

## Phases

- **Phase A (quick figure):** everything below, at a single 5.0 Å cutoff, timed on
  lestrade.
- **Phase B (reproducible):** a cutoff sweep (4.5 / 5.0 / 5.5 Å, every family),
  Modal A100 rows, silicon-testing-framework property checks (elastic
  constants, vacancy and surface energies), and the CI tests below hardened.
- **Out of scope:** the W+He dataset (Zenodo 10.5281/zenodo.18838361). It
  needs a separate spec once the Si pipeline exists.

## Families

| family | code | size knob |
|---|---|---|
| linear ACE | ace-jax `aj fit --m-per-species 0` | `--order 3–4`, `--max-degree` swept for about 50 → 1,500 basis functions (~6 points), plus a top point o4d20 (5,456 functions) |
| linear ACE, learned radials | ace-jax `aj fit --learn-radial` | the same ladder as linear ACE; timed as deployed (lean, splined at `spline_tol="auto"`) |
| ACE+GP | ace-jax `aj fit` | 2–3 linear ACE sizes × `--m-per-species` small / large (~5 points) |
| PACE | pacemaker (ladder fit) | 50 / 100 / 200 / 400 / 800 functions per element, taken from ladder checkpoints; default FS embedding |
| MACE | `mace_run_train`, from scratch | channels 8 / 16 / 32 / 64 × L = 0 / 1; max_ell 3, correlation 3, 2 layers (~6 points) |
| GAP | `gap_fit` (quippy-ase wheel) | GAP-18 published hyperparameters, SOAP n_sparse 1k / 3k / 9k |

The ladder top o4d20 (order 4, total degree 20) matches the basis size of ACEpotentials' published Si fit (Silicon_2023), but with the same GAP-18-derived weights as every other point, not that fit's hand-tuned ones.

PACE is evaluated by both ML-PACE and ace-jax `PACEModel` on the same `.yace`
files. The published GAP-18 model is not refitted. It appears only as a
reference line, because it was trained on the test configurations as well.

## Fairness rules

These are stated in the write-up exactly as here.

- **The same data.** Every family uses one train file and one test file.
- **The same cutoff.** 5.0 Å in Phase A (GAP-18's SOAP cutoff). MACE's 2-layer
  receptive field (10 Å) is inherent to the architecture and is stated, not
  corrected for.
- **The same weights.** GAP-18's per-`config_type` sigmas (`default_sigma`
  scheme), converted once by `weights.py` (weight = 1/σ) into each code's format.
- **One noise scale for ace-jax.** The `ace` and `ace_lr` fits use `--noise shared`, because per-quantity noise cancels the weights at a converged MAP (issue #64).
- **The same E0.** The `isolated_atom` energy, passed explicitly to every
  fitter.
- **The same tuning budget.** Each code's documented defaults, with only the
  size knob varied. Nothing is tuned per family. The fitting knobs used are
  recorded per run.
- **The same scorer.** One `evaluate.py` scores every model through ASE
  calculators. Metrics a fitter reports itself are ignored.
- **Fit time is not speed.** Fit wall times are tabulated separately and never
  enter the throughput axis.

## Data

- **Source:** the GAP-18 training set, CASTEP labels:
  `libAtoms/silicon-testing-framework` `models/GAP/gp_iter6_sparse9k.xml.xyz`,
  pinned to a commit and a sha256 in `bench/accuracy_cost/data.md`. That file
  also records the published GAP-18 errors used as reference lines, with their
  citation. The local copy under `Si_DATASET_RERUN_FHIaims/` has the same
  sha256 as the pinned GitHub file and holds the original CASTEP labels, so
  it is the same data, not a rerun. Virials (1,843 frames: 1,783 `dft_virial`
  plus the 60 upper-case `sh` frames) are fitted by every code that supports
  them, with GAP-18's virial sigmas where the code takes per-type weights.
  GAP-18's fit uses a glue core potential (`core_param_file=glue.xml`); the
  GAP refit keeps it.
- **Split** (`split.py`): 90/10, stratified by `config_type`, fixed seed.
  `config_type`s with fewer than 10 configurations go entirely to train
  (`isolated_atom`, `surface_111_3x3_das` and `crack_110_1-10`), because one or
  two test configurations would make their per-type RMSE pure noise.
  `crack_111_1-10` (10 frames) and `111adatom` (11) sit at the threshold and
  are stratified, with one test frame each (read their per-type RMSE as noise).
- **Normalised files:** `split.py` renames the label keys to plain `energy` /
  `forces` / `virial` (the source uses its own key prefixes) and drops the
  stored GAP predictions. It writes `train.xyz` and `test.xyz` to
  `/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost/`, never
  committed, along with a split hash that every result row records.
- **Reading:** ace-jax-side code uses `fit/xyz.py` (libAtoms extxyz), never
  `ase.io.read`.

## Fitting

All files are under `bench/accuracy_cost/`. `sweep.yaml` lists family × size
entries. Each `fit_<family>.py` takes one entry and writes
`runs/<family>/<size>/`: the model, the exact command or config, the log, fit
wall time, peak memory and the host. Run directories live on /storage. A run
whose model file already exists is skipped (resumable). A crash or divergence
writes `status: failed` with the log tail, and nothing is retried.

| family | environment | host |
|---|---|---|
| linear ACE, ACE+GP | project venv | moriarty GPU |
| PACE | `envs/pacemaker` uv venv (TensorFlow), on /storage | moriarty GPU |
| MACE | `envs/mace` uv venv, on /storage | moriarty GPU |
| GAP | `envs/quip` uv venv with `quippy-ase` | mnf148, `setsid nohup`, `free -g` checked first |

- One GPU fit runs at a time on moriarty: agents there share a 16 GB swap-free
  cap.
- `gap_fit` at 9k sparse points needs tens of GB, which is why it runs on
  mnf148.

## Scoring

`evaluate.py` loads any fitted model as an ASE calculator:
- `ACECalculator` or `GPCalculator` for ace-jax;
- pyace for PACE, cross-checked against ace-jax `PACEModel`;
- `MACECalculator` for MACE;
- quippy `Potential` for GAP.

It writes one row per model to `results/metrics.jsonl`:
- test force RMSE (the ranking metric) and energy RMSE per atom;
- both, broken down by `config_type`;
- the number of parameters;
- the split hash.

## Timing

All timing runs on **lestrade** (32 cores, RTX 4000 Ada). The absolute numbers
are not comparable with the moriarty and A100 rows in `benchmarks.md`.

- **Structure:** diamond Si with 0.05 Å random displacements (seeded). Each
  device uses a main size N and a check size N/8:
  - GPU: N = 32,768 (16³ conventional cells), with 4,096 as the check;
  - CPU: N = 4,096, with 512 as the check.

  A row is flagged `not_plateau` when per-atom throughput differs by more than
  20% between N and N/8.
- **Mode:** each code uses the best mode available on each device:

  | | CPU (32 cores) | GPU (RTX 4000 Ada) |
  |---|---|---|
  | linear ACE, PACE via ace-jax | standalone | LAMMPS `jax/kk` |
  | ACE+GP | standalone | standalone (GP has no LAMMPS export) |
  | ML-PACE | LAMMPS `pace`, 32 MPI ranks | LAMMPS `pace/kk` |
  | MACE | LAMMPS Symmetrix | LAMMPS Symmetrix |
  | GAP | LAMMPS `pair quip`, 32 MPI ranks | — |

- **Precision:** float64 for ace-jax, ML-PACE and GAP. MACE is timed in float32
  and float64; Symmetrix's LAMMPS rows are float64 only, as in `benchmarks.md`.
- **Harness:** `timing.py` reuses `run_case` from `bench/scaling/run_lammps.py` and
  `run_standalone.py`, which provide warm-up, step-count sizing, ~60 s timed
  segments and OOM markers. The only change in `bench/scaling` is a hook that
  accepts a fitted model path in place of a manifest row. Each case runs 3
  times, and the plot shows the median.
- **Parity gate:** before any timing, each model's LAMMPS energy and forces must
  match its own standalone or ASE evaluation on 3 test configurations. This uses
  the existing `parity.py` gates plus a new `gap` gate (quippy against
  `pair quip`). A model that fails is recorded as `status: parity_failed` and
  not timed.
- **Idle check:** `timing.py` refuses to start when lestrade's 1-minute load is
  above 2 or another user has a GPU process. Every row records the load and the
  GPU occupancy.
- **Lestrade build** (`envs/lestrade.sh`): a parametrised copy of
  `bench/scaling/envs/moriarty.sh` with `Kokkos_ARCH_ADA89` in place of
  `AMPERE86`. It builds the same two LAMMPS trees, adds `PKG_ML-QUIP` to the
  Symmetrix/ML-PACE tree, and installs the lammps-jax plugin. Build trees go to
  /storage.

## Outputs

- `results/lestrade-{cpu,gpu}.jsonl` uses the `bench/scaling` row schema plus
  `family`, `size`, `n_params`, `test_rmse_F`, `test_rmse_E`, `not_plateau`
  and `split_hash`.
- `plot.py` writes three figures to `figs/`:
  1. **The Pareto plot:** test force RMSE (log x) against throughput in
     atom-steps/s (log y), one line per family, with CPU and GPU panels.
     Horizontal bands mark the two targets. Vertical reference lines mark the
     refit GAP (9k) and the published GAP-18 errors.
  2. **Smallest model at target:** per family and target, the smallest model
     whose test force RMSE is at or below the target, with its size, CPU and
     GPU throughput, and its ratio to linear ACE. A family that never reaches
     the target is shown as "not reached", never extrapolated.
  3. **Per-`config_type` RMSE heatmap** for the models in that table.
- **Targets:** two force-RMSE values that bracket the refit GAP's test error,
  fixed after the curves are seen and recorded in the write-up. The Pareto plot
  shows the full curves regardless of the targets.
- **Write-up:** `docs/dev/accuracy-cost.md` has the figures, the fairness rules,
  the fit-time table, the caveat that published GAP-18 was trained on the test
  set, and the hardware note.
- **`docs/dev/benchmarks.md`:** gains a pointer to the new page, and a caveat on
  its MACE-medium and same-size PACE rows linking to it. Those rows are kept.

## Testing

`tests/test_bench_accuracy_cost.py` is fast and CPU-only, and runs on a
20-configuration slice of the data:
- the split is deterministic and stratified, and rare types go entirely to
  train;
- `weights.py` emits mutually consistent tables in all four code formats;
- `evaluate.py` reproduces a known model's RMSE on a tiny set;
- the existing `tests/test_bench_scaling.py` still passes with the model-path
  hook.

Fitting and timing need GPUs, pacemaker, MACE and QUIP, so they are not run in
CI. They are validated by:
- the parity gate;
- a sanity check against published errors on this dataset: refit GAP (9k) is
  close to published GAP-18, and ACE and MACE are in the range of published Si
  results on the same data.

## Layout

```
bench/accuracy_cost/
  README.md  data.md  sweep.yaml
  split.py  weights.py  evaluate.py
  fit_ace.py  fit_pace.py  fit_mace.py  fit_gap.py
  timing.py  plot.py
  envs/lestrade.sh  envs/{pacemaker,mace,quip}/pyproject.toml
  results/  figs/
```
