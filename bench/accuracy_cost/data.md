# Accuracy-cost data: GAP-18 silicon training set

## Source

- URL: `https://raw.githubusercontent.com/libAtoms/silicon-testing-framework/ec91b17fbbcd6c4a06a257acbfcba77734c0080c/models/GAP/gp_iter6_sparse9k.xml.xyz`
- Commit: `ec91b17fbbcd6c4a06a257acbfcba77734c0080c` (libAtoms/silicon-testing-framework)
- sha256: `74ac5f387aa0149ec9ea8f3e275d1087d7287db68c15ba8bc4c28a450de67e2a` (checked by `split.fetch()`)
- 2,475 frames; 2,227 train and 248 test after the split (`split.py`, seed 0,
  hash recorded in `DATA/split.json`; current hash `c433c98d49bb`).
- The `.xml` model and `glue.xml` are fetched alongside, for reference.

## Findings

- **Local copy identity.** `/storage/eng/essswb/ownCloud/Si_DATASET_RERUN_FHIaims/gp_iter6_sparse9k.xml.xyz`
  has the same sha256 as the pinned GitHub file. Despite its directory name it
  holds the original CASTEP labels, so it is not an FHI-aims dataset.
- **Glue core.** GAP-18's fit uses a glue core potential
  (`core_param_file=glue.xml`). The GAP refit keeps it.
- **Upper-case keys.** The 60 `sh` frames use upper-case label keys
  (`DFT_energy`, `DFT_force`, `DFT_virial`); `split.normalise` matches
  label keys case-insensitively.
- **Virials.** 1,843 frames carry a DFT virial: 1,783 with `dft_virial` plus
  the 60 upper-case `sh` frames (the brief's 1,783 counts the lower-case key
  only). They are kept as `info["virial"]` (3x3, eV) and fitted where the code
  supports it. The stored GAP predictions (`gap_*`) and a stray `virial` key
  on 78 frames are dropped.
- **Virial file convention.** The normalised files write the 9-vector
  column-major, because extxyz reads a 9-vector as a Fortran-order 3x3.
- **Rare types.** Types with fewer than 10 frames go whole to train:
  `isolated_atom` (1), `surface_111_3x3_das` (1), `crack_110_1-10` (7).
  `crack_111_1-10` (10 frames) and `111adatom` (11) are stratified, one test
  frame each.

## Published GAP-18 errors

Bartok, Kermode, Bernstein and Csanyi, *Phys. Rev. X* **8**, 041048 (2018).
GAP-18's stored gap_* predictions cover 240/2475 frames and are stale; they are
not used. Published GAP-18 errors are computed by evaluating the published XML
with quippy (bench Task 8).

## PACE (pacemaker) environment and fit

- Venv `/storage/eng/essswb/venvs/ac-pacemaker`, built from `envs/pacemaker/pyproject.toml`
  (`uv sync -p 3.9`): Python 3.9.25 (python-ace 0.2.8 ships cp39 wheels only; `extxyz`
  has none, so `split.read_atoms` falls back to `_read_plain` there), pacemaker/pyace 0.2.8,
  ace_evaluator 2021.10.25, tensorflow 2.15.1, numpy 1.26.4, pandas 2.3.3, setuptools<81
  (`pkg_resources`), ruamel.yaml 0.17.40 (<0.18: pacemaker calls `safe_load`).
  `tensorpotential` (the tensorpot backend, not on the python-ace dependency list) is installed
  by hand: `uv pip install --no-deps .` from a clone of ICAMS/TensorPotential at
  `1e44b2558356800ae070658c0bb856ff9bf74538`.  ace-jax is not installed there (needs Python 3.11);
  run with `cd bench` so `accuracy_cost` imports.
- **Stress: PACE fits E+F only.** The installed tensorpot loss (`tensorpotential/tensorpot.py`
  `compute_loss`) has energy and force terms; stress is only predicted. PACE therefore sees no virials.
- **Energy loss is per atom** (`tensorpot.py:283-288`, `LOSS_TYPE: 'per-atom'` set at
  `pyace/fitadapter.py:190`): `eweights * ((E_pred - E_ref)/N)^2`, so `w_energy = 1/sigma_E^2` with GAP-18's
  per-atom sigma needs no 1/N factor.  `normalize_energy_forces_weights` rescales the weights to sum 1.
- **E0:** pyace 0.2.8 has no `data.reference_energy`; the dataframe carries `energy_corrected = E - N*E0`
  and `with_e0` writes E0 into the final `.yace`.  The isolated-atom frame is dropped (no neighbours).
- **Checkpoints** are `interim_potential_ladder_step_<k>.yaml`, k counting ladder steps; `checkpoint_for`
  opens each and takes the first with >= n functions; `n_params` is the actual count.

## MACE (from scratch) environment and fit

- Venv `/storage/eng/essswb/venvs/ac-mace`, built from `envs/mace/pyproject.toml` (`UV_PROJECT_ENVIRONMENT=... uv sync`):
  Python 3.12.8, **mace-torch 0.3.16**, torch 2.14.1+cu130; ace-jax installed with `-e --no-deps` (jax is absent, so
  `split.read_atoms` uses `_read_plain` there); `symmetrix` 0.0.1 built CPU-only from `~/bench-scaling/symmetrix/symmetrix`
  with the `symmetrix_py` step's `SKBUILD_CMAKE_ARGS` (`symmetrix_extract_mace` is in the venv).
- Every flag in the plan exists in `mace_run_train --help` under its plan name; `--loss` choices include `virials`
  (`WeightedEnergyForcesVirialsLoss`, virials_weight 1, stage two 10).  Frames without a `virial` are tolerated: a 92-frame
  smoke train set with 42 virials logged `virials: 42` against `energy: 92, forces: 92` and trained without error.
- `--swa` (stage two) makes `mace_run_train` write the stage-one model as `model.model` and the stage-two final as
  `model_stagetwo.model`; `fit_mace` keeps stage one as `model_stage1.model` and deploys stage two as `model.model`.
- `fit_mace` puts the venv's `bin` on PATH; `ACEJAX_AC_DEVICE` overrides `--device` (default `cuda`).

## GAP (gap_fit refit) environment and fit

- Venv `/storage/eng/essswb/venvs/ac-quip`, built from `envs/quip/pyproject.toml` on lestrade
  (`UV_PROJECT_ENVIRONMENT=... uv sync`) and used unchanged on mnf148: Python 3.12.8, **quippy-ase 0.10.3**
  (QUIP v0.10.3, `gap_fit` in the venv's `bin`), ase 3.29.0, numpy 2.5.3; ace-jax `-e --no-deps`
  (`split.read_atoms` uses the libAtoms reader there).
- `fit_gap.gap_fit_exe()` takes `gap_fit` from `dirname(sys.executable)`: running `venv/bin/python` does not put
  `venv/bin` on PATH.  `args.split()` is harmless: `cmd.txt` is the GAP-18 line with only data, `n_sparse`, `gp_file`
  and the three label names changed, and gap_fit accepted it.
- Smoke (lestrade, 50 train frames strided over the 2,227, `n_sparse=50`, scratch `ACEJAX_AC_ROOT`): 26 s, 3.4 GB;
  gap_fit read 50 energies, 10,014 force and 210 virial components and 3,338 descriptors, loaded `glue.xml`;
  `xml_label` read `GAP_2026_10_4_60_22_20_46_551`; `load_calc("gap", ...)` scored it on test.
- Virials are fitted (`virial_parameter_name=virial`) with GAP-18's own sigmas.
- Memory: `gap_fit` on the full train set at `n_sparse=1000` holds about 155 GB RSS (derivative storage scales with the
  data, not with `n_sparse`); run on mnf148 (251 GB) only, one fit at a time.

## GAP-18 published (saw every frame)

The published `gp_iter6_sparse9k.xml` (plus its `.sparseX.GAP_2017_6_17_60_4_3_56_1651`, fetched from the pinned commit
and decompressed; sha256 `206e72b6...1051`) scored with quippy through `evaluate.summarise(errors(...))`.  It was fitted
on every frame, test included, so it is a reference line, not a held-out score.  Row `gap18_published` in
`results/metrics.jsonl` (test metrics; train under `train`); full scores in `DATA/gap18_published_scores.json`.

| set | rmse_E (meV/atom) | rmse_F (eV/A) | rmse_V (eV/atom) |
|---|---|---|---|
| test (248) | 1.281 | 0.0825 | 0.0201 |
| train (2,227) | 1.373 | 0.0825 | 0.0223 |

### GAP E0 handling

- GAP-18's line has no `e0_method`, so gap_fit's default `isolated` applies: E0 is read from the `isolated_atom`
  frame (our `-158.54496821`). The smoke log shows `e0_method = isolated`, `e0_offset = 2.0`, and the written XML
  holds `<e0 Z="14" value="-156.54496821"/>`: **`e0_offset` is added** (E0 = E_isolated + 2.0), as in GAP-18.
- The glue core does not enter: evaluated alone (`IP Glue`, `glue.xml`) on an isolated Si atom it returns exactly 0.0
  eV (no neighbours), so it cannot account for any E0 difference.
- The published model's `<e0 Z="14">` is `-160.73263835`. By the same rule that implies an isolated-atom energy of
  `-162.73263835` in GAP-18's own database, i.e. 4.1877 eV below ours (`-158.54496821`). The suggested netting
  (`-158.54496821 - 2.0 - E_glue = -160.54496821`) misses the published value by 0.1877 eV, and the sign of the
  offset is the other way, so the published and refit `<e0>` values are not expected to match: they carry different
  isolated-atom reference energies. This does not affect scoring: the published model reproduces our energies to
  1.28 meV/atom on test, and the refit takes its E0 from our own data. Recorded as an unexplained difference in the
  provenance of GAP-18's isolated-atom energy, not a defect of the refit.

## PACE and MACE fits on Modal (`modal_fits.py`)

- Images: the `envs/pacemaker` and `envs/mace` uv.locks synced into `/opt/<env>` (Python 3.9.23 vs 3.9.25 locally),
  tensorpotential at `1e44b25` for pacemaker; TF 2.15.1 brings its own pip CUDA 12.2 / cuDNN 8.9 and sees the GPU
  (checked at the start of every `fit_pace` call). `envs/pacemaker/uv.lock` was stale against its pyproject
  (setuptools 82, ruamel.yaml 0.19): relocked to setuptools 80.10.2 / ruamel.yaml 0.17.40, which the local venv has.
- GPU: A100-40GB (A100-80GB fallback). Both fits run in float64, where an L40S is ~1/64 rate.
- Volume `ace-jax-accuracy-cost`: `data/si_accuracy_cost/{train,test}.xyz, split.json, weights_mace.json,
  gp_iter6_sparse9k.xml` and `runs/<root>/{pace,mace}`. `status.json` has `host: "modal"` and `gpu`;
  `evaluate` copies the latter to `fit_gpu`.
- Symmetrix is not built in the image: `--mode collect` exports `model.json` with the local `ac-mace` venv (CPU).
- PACE probe (`--mode probe`: full ladder, `maxiter` 3): 13 us/atom/eval at 50 functions to 28 at 761, i.e. 2-4.4 s
  per eval over the 154k training atoms; at 1500 BFGS iterations per rung the ladder is ~22-26 h, past Modal's 24 h,
  so `fit_pace` resumes an interrupted ladder from its last checkpoint (`ladder/resume<k>/`, `initial_potential`);
  `fit_wall_s` sums the segments (cmd.txt to fit.log's last write).

## GRACE (gracemaker, from scratch) environment and fit

- Venv `/storage/eng/essswb/venvs/ac-grace` (Sulis: `$ROOT/venv-grace`), built from `envs/grace/pyproject.toml`
  (`UV_PROJECT_ENVIRONMENT=... uv sync -p 3.12`): Python 3.12, **tensorpotential (gracemaker) 0.6.2.1**, tensorflow
  2.20.0 with pip CUDA 12.5 / cuDNN 9 (`tensorflow[and-cuda]`), keras 3.15.1 (tf-keras 2.20.1), ase 3.29.0, numpy 2.5.4,
  phonopy 4.8.3 (the QoI tests run in this venv: `SI_PY=/storage/eng/essswb/venvs/ac-grace/bin/python`). No ace-jax
  there: `split.read_atoms` uses `_read_plain`.
- `fit_grace.py` writes gracemaker's input as the `gracemaker -t` from-scratch answers (`DEFAULT_FIT`), plus cutoff 5,
  `reference_energy: {Si: E0_SI}`, `param_dtype: float64` (float32 is gracemaker's default; float64 matches MACE's
  `--default_dtype=float64` and the float64 evaluation and timing), and 5% of train as gracemaker's "test" set, which
  picks the best checkpoint (the benchmark's test split never reaches the fit).
- **Weights.** GAP-18 sigmas as type ratios to the default sigma, squared (gracemaker weights squared or Huber
  residuals), per structure: energy, force (times gracemaker's default 1/(3N) per atom) and virial (0 without a DFT
  virial). Stress is fitted (`stress` = -virial/volume, Huber, weight 128: the template's stress weight).
- **NaN with stress.** gracemaker divides the stress term by the batch's summed virial weights: a batch with no
  virial frames (its buckets group surfaces, `amorph`, `sp`...) gives 0/0 and every loss is NaN from epoch 1.
  `fit_grace` sets the stress loss's `normalize_by_samples: False` and divides each virial weight by 6 x the mean
  batch sum instead. Also: a frame whose `stress` is None crashes the data builder (it multiplies `np.array(None)`),
  so frames without a virial carry zeros at weight 0; the isolated atom (no neighbours) also gives NaN and is dropped,
  as for PACE. This departure from gracemaker's defaults was reviewed and accepted (2026-10-21): every family that
  can fit virials does (pacemaker's tensorpot loss cannot), and dropping the stress term was the alternative.
- **E0.** The model predicts E - N E0; `fit_grace.calculator` (and the QoI `model.py`s) add N E0 back. LAMMPS runs
  the model as trained (parity compares unshifted energies).
- **LAMMPS.** yury-lysogorskiy/lammps `grace` branch b1f4a6c + lammps-user-pace v.2025.12.4.p1, no TensorFlow
  (`-D NO_GRACE_TF=ON`; `scaling/envs/sulis-grace.sh`). GRACE-1L: `grace_utils export_kokkos` .npz, `pair_style
  grace/1l/kk` (GPU) / `grace/1l/cpu/kk` (CPU), full float64 by default. GRACE-FS: `grace_utils export -sf` YAML,
  `grace/fs/kk` / `grace/fs`. The Kokkos 1L kernel caps n_rad_max at 32 and MLP widths at 64, and reads an order-4 (AA2)
  block unconditionally: an order-3 model exports but dies of SIGFPE in `copy_weights_to_device`, so the ladder keeps
  max_order 4 and the o3 ablation is accuracy only. Linear readout (`LinearOut2Target`) is outside the 1L exporter.
- A serial Kokkos-OpenMP build for parity checks on the desktops: `/storage/eng/essswb/projects/ace-jax/grace-lammps/build-cpu-local/lmp`
  (`build-cpu-local.sh`, mnf148). `python -m accuracy_cost.fit_grace FAMILY SIZE --seed K --parity LMP` runs the
  timing gate's check (three rattled 64-atom cells) and writes `parity-cpu/parity.json` in the run directory.
- **Pilot** (c16l3 seed 0, moriarty RTX A4500, 2,490 s): test rmse_E 5.91 meV/atom, rmse_F 0.0878 eV/A, rmse_V
  0.0291 eV/atom (liq 0.154 eV/A); QoI error 0.74 (21/21; point defects 0.6-1.0 eV low). Parity (64 atoms, three
  cells) 1.4e-11 eV/atom, 2.7e-10 eV/A with grace/1l/cpu/kk (mnf148, Sulis CPU) and grace/1l/kk (Sulis A100).
- **A100 probe** (`envs/sulis-grace-probe.sbatch`, job 2257046, A100-PCIE-40GB, 10 epochs each), s/epoch and the
  projected 376-epoch fit: c8l3 1.62 (0.19 h), c16l3 2.12 (0.24 h), c16l4 3.0 (0.33 h), c32l4 5.0 (0.54 h), c64l4 8.25
  (0.88 h), fs_small / fs_medium 1.38 (0.18 h), c16l4 linro / frozenR 2.88 (0.32 h), c16l4_o3 2.38 (0.27 h).
