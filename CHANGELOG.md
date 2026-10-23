# Changelog

## 0.4.1 (2026-10-22)

**Highlights.** Linear ACE runs in LAMMPS on CPUs with a C++ kernel, 10 to 23
times faster than the compiled JAX program: `aj export --platform cpu --ffi`.
The Performance page has an interactive accuracy-cost figure, and the
benchmark adds GRACE. Long MAP fits and large `--learn-radial` fits on a GPU no
longer stall or run out of memory.

**Upgrading from 0.4.0.** Saved models and bundles are unchanged.
- The `lammps` extra requires lammps-jax 0.2.1 or later. For the C++ kernel,
  build the lammps-jax LAMMPS plugin at 0.2.1 or later as well.
- A MAP fit can stop its Newton polish sooner (see Fixed). Its parameters agree
  with 0.4.0 to approximately 1e-7.

### Added

- **A C++ kernel for linear ACE in CPU LAMMPS bundles.**
  `export_lammps(..., platform="cpu", ffi=True)` and `aj export --platform cpu
  --ffi` put one XLA FFI custom call in the bundle in place of the compiled JAX
  program (`ace_jax.eval.ffi`). Positions and the LAMMPS neighbour list go in;
  atom energies and forces come out.
  - On a 128-core node (Sulis, 4,096 Si atoms, one MPI rank per core) it is 10 to 23
    times faster than the XLA bundle; linear ACE o4d20 costs 3.29 µs/atom-step
    (ML-PACE: 5.31).
  - It needs a linear ACE model, float64 and the matrix layout. It tabulates
    the radial basis in r: energies agree with the calculator to approximately
    1e-12 relative, and forces to 1.5e-8 of the largest force.
  - The export compiles the kernel on first use with your C++17 compiler
    (`$CXX`, `c++`, `g++` or `clang++`; flags `-O3 -march=native`, or
    `--ffi-flags` / `ACEJAX_FFI_FLAGS`) and keeps it in a cache
    (`~/.cache/ace-jax/ffi`, or `ACEJAX_FFI_CACHE`). The bundle records it as
    `ace_jax.ffi`, and `aj export` prints the `LAMMPS_JAX_FFI_HANDLERS` value
    that lets LAMMPS load it. Run LAMMPS with `newton on`.
  - `jax.grad` and `jax.vjp` with respect to positions work through the
    kernel's energy function.
  - See "CPU kernel" in [Export a model to LAMMPS](docs/user/howto/lammps.md).

### Changed

- **The `lammps` extra requires lammps-jax 0.2.1.** Its LAMMPS plugin
  registers custom calls for the CPU, which the C++ kernel needs.
- **Dense rows run in blocks of 4,096 on the GPU** (`energy_forces_virial_dense`
  without `chunk`; 16,384 elsewhere, unchanged). Linear ACE o4d20 with 32,768
  atoms on an A100 uses 1.12 GB for the program, against 4.06 GB before. LAMMPS
  bundles and `ace_jax.jaxmd` size their own blocks and are unchanged.

### Fixed

- **The MAP Newton polish could step for hours.** With the gradient floor, it
  kept stepping while each step gained less than the evidence's roundoff
  (observed: 10 hours at o4d20). It now stops after three such steps in a row
  and reports `stalled`. The ARD evidence polish is unchanged.
- **`--learn-radial` ran out of GPU memory at large bases.** Each radial round
  kept the compiled constants of its hyperparameter re-fit (about 0.7 GB at
  o4d20), until an A100 ran out. They are now released after each round, at
  approximately 11% more time per round.

### Benchmarks

- The Performance page has an interactive version of the accuracy-cost figure:
  presets, a legend that shows and hides families, and the code of each timing
  on hover.
- GRACE-1L and GRACE-FS (gracemaker) are in the silicon benchmark. GRACE-1L
  reaches the target only at its largest size, at 4 to 27 times the cost of
  linear ACE or MACE c16L0.
- The figure shows linear ACE with and without the C++ kernel. All figures
  come from one CPU and one GPU cost file, built by the reproduce block of
  `docs/dev/accuracy-cost-results.md`.
- A GPU memory report (`docs/dev/gpu-memory-results.md`): linear ACE o4d20 runs
  4.4 million atoms on one A100-40GB.

## 0.4.0 (2026-10-19)

**Highlights.** float32 is now a real float32 calculation in LAMMPS bundles
and calculators, and a new page tells you when it is safe. The linear fit can
calculate its design rows in float32 (`--row-precision float32`). MD runs on
the device with JAX-MD (`ace_jax.jaxmd`). Larger models use the DAG product
basis automatically. `aj fit` fits linear ACE by default.

**Upgrading from 0.3.0.** Saved models are unchanged.
- `aj fit` without `--m-per-species` now fits linear ACE (it was the hybrid
  ACE + GP model). Give `--m-per-species` > 0 for the hybrid model.
- A float32 bundle or calculator made with 0.3.0 from a model loaded under
  x64 calculated in float64. Export it again for the float32 speed.
- Models with 4,000 or more chain-form multiplications per atom use the DAG
  product basis. Results agree with 0.3.0 to round-off. Set `aa_dag=None`
  for the 0.3.0 form.

### Fixed

- **float32 export and calculators calculated in float64.** With x64 enabled,
  `export_lammps(dtype="float32")`, `aj export --dtype float32` and
  `ACECalculator(dtype=float32)` on a model already loaded traced a float64
  program. Only the inputs and outputs were float32, so there was no speed-up.
  The model is now converted to float32 after the lean form and the splines
  are made in float64 (`ace_jax.eval.with_dtype`).
- **float32 spline tables lost the precision of their derivatives.** Splined
  and learned radials and radial tables in float32 had a relative derivative
  error of up to 5e-2. The tables are now kept in a local difference form,
  made in float64: 8e-7.
- The design rows of a fit are calculated at the highest matmul precision.
  TF32 on an A100 corrupted float32 force rows.

### Changed

- **The recursive (DAG) product basis is automatic.** `ACECalculator` and
  `export_lammps` default to `aa_dag="auto"`. It uses the DAG for models with
  4,000 or more chain-form multiplications per atom, and the chain form for
  smaller models. Results below the threshold are unchanged. Larger models run
  1.2-2.7x faster on a CPU node and 1.1-1.8x faster on an A100 (linear ACE
  o4d20: 2.7x and 1.8x). `calc.aa_dag` and the bundle's `ace_jax.aa_dag` give
  the form used. Set `aa_dag=None` for the chain form.
- **`aj fit` fits linear ACE by default.** The default of `--m-per-species`
  is now 0 (it was 500). The hybrid ACE + GP model is experimental. To use
  it, give `--m-per-species` > 0. The GP options (`--kernel`, `--no-bump`,
  `--density`, `--pca-d`, `--embedding`) without `--m-per-species` > 0 are
  now an error. `FitConfig` in Python is not changed: set `arm` explicitly.

### Added

- **float32 design rows for the linear fit.** `aj fit --row-precision
  float32` (`FitConfig.row_precision`) calculates the design rows, and on a GPU
  the QR merges, in float32. The evidence, the MAP and the posterior stay in
  float64. A silicon o4d16 fit took 169 s instead of 265 s on an RTX A4500,
  with the RMSE within 1%. The GP arm and `--learn-radial` refuse it.
  `--row-check` is an optional stability check of the result.
- **The float32 error of a model.** `ace_jax.calc.precision.precision_check`
  compares float32 with float64 on a structure. `export_lammps(...,
  precision_structure=atoms)` and `aj export --dtype float32 --structure`
  record it in the bundle and warn above 0.1 meV/atom or a relative force
  error of 1e-4.
- **A new page, "Choose float64 or float32"** (`howto/precision`), gives the
  speed-up by GPU (2.1-4.5x on an RTX A4500, 1.3-1.5x on an A100), the errors,
  and which tasks can use float32. Energy line searches fail in float32: use
  a force-only optimiser, for example ASE `ODE12r`.
- **MD with JAX-MD (`ace_jax.jaxmd`, the new `jaxmd` extra).**
  `jaxmd.potential` gives the energy and forces of a model to the
  integrators of JAX-MD. The neighbour list, the model and the integrator
  then run on the device in one compiled program. The energy is the energy
  of `ACECalculator` with the same options. `K` sets the slots for each
  atom. If an atom has more neighbours than `K`, the energy and forces are
  NaN, and `overflow` tells you. The model runs in blocks of atoms of the
  size that `export_lammps` selects, so the memory is approximately that of a
  LAMMPS bundle. See [Run MD with JAX-MD](https://acesuit.github.io/ace-jax/howto/jax-md/).
- **`write_yace(model, path, summed=True)`** writes one A basis for all
  neighbour species, as ace-jax evaluates the model. The default writes each
  product once for each neighbour species. The summed form has far fewer
  products for multi-element models: 849 instead of 15,021 for a
  three-element embedded model at order 3. ML-PACE reads it only with the
  `summed_neighbour_A` addition. ace-jax does not read summed files.

### Testing

- The lock builds matscipy-neighbours (`fast-neighbours`) from its `main`
  branch at a fixed commit, as 1.1.0 is not on PyPI yet. A pip install of
  the extra is not changed. A test that compared three evaluations bitwise
  now allows 16 ulp: on a GPU, the order of a sum can change between calls.

- The `pace-parity` CI job builds ML-PACE from
  `jameskermode/lammps-user-pace` (ICAMS/lammps-user-pace#31): the
  `v.2025.12.4.p1` release plus tabulated radials, recursive-evaluator
  speed-ups and `summed_neighbour_A`.
  `tests/test_pace_mlpace_yace.py` evaluates four multi-element exports with
  ML-PACE, expanded and summed, and compares them with ace-jax.

### LAMMPS export

- Bundle row blocks are sized to an element target scaled by the memory of
  the target GPU. o4d16 runs up to 2.65x faster on an RTX A4500.
- lammps-jax is on PyPI. Install it with the new `lammps` extra:
  `pip install "ace-jax[lammps]"`. The CI tests the export with lammps-jax
  0.2.0.
- lammps-jax excludes jaxlib 0.10.2, because its GPU scatter-add is
  approximately 4x slower. With the `lammps` extra, Python 3.11 gets jax
  0.10.1.
- The bundle no longer writes `ace_jax.platform`. lammps-jax 0.2.0 reads the
  `platform` key from anywhere in the bundle. Read the platform from
  `export_info.platform`.

### Benchmarks

- The silicon accuracy-cost figure adds linear ACE run by ML-PACE with the
  speed-ups proposed upstream (ICAMS/lammps-user-pace#31 on the CPU, and
  prototype `pace/kk` kernels in `pace_ref/fast` on the A100). Against stock
  ML-PACE on the same node it is 1.5-1.8x faster on the CPU and 1.1-1.7x
  faster on the A100 (o4d20: 2.89 against 5.31 µs/atom-step on 128 cores,
  2.42 against 4.13 on the A100).

- The silicon accuracy-cost benchmark adds linear ACE run by ACEpotentials.jl
  on the CPU: each `aj fit` model rebuilt as an ACEpotentials `ace_model`
  (`bench/scaling/julia/build_trim_acejax.jl`, gated to 1e-10 against ace-jax),
  compiled with `juliac --trim` (ACEpotentials PR 309) and run in LAMMPS. It is
  the fastest CPU code for linear ACE up to o4d14, ties ML-PACE at o4d16 and is
  2x slower than ML-PACE at o4d20 (`bench/accuracy_cost/build_trim.py`,
  `envs/sulis-cpu-trim.sbatch`).
- The benchmark adds NEP (NEP4, trained with GPUMD). On the CPU it runs in
  LAMMPS with `pair_style nep` in float64. On the A100 it runs in GPUMD in mixed
  precision, and the figure labels these points. NEP is the cheapest family on
  the A100. On the CPU, PACE reaches the same force error at a lower cost. Its
  force error stops at approximately 0.10 eV/Å, the published NEP
  result for this dataset, and it does not reach the QoI target
  (`bench/accuracy_cost/fit_nep.py`).

## 0.3.0 (2026-10-14)

**Highlights.** Linear ACE models export to ML-PACE as `.yace` files with
exact tabulated radials, and `aj export` writes LAMMPS models from the command
line. lammps-jax bundles run on CPUs. New PACE-style radial and basis options
(`--radial-basis sbessel`, `--nmax-by-order`, `--lmax-by-order`) and a Newton
optimiser for learned radials. The Performance page compares accuracy and cost
on silicon.

**Upgrading from 0.2.2.** Fits and saved models are unchanged. The
`write_yace(model, spec, path)` argument order gives a deprecation warning: use
`write_yace(model, path, spec=spec)`. `export_lammps(layout="auto")` can now
choose a smaller row block where 0.2.2 chose `sparse`; the bundle records it.

### LAMMPS and ML-PACE export

- **`aj export`** writes a model for LAMMPS. An `--out` that ends in `.json`
  gives a lammps-jax bundle. A bundle needs `--structure`, the initial
  structure of the run. `--platform cpu` and `--procs px,py,pz` give a CPU
  bundle sized for one MPI rank. An `--out` that ends in `.yace` writes a
  `.yace` model again.
- **`write_yace` writes a linear ACE model as a `.yace` file.** The file gives
  the same energies, forces and stresses as the `.npz` model. It keeps the
  radial functions as tables. ML-PACE reads these tables only with its
  tabulated-radial addition (ICAMS/lammps-user-pace#31), which is not yet in ML-PACE releases. ace-jax
  reads the file without it. `aj export --model model.npz --out model.yace`
  does the same.
- **`write_yace(model, path, spec=spec)`** is the new argument order. The old
  order `write_yace(model, spec, path)` still works and gives a warning.
- **`export_lammps(platform="cpu")`** writes a bundle for the lammps-jax CPU
  backend. This needs a lammps-jax version with `export_model(platform=)`.
  Also set `device_memory` to the host memory of one MPI rank.
- **`neighbour_capacity(procs=(px, py, pz))`** gives the capacities of one MPI
  rank on that processor grid. Run LAMMPS with `processors px py pz` to match.
- **`export_lammps(device_memory=...)`** sets the memory of the GPU that runs
  LAMMPS, in bytes that JAX can use. Use it when you export on a different
  machine. `block_rows=` sets the block size. The bundle records both.
- **`export_lammps` keeps the fast layouts for large models.** Before, if one
  32768-row block did not fit in GPU memory, `layout="auto"` used `sparse`.
  That was 15 times slower for a 1461-function silicon model on an A100. Now
  it makes the block smaller, down to 1024 rows, and uses `sparse` only if no
  block fits.
- **`aa_dag="greedy"` or `"prefix"`** on `ACECalculator` and `export_lammps`
  evaluates the product basis recursively. Each product of order k is one
  product of order k - 1 times one A entry. This is the scheme that ML-PACE
  uses. The results agree with the default to roundoff. `"prefix"` gives the
  same products bit for bit. The default (`None`) does not change.

### Basis and fitting

- **`aj fit --radial-basis sbessel`** builds the tensor radials from the
  spherical Bessel functions that pacemaker uses. The default, `poly`, is
  unchanged. The radial learner works with the two bases. Saved models
  record the basis, and older models load as `poly`.
- **`--nmax-by-order` and `--lmax-by-order`** (`aj fit`, `aj basis`) set a
  maximum radial and angular degree for each correlation order, as
  pacemaker's `nradmax_by_orders` and `lmax_by_orders` do. Without them, the
  basis is unchanged.
- **`aj fit --learn-radial --radial-opt newton`: a Newton optimiser for the
  radials.** It is a trust-region Newton method. Each iteration takes up to
  `--radial-cg-iters` (default 20) exact Hessian-vector products of the
  objective with the coefficients projected out, each about two gradients.
  The learned-radial objective is badly conditioned, and L-BFGS makes slow
  progress on it. In a test with recoverable radials, 8 Newton iterations
  reached 20.6 from 7295, against 77.9 for 8 L-BFGS-B iterations. The default
  stays `lbfgs`.
- **`aj fit --weights`** also takes the path of a `.json` file.

### Benchmarks

- **The Performance page compares accuracy and cost on silicon.** Fitted
  models of each family (linear ACE, PACE, MACE, GAP) are compared at the
  same accuracy, timed in LAMMPS on an A100 and on a 128-core CPU node.

### Fixes

- **`aj fit --learn-radial` keeps the learned radials if a later stage
  fails.** The fit writes `fit/radial_lam=<weight>.npz` for each roughness
  weight when its learning stops, before the gate. Before, a failure in the
  gate lost all the learning. The fit also frees the JAX caches before the
  gate, which caused that failure on GPUs.
- **`aj calibrate` refuses an `--uq ard-gp` posterior** (or a `gp_model.npz`)
  with a clear message. Before, it failed later with a missing `model.npz` or
  a `KeyError`.
- **`--uq ard-gp` fits no longer build the support reference.** Only the
  linear calculator serves the support flag, so the reference was not used.
- `--ard-mode sequential` saves memory with `--uq ard` only. The help text and
  the force-uncertainty guide now say so.

## 0.2.2 (2026-10-07)

**Upgrading from 0.2.1.**
- **Refits give different results; refit to get them.** The linear hyperparameter MAP is now converged by default (L-BFGS and a Newton refinement, checked), and the linear evidence and posterior are computed by QR. A fit that passes `--weights` without `--noise` now uses one shared noise scale, so the weights set the E:F:V balance; use `--noise per-quantity` for the old behaviour. Saved models still load and evaluate unchanged.
- **`--learn-radial` results change.** Radial learning now uses the QR statistics and a different optimiser, so learned radials differ from 0.2.1.
- **`support_ok` now flags atoms.** `forces_support` used the wrong quantile in 0.2.0 and 0.2.1. No refit is needed.

- **`--weights` now sets the E:F:V balance by default.** `aj fit --noise`
  takes a new default, `auto`: shared noise (one σ for every weighted row,
  as ACEpotentials' BLR) when you give `--weights` (or weight factors), and
  per-quantity noise otherwise. Before, the default was always per-quantity,
  where each learned σ cancels its quantity's weight at a converged MAP, so
  explicit weights had no effect (#64). **Fits that pass `--weights` without
  `--noise` change**: use `--noise per-quantity` to keep the old behaviour.
  Fits without weights are unchanged. `auto` uses per-quantity where shared
  noise is not available (`--sigma-type`, joint `--uq ard`, `--solver
  lstsq`). `config.json` and `fit.yaml` record the resolved mode.

- **`--learn-radial` works on large bases.** Radial learning computed its
  objective, the hyperparameter re-fit and the candidate scores from the Gram
  matrix, whose condition number is the square of the design's. On Si order
  4, degree 16, that Gram had negative computed eigenvalues: the objective was
  NaN once σ_c passed e^5, and the fit failed after six hours. All three now
  use the QR statistics, as the main fit does. The objective is summed
  directly as the residual at a QR-solved readout, and its gradient is exact
  at that readout. On o4d16 at the fit's hyperparameters (where the Gram form
  is NaN) the gradient agrees with finite differences to 5e-5. Each round is
  now a SciPy L-BFGS-B run, because the QR pass runs on the host on CPUs.

- **Calibrated force uncertainty on the GP arm: `aj fit --uq ard-gp`
  (experimental).** The `--uq ard` stage (evidence fit, PRESS jackknife
  shape, per-group rms and conformal scales) now also runs on the hybrid
  ACE + GP model, over the joint design [B | k(B, B_M)] at the MAP
  hyperparameters. The GP block gets one ARD scale on its prior K_MM. The
  fit writes `gp_model.npz` with the ARD posterior mean and `posterior.npz`;
  `GPCalculator.from_file(gp_model, posterior=...)` and
  `aj eval --posterior` serve `forces_std`, `forces_cov`, `forces_q`,
  `forces_q_mahal` and `forces_group`. The support flag and `aj calibrate`
  remain linear-only. `--uq ard` on the linear model is unchanged.
  Experimental: on the Cantor benchmark it lifts crack-tip coverage (0.902
  against 0.893 for `--uq ard`) but ranks errors slightly less well
  (Spearman ρ 0.006–0.009 lower), so `--uq ard` stays the recommended option
  (bench/defect_uq/results/2026-10-07_ard_gp_acceptance.md).

- **Smaller `--uq ard` and `--uq ard-gp` files.** `posterior.npz` stores the
  L x L Cholesky factor only for `--ard-variance kappa`, which reads it, and
  then packed (its lower triangle). The default sandwich shape serves forces
  from its shape factor alone. On the Cantor basis (L = 15 035) a
  `posterior.npz` shrinks from 1.14 GB to 0.24 GB, and `ACECalculator` no
  longer holds an unused 1.8 GB float64 copy on the device. An `--uq ard-gp`
  `gp_model.npz` no longer carries the full-posterior factor (1.9 GB to
  10 MB): its force uncertainty comes from `posterior.npz`, and on its own
  the model gives energies and forces without `energy_std`. Old posterior
  files load unchanged. `predict_ard` on a saved sandwich posterior reports
  NaN for the uncalibrated energy and virial variances.

- **Follow a running fit.** `aj fit` writes `out/progress.jsonl` as it runs:
  one JSON line per stage, L-BFGS evaluation, Newton refinement and
  radial-learning round, with timings, so a script or `tail -f` can track a
  long fit. Its standard output is now line-buffered, so a log written to a
  file is current, not empty until the fit ends. The `--learn-radial` round
  line now gives the objective's change within the round and the evidence
  (LML) after the hyperparameters are re-fitted. The objective alone could
  rise from one round to the next while the radials improved, because each
  round re-weights it with the re-fitted noise.

- **`aj fit` no longer repeats its statistics pass.** A linear fit streamed
  the training set three times: once for the evidence, again for the
  predictions, and again to write `model.npz`. Prediction and the model file
  now reuse the fit's QR statistics and its MAP posterior, which are bit for
  bit what the extra passes computed. The GP and Cholesky fits reuse their
  cached linear Gram and the MAP posterior. Each configuration's neighbour list
  is built once, not twice. Without `--test`, the test dataset is the training
  dataset, not a second copy of it (`config.json` then has
  `"test_is_train": true`). Results are unchanged, bit for bit.
  `FitConfig(predict_stats="auto")` gives the command-line behaviour.

- **Faster QR statistics on large bases.** The linear fit's QR evidence
  merged every batch into its triangular factors, about L^3 work per batch
  and quantity. It now buffers the nonzero rows of many batches and merges
  once per approximately L rows, without forming Q. At basis size 5477
  (Si order 4, degree 20) one pass needs 5e13 instead of 1.2e15
  floating-point operations: on a 32-core CPU the merges take an estimated
  ~13 min instead of ~5 h. Results agree with per-batch merging to roundoff,
  not bit for bit.
- **Faster, smaller fits on large bases: the A2B coupling is sparse on the
  fitting path.** Models now hold A2B as its nonzeros only (it is 0.025-2%
  occupied), and the fit's design rows and their Jacobian use them directly.
  At basis size 5456 (Si order 4, degree 20) one 224-atom batch of rows takes
  1.9 s instead of 88 s on a 32-core CPU, and the 2.6 GiB dense A2B (held
  2-3 times) is gone. Results equal the dense contraction to roundoff, not
  bit for bit. `load(..., a2b_sparse=False)` keeps the dense form.
- **Fixed: linear fits ran out of memory on large bases.** `aj fit
  --m-per-species 0` (and `--learn-radial`) no longer computes the
  site-feature tensor of the whole training set, which only the GP arm reads.
  At basis size 5456 on 154k atoms that tensor alone was 7.2 GiB of GPU
  memory. Linear fit results are unchanged, bit for bit.
- **Fixed: `forces_support` used the wrong quantile.** In 0.2.0 and 0.2.1,
  `ACECalculator` (and so `aj eval --support`) computed `support_q` as the
  weighted α quantile of the calibration scores instead of the 1 − α
  quantile the documentation defines (α = 1 − `--ard-coverage`, 0.1 by
  default). `support_ok` was therefore `False` only when an atom's own weight
  exceeded 90 % of the pool, so the flag almost never fired. It now flags
  atoms at the documented level. No refit is needed: the posterior is
  unchanged. Force uncertainties (`forces_std`, `forces_q`, `forces_cov`)
  were not affected.
- `--shape-path committee` on `aj eval` and `aj calibrate`
  (`ACECalculator(..., shape_path="committee")`): the `--uq ard` force
  uncertainty without the whole cell's force design rows. The shape is
  evaluated as the forces of a linear ACE model with one coefficient vector
  per column of the shape factor, so memory scales with that factor's rank
  instead of the basis size. Values equal the default path to roundoff. It
  is for cells much larger than 3–4k atoms, where the default path is better.
  `shape_tau=` and `shape_rank=` truncate the factor: the ranking of atoms
  survives but coverage does not, since the scales are fitted at full rank.
- `aj fit --uq ard --ard-support-features normalised`: the support
  reference on the unit-norm site descriptor plus log-norm channels per body
  order, so atoms losing neighbours (a descriptor shrinking towards zero)
  stay visible to the support classifier. The default stays `raw`.
- **`aj fit --noise shared`: one noise scale for the energy, force and
  virial rows, as in ACEpotentials' BLR, so the E:F:V weights set the
  balance.** The default stays `--noise per-quantity` (σ_E, σ_F and σ_V
  learned separately). `FitConfig(noise="shared")` and the `fit.yaml` key
  `noise` set it too.
  - Why: with three free noise scales, each cancels its quantity's weight
    at a converged MAP, so weights such as ACEpotentials' E 30 / F 1 / V 1
    stop mattering. Force rows far outnumber energy rows, so the evidence
    then pushes σ_E up and underweights the energies. On GAP-18 silicon
    (linear, order 4, degree 12, ACEpotentials' weights) per-quantity noise
    learns σ_E = 2.7 against σ_F = 0.10 and gives 28.2 meV/atom and
    0.125 eV/Å on the test set; shared noise (σ = 0.137) gives 10.6 meV/atom
    and 0.144 eV/Å. ACEpotentials' BLR on a like-for-like basis (an
    `ace_model` configured as ace-jax builds it) gives 10.2 meV/atom,
    0.135 eV/Å and 70 meV/atom in the virial (9.9 meV/atom and 0.132 eV/Å
    on its `ace1` basis).
  - Use it whenever the weights are meant to set the balance, as in
    ACEpotentials. On Si_tiny the shared-noise evidence optimum reproduces
    ACEpotentials' own BLR fit (`acefit!`, default weights) to 4×10⁻⁷ in the
    coefficients; per-quantity noise ends 55% away.
  - It is a tie, not a second objective: σ_E and σ_V copy σ_F (which keeps
    its hyperprior), so the L-BFGS MAP, the Newton polish, the convergence
    check and the Laplace, VI and NUTS rungs see one noise coordinate.
    `--learn-radial` ties the radial stage's inner MAPs too.
    `map_convergence.json` records `noise` and `tied`, and the resolved
    `fit.yaml` records `noise`. It is not available with `sigma_type`,
    joint ARD (`--uq ard --ard-mode joint` refits the noise itself; use
    `sequential`) or `--solver lstsq`.
  - `map_convergence.json` now also records `log_evidence`, the log
    marginal likelihood at the MAP (every mode). Compare fits across noise
    modes by it, not by `logpost`: shared noise leaves out the hyperpriors
    of the two tied scales, so its `logpost` is not on the same footing.

- **Linear `aj fit` now converges the hyperparameter MAP by default, and
  every fit is checked, so refits give different (better) results.** The
  default `--opt` is now `lbfgs` (bounded L-BFGS-B) instead of `adam`.
  - The old default, 500 Adam steps, moved each log-hyperparameter at most
    about 5 units from the prior mean, so fits on large data were **not
    converged**, and nothing said so. On GAP-18 silicon (linear o4d12,
    measured on a 0.2.0 basis) its log-posterior was −1.04×10⁷ against
    +1.92×10⁵ at the optimum, σ_E was 100× too small, and the test force
    RMSE was 0.345 eV/Å against 0.163 at the converged MAP. L-BFGS took
    63 evaluations (0.2 s on an RTX 4000 Ada) to reach it.
  - Linear fits on the cached-Gram LML then take a Newton polish to a
    stationary point (`--map-polish`, default `auto`). Its Hessian is
    exact. It is built one column at a time, one Hessian-vector product per
    free hyperparameter, which needs about 2.3× the gradient's memory
    (515 MB against 225 MB at 2,053 basis functions). `jax.hessian` would
    need about 10× (2.2 GB).
  - GP fits (`--m-per-species` > 0) are not polished by default, since one
    GP gradient took about 70 s on GAP-18 Si (o3d12, 100 inducing sites).
    They are checked and warned. A line-search stop of L-BFGS-B
    (`ABNORMAL`) now restarts it once, for at most 50 evaluations.
    `--map-polish on` polishes a GP fit, at one Hessian-vector product per
    free hyperparameter per Newton step. On the GP arm each product is a
    forward-over-reverse pass through the streamed objective, so a polish
    can take hours at large scale.
  - **The check.** The fit estimates what one more Newton step would gain
    in log-posterior. If that exceeds 10⁻³ nats, it logs
    `WARNING: MAP did not converge` and warns; `--strict` makes that an
    error (`MapNotConverged`). The estimate costs no extra evaluation: it
    uses the polish Hessian, or else L-BFGS-B's own limited-memory
    inverse-Hessian estimate. That estimate is approximate, so on a GP fit
    the gain is too. A polished fit whose log-posterior is too noisy to
    resolve the last 10⁻³ nats can pass on a roundoff floor of at most
    0.1 nats, and only when every gradient component is within 10× its
    measured roundoff; it is then recorded as `resolution_limited`. The
    record is written to `map_convergence.json`.
  - **Laplace rung.** It now holds hyperparameters that sit on their box
    bound at the MAP fixed. A converged MAP can run a noise scale to its
    bound, where the Hessian is singular.
  - **Old run files.** `--opt adam` is still available. `FitConfig`
    (Python) already defaulted to `opt="lbfgs"`, and it now polishes linear
    fits too; `map_polish="off"` restores the old endpoint. A resolved
    `fit.yaml` from an earlier run that says `opt: lbfgs` also polishes
    when rerun.
  - `sigma_type` fits (Python) still use Adam, unchecked: the fit warns
    that `map_polish` and `strict` do not apply.
- **Changed: the linear fit's evidence and posterior are computed by QR.** The
  evidence fit factored the normal equations G + Λ by Cholesky, which squares the
  condition number of the design. Fits that nearly interpolate their data (smooth
  labels from a reference model, a well-matched basis) drive the noise level to its
  floor and the prior towards flat, and there the Cholesky lost accuracy without
  failing: on the repository's own Si fixture the saved weights were off by about
  0.1%, and on a 685-function Si basis the evidence was off by 7%. In the worst
  cases it returned NaN and wrote a model of NaN weights (the GaAs fit of the
  bring-your-own-data tutorial). The linear arm (`--m-per-species 0`) now
  compresses each quantity's design rows by QR once and evaluates the evidence,
  its gradient, the predictions and the saved readout from one QR factorisation,
  at the condition number of the design rather than its square. Where the
  Cholesky was accurate the results agree to roundoff. An evidence evaluation costs
  about 6× the Cholesky's (a few seconds per fit at a few hundred functions, more
  at several thousand); `--lml-solver cholesky` (`FitConfig(lml_solver=...)`)
  restores the old path. The GP arm still uses the Cholesky, now falling back to
  QR when it fails.
- **Fixed: a model with non-finite weights is never written.** A failed solve now
  raises instead of saving NaN weights.

## 0.2.1 (2026-10-05)

**Upgrading from 0.2.0.**
- **Rebuild built bases and refit.** A basis built by `aj basis`, `aj fit --order/--max-degree` or `build_basis` in 0.2.0 was not rotation invariant (see the first entry below). Saved `.npz` and `gp_model.npz` files keep the bad coupling and loading cannot detect it, so rebuild the basis and refit. Embedding bases (`--embedding`) and models exported from Julia are unaffected.

- **Fixed: built bases were not rotation invariant.** Since the compiled
  coupling library arrived (0.2.0), `aj basis`, `aj fit --order/--max-degree`
  and `build_basis` paired the coupling's columns with the wrong products of
  the A basis. The B functions were therefore not rotation invariant: at
  order 2, degree 6 for Si, 5 of 17 changed under a rotation, and at
  order 4, degree 12, 272 of 338. The smoothness prior also applied to the
  wrong columns. Fits from these bases lost accuracy: on GAP-18 Si at
  order 4, degree 12, the force RMSE floor was about 0.19 eV/Å, against 0.13
  for ACEpotentials. Built bases are now invariant to roundoff, and the
  invariance and parity with ACEpotentials are tested. A saved `.npz` or
  `gp_model.npz` keeps its bad coupling, and loading cannot detect it:
  **rebuild the basis and refit**. Stale entries in the coupling cache are
  corrected on use. Embedding bases (`--embedding`) and models exported from
  Julia are unaffected.
- Faster CPU evaluation. On the CPU, the product basis is now an explicit
  feature-major chain of multiplies. Before, `jnp.prod`'s reverse mode
  compiled to strided scalar copies. Other backends are unchanged, and
  energies and forces agree with the old form to roundoff. Each model's
  medium benchmark at 2,048 atoms, on an i9-14900K with 8 P-cores: linear
  ACE 2.9× faster on SiGe and 1.2× on Cantor; PACE 1.3× on SiGe and 1.1×
  on Cantor.
- Faster CPU evaluation of linear ACE models with three or more species.
  On the CPU, the lean evaluation form now pools A per neighbour species
  with a segment sum. Before, it expanded every edge's radial values over
  all species with a one-hot. Two-species models, PACE models and other
  backends are unchanged. Cantor (5 species) at 2,048 atoms, on an
  i9-14900K with 8 P-cores: 1.37× faster for the medium model, 1.31× for
  the small and 1.12× for the large.
- Opt-in radial tables for faster CPU evaluation: `radial_table=True` on
  `ACECalculator`, `export_lammps` and `lean` (or an interval count; the
  default is off). Each species pair's radial functions are tabulated as a
  cubic spline in r on [0.5 Å, rcut] with 4,000 intervals, so an edge reads a
  table instead of evaluating transcendentals: for ACE, R_nl (transform and
  envelope included) and the pair radial; for PACE, the radial basis g_k.
  Beyond each pair's cutoff the tables are exactly zero. This is an
  approximation, not roundoff: on the medium benchmark models, energies agree
  to at most 1.8e-11 relative (≤ 1e-12 eV/atom) and forces to at most 1.7e-8
  of max|F|. Compiled skin step at 2,048 atoms, on an i9-14900K, one core /
  8 P-cores: ACE 1.24× / 1.08× faster on SiGe and 1.39× / 1.14× on Cantor;
  PACE 1.09× / 1.02× on SiGe and 1.22× / 1.08× on Cantor.

## 0.2.0 (2026-10-03)

**Upgrading from 0.1.x.**
- **`aj fit --uq ard` gives new results.**
  - `forces_std` now uses per-group scales with a transfer correction instead of one scalar λ.
  - The default `--force-shape` is `aniso`.
  - ARD fits E0 jointly under the default `--e0 lsq`, so refit ARD models change slightly (energies included).
  - Pass `--force-shape iso` to get the spherical region.
- **Posteriors are now schema 3.** Schema-1 and schema-2 `posterior.npz` files still load and serve `forces_std`. `forces_q`, `forces_cov`, `forces_group` and `forces_support` need a refit.
- **Energy and virial variances under ARD are not calibrated.** Only forces are.

- Calibrated per-atom force uncertainty, `aj fit --uq ard` revision 2. The
  uncertainty is a shape times per-group scales:
  - the shape is the per-atom 3×3 block of an exact, centred
    delete-one-cluster (PRESS) jackknife covariance, with large training cells
    split into spatial blocks (`--ard-cluster-size`, default 3 r_cut;
    `--ard-press exact|block`);
  - the scales are fitted per Mondrian group (shell distortion band ×
    coordination, `--ard-groups`, `--ard-n-min`) on a stratified hold-out
    (`--ard-val-frac`): an rms scale for `forces_std`/`forces_cov` and a
    configuration-weighted conformal quantile for `forces_q` at
    `--ard-coverage` (default 0.9);
  - the hold-out scale is carried to the served posterior by a transfer
    exponent fitted per run from a second hold-out fit
    (`--ard-transfer exponent|sqrt|none`);
  - `--force-shape aniso` (the default) serves an ellipsoidal region and
    `forces_q_mahal`; `iso` a spherical one.

  `ACECalculator(model, posterior=...)` serves `forces_std`, `forces_cov`,
  `forces_q`, `forces_q_mahal`, `forces_group` and the covariate-shift
  support flag `forces_support` (`--no-ard-support` skips it); `aj eval
  --posterior P --per-atom out.xyz [--support]` writes them per atom. The new
  `aj calibrate` recomputes the scales on labelled target-regime cells
  (per-group replace by default, `--append`, `--replace`). On the Cantor
  benchmark the default meets every coverage target, in distribution and on
  unseen crack and dislocation cells. Posteriors are schema 3; schema-2 files
  serve only the scalar `forces_std`. User docs: a how-to page and a page on
  the mathematics.
- Size-aware batching (`aj fit --batch-pack auto|on|off`, `build_dataset(pack=)`):
  configurations are packed by an atom budget when a training set mixes small
  and large cells, instead of padding every batch to the largest.
- Training-side design rows and statistics are node-chunked over a memory
  budget, so fits with cells of thousands of atoms fit on one GPU; the served
  uncertainty is chunked over atoms.
- `GPCalculator(deriv_dtc=False)` (`aj eval --no-deriv-dtc`) gives an SoR-only
  `forces_std` without the derivative-DTC term, whose whole-cell arrays may not
  fit for a big cell; over budget the term now warns instead of raising.
- ARD now fits E0 jointly under `--e0 lsq` (as BLR): one E0 column per species
  with BLR's fixed prior, outside the ARD body-order groups; `model.npz` holds
  the fitted E0. Force uncertainties are unaffected, and posteriors written
  before still load.
- `run_ard_stage` compiles its programs once (no per-call recompilation).
- ARD evidence fits are fully converged and deterministic:
  - L-BFGS is followed by a bounded projected-Newton polish with the exact Hessian;
  - convergence is judged on measured roundoff;
  - two runs give bit-identical results.
  The fit reports its evidence and gradient roundoff and logs cond(S), warning above `ard_cond_max`.
- The total energy is a compensated sum of the site energies (correctly rounded,
  independent of atom order and layout), and a float32 model returns it in
  float64 when x64 is enabled instead of a float32 total quantised at its own
  ulp (1 eV at 60k atoms), which stalled ASE line searches. Forces are unchanged.
- `ACECalculator(..., energy_reference="E0")` reports energies relative to the
  isolated atoms (`results["e0_offset"]` holds the subtracted constant). The
  large per-atom E0 otherwise limits the float64 resolution of the total and
  stalls energy-based line searches in tight relaxations of large cells.

## 0.1.2 (2026-10-03)

- Tutorials link the public [MLIP School](https://mlipschool.uk/) pages.
- `ace_jax.fit.pipeline.fit` refuses to run without float64 (`jax_enable_x64`),
  instead of fitting silently in float32. `aj fit` already enables it.

## 0.1.1 (2026-10-03)

- User docs: a Performance page (CPU float64, GPU float64/float32; SiGe and
  Cantor; every evaluator).
- Tutorials 6, 7 and 9, adapted from the MLIP school's E2, E3 and D notebooks:
  surfaces (coverage, repair and stable relaxations), automating curation
  (random, novelty and ace-jax's ARD-uncertainty selection), and bring your own
  data (GaAs demo). Their labels ship; `ace_jax.tutorials` gains the builders,
  `campaign` (descriptors, MD pools, novelty) and `curation` (the campaign loop).
- `e0="lsq"` (`aj fit --e0 lsq`) now fits E0 jointly with the readout: one
  column per species with a 1 eV prior around the least-squares E0, folded into
  E0 on export (isolated training atoms still pin their species; otherwise E0
  is a reference level that trades off against the basis). An energy
  offset the basis carries is absorbed instead of being paid as an energy error;
  on a five-element alloy with a one-channel embedding the test energy error
  falls from 1188 to 9 meV/atom. Every `e0="lsq"` fit changes slightly. The old
  behaviour is `e0="prefit"`, and ARD and POPS fits keep it.
- Fixed: a built basis keeps its smoothness prior when saved, so fits from a
  `BasisSpec` or `Basis` no longer print "gamma missing ... rebuilt"; the
  notice for older exports goes through the caller's `log`.
- `embedding="identity"` with fewer channels than elements is an error with an
  explanation (it used to fail with "zero embedding row"); embedding tables that
  make two elements indistinguishable (e.g. `d_max=1`) raise a warning.
- Tutorial 3, multi-element fits: categorical against embedded species bases on a
  five-element alloy. The school tutorials are renumbered 4, 5 and 8.

## 0.1.0 (2026-10-02)

First release on PyPI.

- Build ACE bases in Python (`aj basis`, `ace_jax.basis.build_basis`); the coupling
  coefficients come from the compiled `ace-jax-coupling` wheel, with no Julia.
- Fit linear ACE and hybrid ACE + GP models (`aj fit`, `ace_jax.fit.pipeline`):
  Bayesian linear regression with evidence-maximised noise and prior
  hyperparameters, a calibrated UQ ladder (MAP, Laplace, Pathfinder, VI, NUTS),
  POPS and ARD, learned radial bases (`--learn-radial`), `fit.yaml` run files, and
  `--solver lstsq` for plain least squares.
- Evaluate ACE `.npz` and PACE `.yace` models: `ACECalculator` and `GPCalculator`
  for ASE, `aj eval`, and `export_lammps` bundles for lammps-jax.
- Fit from extxyz files or lists of `ase.Atoms`; stress labels are converted to
  virials (`--stress-key`).
- Documentation and tutorials at <https://acesuit.github.io/ace-jax/>.
