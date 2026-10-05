# Changelog

## Unreleased

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
    central differences of the compiled gradient over the free
    hyperparameters, so it needs no more memory than the gradient.
    `--map-polish exact` uses `jax.hessian` instead: about 10× the
    gradient's temporaries (2.2 GB at 2,053 basis functions), and used only
    if it fits in free memory.
  - GP fits (`--m-per-species` > 0) are not polished by default, since one
    GP gradient took about 70 s on GAP-18 Si (o3d12, 100 inducing sites).
    They are checked and warned. A line-search stop of L-BFGS-B
    (`ABNORMAL`) now restarts it once, for at most 50 iterations.
    `--map-polish on` polishes a GP fit (2 gradient evaluations per free
    hyperparameter per Newton step).
  - **The check.** The fit estimates what one more Newton step would gain
    in log-posterior. If that exceeds 10⁻³ nats, it logs
    `WARNING: MAP did not converge` and warns; `--strict` makes that an
    error (`MapNotConverged`). The estimate costs no extra evaluation: it
    uses the polish Hessian, or else L-BFGS-B's own inverse-Hessian
    estimate. The record is written to `map_convergence.json`.
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
