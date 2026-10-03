# Changelog

## Unreleased

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
