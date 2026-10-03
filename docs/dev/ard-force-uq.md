# Calibrated per-atom force uncertainty (`--uq ard`, revision 2): usage notes

This is the user-facing description of `--uq ard` revision 2 (PR #34) as it stood in
the top-level README before the README became a landing page (release 0.1.0). The user
documentation is now the MkDocs pages
[`docs/user/howto/force-uncertainty.md`](../user/howto/force-uncertainty.md) (usage) and
[`docs/user/concepts/force-uncertainty-maths.md`](../user/concepts/force-uncertainty-maths.md)
(mathematics); this file is kept as a compact developer summary.
Design: `specs/2026-09-30-conformal-force-sigma-design.md`; mathematics:
`specs/tex/force-uq-math-pipeline.tex`.

```bash
# calibrated per-atom force uncertainty: ARD posterior (linear model)
aj fit --model si.npz --train train.xyz --test test.xyz $K \
    --m-per-species 0 --uq ard --opt lbfgs --r0 2.35 --out out_ard
aj eval --model out_ard/model.npz --posterior out_ard/posterior.npz --data big.xyz $K \
    --per-atom atoms_std.xyz --support   # per-atom forces_std + support flags, e.g. to colour a crack tip
```

`--uq ard` fits prior scales per body order and the noise scales by evidence (joint type-II ML).
The force uncertainty has a shape and two scales (mathematics:
[`specs/tex/force-uq-math-pipeline.tex`](specs/tex/force-uq-math-pipeline.tex)):

- **Shape.** The centred delete-one-cluster (PRESS) jackknife covariance of the force at each
  atom, with spatial clusters of `--ard-cluster-size` r_cut (default 3; `inf` = whole
  configurations), so large cells are split into ~3 r_cut blocks by default. The default
  `--force-shape aniso` keeps a full 3x3 shape and Mahalanobis conformal scores; `--force-shape iso`
  gives the spherical radius instead. The default `--ard-variance sandwich` now means this
  jackknife shape (`--ard-press exact|block` picks the exact per-cluster correction or a block
  approximation); `--ard-variance kappa` keeps the posterior shape A⁻¹ instead and still gets the
  per-group scales below.
- **Two scales, per group.** `forces_std` / `forces_cov` use a per-group rms factor (the scale at
  which the standardised error has unit rms). `forces_cov` is always the full 3x3. `forces_q` is the per-group conformal radius at
  `--ard-coverage` (default 0.9): `|F_err| <= forces_q` with that probability for atoms
  exchangeable with the group's calibration configurations (with the default `--force-shape aniso` the score is
  Mahalanobis: `forces_q` is then the largest semi-axis q_g sqrt(lambda_max(M)) of the region and
  `forces_q_mahal` the Mahalanobis radius q_g; `--force-shape iso` gives the spherical radius).
  On the bench365 v3 validation aniso is the only uncalibrated variant meeting the targets, and
  `aj calibrate` with labelled target cells remains the most reliable route when such data exist.
  Scores come from a stratified hold-out (`--ard-val-frac`) scored with the hold-out posterior, then
  carried to the served posterior by (N/N_fit)^β, with β fitted per run from a second, smaller hold-out fit and clipped to [0, ½]
  (`--ard-transfer exponent`, the default; `sqrt` fixes β = ½, `none` β = 0).
- **Groups.** 8 Mondrian groups = 4 distortion bands x [coordination = modal]
  (`--ard-groups distortion|none`); groups with fewer than `--ard-n-min` (default 20)
  calibration configurations borrow from a neighbour (the threshold rises to ⌈(1−α)/α⌉, e.g. 99 at
  `--ard-coverage 0.99`, so that a finite `q` is reachable). If even all groups pooled have fewer than ⌈(1−α)/α⌉ configurations (a configuration counts once in the pool, however many groups it spans) for
  the coverage, `q` (and `forces_q`) is infinite and the fit logs a WARNING. `forces_group` gives
  each atom's group.
  `posterior.npz` holds the per-group table: `lam_rms`, `q`, `r = q/(lam_rms·χ₃⁻¹(0.9))`
  (r near 1 means the Gaussian shape fits), `n_cfg` (T_val / U), merges and sources.
- **Support.** `forces_support` (`support_ok`, `support_q`, `n_eff`) flags atoms the calibration
  cannot certify (covariate shift); `aj eval --posterior P --per-atom out.xyz --support` writes
  it. `--no-ard-support` skips the reference at fit time.

For a regime the fit data does not cover (cracks, interfaces), recalibrate on labelled cells of
that kind; only the scales change, the model is untouched. A group is recalibrated on the new cells
alone only when they contribute at least `--ard-n-min` (default 20) configurations to it; with fewer,
their scores are pooled with the stored hold-out scores, so a few cells move that group's scale only
slightly. The labelled cells must not be in the training set (`aj calibrate` does not check this),
or the scales come out too small:

```bash
aj calibrate --model out_ard/model.npz --posterior out_ard/posterior.npz \
    --data crack_cells.xyz --out crack_posterior.npz
aj eval --model out_ard/model.npz --posterior crack_posterior.npz --data big.xyz $K --per-atom atoms_std.xyz
# or ACECalculator(model, posterior="crack_posterior.npz")
```

By default, groups where the new cells have at least `--ard-n-min` (default 20; ⌈(1−α)/α⌉ if that is
larger) configurations use them alone; every other group pools the stored hold-out scores with them. `--append` pools in every
group; `--replace` uses the new cells alone in every group. Posteriors are schema 3;
older schema-2 posteriors serve only the old scalar `forces_std` (the new properties raise and ask
you to refit with `--uq ard`). The validation programme for this revision is described in
[`bench/defect_uq/README.md`](../../bench/defect_uq/README.md) and has been run: the default
(aniso + transfer exponent) meets every coverage target, in distribution and on the uncalibrated
crack and dislocation cells; results in
[`bench/defect_uq/results/2026-10-03_rev2_acceptance.md`](../../bench/defect_uq/results/2026-10-03_rev2_acceptance.md). Only the force uncertainty is calibrated
(`ard.json` `tempered_quantities: ["F"]`); energy and virial variances are the uncalibrated
posterior ones. `--uq ard` changes the mean as well as the uncertainty: `model.npz` holds the ARD
posterior mean, not the BLR/MAP mean. `posterior.npz` stores the float32 posterior factor, ~0.9 GB
at L = 15k, plus the cluster factors. `--ard-mode sequential` is the low-memory fallback. The calculator's
`forces_std` holds the whole cell's force design rows, about N·3·L·8 bytes (N atoms, L columns;
7 GB for 100k atoms at L = 3k), so size cells to fit them.
