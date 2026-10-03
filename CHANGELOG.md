# Changelog

## Unreleased

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
