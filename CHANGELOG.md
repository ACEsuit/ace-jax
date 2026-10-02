# Changelog

## Unreleased

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
