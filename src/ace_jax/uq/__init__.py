"""Uncertainty quantification of a fitted linear model, generic in its quantities: the ARD evidence and
posterior (`ard`), the cluster jackknife (`jackknife`, `clusters`), conformal calibration (`conformal`),
POPS (`pops`) and the support check (`support`).  `ace_jax.fit.{ard, jackknife}` apply them to the energy,
force and virial fit."""
