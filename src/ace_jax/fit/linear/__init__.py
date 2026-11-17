"""Generic Bayesian linear fitting: a row generator gives (Phi, Y, w) per quantity (`stats`), streamed to
Gram or QR statistics; the evidence and posterior from them (`objective`, `lml_host`, `lml_svd`); the prior
root (`prior_root`); and the optimisers a MAP uses (`multistart`, `newton`, `lbfgs`, `block_lbfgs`, `varpro`).
Nothing here knows a model, or energies and forces: `ace_jax.fit.{stats, objective, solve}` are the
potential fit built on it."""
