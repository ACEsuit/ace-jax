"""Variable projection (VarPro) driver:  learn nonlinear feature
parameters eta by minimising the PROJECTED residual, with the linear
coefficients solved exactly (projected out) at each outer step by a stable
inner least-squares solve.  This is the FS-density pre-training of
spike_fs/varpro_core.jl ported to the JAX inner solve -- the outer
optimiser is an alternative ladder driver over the same inner solvers as
objective.posterior / solve.py: "learn the nonlinear part once, cheaply, by
VarPro; freeze; convex fit at scale".

The gradient d/deta of the projected residual is obtained by autodiff THROUGH
the inner solve (JAX differentiates the QR/lstsq exactly), i.e. the
Golub-Pereyra/Kaufman VarPro gradient with no hand-coded formula.  The inner
solve is the stable stacked least squares [ (w) Phi ; Lam_sqrt ] (kappa, not
kappa^2), matching ace_jax.fit.solve.
"""
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize


def inner_solve(Phi, y, Lam_sqrt):
    """Regularised least squares min ||Phi c - y||^2 + ||Lam_sqrt c||^2 via the
    stable stacked QR (jnp.linalg.lstsq), differentiable in Phi (hence in eta)."""
    A = jnp.concatenate([Phi, Lam_sqrt], 0)
    b = jnp.concatenate([y, jnp.zeros(Lam_sqrt.shape[0])])
    return jnp.linalg.lstsq(A, b, rcond=None)[0]


def projected_residual(eta, design_fn, Lam_sqrt):
    """||Phi(eta) c*(eta) - y||^2 with c* the projected inner solution."""
    Phi, y = design_fn(eta)
    c = inner_solve(Phi, y, Lam_sqrt)
    r = Phi @ c - y
    return r @ r


def learn(eta0, design_fn, Lam_sqrt, maxiter=200):
    """Outer VarPro: minimise the projected residual over eta by BFGS (gradient
    by autodiff through the inner solve).  Returns (eta_star, result)."""
    loss = lambda e: projected_residual(e, design_fn, Lam_sqrt)
    res = minimize(loss, eta0, method="BFGS", options=dict(maxiter=maxiter))
    return res.x, res


def fit_frozen(eta, design_fn, Lam_sqrt):
    """Convex fit at a frozen eta: the projected inner solution and its residual.
    Used for the transfer test (freeze eta learned on one split, fit another)."""
    Phi, y = design_fn(eta)
    c = inner_solve(Phi, y, Lam_sqrt)
    r = Phi @ c - y
    return c, float(r @ r)
