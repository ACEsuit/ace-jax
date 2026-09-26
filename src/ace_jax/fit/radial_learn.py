"""Learned tensor-basis radials by variable projection (VarPro).

The optimiser moves the analytic-branch mixing weights Wnlq; the linear readout
is projected out exactly.  At fixed hyperparameters theta the projected ridge
residual

    r(W) = min_c ||Phi(W) c - y||^2_w + c^T Lambda c = yy - b^T (G + Lambda)^{-1} b,
    Lambda = diag(gamma^2 / sigma_c^2),

is closed form in the streamed linear statistics (G, b, yy) of the model with
radials W, so no design matrix is ever materialised.  Its W-gradient is the
exact Golub-Pereyra/Kaufman gradient (envelope theorem), taken by autodiff
through the checkpointed `linear_statistics` scan.  M = 0 throughout: the
residual GP is fitted afterwards on the frozen learned model.
See docs/specs/2026-09-26-learned-radial-varpro-design.md.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.linalg import solve_triangular

from .hypers import from_array, to_array
from .ladder import run_map
from .objective import combine, log_marginal_likelihood
from .radial_model import (normalise, radial_gram, require_analytic, roughness,
                           roughness_matrix, row_active, with_radial)
from .stats import linear_statistics


def projected_residual_from_stats(theta, lin, gamma):
    """yy - b^T (G + Lambda)^{-1} b from linear statistics `lin` (M = 0).
    A failed Cholesky yields NaN; the learner treats that as a stop signal."""
    G, b, yy, _, _ = combine(theta, lin)
    lam = gamma ** 2 * jnp.exp(-2.0 * theta.log_sigma_c)
    L = jnp.linalg.cholesky(G + jnp.diag(lam))
    v = solve_triangular(L, b, lower=True)
    return yy - v @ v


def projected_residual(W, theta, prob, ds):
    """VarPro objective of the linear ACE with tensor radials W (one full
    streaming pass over ds)."""
    lin = linear_statistics(with_radial(prob.model, W), prob.cfg, ds)
    return projected_residual_from_stats(theta, lin, prob.gamma)


def require_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError("learned radials need float64: jax.config.update('jax_enable_x64', True)")


def lbfgs_loop(f, x0, *, steps, tol=1e-6, patience=3, memory_size=10):
    """Minimise f (x -> scalar, jittable) by optax L-BFGS with zoom line search.

    Stops after `steps` iterations ("steps"); when the relative decrease stays
    below `tol` for `patience` consecutive iterations ("converged"); when an
    iteration fails to decrease f ("linesearch"); or when f, its gradient or
    the iterate go non-finite ("nonfinite").  Always returns the best finite
    iterate seen: (x_best, f_best, trace, reason), trace = f after each
    accepted step (strictly decreasing)."""
    x_best, f_best = x0, float(f(x0))
    if int(steps) <= 0:
        return x_best, f_best, [], "steps"
    if not np.isfinite(f_best):
        return x_best, f_best, [], "nonfinite"
    opt = optax.lbfgs(memory_size=memory_size)
    vg = optax.value_and_grad_from_state(f)
    x, state = x0, opt.init(x0)
    trace, prev, small = [], f_best, 0
    for _ in range(int(steps)):
        value, grad = vg(x, state=state)
        if not (np.isfinite(float(value)) and bool(jnp.all(jnp.isfinite(grad)))):
            return x_best, f_best, trace, "nonfinite"
        updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=f)
        x = optax.apply_updates(x, updates)
        fx = float(optax.tree_utils.tree_get(state, "value"))
        if not (np.isfinite(fx) and bool(jnp.all(jnp.isfinite(x)))):
            return x_best, f_best, trace, "nonfinite"
        if fx >= prev:                       # rejected step: not recorded, best kept
            return x_best, f_best, trace, "linesearch"
        trace.append(fx)
        x_best, f_best = x, fx
        small = small + 1 if (prev - fx) <= tol * max(abs(prev), 1e-300) else 0
        prev = fx
        if small >= patience:
            return x_best, f_best, trace, "converged"
    return x_best, f_best, trace, "steps"


def theta_map_linear(prob, ds, W, *, steps=300, seed=0, init=None):
    """theta-MAP of the M = 0 LML for the model with radials W (one streaming
    pass for the statistics, then run_map on the cached Gram).  init: optional
    theta array to warm-start from.  Returns the theta array."""
    lin = linear_statistics(with_radial(prob.model, W), prob.cfg, ds)
    lml = jax.jit(lambda a: log_marginal_likelihood(from_array(a), lin, prob))
    h = run_map(lml, prob.prior, steps=steps, seed=seed,
                init=None if init is None else from_array(jnp.asarray(init)))
    return to_array(h)


@partial(jax.jit, static_argnames=("cfg",))
def _objective(V, a, model, ds, gamma, Q, active, D2, wn, lam, cfg):
    theta = from_array(a)
    W = normalise(V, Q, active)
    lin = linear_statistics(with_radial(model, W), cfg, ds)
    return projected_residual_from_stats(theta, lin, gamma) + lam * roughness(W, D2, wn)


def learn_radial(prob, ds, W0, *, theta0=None, profile=True, lam_rough=0.0, rough_weights=None,
                 steps=200, reprofile_every=10, tol=1e-6, patience=3, map_steps=300,
                 n_prior=10.0, seed=0):
    """VarPro-learn the tensor radials of prob.model (analytic branch, M = 0).

    Minimises  r(W; theta) + lam * roughness(W)  over V with W = normalise(V)
    (unit empirical norm per radial; rows that are zero in W0 stay zero), by
    L-BFGS in rounds of `reprofile_every` steps.  With profile=True theta is
    re-MAP'd on the M = 0 LML after every round (and at the start unless
    theta0 is given), and L-BFGS restarts because the objective changed.
    lam_rough is RELATIVE: lam = lam_rough * r(W0) / roughness(W0).  Stops at
    `steps` total or at the first round that ends early (converged, line
    search, non-finite).  Returns (W, info); steps=0 returns normalise(W0)."""
    require_x64()
    require_analytic(prob.model)
    W0 = jnp.asarray(W0, jnp.float64)
    active = row_active(W0)
    Q = radial_gram(prob.model, ds, n_prior=n_prior)
    D2 = roughness_matrix(prob.model)
    wn = jnp.ones(W0.shape[2]) if rough_weights is None else jnp.asarray(rough_weights, jnp.float64)
    V = normalise(W0, Q, active)
    if theta0 is not None:
        a = to_array(theta0)
    elif profile:
        a = theta_map_linear(prob, ds, V, steps=map_steps, seed=seed)
    else:
        raise ValueError("learn_radial: profile=False needs theta0")
    r0 = float(projected_residual_from_stats(
        from_array(a), linear_statistics(with_radial(prob.model, V), prob.cfg, ds), prob.gamma))
    rough0 = float(roughness(V, D2, wn))
    lam = float(lam_rough) * r0 / max(rough0, 1e-300) if lam_rough else 0.0
    info = {"trace": [], "reasons": [], "theta": [np.asarray(a)], "lam_abs": lam,
            "lam_rough": float(lam_rough), "steps": 0}
    done = 0
    while done < int(steps):
        n = min(int(reprofile_every), int(steps) - done)
        f = lambda X, a=a: _objective(X, a, prob.model, ds, prob.gamma, Q, active, D2, wn,
                                      lam, prob.cfg)
        V, _, trace, reason = lbfgs_loop(f, V, steps=n, tol=tol, patience=patience)
        info["trace"].extend(trace)
        info["reasons"].append(reason)
        done += n
        # Canonicalise once per round (lbfgs_loop moves V, not W = normalise(V));
        # persisting it back avoids a second normalise() at the return below,
        # which for steps=0 would perturb the already-normalised V0 by ~1 ULP
        # (rsqrt(1 + eps) != 1 bit-for-bit) and break exact-equality recovery
        # of normalise(W0) against an independently-computed reference.
        V = normalise(V, Q, active)
        if profile:
            a = theta_map_linear(prob, ds, V, steps=map_steps, seed=seed, init=a)
            info["theta"].append(np.asarray(a))
        if reason != "steps":
            break
    info["steps"] = done
    info["theta_final"] = np.asarray(a)
    return V, info
