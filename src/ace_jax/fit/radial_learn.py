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
See docs/dev/specs/2026-09-26-learned-radial-varpro-design.md.
"""
import itertools
import json
import pathlib
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.linalg import solve_triangular

from .hypers import from_array, log_prior, to_array
from .ladder import run_map
from .objective import combine, log_marginal_likelihood, posterior
from .radial_model import (data_r_range, gap_penalty, normalise, radial_gram, require_analytic,
                           roughness, roughness_matrix, row_active, spectral_penalty,
                           spectral_weights, uniform_gram, with_radial)
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


@partial(jax.jit, static_argnames=("f", "statics", "memory_size"))
def _lbfgs_step(x, state, args, *, f, statics, memory_size):
    """One optax L-BFGS + zoom-line-search step of f(y, *args, *statics).

    `f` and `statics` are static (baked into the compiled program; `f` must be
    a stable, hashable callable such as a module-level function, and `statics`
    a tuple of hashable objects), `x`, `state` and `args` are traced.  Reusing
    this single jitted step across rounds -- rather than re-jitting a fresh
    closure over the (theta, model, ...) each round -- means only a change in
    shape/dtype (never in value) of `x`/`state`/`args`, or in `f`/`statics`/
    `memory_size` themselves, triggers a recompile."""
    fa = lambda y: f(y, *args, *statics)
    opt = optax.lbfgs(memory_size=memory_size)
    value, grad = optax.value_and_grad_from_state(fa)(x, state=state)
    updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=fa)
    return optax.apply_updates(x, updates), state, value, grad


def lbfgs_loop(f, x0, *, steps, args=(), statics=(), tol=1e-6, patience=3, memory_size=10):
    """Minimise f(x, *args, *statics) -> scalar by optax L-BFGS with zoom line
    search, via the single compiled `_lbfgs_step` (no per-call closure is
    jitted here, so calling this repeatedly with the same `f`/`statics` and
    matching `x0`/`args` shapes -- e.g. once per VarPro round -- compiles
    once and reuses the executable thereafter).  `args` may change VALUE
    between calls (e.g. re-profiled theta) without forcing a recompile, since
    only their shape/dtype is baked into the trace; `f`, `statics` and
    `memory_size` must stay fixed (they are static, part of the cache key).

    Stops after `steps` iterations ("steps"); when the relative decrease stays
    below `tol` for `patience` consecutive iterations ("converged"); when an
    iteration fails to decrease f ("linesearch"); or when f, its gradient or
    the iterate go non-finite ("nonfinite").  Always returns the best finite
    iterate seen: (x_best, f_best, trace, reason), trace = f after each
    accepted step (strictly decreasing)."""
    if int(steps) <= 0:
        return x0, float(f(x0, *args, *statics)), [], "steps"
    opt = optax.lbfgs(memory_size=memory_size)
    # optax.lbfgs's init leaves a few ZoomLinesearchInfo fields weak-typed
    # (Python-literal defaults), while every _lbfgs_step call returns them
    # strongly typed; left alone, that mismatch is a second, permanent
    # abstract signature -- one extra (harmless but avoidable) compile the
    # first time this shape/dtype combination is ever seen.  Stripping it
    # up front means the very first iteration of the very first round
    # already matches every later call's signature: one compile, not two.
    state = jax.tree.map(lambda a: jax.lax.convert_element_type(a, a.dtype)
                         if hasattr(a, "dtype") else a, opt.init(x0))
    x = x0
    x_best, f_best, trace, prev, small = x0, None, [], None, 0
    for i in range(int(steps)):
        x_new, state, value, grad = _lbfgs_step(x, state, args, f=f, statics=statics,
                                                 memory_size=memory_size)
        value = float(value)               # f(x) at the pre-step x (f(x0) when i == 0)
        if i == 0:
            f_best = prev = value
        if not (np.isfinite(value) and bool(jnp.all(jnp.isfinite(grad)))):
            return x_best, f_best, trace, "nonfinite"
        fx = float(optax.tree_utils.tree_get(state, "value"))
        if not (np.isfinite(fx) and bool(jnp.all(jnp.isfinite(x_new)))):
            return x_best, f_best, trace, "nonfinite"
        if fx >= prev:                       # rejected step: not recorded, best kept
            return x_best, f_best, trace, "linesearch"
        x = x_new
        trace.append(fx)
        x_best, f_best = x, fx
        small = small + 1 if (prev - fx) <= tol * max(abs(prev), 1e-300) else 0
        prev = fx
        if small >= patience:
            return x_best, f_best, trace, "converged"
    return x_best, f_best, trace, "steps"


def theta_map_linear(prob, ds, W, *, steps=300, seed=0, init=None, return_stats=False, lin=None):
    """theta-MAP of the M = 0 LML for the model with radials W (one streaming
    pass for the statistics, then run_map on the cached Gram).  init: optional
    theta array to warm-start from.  `lin`: the statistics of ds if the caller
    already has them (then W is unused and may be None).  Returns the theta array; with
    return_stats=True returns (a, lin, diag): `lin` the linear statistics of
    ds it streamed (so a caller can reuse them without another pass) and
    `diag` a MAP convergence diagnostic computed on the cached `lin` (no extra
    pass): the final LML, the change in the SVI loss (-log posterior) over the
    last min(10, steps) steps, and the norm of the log-posterior gradient at
    the returned theta."""
    if lin is None:
        lin = linear_statistics(with_radial(prob.model, W), prob.cfg, ds)
    lml = jax.jit(lambda a: log_marginal_likelihood(from_array(a), lin, prob))
    h, losses = run_map(lml, prob.prior, steps=steps, seed=seed, return_losses=True,
                        init=None if init is None else from_array(jnp.asarray(init)))
    a = to_array(h)
    if not return_stats:
        return a
    logpost = lambda x: lml(x) + log_prior(from_array(x), prob.prior)
    k = min(10, len(losses) - 1)
    diag = {"lml": float(lml(a)), "map_steps": int(steps),
            "dloss_last": float(losses[-1] - losses[-1 - k]) if k > 0 else float("nan"),
            "grad_norm": float(jnp.linalg.norm(jax.grad(logpost)(a)))}
    return a, lin, diag


def _objective(V, a, model, ds, gamma, Q, active, D2, wn, lam, W_ref, sw, lam_spec, U, lam_gap, cfg):
    """The VarPro-plus-priors objective, module-level so it is a stable,
    hashable `f` for `lbfgs_loop`/`_lbfgs_step` (a fresh per-round closure
    over the same computation would be a distinct object each round and
    force a recompile of the L-BFGS step every round -- see `learn_radial`).
    `cfg` is meant to be passed through `lbfgs_loop`'s `statics`, not `args`;
    this function does not need to be jitted itself, since `_lbfgs_step` is
    the sole jit boundary and traces straight through it.  `W_ref`, `sw` and
    `lam_spec` (traced `args`, not `statics`) add the spectral prior on the
    change from the reference radials W_ref, more strongly at high degree.
    `U` and `lam_gap` (also traced `args`) add the data-gap prior: the same
    change W_ref, but measured under the uniform-in-r Gram U rather than
    per-degree weights, so it costs change in low-data gaps between
    coordination shells rather than change at high polynomial degree."""
    theta = from_array(a)
    W = normalise(V, Q, active)
    lin = linear_statistics(with_radial(model, W), cfg, ds)
    return (projected_residual_from_stats(theta, lin, gamma) + lam * roughness(W, D2, wn)
            + lam_spec * spectral_penalty(W, W_ref, sw) + lam_gap * gap_penalty(W, W_ref, U))


def require_linear(prob):
    if prob.ind.XM.shape[0] != 0:
        raise ValueError(f"learned radials support the linear model only (M = 0 inducing points), "
                         f"got M = {prob.ind.XM.shape[0]}; fit the residual GP afterwards")


def relative_lambda(lam_rough, r0, rough0):
    """Absolute roughness weight lam_rough * r0 / rough0 (0 for lam_rough = 0).
    Raises if the init is (numerically) perfectly smooth, where the relative
    weight would be effectively infinite."""
    if not lam_rough:
        return 0.0
    if rough0 < 1e-12 * max(abs(r0), 1.0):
        raise ValueError(f"relative lam_rough={lam_rough:g} is undefined: roughness of the init "
                         f"rough0={rough0:.3e} is ~0 relative to r0={r0:.3e}; use lam_rough=0")
    return float(lam_rough) * r0 / rough0


def relative_lambda_spec(lam_spec, r0, n_active):
    """Absolute spectral weight lam_spec * r0 / n_active (0 for lam_spec = 0),
    the analogue of `relative_lambda` for the spectral prior on the radial
    change: n_active is the number of active radials (row_active count), not
    a roughness scale.

    Scale: the weights (1+q)^p are large (~9e5 at q=30, p=4), so the useful
    window is narrow. Measured on si_tiny with n_q=30, p=4 and a 5%
    flat-spectrum change, the penalty/r0 ratio is about lam_spec * 5.5e3:
    1e-6 is negligible, 1e-5..3e-5 is comparable to lam_rough=1e-2, and 1e-4
    already dominates the fit. Sweep roughly 1e-6..1e-4, not the lam_rough
    grid. The window moves by about n_q^dp per unit change of p."""
    if lam_spec < 0:
        raise ValueError(f"lam_spec must be >= 0, got {lam_spec}")
    if not lam_spec:
        return 0.0
    if n_active <= 0:
        raise ValueError("spectral prior: no active radials in W0 (all rows zero)")
    return float(lam_spec) * r0 / n_active


def relative_lambda_gap(lam_gap, r0, n_active):
    """Absolute data-gap weight lam_gap * r0 / n_active (0 for lam_gap = 0),
    the analogue of `relative_lambda_spec` for the data-gap prior on the
    radial change: n_active is the number of active radials (row_active
    count), not a roughness or spectral scale.

    Scale, measured on the real learned change (n_q=12, lam_gap=0, 40 steps):
    gap_penalty(W_learned, W_ref, U) / n_active = 0.009 (SiGe) and 0.017
    (Cantor), so the penalty/r0 ratio is about lam_gap * 0.01-0.02. Learning
    gains ~0.5 r0 in the fit term, so lam_gap ~1 is weak, ~10 is comparable,
    and ~30 or more is strong. Sweep roughly 1..30. A synthetic iid-random 25%
    change on si_tiny gave 0.12; that overstates the scale about 10x
    because it spreads change over every degree. The weight is not comparable
    to lam_rough or lam_spec, which act on other quantities."""
    if lam_gap < 0:
        raise ValueError(f"lam_gap must be >= 0, got {lam_gap}")
    if not lam_gap:
        return 0.0
    if n_active <= 0:
        raise ValueError("gap prior: no active radials in W0 (all rows zero)")
    return float(lam_gap) * r0 / n_active


_I_SIGMA_E = None


def _learn_theta(a, mult):
    """theta array for the radial objective: log_sigma_E shifted by log(mult)."""
    global _I_SIGMA_E
    if mult == 1.0:
        return a
    if _I_SIGMA_E is None:
        from .hypers import Hypers
        _I_SIGMA_E = Hypers._fields.index("log_sigma_E")
    return jnp.asarray(a).at[_I_SIGMA_E].add(float(np.log(mult)))


def learn_radial(prob, ds, W0, *, theta0=None, profile=True, lam_rough=0.0, rough_weights=None,
                 lam_spec=0.0, spec_p=4.0, lam_gap=0.0, steps=200, reprofile_every=10, tol=1e-6,
                 patience=3, map_steps=300, n_prior=None, seed=0, log=None, Q=None, D2=None,
                 r0=None, U=None, learn_sigma_e_mult=1.0):
    """VarPro-learn the tensor radials of prob.model (analytic branch, M = 0).

    learn_sigma_e_mult scales sigma_E inside the radial objective only (after
    every theta re-MAP): > 1 learns the radials under a force-heavier weighting
    than the evidence picks. Everything downstream (gate, final linear fit) keeps
    the plain MAP theta. With a multiplier != 1, r0 (the reference scale of the
    relative priors) is taken at the scaled theta.

    Minimises  r(W; theta) + lam * roughness(W) + lam_spec_abs * spectral_penalty(W, W_ref)
              + lam_gap_abs * gap_penalty(W, W_ref, U)
    over V with W = normalise(V) (unit empirical norm per radial; rows that
    are zero in W0 stay zero), by L-BFGS in rounds of `reprofile_every`
    steps.  With profile=True theta is re-MAP'd on the M = 0 LML after every
    round (and at the start unless theta0 is given), and L-BFGS restarts
    because the objective changed.  lam_rough is RELATIVE: lam = lam_rough *
    r(W0) / roughness(W0).  lam_spec is also RELATIVE: lam_spec_abs =
    lam_spec * r0 / n_active, n_active = the number of active (row_active)
    radials; the spectral prior penalises the departure of the normalised W
    from W_ref = normalise(W0) (the starting V), weighted per Legendre degree
    q by spectral_weights(n_q, spec_p) = (1 + q)^spec_p, so it grows with
    degree -- the analogue of the Gamma smoothness prior's degree weighting,
    but on the CHANGE rather than the absolute radial.  lam_gap is also
    RELATIVE (relative_lambda_gap): lam_gap_abs = lam_gap * r0 / n_active; the
    data-gap prior penalises the same departure from W_ref, but measured under
    U = uniform_gram(prob.model, 0.8 * r_min, rcut) (r_min from
    data_r_range(ds), rcut = prob.cfg.rcut) -- a uniform-in-r measure, so it
    costs change equally everywhere on that range, including gaps between
    coordination shells where the data-weighted gauge Q costs nothing.  Stops
    at `steps` total or at the first round that ends early (converged, line
    search, non-finite).  Returns (W, info); steps=0 returns normalise(W0).

    `log`: optional one-arg callable given one-line progress strings (start
    hyperparameters, then a line per round); None (default) is silent and
    leaves all other behaviour unchanged.

    Precomputed pieces (fit_radial passes them so a lambda grid shares them;
    each is computed here when None): Q = radial_gram(prob.model, ds,
    n_prior) (n_prior None -> radial_gram's default), D2 =
    roughness_matrix(prob.model), r0 = r(normalise(W0); theta0), the
    projected residual at the start -- only valid together with that theta0 --
    and U = uniform_gram(...) as above (only computed, an extra streaming pass
    over ds via data_r_range, when lam_gap > 0; a zeros array of the right
    shape otherwise, so the default path adds no streaming passes and the
    traced arg structure of `_objective` stays fixed).  M > 0 (inducing
    points) is unsupported and raises."""
    require_x64()
    require_analytic(prob.model)
    require_linear(prob)
    W0 = jnp.asarray(W0, jnp.float64)
    active = row_active(W0)
    if Q is None:
        Q = radial_gram(prob.model, ds) if n_prior is None else radial_gram(prob.model, ds, n_prior=n_prior)
    if D2 is None:
        D2 = roughness_matrix(prob.model)
    if U is None:
        if lam_gap:
            r_min, _ = data_r_range(ds)
            U = uniform_gram(prob.model, 0.8 * r_min, prob.cfg.rcut)
        else:
            NZ, n_q = W0.shape[0], W0.shape[-1]
            U = jnp.zeros((NZ, NZ, n_q, n_q), jnp.float64)
    wn = jnp.ones(W0.shape[2]) if rough_weights is None else jnp.asarray(rough_weights, jnp.float64)
    V = normalise(W0, Q, active)
    W_ref = V                              # reference for the spectral/gap priors: the starting V
    sw = spectral_weights(W0.shape[-1], spec_p)
    lin0 = None
    if theta0 is not None:
        a = to_array(theta0)
    elif profile:
        a, lin0, _ = theta_map_linear(prob, ds, V, steps=map_steps, seed=seed, return_stats=True)
    else:
        raise ValueError("learn_radial: profile=False needs theta0")
    if learn_sigma_e_mult <= 0:
        raise ValueError(f"learn_sigma_e_mult must be > 0, got {learn_sigma_e_mult}")
    if r0 is None or learn_sigma_e_mult != 1.0:
        if lin0 is None:
            lin0 = linear_statistics(with_radial(prob.model, V), prob.cfg, ds)
        r0 = projected_residual_from_stats(from_array(_learn_theta(a, learn_sigma_e_mult)), lin0, prob.gamma)
    r0 = float(r0)
    rough0 = float(roughness(V, D2, wn))
    lam = relative_lambda(lam_rough, r0, rough0)
    n_active = int(jnp.sum(active))
    lam_spec_abs = relative_lambda_spec(lam_spec, r0, n_active)
    lam_gap_abs = relative_lambda_gap(lam_gap, r0, n_active)
    info = {"trace": [], "reasons": [], "theta": [np.asarray(a)], "lam_abs": lam,
            "lam_rough": float(lam_rough), "r0": r0, "rough0": rough0, "steps": 0,
            "round_lengths": [], "lam_spec": float(lam_spec), "lam_spec_abs": lam_spec_abs,
            "spec_p": float(spec_p), "lam_gap": float(lam_gap), "lam_gap_abs": lam_gap_abs,
            "learn_sigma_e_mult": float(learn_sigma_e_mult)}
    if log is not None:
        log(f"learn_radial: lam_rough={float(lam_rough):g} lam_abs={lam:.6e} "
            f"lam_spec={float(lam_spec):g} lam_spec_abs={lam_spec_abs:.6e} spec_p={float(spec_p):g} "
            f"lam_gap={float(lam_gap):g} lam_gap_abs={lam_gap_abs:.6e} "
            f"r0={r0:.6e} profile={bool(profile)} learn_sigma_e_mult={float(learn_sigma_e_mult):g}")
    lam = jnp.asarray(lam, jnp.float64)
    lam_spec_abs = jnp.asarray(lam_spec_abs, jnp.float64)
    lam_gap_abs = jnp.asarray(lam_gap_abs, jnp.float64)
    done = 0
    round_idx = 0
    while done < int(steps):
        round_idx += 1
        n = min(int(reprofile_every), int(steps) - done)
        t_round = time.perf_counter()
        # `_objective` (module-level) + fixed `statics=(prob.cfg,)` is the same
        # `f`/static pair every round, so `_lbfgs_step` compiles once across
        # the whole learn_radial call and every round below just reuses it;
        # `args` change VALUE each round (a is re-profiled) but not shape/dtype.
        V, f_best, trace, reason = lbfgs_loop(
            _objective, V, steps=n, tol=tol, patience=patience,
            args=(_learn_theta(a, learn_sigma_e_mult), prob.model, ds, prob.gamma, Q, active, D2, wn, lam,
                  W_ref, sw, lam_spec_abs,
                  U, lam_gap_abs),
            statics=(prob.cfg,))
        info["trace"].extend(trace)
        info["reasons"].append(reason)
        info["round_lengths"].append(len(trace))
        done += n
        # Canonicalise once per round (lbfgs_loop moves V, not W = normalise(V));
        # persisting it back avoids a second normalise() at the return below,
        # which for steps=0 would perturb the already-normalised V0 by ~1 ULP
        # (rsqrt(1 + eps) != 1 bit-for-bit) and break exact-equality recovery
        # of normalise(W0) against an independently-computed reference.
        V = normalise(V, Q, active)
        profile_dt = 0.0
        if profile:
            t_profile = time.perf_counter()
            a = theta_map_linear(prob, ds, V, steps=map_steps, seed=seed, init=a)
            profile_dt = time.perf_counter() - t_profile
            info["theta"].append(np.asarray(a))
        # round_dt is measured after the (optional) re-profile above, so it
        # covers the whole round -- theta_map_linear does its own full
        # streaming pass and must not be left out of the wall time.
        round_dt = time.perf_counter() - t_round
        if log is not None:
            theta_msg = ""
            if profile:
                th = from_array(a)
                theta_msg = (f" log_sigma_c={float(th.log_sigma_c):.4f} "
                             f"log_sigma_E={float(th.log_sigma_E):.4f} "
                             f"log_sigma_F={float(th.log_sigma_F):.4f}")
            eta_msg = ""
            if round_idx == 1 and done < int(steps):
                # projected from round 1 (includes compile, so conservative)
                eta = round_dt * (int(steps) - done) / n
                eta_msg = f" eta={eta:.0f}s (this lam, from round 1)"
            log(f"learn_radial: round {round_idx} steps={done}/{int(steps)} "
                f"accepted={len(trace)} obj={f_best:.6e} reason={reason} "
                f"time={round_dt:.1f}s profile_time={profile_dt:.1f}s" + theta_msg + eta_msg)
        if reason != "steps":
            break
    info["steps"] = done
    info["theta_final"] = np.asarray(a)
    return V, info


def _val_score(c, val, a_norm):
    """sum_{t in E, F} SSE_t / (n_t sigma_t^2) of readout c on the validation
    statistics `val`, sigma from a_norm; SSE_t = yy - 2 c.b + c.G.c.  A type
    with no rows is skipped."""
    th = from_array(a_norm)
    score = 0.0
    for t in "EF":
        n = float(getattr(val, f"n_{t}"))
        if n == 0:
            continue
        G, b, yy = getattr(val, f"G_{t}"), getattr(val, f"b_{t}"), getattr(val, f"yy_{t}")
        sse = float(yy - 2.0 * c @ b + c @ G @ c)
        score += sse / (n * float(jnp.exp(2.0 * getattr(th, f"log_sigma_{t}"))))
    return score


def holdout_score(W, a_fit, a_norm, prob, ds_fit, ds_val, *, lin_fit=None, lin_val=None,
                  return_readout=False):
    """Validation error of the M = 0 linear fit with radials W: the readout c is
    the posterior mean on ds_fit at theta a_fit; the score is
    sum_{t in E, F} SSE_t / (n_t sigma_t^2) on ds_val with sigma from a_norm
    (fixed across candidates so scores are comparable), from ds_val's weighted
    linear statistics -- no design matrix.  A type with no rows in ds_val is
    skipped.  lin_fit / lin_val: the statistics of ds_fit / ds_val at W if the
    caller already has them (each saves one streaming pass).  With
    return_readout=True returns (score, c); W may be None when both statistics are given."""
    if lin_fit is None or lin_val is None:
        model = with_radial(prob.model, W)
    if lin_fit is None:
        lin_fit = linear_statistics(model, prob.cfg, ds_fit)
    if lin_val is None:
        lin_val = linear_statistics(model, prob.cfg, ds_val)
    c, _ = posterior(from_array(a_fit), lin_fit, prob)
    score = _val_score(c, lin_val, a_norm)
    return (score, c) if return_readout else score


def gate(candidates, score):
    """Score every candidate (lower is better) as score(label, W); ties resolve
    to insertion order, so put the conservative choice ("init") first.
    Returns (label, scores)."""
    scores = {k: float(score(k, w)) for k, w in candidates.items()}
    order = list(scores)
    return min(order, key=lambda k: (scores[k], order.index(k))), scores


def fit_radial(prob, ds_fit, ds_val, W0, *, lam_grid=(0.0, 1e-3, 1e-2, 1e-1), spec_grid=(0.0,),
               gap_grid=(0.0,), theta0=None, map_steps=300, log=None, checkpoint=None, **learn_kw):
    """learn_radial on ds_fit once per (roughness, spectral, data-gap weight)
    triple in lam_grid x spec_grid x gap_grid, then keep the best of {init,
    learned per triple} on the disjoint ds_val (ties -> init).

    Candidate labels are "learned_lam=<l>" with a "_spec=<s>" suffix appended
    when spec_grid has more than one value, and a "_gap=<g>" suffix appended
    when gap_grid has more than one value (each independently; backward
    compatible with plain "learned_lam=<l>" when both grids are single-valued,
    an implicit spec=0/gap=0); checkpoints use the same scheme without the
    "learned_" prefix (lam_label = f"{l:g}", plus "_spec=..."/"_gap=..." under
    the same conditions).

    Every candidate, init included, is scored by ONE procedure: a_fit =
    theta_map_linear(ds_fit, W, map_steps, init=a0), readout = posterior mean
    on ds_fit at a_fit, score on ds_val with sigma from a0 (a0 = theta0, or the
    theta-MAP at the normalised init).  So with steps=0 (learned == init) the
    scores are identical and the tie goes to init.  Also recorded:
    info["scores_at_a0"] (every candidate's readout at the common a0, no theta
    optimisation involved), info["map_diag"] (per-candidate MAP convergence
    diagnostic, see theta_map_linear), info["theta_fit"] and info["readout"]
    (the SELECTED candidate's M = 0 readout on ds_fit at its a_fit, length
    len_basis, species-blocked as fit.rows._place).  Each candidate costs one
    ds_fit pass plus one ds_val pass.  Returns (W_sel, info).

    `log`: optional one-arg callable given one-line progress strings (forwarded
    to `learn_radial`, plus fit_radial's own per-candidate and gate-score
    lines); None (default) is silent and leaves all other behaviour unchanged.

    `checkpoint`: optional callable checkpoint(lam_label, W, run_info), called
    as soon as each (lam, spec, gap) triple's learn_radial finishes, so an
    interrupted grid keeps its finished runs (e.g. save_result without
    src_npz).  Q, D2, U and the relative-lambda/spec/gap reference are
    computed once here and shared by every candidate (U only when some
    gap_grid value is > 0, via one extra streaming pass for data_r_range); M >
    0 raises."""
    require_x64()
    require_analytic(prob.model)
    require_linear(prob)
    multi_spec = len(spec_grid) > 1
    multi_gap = len(gap_grid) > 1
    combos = list(itertools.product(lam_grid, spec_grid, gap_grid))

    def cand_label(lam, spec, gap):
        label = f"learned_lam={float(lam):g}"
        if multi_spec:
            label += f"_spec={float(spec):g}"
        if multi_gap:
            label += f"_gap={float(gap):g}"
        return label

    def run_key(lam, spec, gap):
        key = f"{float(lam):g}"
        if multi_spec:
            key += f"_spec={float(spec):g}"
        if multi_gap:
            key += f"_gap={float(gap):g}"
        return key

    labels = [cand_label(*c) for c in combos]
    if len(set(labels)) != len(labels):
        raise ValueError(f"fit_radial: duplicate combinations in lam_grid x spec_grid x gap_grid {combos}")
    W0 = jnp.asarray(W0, jnp.float64)
    # shared by every candidate: the gauge Gram, the roughness matrix, the
    # uniform-in-r Gram (if needed), and the relative-lambda reference
    # r0 = r(W_init; a0) (all runs start there)
    n_prior = learn_kw.pop("n_prior", None)
    Q = radial_gram(prob.model, ds_fit) if n_prior is None else radial_gram(prob.model, ds_fit, n_prior=n_prior)
    D2 = roughness_matrix(prob.model)
    U = None
    if any(gap_grid):
        r_min, _ = data_r_range(ds_fit)
        U = uniform_gram(prob.model, 0.8 * r_min, prob.cfg.rcut)
    W_init = normalise(W0, Q, row_active(W0))
    if theta0 is not None:
        a0 = to_array(theta0)
        lin0 = linear_statistics(with_radial(prob.model, W_init), prob.cfg, ds_fit)
    else:
        a0, lin0, _ = theta_map_linear(prob, ds_fit, W_init, steps=map_steps, return_stats=True)
    r0 = float(projected_residual_from_stats(from_array(a0), lin0, prob.gamma))
    rw = learn_kw.get("rough_weights")
    rough0 = float(roughness(W_init, D2, jnp.ones(W0.shape[2]) if rw is None
                             else jnp.asarray(rw, jnp.float64)))
    for lam in lam_grid:
        relative_lambda(lam, r0, rough0)          # fail before any learning, not mid-grid
    cands, runs = {"init": W_init}, {}
    for lam, spec, gap in combos:
        label, key = cand_label(lam, spec, gap), run_key(lam, spec, gap)
        if log is not None:
            log(f"fit_radial: lam={lam:g} spec={spec:g} gap={gap:g} starting")
        W, info = learn_radial(prob, ds_fit, W0, theta0=from_array(a0), lam_rough=lam,
                               lam_spec=spec, lam_gap=gap, map_steps=map_steps, log=log, Q=Q,
                               D2=D2, r0=r0, U=U, **learn_kw)
        cands[label] = W
        runs[key] = info
        if checkpoint is not None:
            checkpoint(key, W, info)

    at_a0, theta_fit, map_diag, readouts = {}, {}, {}, {}

    def score(label, W):
        a_fit, lin_fit, diag = theta_map_linear(prob, ds_fit, W, steps=map_steps, init=a0,
                                                return_stats=True)
        lin_val = linear_statistics(with_radial(prob.model, W), prob.cfg, ds_val)
        s, c = holdout_score(W, a_fit, a0, prob, ds_fit, ds_val, lin_fit=lin_fit,
                             lin_val=lin_val, return_readout=True)
        at_a0[label] = holdout_score(W, a0, a0, prob, ds_fit, ds_val, lin_fit=lin_fit,
                                     lin_val=lin_val)
        theta_fit[label], map_diag[label], readouts[label] = np.asarray(a_fit), diag, np.asarray(c)
        if log is not None:
            log(f"fit_radial: gate {label} score={s:.6e} score_at_a0={at_a0[label]:.6e} "
                f"map_dloss_last={diag['dloss_last']:.3e} map_grad_norm={diag['grad_norm']:.3e}")
        return s

    label, scores = gate(cands, score)
    if log is not None:
        log(f"fit_radial: selected {label}")
    return cands[label], {"selected": label, "scores": scores, "scores_at_a0": at_a0,
                          "map_diag": map_diag, "theta_fit": theta_fit, "runs": runs,
                          "theta_init": np.asarray(a0), "readout": readouts[label]}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def save_result(out_dir, W, info, *, src_npz=None, model=None, readout=None):
    """Write rnl_Wnlq.npy and radial_info.json to out_dir.  With src_npz and
    model (the analytic model W belongs to, e.g. after widen_radial /
    to_analytic) also write model.npz = src_npz with the learned radial AND
    the readout fitted for it (`readout`, default info["readout"] as returned
    by fit_radial; also saved as readout.npy).  The source npz's WB/Wpair
    belong to the old radials, so writing model.npz without a readout is
    refused rather than silently stale.  info["readout"] is kept out of the
    JSON (it is len_basis long).  model.npz is marked meta "radial_learned"
    (so `lean` splines it by default) unless info["selected"] == "init"."""
    if readout is None:
        readout = info.get("readout")
    if src_npz is not None:
        if model is None:
            raise ValueError("save_result: src_npz needs the analytic `model` W belongs to")
        if readout is None:
            raise ValueError("save_result: writing model.npz needs the readout fitted for W "
                             "(fit_radial's info['readout']); the source WB/Wpair are stale")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rnl_Wnlq.npy", np.asarray(W))
    (out / "radial_info.json").write_text(
        json.dumps(_jsonable({k: v for k, v in info.items() if k != "readout"}), indent=1))
    if readout is not None:
        np.save(out / "readout.npy", np.asarray(readout))
    if src_npz is not None:
        from ..basis.export import patch_radial_npz
        # the held-out gate may keep the initial radial: then nothing was learned
        learned = info.get("selected") != "init"
        patch_radial_npz(src_npz, out / "model.npz", with_radial(model, W, learned=learned),
                         readout=readout)
