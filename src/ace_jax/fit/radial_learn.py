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


def theta_map_linear(prob, ds, W, *, steps=300, seed=0, init=None, return_stats=False):
    """theta-MAP of the M = 0 LML for the model with radials W (one streaming
    pass for the statistics, then run_map on the cached Gram).  init: optional
    theta array to warm-start from.  Returns the theta array; with
    return_stats=True returns (a, lin, diag): `lin` the linear statistics of
    ds it streamed (so a caller can reuse them without another pass) and
    `diag` a MAP convergence diagnostic computed on the cached `lin` (no extra
    pass): the final LML, the change in the SVI loss (-log posterior) over the
    last min(10, steps) steps, and the norm of the log-posterior gradient at
    the returned theta."""
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


def _objective(V, a, model, ds, gamma, Q, active, D2, wn, lam, cfg):
    """The VarPro-plus-roughness objective, module-level so it is a stable,
    hashable `f` for `lbfgs_loop`/`_lbfgs_step` (a fresh per-round closure
    over the same computation would be a distinct object each round and
    force a recompile of the L-BFGS step every round -- see `learn_radial`).
    `cfg` is meant to be passed through `lbfgs_loop`'s `statics`, not `args`;
    this function does not need to be jitted itself, since `_lbfgs_step` is
    the sole jit boundary and traces straight through it."""
    theta = from_array(a)
    W = normalise(V, Q, active)
    lin = linear_statistics(with_radial(model, W), cfg, ds)
    return projected_residual_from_stats(theta, lin, gamma) + lam * roughness(W, D2, wn)


def learn_radial(prob, ds, W0, *, theta0=None, profile=True, lam_rough=0.0, rough_weights=None,
                 steps=200, reprofile_every=10, tol=1e-6, patience=3, map_steps=300,
                 n_prior=10.0, seed=0, log=None):
    """VarPro-learn the tensor radials of prob.model (analytic branch, M = 0).

    Minimises  r(W; theta) + lam * roughness(W)  over V with W = normalise(V)
    (unit empirical norm per radial; rows that are zero in W0 stay zero), by
    L-BFGS in rounds of `reprofile_every` steps.  With profile=True theta is
    re-MAP'd on the M = 0 LML after every round (and at the start unless
    theta0 is given), and L-BFGS restarts because the objective changed.
    lam_rough is RELATIVE: lam = lam_rough * r(W0) / roughness(W0).  Stops at
    `steps` total or at the first round that ends early (converged, line
    search, non-finite).  Returns (W, info); steps=0 returns normalise(W0).

    `log`: optional one-arg callable given one-line progress strings (start
    hyperparameters, then a line per round); None (default) is silent and
    leaves all other behaviour unchanged."""
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
            "lam_rough": float(lam_rough), "steps": 0, "round_lengths": []}
    if log is not None:
        log(f"learn_radial: lam_rough={float(lam_rough):g} lam_abs={lam:.6e} "
            f"r0={r0:.6e} profile={bool(profile)}")
    lam = jnp.asarray(lam, jnp.float64)
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
            args=(a, prob.model, ds, prob.gamma, Q, active, D2, wn, lam),
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
            log(f"learn_radial: round {round_idx} steps={done}/{int(steps)} "
                f"accepted={len(trace)} obj={f_best:.6e} reason={reason} "
                f"time={round_dt:.1f}s profile_time={profile_dt:.1f}s" + theta_msg)
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
    return_readout=True returns (score, c)."""
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


def fit_radial(prob, ds_fit, ds_val, W0, *, lam_grid=(0.0, 1e-3, 1e-2, 1e-1), theta0=None,
               map_steps=300, log=None, **learn_kw):
    """learn_radial on ds_fit once per relative roughness weight in lam_grid,
    then keep the best of {init, learned per lam} on the disjoint ds_val
    (ties -> init).

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
    to `learn_radial`, plus fit_radial's own per-lambda and gate-score lines);
    None (default) is silent and leaves all other behaviour unchanged."""
    require_x64()
    require_analytic(prob.model)
    labels = [f"learned_lam={float(lam):g}" for lam in lam_grid]
    if len(set(labels)) != len(labels):
        raise ValueError(f"fit_radial: duplicate values in lam_grid {tuple(lam_grid)}")
    W0 = jnp.asarray(W0, jnp.float64)
    Q = radial_gram(prob.model, ds_fit, n_prior=learn_kw.get("n_prior", 10.0))
    W_init = normalise(W0, Q, row_active(W0))
    a0 = to_array(theta0) if theta0 is not None else theta_map_linear(prob, ds_fit, W_init, steps=map_steps)
    cands, runs = {"init": W_init}, {}
    for lam, label in zip(lam_grid, labels):
        if log is not None:
            log(f"fit_radial: lam={lam:g} starting")
        W, info = learn_radial(prob, ds_fit, W0, theta0=from_array(a0), lam_rough=lam,
                               map_steps=map_steps, log=log, **learn_kw)
        cands[label] = W
        runs[f"{float(lam):g}"] = info

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


def save_result(out_dir, W, info, *, src_npz=None, model=None):
    """Write rnl_Wnlq.npy and radial_info.json to out_dir; with src_npz and
    model (the analytic model W belongs to, e.g. after widen_radial /
    to_analytic) also write model.npz = src_npz with the learned radial."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rnl_Wnlq.npy", np.asarray(W))
    (out / "radial_info.json").write_text(json.dumps(_jsonable(info), indent=1))
    if src_npz is not None:
        if model is None:
            raise ValueError("save_result: src_npz needs the analytic `model` W belongs to")
        from ..construct.export import patch_radial_npz
        patch_radial_npz(src_npz, out / "model.npz", with_radial(model, W))
