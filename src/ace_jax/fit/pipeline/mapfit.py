import time
import warnings
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ..hypers import Hypers, from_array, to_array
from ..ladder import run_map
from ..multistart import multistart_map, prior_starts
from ..newton import _evidence_noise, newton_polish

# log-space boxes: generous, but keep the Cholesky away from sigma -> 0; A up to
# 1e3 (the full/PCA-descriptor GP sat at the old bound of 10, Cantor-1k)
LBFGS_LO = np.log([0.05, 1e-3, 0.1, 1.5, 1e-3, 0.1, 1e-2, 1e-4, 1e-4, 1e-4])
LBFGS_HI = np.log([50.0, 1e3, 50.0, 4.0, 50.0, 100.0, 1e4, 10.0, 10.0, 10.0])


# Stationarity of the MAP: the largest component of the (bound-projected) log-posterior gradient,
# in nats per unit of a log-hyperparameter, must be within MAP_GTOL -- or within 10x its own
# measured roundoff, which exceeds MAP_GTOL on large data (5e-3 at 4.6e5 force rows, GAP-18 Si)
MAP_GTOL = 1e-2


class MapNotConverged(RuntimeError):
    """The MAP ended away from a stationary point (raised under `strict`; otherwise a warning)."""


class MapFit(NamedTuple):
    theta: object; restarts: object; sigma_type_ratios: object; timings: dict
    log_evidence: object = None          # log marginal likelihood at theta (no hyperprior); None for sigma_type
    convergence: object = None           # dict: optimiser, polish, logpost, pgrad_inf, gnoise_max, gtol, converged


def _log_evidence(obj, theta):
    """The LML at theta, comparable across bases on the same data (the logged L-BFGS
    'logpost' adds the hyperprior, so it is not used)."""
    return float(obj.lik(to_array(theta)))


def _bound_gradient(x, g, lo, hi):
    """The gradient g of the maximised log-posterior with the components zeroed that push x out of
    the box at a bound it sits on (the KKT residual of a bounded maximum; lo/hi None: unbounded)."""
    if lo is None:
        return g
    return np.where(((x <= lo) & (g < 0)) | ((x >= hi) & (g > 0)), 0.0, g)


class _LogPosterior:
    """The `newton.newton_polish` interface over the MAP objective: value_and_grad is the compiled
    log-posterior gradient, the Hessian the exact `jax.hessian` of the log-posterior.  On the linear
    arm (GAP-18 Si, o4d12) it compiled in 3.9 s and evaluates in 0.05 s against a 4 ms gradient;
    through the GP arm's streamed scan it costs far more (see `ladder.run_laplace_fd` and
    FitConfig.map_polish), so 'auto' polishes the linear arm only."""

    def __init__(self, vg_host, logpost):
        import jax
        self.vg, self._hess = vg_host, jax.jit(jax.hessian(logpost))

    def value_and_grad(self, x):
        return self.vg(x)

    def hessian(self, x):
        H = np.asarray(self._hess(jnp.asarray(x)), float)
        return 0.5 * (H + H.T)


def polish_wanted(cfg):
    """map_polish 'auto' polishes the linear arm only (see FitConfig.map_polish)."""
    return cfg.map_polish == "on" or (cfg.map_polish == "auto" and cfg.arm == "linear")


def check_stationary(vg_host, x, lo=None, hi=None, *, strict=False, log=print, what="MAP", info=None):
    """Measure how stationary the MAP is and say so: warn (or raise MapNotConverged when strict)
    if the largest bound-projected gradient component exceeds max(MAP_GTOL, 10 x its roundoff).
    The roundoff comes from four probes at x (`newton._evidence_noise`), so the test is
    meaningful at any data size; `info` (a polish record) supplies it when already measured, and a
    polish that converged by its own rule (no Newton gain resolvable above roundoff) counts too."""
    v, g = vg_host(x)
    gb = _bound_gradient(x, g, lo, hi)
    if info is not None and "gnoise" in info:
        gnoise = np.asarray(info["gnoise"], float)
    else:
        _, gnoise = _evidence_noise(lambda z: tuple(-np.asarray(t) for t in vg_host(z)), x, -v, -g,
                                    np.zeros((x.size, x.size)))
    tol = max(MAP_GTOL, 10 * float(gnoise.max()))
    pgi = float(np.abs(gb).max())
    worst = Hypers._fields[int(np.argmax(np.abs(gb)))]
    # or the polish found no resolvable Newton gain (decrement^2 within 10x the logpost's measured
    # roundoff): on an ill-conditioned optimum the gradient roundoff probes understate the noise
    floor = bool(info is not None and info.get("converged"))
    rec = {"logpost": float(v), "pgrad_inf": pgi, "pgrad_worst": worst, "gnoise_max": float(gnoise.max()),
           "gtol": tol, "converged": bool(np.isfinite(v) and (pgi <= tol or floor))}
    if rec["converged"]:
        why = f"|grad|_inf {pgi:.2e} <= {tol:.1e}" if pgi <= tol else f"Newton polish: {info['message']}"
        log(f"{what}: stationary ({why}), logpost {float(v):.10g}")
        return rec
    msg = (f"{what} did not converge: |d logpost / d theta|_inf = {pgi:.3g} (at {worst}) > tolerance "
           f"{tol:.2g}, logpost {float(v):.10g}. The hyperparameters, and every prediction made with "
           f"them, are not at the optimum: use --opt lbfgs (the default) and/or raise --map-steps")
    if strict:
        raise MapNotConverged(msg)
    log("WARNING: " + msg)
    warnings.warn(msg, UserWarning, stacklevel=2)
    return rec


def _rho_fix(cfg, prob):
    if cfg.fix_rho == "auto":
        XM = np.asarray(prob.ind.XM)
        d2 = ((XM[:, None, :] - XM[None, :, :]) ** 2).mean(-1)
        np.fill_diagonal(d2, np.inf)
        return float(np.median(np.sqrt(d2.min(1))))
    return float(cfg.fix_rho)


def fit_map(cfg, d, b, obj, log=print):
    prob, t = b.prob, time.time()
    init = None if cfg.init is None else Hypers(**cfg.init)
    if cfg.sigma_type:
        from ..ladder import run_map_ps
        from ..paramset import build_fit_paramset
        n_types = int(np.asarray(d.ds_train.cfg_type).max()) + 1
        ps0 = build_fit_paramset(init or prob.prior.mu, prob.prior, n_types=n_types,
                                 sigma_type=True, route=cfg.route)
        ps = run_map_ps(ps0, prob, d.ds_train, steps=cfg.map_steps, lr=cfg.map_lr, seed=cfg.seed)
        ratios = ps.sigma_type_ratios()
        return MapFit(from_array(ps.block("hypers").value), None,
                      None if ratios is None else np.asarray(ratios), {"map": time.time() - t})
    def vg_host(x):
        v, g = obj.vg(jnp.asarray(x)); g.block_until_ready()
        return float(v), np.asarray(g, float)
    if cfg.opt == "adam":
        theta = run_map(obj.lik, prob.prior, steps=cfg.map_steps, lr=cfg.map_lr, seed=cfg.seed, init=init)
        conv = check_stationary(vg_host, np.array(to_array(theta), float), strict=cfg.strict, log=log)
        return MapFit(theta, None, None, {"map": time.time() - t}, _log_evidence(obj, theta),
                      {"optimiser": "adam", "polish": None, **conv})
    x0 = np.array(to_array(init or prob.prior.mu), float)       # a copy: fix_rho writes into it
    lo, hi = LBFGS_LO.copy(), LBFGS_HI.copy()
    if cfg.fix_rho is not None:
        lo[5] = hi[5] = x0[5] = np.log(_rho_fix(cfg, prob))
        log(f"fix-rho: rho pinned at {float(np.exp(lo[5])):.4f}")
    tick = [time.time()]
    def log_eval(k, i, v):
        log(f"  lbfgs start {k} eval {i}  logpost = {v:.6g}  ({time.time() - tick[0]:.0f} s)")
        tick[0] = time.time()
    starts = prior_starts(prob.prior, cfg.map_restarts, lo, hi, x0, seed=cfg.seed)
    best, runs = multistart_map(vg_host, starts, lo, hi, cfg.map_steps, log=log_eval)
    for r in runs:
        log(f"L-BFGS start {r['start']}: logpost {r['value']:.6g}  nfev {r['nfev']}  {r['message']}")
    restarts = [{"start": r["start"], "logpost": r["value"], "nfev": r["nfev"], "message": r["message"],
                 "x0": r["x0"].tolist(), "x": r["x"].tolist()} for r in runs]
    log(f"L-BFGS: best of {len(runs)} start(s) = start {best['start']}, logpost {best['value']:.6g}")
    x, pol = np.asarray(best["x"], float), None
    if polish_wanted(cfg):
        tp = time.time()
        from ..hypers import log_prior
        logpost = lambda a: obj.lik(a) + log_prior(from_array(a), prob.prior)   # noqa: E731
        x, pol = newton_polish(_LogPosterior(vg_host, logpost), x, lo, hi)
        log(f"Newton polish: {pol['message']}; {pol['steps']} step(s), {pol['hessian_evals']} Hessian(s), "
            f"{time.time() - tp:.1f} s")
    theta = Hypers(*[float(v) for v in x])
    conv = check_stationary(vg_host, x, lo, hi, strict=cfg.strict, log=log, info=pol)
    pol_rec = None if pol is None else {k: v for k, v in pol.items() if k != "gnoise"}
    return MapFit(theta, restarts, None, {"map": time.time() - t}, _log_evidence(obj, theta),
                  {"optimiser": "lbfgs", "nfev": int(sum(r["nfev"] for r in runs)), "lbfgs_message": best["message"],
                   "polish": pol_rec, **conv})
