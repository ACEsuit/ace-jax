import time
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..hypers import Hypers, from_array, log_prior, to_array
from ..ladder import run_map
from ..multistart import lbfgs_map, multistart_map, prior_starts
from ..newton import _newton_step, newton_polish
from ..paramset import SHARED_NOISE, tie_noise

# log-space boxes: generous, but keep the Cholesky away from sigma -> 0; A up to
# 1e3 (the full/PCA-descriptor GP sat at the old bound of 10, Cantor-1k)
LBFGS_LO = np.log([0.05, 1e-3, 0.1, 1.5, 1e-3, 0.1, 1e-2, 1e-4, 1e-4, 1e-4])
LBFGS_HI = np.log([50.0, 1e3, 50.0, 4.0, 50.0, 100.0, 1e4, 10.0, 10.0, 10.0])

# Stationarity of the MAP is judged by the PREDICTED GAIN of one more (box-constrained) Newton step,
# 1/2 g^T B^-1 g in nats over the free coordinates -- scale-free, unlike a gradient norm (a gradient
# of 3 nats per log-unit was a 1e-5-nat gain at 4.6e5 force rows, GAP-18 Si).  B is the polish
# Hessian when there is one (exact), else L-BFGS-B's limited-memory inverse-Hessian estimate (an
# ESTIMATE: on the GP arm the gain is approximate).  The MAP counts as stationary when the gain is
# <= MAP_GAIN_TOL.  A polished MAP may also pass on the roundoff floor -- only when every free
# gradient component is within 10x its own measured roundoff gnoise_i, and with the gain within the
# gain that gradient noise alone explains, 1/2 (10 gnoise)^T |H|^-1 (10 gnoise), capped at
# MAP_FLOOR_CAP -- recorded as `resolution_limited`.
MAP_GAIN_TOL = 1e-3
MAP_FLOOR_CAP = 0.1      # nats: the most the roundoff floor may excuse
ADAM_GTOL = 1e-2         # Adam on a path with no cheap Hessian: |grad|_inf, nats per log-unit
RESTART_EVALS = 50       # L-BFGS-B evaluations (and iterations) after a line-search ('ABNORMAL') stop
# the kernel hyperparameters: on the linear arm the LML never reads them, only their hyperprior does
KERNEL = np.isin(Hypers._fields, ("log_ell", "log_A", "log_alpha", "log_r0", "log_eps", "log_rho"))


class MapNotConverged(RuntimeError):
    """The MAP ended away from a stationary point (raised under `strict`; otherwise a warning)."""


class MapFit(NamedTuple):
    theta: object; restarts: object; sigma_type_ratios: object; timings: dict
    log_evidence: object = None          # log marginal likelihood at theta (no hyperprior); None for sigma_type
    convergence: object = None           # dict: optimiser, restart, polish, logpost, gain, converged, ...
    fixed: object = None                 # (10,) bool: hyperparameters held at a bound (fix_rho, or bound-active
                                         # at the MAP); the Laplace rung holds them fixed


def _log_evidence(obj, theta):
    """The LML at theta, comparable across bases and across noise modes on the same data (the logged
    L-BFGS 'logpost' adds the hyperprior, which shared noise counts for one noise coordinate, not
    three, so it is not used).  Written to map_convergence.json as `log_evidence`."""
    return float(obj.lik(to_array(theta)))


def _free(x, g, lo, hi):
    """Coordinates the MAP is free in: not pinned (lo == hi) and not bound-active (on a bound with
    the log-posterior gradient g pushing outward).  lo None: unbounded, all free."""
    if lo is None:
        return np.ones(x.size, bool)
    return ~((lo >= hi) | ((x <= lo) & (g <= 0)) | ((x >= hi) & (g >= 0)))


def _bound_active(x, lo, hi):
    return (lo >= hi) | np.isclose(x, lo, rtol=0, atol=1e-10) | np.isclose(x, hi, rtol=0, atol=1e-10)


class _HVPLogPosterior:
    """The `newton.newton_polish` interface over the MAP objective: value_and_grad is the compiled
    log-posterior gradient; the Hessian is EXACT, built one column at a time over the free
    coordinates by a compiled Hessian-vector product (forward-over-reverse, `jax.jvp` of `jax.grad`,
    one compile reused per column).  One column needs ~2.3x the gradient's temporaries (16 against
    7 P^2 doubles at readout length P = 2,053), where `jax.hessian` pushes all 10 tangents at once
    (69 P^2, ~10x).  Pinned and bound-active coordinates are not differentiated: decoupled with unit
    curvature, the box-constrained Newton step leaves them on their bounds.  A finite-difference
    Hessian (step 1e-4) was tried first: its error ~gnoise/step swamped the weakly curved directions
    and the polish stopped 2.9 nats short (12 Si configs).

    prior_only (10,) bool, with logprior: coordinates the likelihood never reads (the kernel
    hyperparameters on the linear arm).  Their columns are the hyperprior's alone, from a compiled HVP
    of `logprior` (microseconds): the likelihood's part of those columns is exactly zero, and a full
    column costs ~4 gradients (~56 s at o4d20 on an RTX 4000 Ada, against 16 s for a gradient).
    n_hvp counts the full columns only."""

    def __init__(self, vg_host, lo, hi, logpost, prior_only=None, logprior=None):
        self.vg, self.lo, self.hi, self.n_eval, self.n_hvp = vg_host, lo, hi, 0, 0
        self._hvp = jax.jit(lambda a, v: jax.jvp(jax.grad(logpost), (a,), (v,))[1])
        self.prior_only = None if prior_only is None or logprior is None else np.asarray(prior_only, bool)
        self._phvp = None if self.prior_only is None else \
            jax.jit(lambda a, v: jax.jvp(jax.grad(logprior), (a,), (v,))[1])
        self._last = (None, None)

    def value_and_grad(self, x):
        x = np.asarray(x, float)
        if self._last[0] is not None and np.array_equal(self._last[0], x):
            return self._last[1]
        self.n_eval += 1
        r = self.vg(x)
        self._last = (x.copy(), r)
        return r

    def hessian(self, x):
        x = np.asarray(x, float)
        _, g = self.value_and_grad(x)
        free = _free(x, g, self.lo, self.hi)
        H = np.zeros((x.size, x.size))
        xj = jnp.asarray(x)
        for i in np.flatnonzero(free):
            e = jnp.zeros_like(xj).at[i].set(1.0)
            if self.prior_only is not None and self.prior_only[i]:
                H[:, i] = np.asarray(self._phvp(xj, e), float)
                continue
            self.n_hvp += 1
            H[:, i] = np.asarray(self._hvp(xj, e), float)
        Hf = H[np.ix_(free, free)]
        H = np.zeros_like(H)
        H[np.ix_(free, free)] = 0.5 * (Hf + Hf.T)
        H[~free, ~free] = -1.0                       # decoupled, concave: Newton holds them
        return H


def polish_wanted(cfg):
    """'auto' polishes only the linear arm on its cached LML (QR or Gram; objective lml, one device, device
    engine), where a gradient costs milliseconds; a GP gradient costs ~70 s on GAP-18 Si (o3d12,
    M = 100), so there 'on' is opt-in."""
    if cfg.map_polish in ("on", "off"):
        return cfg.map_polish == "on"
    return cfg.arm == "linear" and cfg.objective == "lml" and cfg.devices == 1 and cfg.lml == "device"


def _gain(g, H, x, lo, hi):
    """Predicted gain (nats) of the box-constrained Newton step for the maximised log-posterior with
    gradient g and Hessian H (of the log-posterior; modified to negative definite in _newton_step)."""
    lo = np.full(x.size, -np.inf) if lo is None else lo
    hi = np.full(x.size, np.inf) if hi is None else hi
    _, dec = _newton_step(-H, -g, x, lo, hi)
    return 0.5 * dec


def _noise_gain(gnoise, H, free):
    """The Newton gain a gradient of 10x its roundoff explains: 1/2 (10 gnoise)^T |H_ff|^-1 (10 gnoise)
    over the free block (|H|: eigenvalues by magnitude, floored like _newton_step), capped."""
    if not free.any():
        return 0.0
    w, V = np.linalg.eigh(-H[np.ix_(free, free)])
    w = np.maximum(np.abs(w), 1e-10 * max(np.abs(w).max(), 1e-300))
    u = V.T @ (10 * np.asarray(gnoise, float)[free])
    return min(MAP_FLOOR_CAP, 0.5 * float(u @ (u / w)))


def judge(v, g, x, lo, hi, *, H=None, gnoise=None, strict=False, log=print, what="MAP"):
    """Say whether the MAP is stationary: the predicted Newton gain (`_gain`, needs H) is within
    MAP_GAIN_TOL; or (gnoise given: a polished MAP) every free gradient component is within 10x its
    roundoff gnoise_i and the gain within `_noise_gain` (resolution-limited); with no Hessian,
    |grad|_inf (bound-projected) within ADAM_GTOL.  Warns, or raises MapNotConverged when strict.
    Costs no evaluation: the caller passes v, g (and H, gnoise)."""
    pg = np.where(_free(x, g, lo, hi), g, 0.0) if lo is not None else g
    pgi = float(np.abs(pg).max()) if np.all(np.isfinite(pg)) else float("inf")
    worst = Hypers._fields[int(np.argmax(np.abs(np.nan_to_num(pg, nan=np.inf))))]
    rec = {"logpost": float(v), "pgrad_inf": pgi, "pgrad_worst": worst,
           "gnoise_max": None if gnoise is None else float(np.max(gnoise))}
    if H is not None and np.isfinite(v) and np.all(np.isfinite(g)):
        gain = _gain(g, H, x, lo, hi)
        floor = 0.0
        if gnoise is not None:
            free = _free(x, g, lo, hi) if lo is not None else np.ones(x.size, bool)
            if np.all(np.abs(pg)[free] <= 10 * np.asarray(gnoise, float)[free]):
                floor = _noise_gain(gnoise, H, free)
        ok = gain <= max(MAP_GAIN_TOL, floor)
        rec.update(gain=gain, gain_tol=MAP_GAIN_TOL, noise_floor=floor,
                   resolution_limited=bool(ok and gain > MAP_GAIN_TOL), converged=bool(ok))
        crit = f"predicted gain {gain:.2g} nats" + (f" (within the gradient-roundoff floor {floor:.1g})"
                                                     if rec["resolution_limited"] else "")
        bad = f"predicted gain of another Newton step {gain:.3g} nats > {max(MAP_GAIN_TOL, floor):.2g}"
    else:
        ok = bool(np.isfinite(v) and pgi <= ADAM_GTOL)
        rec.update(gain=None, gtol=ADAM_GTOL, resolution_limited=False, converged=ok)
        crit = f"|grad|_inf {pgi:.2e}"
        bad = f"|d logpost / d theta|_inf {pgi:.3g} (at {worst}) > {ADAM_GTOL}"
    if ok:
        log(f"{what}: stationary ({crit}), logpost {float(v):.10g}")
        return rec
    msg = (f"{what} did not converge: {bad}, logpost {float(v):.10g}, largest gradient at {worst}. The "
           f"hyperparameters, and every prediction made with them, are not at the optimum. Try "
           f"--map-polish on (a Newton polish: one Hessian-vector product per free hyperparameter per step; "
           f"on the GP arm each is a forward-over-reverse pass through the streamed objective, which can "
           f"take hours at large scale), --map-restarts N, "
           f"or more --map-steps; --strict makes this an error")
    if strict:
        raise MapNotConverged(msg)
    log("WARNING: " + msg)
    warnings.warn(msg, UserWarning, stacklevel=3)
    return rec


def _noise_record(tied):
    """map_convergence.json's noise entries: only for shared noise (the per-quantity record, part of
    the bit-exact pipeline goldens, is unchanged)."""
    if tied is None:
        return {}
    return {"noise": "shared", "tied": {Hypers._fields[i]: SHARED_NOISE for i in np.flatnonzero(tied)}}


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
        msg = ("sigma_type: the per-config-type MAP runs Adam over the extended vector and is not checked "
               "for convergence (map_polish and strict do not apply)")
        log("WARNING: " + msg); warnings.warn(msg, UserWarning, stacklevel=2)
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
    polish = polish_wanted(cfg)
    tied = getattr(obj, "tied", None)               # shared noise: log_sigma_E/V copy log_sigma_F
    free = None if tied is None else ~tied
    logprior = lambda a: log_prior(from_array(a), prob.prior, free)                  # noqa: E731
    logpost = lambda a: obj.lik(a) + logprior(a)                                     # noqa: E731
    prior_only = KERNEL if cfg.arm != "gp" else None
    hvp_ev = lambda: _HVPLogPosterior(vg_host, lo, hi, logpost, prior_only, logprior)  # noqa: E731
    if cfg.opt == "adam":
        theta = run_map(obj.lik, prob.prior, steps=cfg.map_steps, lr=cfg.map_lr, seed=cfg.seed, init=init)
        x = np.array(to_array(theta), float)
        lo = hi = None
        if tied is not None:      # tie, then pin the tied coordinates: they are not free hyperparameters
            x = tie_noise(x)
            lo, hi = np.where(tied, x, -np.inf), np.where(tied, x, np.inf)
            theta = Hypers(*[float(val) for val in x])
        v, g = vg_host(x)                               # one evaluation; a Hessian only where it is cheap
        H = hvp_ev().hessian(x) if polish else None
        conv = judge(v, g, x, lo, hi, H=H, strict=cfg.strict, log=log)
        ev = _log_evidence(obj, theta)
        return MapFit(theta, None, None, {"map": time.time() - t}, ev,
                      {"optimiser": "adam", "restart": None, "polish": None, **conv, "log_evidence": ev,
                       **_noise_record(tied)})
    x0 = np.array(to_array(init or prob.prior.mu), float)       # a copy: fix_rho writes into it
    lo, hi = LBFGS_LO.copy(), LBFGS_HI.copy()
    if tied is not None:          # the tied coordinates are pinned (lo == hi): only log_sigma_F moves
        x0 = tie_noise(x0)
        lo[tied] = hi[tied] = x0[tied]
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
    restart = None
    if "ABNORMAL" in best["message"]:        # a line-search stop, not convergence: once more from there
        n = min(cfg.map_steps, RESTART_EVALS)       # evaluations, not iterations: ~1 min each on GP
        r = lbfgs_map(vg_host, best["x"], lo, hi, n, maxfun=n, log=lambda i, v: log_eval("restart", i, v))
        restart = {"iters": n, "nfev": r["nfev"], "message": r["message"], "logpost_before": best["value"],
                   "logpost": r["value"]}
        log(f"L-BFGS restart (line-search stop): logpost {best['value']:.6g} -> {r['value']:.6g}  "
            f"nfev {r['nfev']}  {r['message']}")
        if r["value"] >= best["value"]:
            best = r
    x, v, g, pol, H, gnoise = np.asarray(best["x"], float), best["value"], best["grad"], None, None, None
    if polish:
        tp = time.time()
        ev = hvp_ev()
        x, pol = newton_polish(ev, x, lo, hi, gradient_floor=True)
        v, g = ev.value_and_grad(x)
        H, gnoise = pol.pop("hessian"), np.asarray(pol["gnoise"], float)    # the polish's own, at x
        if H is None or not np.all(np.isfinite(gnoise)):
            gnoise = None
        log(f"Newton polish (exact Hessian, column-wise HVPs): {pol['message']}; {pol['steps']} step(s), "
            f"{pol['hessian_evals']} Hessian(s), {ev.n_hvp} HVPs, {ev.n_eval} gradient evaluations, "
            f"{time.time() - tp:.1f} s")
        # no wall time in the record: map_convergence.json is part of the bit-exact pipeline goldens
        pol = {**{k: val for k, val in pol.items() if k != "gnoise"}, "hvps": ev.n_hvp, "evals": ev.n_eval}
    elif best.get("hess_inv") is not None and g is not None:
        H = -np.linalg.inv(0.5 * (best["hess_inv"] + best["hess_inv"].T))   # L-BFGS-B's estimate, no evaluation
    if g is None:                               # never finite: nothing to judge with
        g = np.full(x.size, np.nan)
    conv = judge(v, g, x, lo, hi, H=H, gnoise=gnoise, strict=cfg.strict, log=log)
    fixed = _bound_active(x, lo, hi)
    if tied is not None:          # the objective reads only the shared coordinate: copy it out
        x, fixed = tie_noise(x), fixed & ~tied
        log(f"shared noise: sigma_E = sigma_F = sigma_V = {float(np.exp(x[8])):.6g}")
    theta = Hypers(*[float(val) for val in x])
    ev = _log_evidence(obj, theta)
    return MapFit(theta, restarts, None, {"map": time.time() - t}, ev,
                  {"optimiser": "lbfgs", "nfev": int(sum(r["nfev"] for r in runs)), "lbfgs_message": best["message"],
                   "restart": restart, "polish": pol, **conv, "log_evidence": ev, **_noise_record(tied)},
                  fixed)
