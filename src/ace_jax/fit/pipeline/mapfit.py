import os
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

# log-space boxes: generous, but keep the Cholesky away from sigma -> 0; A up to
# 1e3 (the full/PCA-descriptor GP sat at the old bound of 10, Cantor-1k)
LBFGS_LO = np.log([0.05, 1e-3, 0.1, 1.5, 1e-3, 0.1, 1e-2, 1e-4, 1e-4, 1e-4])
LBFGS_HI = np.log([50.0, 1e3, 50.0, 4.0, 50.0, 100.0, 1e4, 10.0, 10.0, 10.0])

# Stationarity of the MAP is judged by the PREDICTED GAIN of one more (box-constrained) Newton step,
# 1/2 g^T B^-1 g in nats over the free coordinates -- scale-free, unlike a gradient norm (a gradient
# of 3 nats per log-unit was a 1e-5-nat gain at 4.6e5 force rows, GAP-18 Si).  B is the polish
# Hessian when there is one, else L-BFGS-B's inverse-Hessian estimate.  The MAP counts as stationary
# when the gain is <= MAP_GAIN_TOL, or <= 10x the log-posterior's measured roundoff (a polished MAP
# only): that secondary floor passes a point the objective cannot resolve, and the record says so
# (`resolution_limited`).
MAP_GAIN_TOL = 1e-3
ADAM_GTOL = 1e-2         # Adam on a path with no cheap Hessian: |grad|_inf, nats per log-unit
FD_EPS = 1e-4            # central-difference step of the polish Hessian (log units)
RESTART_ITERS = 50       # extra L-BFGS-B iterations after a line-search ('ABNORMAL') stop
EXACT_HEADROOM = 0.5     # map_polish 'exact': the Hessian's temporaries must fit in this share of free memory


class MapNotConverged(RuntimeError):
    """The MAP ended away from a stationary point (raised under `strict`; otherwise a warning)."""


class MapFit(NamedTuple):
    theta: object; restarts: object; sigma_type_ratios: object; timings: dict
    log_evidence: object = None          # log marginal likelihood at theta (no hyperprior); None for sigma_type
    convergence: object = None           # dict: optimiser, restart, polish, logpost, gain, converged, ...
    fixed: object = None                 # (10,) bool: hyperparameters held at a bound (fix_rho, or bound-active
                                         # at the MAP); the Laplace rung holds them fixed


def _log_evidence(obj, theta):
    """The LML at theta, comparable across bases on the same data (the logged L-BFGS
    'logpost' adds the hyperprior, so it is not used)."""
    return float(obj.lik(to_array(theta)))


def _free(x, g, lo, hi):
    """Coordinates the MAP is free in: not pinned (lo == hi) and not bound-active (on a bound with
    the log-posterior gradient g pushing outward).  lo None: unbounded, all free."""
    if lo is None:
        return np.ones(x.size, bool)
    return ~((lo >= hi) | ((x <= lo) & (g <= 0)) | ((x >= hi) & (g >= 0)))


def _bound_active(x, lo, hi):
    return (lo >= hi) | np.isclose(x, lo, rtol=0, atol=1e-10) | np.isclose(x, hi, rtol=0, atol=1e-10)


class _FDLogPosterior:
    """The `newton.newton_polish` interface over the MAP objective, at gradient-level memory: the
    Hessian is central differences (step FD_EPS) of the compiled value-and-grad over the FREE
    coordinates only (2k evaluations; the others are decoupled with a unit curvature, so the
    box-constrained Newton step leaves them on their bounds).  `jax.hessian` through the Cholesky
    LML needs ~10x the gradient's temporaries (69 against 7 P^2 doubles, P the readout length:
    2.2 GB at P = 2,053, ~14 GB at 5,000), so it is the gated opt-in 'exact'; on the linear arm
    (GAP-18 Si, o4d12) the two Hessians agree to 1e-5 relative."""

    def __init__(self, vg_host, lo, hi):
        self.vg, self.lo, self.hi, self.n_eval = vg_host, lo, hi, 0
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
        for i in np.flatnonzero(free):
            e = np.zeros(x.size); e[i] = FD_EPS
            self.n_eval += 2
            H[:, i] = (self.vg(x + e)[1] - self.vg(x - e)[1]) / (2 * FD_EPS)
        Hf = H[np.ix_(free, free)]
        H = np.zeros_like(H)
        H[np.ix_(free, free)] = 0.5 * (Hf + Hf.T)
        H[~free, ~free] = -1.0                       # decoupled, concave: Newton holds them
        return H


class _ExactLogPosterior(_FDLogPosterior):
    """map_polish 'exact': the compiled `jax.hessian` of the log-posterior (the caller has checked
    that its temporaries fit, `_exact_hessian`)."""

    def __init__(self, vg_host, lo, hi, hess):
        super().__init__(vg_host, lo, hi)
        self._hess = hess

    def hessian(self, x):
        H = np.asarray(self._hess(jnp.asarray(x)), float)
        return 0.5 * (H + H.T)


def _free_bytes():
    """Free memory of the default device: the allocator's limit minus its use (GPU), else the
    host's available physical memory (Linux), else None."""
    try:
        st = jax.devices()[0].memory_stats() or {}
        if "bytes_limit" in st:
            return int(st["bytes_limit"]) - int(st.get("bytes_in_use", 0))
    except Exception:                                # noqa: BLE001  (backends without memory_stats)
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _exact_hessian(logpost, x, log):
    """The compiled exact Hessian if its temporaries (XLA's memory analysis) fit in EXACT_HEADROOM of
    the free memory, else None (the caller falls back to finite differences).  Compiling it costs
    seconds on the linear arm, far more through the GP arm's streamed scan."""
    comp = jax.jit(jax.hessian(logpost)).lower(jnp.asarray(x)).compile()
    ma = comp.memory_analysis()
    need, free = getattr(ma, "temp_size_in_bytes", None), _free_bytes()
    if need is None or free is None or need > EXACT_HEADROOM * free:
        log(f"map-polish exact: Hessian temporaries {need} B vs {free} B free: using the finite-difference Hessian")
        return None
    log(f"map-polish exact: Hessian temporaries {need / 2**20:.0f} MiB ({free / 2**20:.0f} MiB free)")
    return comp


def polish_mode(cfg):
    """None | 'fd' | 'exact'.  'auto' polishes only the linear arm on the cached-Gram LML (objective
    lml, one device, device engine), where a gradient costs milliseconds; a GP gradient costs ~70 s
    on GAP-18 Si (o3d12, M = 100), so there 'on' is opt-in."""
    if cfg.map_polish == "off":
        return None
    if cfg.map_polish in ("on", "exact"):
        return "fd" if cfg.map_polish == "on" else "exact"
    cheap = cfg.arm == "linear" and cfg.objective == "lml" and cfg.devices == 1 and cfg.lml == "device"
    return "fd" if cheap else None


def _gain(g, H, x, lo, hi):
    """Predicted gain (nats) of the box-constrained Newton step for the maximised log-posterior with
    gradient g and Hessian H (of the log-posterior; modified to negative definite in _newton_step)."""
    lo = np.full(x.size, -np.inf) if lo is None else lo
    hi = np.full(x.size, np.inf) if hi is None else hi
    _, dec = _newton_step(-H, -g, x, lo, hi)
    return 0.5 * dec


def judge(v, g, x, lo, hi, *, H=None, noise=0.0, strict=False, log=print, what="MAP"):
    """Say whether the MAP is stationary: the predicted Newton gain (`_gain`, needs H) is within
    MAP_GAIN_TOL, or within 10x the measured log-posterior roundoff `noise` (resolution-limited);
    with no Hessian, |grad|_inf (bound-projected) within ADAM_GTOL.  Warns, or raises
    MapNotConverged when strict.  Costs no evaluation: the caller passes v, g (and H)."""
    pg = np.where(_free(x, g, lo, hi), g, 0.0) if lo is not None else g
    pgi = float(np.abs(pg).max()) if np.all(np.isfinite(pg)) else float("inf")
    worst = Hypers._fields[int(np.argmax(np.abs(np.nan_to_num(pg, nan=np.inf))))]
    rec = {"logpost": float(v), "pgrad_inf": pgi, "pgrad_worst": worst, "noise": float(noise)}
    if H is not None and np.isfinite(v) and np.all(np.isfinite(g)):
        gain = _gain(g, H, x, lo, hi)
        ok = gain <= max(MAP_GAIN_TOL, 10 * noise)
        rec.update(gain=gain, gain_tol=MAP_GAIN_TOL, resolution_limited=bool(ok and gain > MAP_GAIN_TOL),
                   converged=bool(ok))
        crit = f"predicted gain {gain:.2g} nats" + (f" (within 10x the roundoff {noise:.1g})"
                                                     if rec["resolution_limited"] else "")
        bad = f"predicted gain of another Newton step {gain:.3g} nats > {max(MAP_GAIN_TOL, 10 * noise):.2g}"
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
           f"--map-polish on (a Newton polish: ~2 x 10 gradient evaluations per step), --map-restarts N, "
           f"or more --map-steps; --strict makes this an error")
    if strict:
        raise MapNotConverged(msg)
    log("WARNING: " + msg)
    warnings.warn(msg, UserWarning, stacklevel=3)
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
    mode = polish_mode(cfg)
    if cfg.opt == "adam":
        theta = run_map(obj.lik, prob.prior, steps=cfg.map_steps, lr=cfg.map_lr, seed=cfg.seed, init=init)
        x = np.array(to_array(theta), float)
        v, g = vg_host(x)                               # one evaluation; a Hessian only where it is cheap
        H = _FDLogPosterior(vg_host, None, None).hessian(x) if mode is not None else None
        conv = judge(v, g, x, None, None, H=H, strict=cfg.strict, log=log)
        return MapFit(theta, None, None, {"map": time.time() - t}, _log_evidence(obj, theta),
                      {"optimiser": "adam", "restart": None, "polish": None, **conv})
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
    restart = None
    if "ABNORMAL" in best["message"]:        # a line-search stop, not convergence: once more from there
        n = min(cfg.map_steps, RESTART_ITERS)
        r = lbfgs_map(vg_host, best["x"], lo, hi, n, log=lambda i, v: log_eval("restart", i, v))
        restart = {"iters": n, "nfev": r["nfev"], "message": r["message"], "logpost_before": best["value"],
                   "logpost": r["value"]}
        log(f"L-BFGS restart (line-search stop): logpost {best['value']:.6g} -> {r['value']:.6g}  "
            f"nfev {r['nfev']}  {r['message']}")
        if r["value"] >= best["value"]:
            best = r
    x, v, g, pol, H, noise = np.asarray(best["x"], float), best["value"], best["grad"], None, None, 0.0
    if mode is not None:
        tp = time.time()
        hess = None
        if mode == "exact":
            hess = _exact_hessian(lambda a: obj.lik(a) + log_prior(from_array(a), prob.prior), x, log)
        ev = _FDLogPosterior(vg_host, lo, hi) if hess is None else _ExactLogPosterior(vg_host, lo, hi, hess)
        x, pol = newton_polish(ev, x, lo, hi)
        v, g = ev.value_and_grad(x)
        H, noise = pol.pop("hessian"), float(pol["noise"])        # the polish's own Hessian, at x
        noise = noise if np.isfinite(noise) else 0.0
        log(f"Newton polish ({'exact' if hess is not None else 'finite-difference'} Hessian): {pol['message']}; "
            f"{pol['steps']} step(s), {pol['hessian_evals']} Hessian(s), {ev.n_eval} gradient evaluations, "
            f"{time.time() - tp:.1f} s")
        pol = {**{k: val for k, val in pol.items() if k != "gnoise"}, "hessian": "exact" if hess is not None else "fd"}
    elif best.get("hess_inv") is not None and g is not None:
        H = -np.linalg.inv(0.5 * (best["hess_inv"] + best["hess_inv"].T))   # L-BFGS-B's estimate, no evaluation
    if g is None:                               # never finite: nothing to judge with
        g = np.full(x.size, np.nan)
    conv = judge(v, g, x, lo, hi, H=H, noise=noise, strict=cfg.strict, log=log)
    theta = Hypers(*[float(val) for val in x])
    return MapFit(theta, restarts, None, {"map": time.time() - t}, _log_evidence(obj, theta),
                  {"optimiser": "lbfgs", "nfev": int(sum(r["nfev"] for r in runs)), "lbfgs_message": best["message"],
                   "restart": restart, "polish": pol, **conv},
                  _bound_active(x, lo, hi))
