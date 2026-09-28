import time
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ..hypers import Hypers, from_array, to_array
from ..ladder import run_map
from ..multistart import multistart_map, prior_starts

# log-space boxes: generous, but keep the Cholesky away from sigma -> 0; A up to
# 1e3 (the full/PCA-descriptor GP sat at the old bound of 10, Cantor-1k)
LBFGS_LO = np.log([0.05, 1e-3, 0.1, 1.5, 1e-3, 0.1, 1e-2, 1e-4, 1e-4, 1e-4])
LBFGS_HI = np.log([50.0, 1e3, 50.0, 4.0, 50.0, 100.0, 1e4, 10.0, 10.0, 10.0])


class MapFit(NamedTuple):
    theta: object; restarts: object; sigma_type_ratios: object; timings: dict


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
    if cfg.opt == "adam":
        theta = run_map(obj.lik, prob.prior, steps=cfg.map_steps, lr=cfg.map_lr, seed=cfg.seed, init=init)
        return MapFit(theta, None, None, {"map": time.time() - t})
    x0 = np.asarray(to_array(init or prob.prior.mu), float)
    lo, hi = LBFGS_LO.copy(), LBFGS_HI.copy()
    if cfg.fix_rho is not None:
        lo[5] = hi[5] = x0[5] = np.log(_rho_fix(cfg, prob))
        log(f"fix-rho: rho pinned at {float(np.exp(lo[5])):.4f}")
    tick = [time.time()]
    def vg_host(x):
        v, g = obj.vg(jnp.asarray(x)); g.block_until_ready()
        return float(v), np.asarray(g, float)
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
    return MapFit(Hypers(*[float(v) for v in best["x"]]), restarts, None, {"map": time.time() - t})
