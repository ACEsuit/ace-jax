import time
from typing import NamedTuple

import numpy as np

from ..hypers import to_array

KNOWN = ("map", "laplace", "pathfinder", "vi", "nuts")


class Rungs(NamedTuple):
    draws: dict; info: dict; timings: dict


def run_rungs(cfg, b, obj, theta, log=print, fixed=None):
    """fixed (10,) bool, from MapFit.fixed: hyperparameters on a bound at the MAP, held fixed by the
    Laplace rung (there the log-posterior has no interior mode; its Hessian is singular).

    Shared noise (obj.tied): the tied log_sigma_E/V carry no likelihood, so Laplace, VI and NUTS hold
    them (the shared scale is one coordinate, counted once); Pathfinder keeps them as independent
    hyperprior directions, which leave the others' marginal unchanged.  Every rung's draws are then
    tied (columns E and V copy F), as the predictions need."""
    bad = [r for r in cfg.rungs if r not in KNOWN]
    if bad:
        raise ValueError(f"unknown rung(s) {bad}; known: {list(KNOWN)}")
    from ..ladder import run_laplace, run_laplace_fd, run_nuts, run_pathfinder, run_vi
    draws, info, tm = {"map": np.asarray(to_array(theta))[None]}, {}, {}
    prior = None if b is None else b.prob.prior
    tied = getattr(obj, "tied", None)
    held = fixed if tied is None else (tied if fixed is None else np.asarray(fixed, bool) | tied)
    if "laplace" in cfg.rungs:
        t = time.time()
        if cfg.laplace == "fd":
            draws["laplace"], info["laplace"] = run_laplace_fd(obj.lik, prior, theta, n_draws=cfg.n_draws,
                                                               seed=cfg.seed, fixed=held)
        else:
            draws["laplace"], _ = run_laplace(obj.lik, prior, n_draws=cfg.n_draws, steps=cfg.map_steps,
                                              seed=cfg.seed, init=theta, fixed=held)
        if fixed is not None and np.any(fixed):
            from ..hypers import Hypers
            log(f"laplace: held at their bound: {[Hypers._fields[i] for i in np.flatnonzero(fixed)]}")
        tm["laplace"] = time.time() - t
    if "pathfinder" in cfg.rungs:
        t = time.time()
        draws["pathfinder"], info["pathfinder"] = run_pathfinder(obj.lik, prior, theta, n_draws=cfg.n_draws,
                                                                 seed=cfg.seed, num_samples=cfg.pf_samples,
                                                                 maxiter=cfg.pf_maxiter)
        tm["pathfinder"] = time.time() - t
    if "vi" in cfg.rungs:
        t = time.time()
        draws["vi"], _ = run_vi(obj.lik, prior, n_draws=cfg.n_draws, steps=cfg.vi_steps, seed=cfg.seed,
                                init=theta, fixed=tied)
        tm["vi"] = time.time() - t
    if "nuts" in cfg.rungs:
        t = time.time()
        draws["nuts"], info["nuts"] = run_nuts(obj.lik, prior, num_warmup=cfg.nuts_warmup,
                                               num_samples=cfg.nuts_samples, num_chains=cfg.nuts_chains,
                                               seed=cfg.seed, init=theta, fixed=tied)
        tm["nuts"] = time.time() - t
    if "map" not in cfg.rungs:
        draws.pop("map")
    if tied is not None:
        from ..paramset import tie_noise
        draws = {k: tie_noise(np.asarray(v, float)) for k, v in draws.items()}
    return Rungs(draws, info, tm)
