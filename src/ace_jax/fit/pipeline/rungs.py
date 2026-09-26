import time
from typing import NamedTuple

import numpy as np

from ..hypers import to_array

KNOWN = ("map", "laplace", "pathfinder", "vi", "nuts")


class Rungs(NamedTuple):
    draws: dict; info: dict; timings: dict


def run_rungs(cfg, b, obj, theta, log=print):
    bad = [r for r in cfg.rungs if r not in KNOWN]
    if bad:
        raise ValueError(f"unknown rung(s) {bad}; known: {list(KNOWN)}")
    from ..ladder import run_laplace, run_laplace_fd, run_nuts, run_pathfinder, run_vi
    draws, info, tm = {"map": np.asarray(to_array(theta))[None]}, {}, {}
    prior = None if b is None else b.prob.prior
    if "laplace" in cfg.rungs:
        t = time.time()
        if cfg.laplace == "fd":
            draws["laplace"], info["laplace"] = run_laplace_fd(obj.lik, prior, theta, n_draws=cfg.n_draws,
                                                               seed=cfg.seed)
        else:
            draws["laplace"], _ = run_laplace(obj.lik, prior, n_draws=cfg.n_draws, steps=cfg.map_steps,
                                              seed=cfg.seed, init=theta)
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
                                init=theta)
        tm["vi"] = time.time() - t
    if "nuts" in cfg.rungs:
        t = time.time()
        draws["nuts"], info["nuts"] = run_nuts(obj.lik, prior, num_warmup=cfg.nuts_warmup,
                                               num_samples=cfg.nuts_samples, num_chains=cfg.nuts_chains,
                                               seed=cfg.seed, init=theta)
        tm["nuts"] = time.time() - t
    if "map" not in cfg.rungs:
        draws.pop("map")
    return Rungs(draws, info, tm)
