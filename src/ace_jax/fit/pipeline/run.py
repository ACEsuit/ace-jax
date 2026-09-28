import time
from typing import NamedTuple

from ...eval import highest_precision
from .mapfit import fit_map
from .objective import make_objective, release
from .predict import predict_splits
from .problem import build_problem
from .rungs import run_rungs


class FitResult(NamedTuple):
    config: object; data: object; built: object; theta: object
    map: object; rungs: object; preds: object; timings: dict


def fit(cfg, data, log=print, on_stage=None):
    """Run the pipeline.  on_stage(name, payload), if given, is called as each
    expensive stage finishes ("data" -> FitData, "map" -> MapFit, "rungs" -> Rungs),
    so a driver can write those results before a later stage (e.g. POPS or
    prediction running out of memory) can lose them."""
    T0 = time.time()
    cfg.validate()
    stage = on_stage or (lambda name, payload: None)
    stage("data", data)
    b = build_problem(cfg, data)
    with highest_precision():
        obj = make_objective(cfg, data, b)
        mf = fit_map(cfg, data, b, obj, log=log)
        stage("map", mf)
        rg = run_rungs(cfg, b, obj, mf.theta, log=log)
        stage("rungs", rg)
        # cached linear statistics (run.py) or a full recompute per draw (the CLI's
        # historical path): equal in exact arithmetic, not in summation order
        stats = obj.stats if cfg.predict_stats == "cached" else None
        if cfg.uq == "pops":
            # POPS needs only the linear statistics, which obj.stats already holds
            # (the device path caches them; host-cache is GP-only, never POPS):
            # drop the LML and its jitted objective, then free their buffers
            obj = obj._replace(lik=None, vg=None, host_cache=None)
            release()
        pr = predict_splits(cfg, data, b, stats, mf.theta, rg.draws, log=log)
    tm = {**b.timings, **obj.timings, **mf.timings, **rg.timings, **pr.timings, "total": time.time() - T0}
    return FitResult(cfg, data, b, mf.theta, mf, rg, pr, tm)
