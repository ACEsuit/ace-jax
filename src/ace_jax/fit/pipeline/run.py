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


def fit(cfg, data, log=print):
    T0 = time.time()
    cfg.validate()
    b = build_problem(cfg, data)
    with highest_precision():
        obj = make_objective(cfg, data, b)
        mf = fit_map(cfg, data, b, obj, log=log)
        rg = run_rungs(cfg, b, obj, mf.theta, log=log)
        stats = obj.stats
        if cfg.uq == "pops":
            # POPS needs only the linear statistics: free the LML's Gram-sized buffers first
            from ..stats import assemble_statistics, linear_statistics, residual_statistics
            import jax
            prob, ds = b.prob, data.ds_train
            obj = obj._replace(lik=None, vg=None, stats=None, host_cache=None)
            release()
            lin = jax.jit(lambda: linear_statistics(prob.model, prob.cfg, ds))()
            stats = lambda th: assemble_statistics(lin, residual_statistics(th, prob.spec, prob.model,
                                                                            prob.ind, prob.cfg, ds))
        pr = predict_splits(cfg, data, b, stats, mf.theta, rg.draws, log=log)
    tm = {**b.timings, **obj.timings, **mf.timings, **rg.timings, **pr.timings, "total": time.time() - T0}
    return FitResult(cfg, data, b, mf.theta, mf, rg, pr, tm)
