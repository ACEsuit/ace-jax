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
    ard: object = None                   # ARDResult when uq == "ard" (last: positional use unaffected)
    radial: object = None                # RadialResult when cfg.learn_radial (last: positional use unaffected)
    readout: object = None               # the least-squares readout when cfg.solver == "lstsq"


def fit(cfg, data, log=print, on_stage=None):
    """Run the pipeline.  on_stage(name, payload), if given, is called as each
    expensive stage finishes ("data" -> FitData, "radial" -> RadialResult when
    cfg.learn_radial, "map" -> MapFit, "rungs" -> Rungs,
    "ard" -> ARDResult and "model" -> the ARD-mean model.npz arrays when uq == "ard"),
    so a driver can write those results before a later stage (e.g. POPS or
    prediction running out of memory) can lose them."""
    T0 = time.time()
    import jax
    if not jax.config.jax_enable_x64:     # a float32 fit is silently poor: the caller chooses precision
        raise RuntimeError("fitting needs float64: call jax.config.update('jax_enable_x64', True) "
                           "before importing anything that uses JAX (aj fit does this for you)")
    cfg.validate()
    if cfg.e0 == "lsq" and not cfg.joint_e0:
        log("e0 lsq: POPS fits with the least-squares E0 fixed (as e0='prefit')")
    stage = on_stage or (lambda name, payload: None)
    stage("data", data)
    radial = None
    if cfg.learn_radial:
        from .radials import learn_radials
        data, radial = learn_radials(cfg, data, log=log)
        stage("radial", radial)
    b = build_problem(cfg, data)
    if cfg.solver == "lstsq":       # no objective, no MAP: one weighted least-squares solve
        from .lstsq import fit_lstsq
        with highest_precision():
            mf, rg, readout = fit_lstsq(cfg, data, b, log=log)
            stage("map", mf)
            pr = predict_splits(cfg, data, b, None, mf.theta, rg.draws, log=log, readout=readout)
        tm = {**b.timings, **mf.timings, **pr.timings, "total": time.time() - T0}
        return FitResult(cfg, data, b, mf.theta, mf, rg, pr, tm, None, radial, readout)
    with highest_precision():
        obj = make_objective(cfg, data, b)
        mf = fit_map(cfg, data, b, obj, log=log)
        stage("map", mf)
        rg = run_rungs(cfg, b, obj, mf.theta, log=log, fixed=mf.fixed)
        stage("rungs", rg)
        ard = None
        if cfg.uq == "ard":
            from ..ard import run_ard_stage
            # joint mode refits on the objective's cached linear statistics (not a second pass);
            # drop everything else the objective holds -- the LML, its jitted objective and the
            # stats closure (ARD predicts from its own posterior) -- then free their buffers
            full = obj.lin if cfg.ard_mode == "joint" else None
            obj = obj._replace(lik=None, vg=None, host_cache=None, stats=None, lin=None)
            release()
            ard = run_ard_stage(cfg, data, b, mf.theta, log=log, full_stats=full)
            del full
            stage("ard", ard)
            from .export import linear_arrays_from_mean, model_file_blocked
            if model_file_blocked(cfg) is None:      # the ARD-mean model.npz, before prediction
                stage("model", linear_arrays_from_mean(data.z, data.E0, b.prob.cfg, ard.posterior.mean))
        # cached linear statistics (run.py) or a full recompute per draw (the CLI's
        # historical path): equal in exact arithmetic, not in summation order
        stats = obj.stats if cfg.predict_stats == "cached" else None
        if cfg.uq == "pops":
            # POPS needs only the linear statistics, which obj.stats already holds
            # (the device path caches them; host-cache is GP-only, never POPS):
            # drop the LML and its jitted objective, then free their buffers
            obj = obj._replace(lik=None, vg=None, host_cache=None)
            release()
        pr = predict_splits(cfg, data, b, stats, mf.theta, rg.draws, log=log, ard=ard)
    tm = {**b.timings, **obj.timings, **mf.timings, **rg.timings, **pr.timings}
    if ard is not None:
        tm["ard"] = ard.report["seconds"]
    if radial is not None:
        tm["radial"] = radial.seconds
    tm["total"] = time.time() - T0
    return FitResult(cfg, data, b, mf.theta, mf, rg, pr, tm, ard, radial)
