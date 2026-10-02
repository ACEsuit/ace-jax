"""Learned tensor radials as a fit-pipeline stage (`aj fit --learn-radial`;
docs/dev/specs/2026-10-01-fit-learned-radials-design.md).

bench/learn_radial/run.py's recipe, run before the fit: a seeded hold-out of the
training configs gates fit_radial's candidates (init + one learned per roughness
weight), and the selected radials, with the readout fitted for them, are patched
into the model's npz arrays in memory.  FitData's model/meta/z are swapped for the
patched ones, the fitted E0 reapplied, and the rest of the pipeline then refits on
the FULL training set exactly as for any other model file."""
import io
import time
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ...basis.export import patch_radial_npz
from ...basis.prior import prior_diagonal
from ...eval import load
from ..data import build_dataset
from ..hypers import default_prior
from ..inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ..kernels import KernelSpec
from ..objective import Problem
from ..radial_learn import fit_radial
from ..radial_model import rnl_degrees, to_analytic, with_radial

RADIAL_MAP_STEPS = 300               # per-candidate theta-MAP steps in fit_radial (the driver's)


class RadialResult(NamedTuple):
    W: np.ndarray                    # the selected rnl_Wnlq
    info: dict                       # fit_radial's info, minus the readout
    n_fit: int; n_val: int
    relres_max: float                # to_analytic's worst relative projection residual
    seconds: float


def learn_radials(cfg, data, log=print):
    """(FitData with the learned model, RadialResult)."""
    t0 = time.time()
    n = len(data.train)
    idx = np.random.default_rng(cfg.seed).permutation(n)       # the ARD split convention
    nval = max(1, int(round(cfg.radial_val_frac * n)))
    if nval >= n:
        raise ValueError(f"radial_val_frac={cfg.radial_val_frac} holds out {nval} of {n} training "
                         f"configs and leaves none to learn the radials on")
    val, fit_ = [data.train[i] for i in idx[:nval]], [data.train[i] for i in idx[nval:]]
    try:
        model, relres = to_analytic(data.model, cfg.radial_n_q)
    except ValueError as e:
        raise ValueError(f"learned radials need an analytic or spline tensor radial, got "
                         f"radial_kind={data.model.radial_kind!r} (embedding models are not supported "
                         f"yet, see issue #31): {e}") from e
    relres_max = float(np.max(relres)) if np.size(relres) else 0.0
    meta = data.meta
    ds_fit, ds_val = (build_dataset(cs, meta, data.E0, cfg.batch) for cs in (fit_, val))
    r0 = cfg.r0 if cfg.r0 is not None else data.r0
    gc = GPConfig(r0=r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                  NZ=len(meta["elements"]), C=cfg.batch)
    X, S = site_features(model, gc, ds_fit)
    ind = select_inducing(X, S, ds_fit.node_z, ds_fit.node_mask, 0, descriptor_scale(X, ds_fit.node_mask))
    prob = Problem(KernelSpec("cosine", True, gc.D), model, ind, gc,
                   jnp.asarray(prior_diagonal(data.z, meta, data.source)), default_prior(r0))
    log(f"learn radials: n_q={cfg.radial_n_q} on {len(fit_)} configs, gate on {nval} "
        f"(to_analytic relres_max={relres_max:.2e})")
    W, info = fit_radial(prob, ds_fit, ds_val, model.rnl_Wnlq, lam_grid=tuple(cfg.radial_lam_grid),
                         steps=cfg.radial_steps, map_steps=RADIAL_MAP_STEPS,
                         rough_weights=1.0 / (1.0 + rnl_degrees(meta)) ** 2, log=log)
    # patch the source arrays in memory: same file-shaped hand-off as a built basis
    src, dst = io.BytesIO(), io.BytesIO()
    np.savez(src, **{k: data.z[k] for k in data.z.files})
    src.seek(0)
    patch_radial_npz(src, dst, with_radial(model, W, learned=info["selected"] != "init"),
                     readout=info["readout"])
    dst.seek(0)
    new_model, new_meta, new_z = load(dst)
    new_model = eqx.tree_at(lambda m: m.E0, new_model, jnp.asarray(data.E0))   # e0='lsq' survives
    rr = RadialResult(np.asarray(W), {k: v for k, v in info.items() if k != "readout"}, len(fit_), nval,
                      relres_max, time.time() - t0)
    log(f"learn radials: selected {info['selected']} in {rr.seconds:.1f}s")
    return data._replace(model=new_model, meta=new_meta, z=new_z,
                         source=(data.source or "model") + " + learned radials"), rr
