"""Fitted-model files.

A linear fit (M = 0) is an ordinary ACE potential: `model.npz` is the input
model file with the fitted readout (WB, Wpair) and E0 written in, so
`ace_jax.load`, `ACECalculator` and `ace-jax eval` read it unchanged.

A GP fit also needs the residual block, so `gp_model.npz` is self-contained:
the ACE arrays (prefixed "ace/", E0 fitted), the GP and kernel configuration,
the inducing set, and per hyperparameter draw the posterior (mu, L) -- the
same factorisation the pipeline's predictions use, so a reloaded
`GPCalculator` reproduces them.  L is (Dt, Dt), Dt = len_basis + M, per draw:
that, not the ACE model, sets the file size.
"""
import dataclasses
import io
import json
import pathlib

import jax.numpy as jnp
import numpy as np

from ..hypers import from_array, to_array
from ..objective import posterior
from ..predict import _train_stats

GP_SCHEMA = 1


def _ace_arrays(res):
    z = np.load(res.config.model)
    out = {k: z[k] for k in z.files}
    out["E0"] = np.asarray(res.data.E0, np.float64)
    return out


def _posterior(res, theta):
    return posterior(theta, _train_stats(theta, res.built.prob, res.data.ds_train, None), res.built.prob)


def _draws(res, n_draws):
    """n_draws = 1: the MAP hyperparameters; more: evenly spaced draws of the last
    rung (the subsampling predict_splits uses)."""
    if n_draws <= 1:
        return np.asarray(to_array(res.theta))[None]
    dr = np.asarray(res.rungs.draws[res.config.rungs[-1]])
    return dr if len(dr) <= n_draws else dr[np.linspace(0, len(dr) - 1, n_draws).astype(int)]


def linear_model_arrays(res):
    """The ACE npz arrays with the posterior-mean readout at the MAP hyperparameters.
    Column layout of the linear block: species-major B blocks, then pair blocks
    (fit/rows.py `_place`), i.e. WB[b, z] = mu[z*n_B + b]."""
    cfg = res.built.prob.cfg
    nB, nP, NZ = cfg.n_B, cfg.n_pair, cfg.NZ
    mu, _ = _posterior(res, res.theta)
    mu = np.asarray(mu)
    out = _ace_arrays(res)
    out["WB"] = mu[:NZ * nB].reshape(NZ, nB).T.copy()
    out["Wpair"] = mu[NZ * nB:NZ * (nB + nP)].reshape(NZ, nP).T.copy()
    return out


def gp_model_arrays(res, n_draws=1):
    prob = res.built.prob
    draws = _draws(res, n_draws)
    mus, Ls = [], []
    for d in draws:
        mu, L = _posterior(res, from_array(jnp.asarray(d)))
        mus.append(np.asarray(mu)); Ls.append(np.asarray(L))
    ind = prob.ind
    out = {f"ace/{k}": v for k, v in _ace_arrays(res).items()}
    out.update(ind_XM=np.asarray(ind.XM), ind_SM=np.asarray(ind.SM), ind_ZM=np.asarray(ind.ZM),
               ind_scale=np.asarray(ind.scale), ind_Pmap=np.asarray(ind.Pmap), ind_embed=np.asarray(ind.embed),
               draws=np.asarray(draws), mu=np.stack(mus), L=np.stack(Ls))
    out["gp_json"] = np.frombuffer(json.dumps({
        "schema_version": GP_SCHEMA, "gpcfg": dataclasses.asdict(prob.cfg),
        "kernel": dataclasses.asdict(prob.spec), "warp": ind.warp}).encode(), np.uint8)
    return out


def save_model(res, out, n_draws=1, log=print):
    """Write the fitted model into directory `out`: model.npz (linear arm) or
    gp_model.npz (GP arm).  Returns the path, or None when the fit cannot be
    represented as a model file (a dimer baseline is added back outside the model;
    a .yace input has no npz schema to write into)."""
    cfg = res.config
    if cfg.baseline is not None or cfg.base_npz is not None:
        log("not saving a model file: the baseline is added outside the model")
        return None
    if str(cfg.model).endswith(".yace"):
        log("not saving a model file: .yace inputs are not supported")
        return None
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    if res.built.prob.ind.XM.shape[0] == 0:
        path = out / "model.npz"
        np.savez(path, **linear_model_arrays(res))
    else:
        path = out / "gp_model.npz"
        np.savez(path, **gp_model_arrays(res, n_draws))
    return path


def load_gp_model(path):
    """(FittedGP, meta) from a gp_model.npz; see GPCalculator.from_file."""
    from ...calc.gp import FittedGP
    from ...eval import load
    from ..inducing import GPConfig, Inducing
    from ..kernels import KernelSpec
    from ..objective import Problem
    z = np.load(path)
    info = json.loads(bytes(z["gp_json"]).decode())
    if info["schema_version"] != GP_SCHEMA:
        raise ValueError(f"unsupported gp_model schema_version {info['schema_version']}")
    buf = io.BytesIO()
    np.savez(buf, **{k[4:]: z[k] for k in z.files if k.startswith("ace/")})
    buf.seek(0)
    model, meta, _ = load(buf)
    ind = Inducing(jnp.asarray(z["ind_XM"]), jnp.asarray(z["ind_SM"]), jnp.asarray(z["ind_ZM"]),
                   jnp.asarray(z["ind_scale"]), jnp.asarray(z["ind_Pmap"]), info["warp"],
                   jnp.asarray(z["ind_embed"]))
    prob = Problem(KernelSpec(**info["kernel"]), model, ind, GPConfig(**info["gpcfg"]), None, None)
    posts = [(jnp.asarray(m), jnp.asarray(L)) for m, L in zip(z["mu"], z["L"])]
    return FittedGP(prob, np.asarray(z["draws"]), posts), meta
