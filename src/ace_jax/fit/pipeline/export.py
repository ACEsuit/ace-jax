"""Fitted-model files.

A linear fit (M = 0) is an ordinary ACE potential: `model.npz` is the input
model file with the fitted readout (WB, Wpair) and E0 written in, so
`ace_jax.load`, `ACECalculator` and `ace-jax eval` read it unchanged.

A GP fit also needs the residual block, so `gp_model.npz` is self-contained:
the ACE arrays (prefixed "ace/", E0 fitted; with joint E0 the pre-fit E0, whose shift is in mu), the GP and kernel configuration,
the inducing set, and per hyperparameter draw the posterior (mu, L) -- the
same factorisation the pipeline's predictions use, so a reloaded
`GPCalculator` reproduces them.  L is (Dt, Dt), Dt = len_basis + M, per draw:
that, not the ACE model, sets the file size.
"""
import dataclasses
import io
import os
import json
import pathlib

import jax.numpy as jnp
import numpy as np

from ..hypers import from_array, to_array
from ..predict import _train_stats, fit_posterior, theta_key

GP_SCHEMA = 1


def _ace_arrays(res):
    return _ace_arrays_from(res.data.z, res.data.E0)


def _ace_arrays_from(z, E0):
    """The input model's npz arrays (`FitData.z`: read from the path or from the
    in-memory basis) with E0 as fitted."""
    out = {k: z[k] for k in z.files}
    out["E0"] = np.asarray(E0, np.float64)
    return out


def linear_arrays_from_mean(z, E0, pcfg, mu):
    """The ACE npz arrays of the input model (`z` = FitData.z) with readout mu, E0 as fitted.
    Column layout: species-major B blocks, then pair blocks (fit/rows.py `_place`)."""
    nB, nP, NZ = pcfg.n_B, pcfg.n_pair, pcfg.NZ
    mu = np.asarray(mu)
    if not np.isfinite(mu).all():           # a NaN model file would only fail far from its cause
        raise ValueError(f"the fitted readout has {int((~np.isfinite(mu)).sum())} non-finite "
                         "coefficients: the solve failed, so no model file is written")
    if getattr(pcfg, "e0_cols", False):      # joint E0: the fitted shift of the pre-fit E0
        E0 = np.asarray(E0, np.float64) + mu[pcfg.len_readout:pcfg.len_readout + NZ]
    out = _ace_arrays_from(z, E0)
    out["WB"] = mu[:NZ * nB].reshape(NZ, nB).T.copy()
    out["Wpair"] = mu[NZ * nB:NZ * (nB + nP)].reshape(NZ, nP).T.copy()
    return out


def model_file_blocked(cfg):
    """Why no model file can represent this fit (None when one can)."""
    if cfg.baseline is not None or cfg.base_npz is not None:
        return "the baseline is added outside the model"
    if isinstance(cfg.model, (str, os.PathLike)) and str(cfg.model).endswith(".yace"):   # a built basis is npz
        return ".yace inputs are not supported"
    return None


def _posterior(res, theta):
    """The training posterior at theta: the one the predictions factored when the fit kept it
    (res.posteriors, the MAP's), else from res.stats (the objective's, where they are bitwise a
    recompute's), else from a statistics pass over the training set."""
    hit = (getattr(res, "posteriors", None) or {}).get(theta_key(theta))
    if hit is not None:
        return hit
    return fit_posterior(theta, _train_stats(theta, res.built.prob, res.data.ds_train, getattr(res, "stats", None)),
                         res.built.prob, res.data.ds_train)


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
    if res.readout is not None:
        mu = res.readout                      # solver lstsq
    elif res.ard is not None:
        mu = res.ard.posterior.mean           # ARD: the posterior mean the predictions use
    elif "mean" in res.preds.pops:
        mu = res.preds.pops["mean"]      # POPS: exactly the mean the predictions used
    else:
        mu, _ = _posterior(res, res.theta)
    return linear_arrays_from_mean(res.data.z, res.data.E0, res.built.prob.cfg, mu)


def _gpcfg_json(gpcfg):
    """GPConfig for gp_json; e0_cols only when set, so prefit and e0='model' files stay
    readable by ace-jax releases that predate joint E0."""
    d = dataclasses.asdict(gpcfg)
    if not d.get("e0_cols"):
        d.pop("e0_cols", None)
    return d


def _ard_gp_posterior(post):
    """([theta], [mean], None) of an ard-gp ARDPosterior: the ARD mean is the model's, so GPCalculator
    predicts the served posterior's mean.  No full-posterior factor: the calibrated force UQ is served from
    posterior.npz, and the Dt x Dt factor (1.9 GB on the Cantor basis) would serve only an untempered
    energy_std."""
    return np.asarray(post.gp_theta)[None], [np.asarray(post.mean)], None


def gp_model_arrays(res, n_draws=1):
    prob = res.built.prob
    if getattr(res, "ard", None) is not None:        # uq ard-gp: the ARD posterior is the model
        draws, mus, Ls = _ard_gp_posterior(res.ard.posterior)
    else:
        draws = _draws(res, n_draws)
        mus, Ls = [], []
        for d in draws:
            mu, L = _posterior(res, from_array(jnp.asarray(d)))
            mus.append(np.asarray(mu)); Ls.append(np.asarray(L))
    # joint E0 (gpcfg.e0_cols): the E0 columns stay in mu and L, so GPCalculator's mean and
    # variance (its rows add the species counts too) are the fit's exactly; ace/E0 is the
    # pre-fit E0 they shift
    gpcfg = prob.cfg
    ind = prob.ind
    out = {f"ace/{k}": v for k, v in _ace_arrays(res).items()}
    out.update(ind_XM=np.asarray(ind.XM), ind_SM=np.asarray(ind.SM), ind_ZM=np.asarray(ind.ZM),
               ind_scale=np.asarray(ind.scale), ind_Pmap=np.asarray(ind.Pmap), ind_embed=np.asarray(ind.embed),
               draws=np.asarray(draws), mu=np.stack(mus))
    if Ls is not None:      # ard-gp: none -- the force UQ is posterior.npz's; L would serve only energy_std
        out["L"] = np.stack(Ls)
    info = {"schema_version": GP_SCHEMA, "gpcfg": _gpcfg_json(gpcfg),
            "kernel": dataclasses.asdict(prob.spec), "warp": ind.warp}
    from .config import resolved_noise
    if resolved_noise(res.config) != "per-quantity":   # only then: per-quantity files unchanged
        info["noise"] = resolved_noise(res.config)
    out["gp_json"] = np.frombuffer(json.dumps(info).encode(), np.uint8)
    return out


def save_model(res, out, n_draws=1, log=print):
    """Write the fitted model into directory `out`: model.npz (linear arm) or
    gp_model.npz (GP arm). Only the model: the fitted hyperparameters are
    `res.theta` (and `res.map.log_evidence`), and `write_outputs(res, out)` also
    writes them (theta_map.json) with the metrics.  Returns the path, or None when the fit cannot be
    represented as a model file (a dimer baseline is added back outside the model;
    a .yace input has no npz schema to write into)."""
    why = model_file_blocked(res.config)
    if why is not None:
        log(f"not saving a model file: {why}")
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
    # an ard-gp gp_model.npz has no L (its UQ is the posterior.npz): the posterior is then (mean, None)
    Ls = z["L"] if "L" in z.files else [None] * len(z["mu"])
    posts = [(jnp.asarray(m), None if L is None else jnp.asarray(L)) for m, L in zip(z["mu"], Ls)]
    return FittedGP(prob, np.asarray(z["draws"]), posts), meta
