import time
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ...construct.prior import prior_diagonal
from ...eval import highest_precision
from ..embedding import load_mace_embedding
from ..hypers import default_prior
from ..inducing import GPConfig, build_pmap, descriptor_scale, select_inducing, site_features
from ..kernels import KernelSpec
from ..objective import Problem


class Built(NamedTuple):
    prob: object; gpcfg: object; X: object; S: object; s_floor: object; timings: dict


def build_problem(cfg, d):
    t = time.time()
    meta = d.meta
    els = [int(e) for e in meta["elements"]]
    gpcfg = GPConfig(r0=cfg.r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                     NZ=len(els), C=cfg.batch)
    with highest_precision():
        X, S = site_features(d.model, gpcfg, d.ds_train)
        scale = descriptor_scale(X, d.ds_train.node_mask)
        m = cfg.m_per_species if cfg.arm == "gp" else 0
        Pmap = build_pmap(gpcfg, scale, density=None if cfg.density == "none" else cfg.density,
                          d=cfg.pca_d, X=np.asarray(X), mask=np.asarray(d.ds_train.node_mask))
        raw = None if cfg.embedding is None else np.asarray(load_mace_embedding(cfg.embedding, els))
        embed = None if raw is None else jnp.asarray(raw)
        ind = select_inducing(X, S, d.ds_train.node_z, d.ds_train.node_mask, m, scale,
                              Pmap=Pmap, warp=cfg.warp, embed=embed, nz=len(els),
                              de=(int(embed.shape[1]) if embed is not None else None))
        s_floor = None
        if cfg.delta_s_floor_q is not None:
            s_live = np.asarray(S)[np.asarray(d.ds_train.node_mask)]
            s_floor = float(np.quantile(s_live, cfg.delta_s_floor_q))
        prob = Problem(KernelSpec(cfg.kernel, cfg.bump, gpcfg.D, s_floor=s_floor), d.model, ind, gpcfg,
                       jnp.asarray(prior_diagonal(d.z, meta, cfg.model)), default_prior(cfg.r0))
    return Built(prob, gpcfg, X, S, s_floor, {"inducing": time.time() - t})
