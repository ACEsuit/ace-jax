import time
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ...basis.prior import prior_diagonal
from ...eval import highest_precision
from ..embedding import load_mace_embedding
from ..hypers import default_prior
from ..inducing import GPConfig, build_pmap, descriptor_scale, select_inducing, site_features
from ..kernels import KernelSpec

# Joint E0 (e0='lsq'): the E0 columns' prior is N(pre-fit E0, E0_PRIOR_STD^2) per species, in eV --
# wide against fit errors, but E0 is only weakly identified against the readout (a constant site
# energy trades off against near-constant basis functions), so the prior does set where it lands
# (tutorial 1: -160.2 eV at 1 eV, -159.7 at 10 eV); without an isolated atom, E0 is a reference
# level, not a free-atom energy
E0_PRIOR_STD = 1.0
E0_PINNED_STD = 1e-8     # a species with an isolated atom in training: its energy is E0 (see lsq_e0)


def _e0_prec(d, els):
    """The joint-E0 columns' prior precisions: wide, or pinned for species whose E0 an
    isolated training atom defines (it is predicted as E0 alone, so lsq_e0 set E0 to it)."""
    from .data import _isolated
    iso = {int(c.numbers[0]) for c in d.train if _isolated(c, float(d.meta["rcut"]))}
    return jnp.asarray([(E0_PINNED_STD if e in iso else E0_PRIOR_STD) ** -2.0 for e in els])
from ..objective import Problem


class Built(NamedTuple):
    prob: object; gpcfg: object; X: object; S: object; s_floor: object; timings: dict


def build_problem(cfg, d):
    t = time.time()
    meta = d.meta
    r0 = cfg.r0 if cfg.r0 is not None else d.r0
    if r0 is None:
        raise ValueError("r0 is not set and the model carries no basis r0: pass r0 (--r0)")
    els = [int(e) for e in meta["elements"]]
    gpcfg = GPConfig(r0=r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                     NZ=len(els), C=int(d.ds_train.y_E.shape[1]),   # the Dataset's C (packing can change it)
                     e0_cols=cfg.joint_e0)
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
                       jnp.asarray(prior_diagonal(d.z, meta, d.source)), default_prior(r0),
                       e0_prec=_e0_prec(d, els) if cfg.joint_e0 else None, lml_solver=cfg.lml_solver)
    return Built(prob, gpcfg, X, S, s_floor, {"inducing": time.time() - t})
