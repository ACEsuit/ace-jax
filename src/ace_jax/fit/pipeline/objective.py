import gc
import time
from typing import NamedTuple

import jax
import numpy as np
from jax.sharding import Mesh

from ..hypers import from_array, log_prior, to_array
from ..objective import make_lml, make_log_density
from ..stats import assemble_statistics, linear_statistics, residual_statistics


class Objective(NamedTuple):
    lik: object; vg: object; stats: object; host_cache: object; prior_mu: object; timings: dict


def _pad_to_multiple(ds, n):
    """Append inert copies of the last batch (weights, cfg_mask, node_mask and
    nbr_mask zeroed) so ds.n_batches is a multiple of n -- the sharded
    sufficient statistics need n_batches % n_devices == 0.  Moved verbatim from
    cli.py (which now imports it from here)."""
    import jax.numpy as jnp
    k = (-ds.n_batches) % n
    if k == 0:
        return ds
    last = jax.tree.map(lambda a: a[-1:], ds)
    last = last._replace(w_E=jnp.zeros_like(last.w_E), w_F=jnp.zeros_like(last.w_F),
                         w_V=jnp.zeros_like(last.w_V), cfg_mask=jnp.zeros_like(last.cfg_mask),
                         node_mask=jnp.zeros_like(last.node_mask),
                         nbr_mask=jnp.zeros_like(last.nbr_mask))
    pad = jax.tree.map(lambda a: jnp.concatenate([a] * k, axis=0), last)
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), ds, pad)


def make_objective(cfg, d, b):
    prob, t = b.prob, time.time()
    if cfg.lml == "host-cache":
        from ..hostcache import HostCachedLML
        lik = HostCachedLML(prob, d.ds_train, chunk=cfg.lml_chunk)
        prior_vg = jax.jit(jax.value_and_grad(lambda a: log_prior(from_array(a), prob.prior)))
        def vg(a):
            v, g = lik.value_and_grad(a); pv, pg = prior_vg(a)
            return v + pv, g + pg
        stats = lambda th: assemble_statistics(lik.lin, lik._residual_stats(to_array(th)))
        host = lik
    else:
        mesh = None
        if cfg.devices > 1:
            devs = jax.devices()[:cfg.devices]
            if len(devs) != cfg.devices:
                raise ValueError(f"devices={cfg.devices} requested, only {len(devs)} available")
            mesh = Mesh(np.array(devs), ("data",))
        if mesh is None and cfg.objective == "lml":
            lik = make_lml(prob, d.ds_train, cache_linear=True)
        else:
            ds_fit = _pad_to_multiple(d.ds_train, cfg.devices) if mesh is not None else d.ds_train
            lik = make_log_density(prob, ds_fit, cfg.objective, mesh=mesh).likelihood
        jax.block_until_ready(lik(to_array(prob.prior.mu)))
        logpost = jax.jit(lambda a: lik(a) + log_prior(from_array(a), prob.prior))
        vg = jax.jit(jax.value_and_grad(logpost))
        lin = jax.jit(lambda: linear_statistics(prob.model, prob.cfg, d.ds_train))()
        stats = lambda th: assemble_statistics(lin, residual_statistics(th, prob.spec, prob.model,
                                                                        prob.ind, prob.cfg, d.ds_train))
        host = None
    return Objective(lik, vg, stats, host, prob.prior.mu, {"stats_once": time.time() - t})


def release():
    """Clear JAX's compile caches and collect: once the caller has dropped its
    references to the likelihood, this frees its Gram-sized buffers (POPS then
    needs only the linear statistics; on the lossless Cantor base the LML holds
    ~150 GB and starved POPS's kernels)."""
    jax.clear_caches(); gc.collect()
