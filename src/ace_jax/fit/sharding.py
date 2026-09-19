"""Batches sharded over a 1-d device mesh; each device streams its shard with
the Task 8 scan and the (Dt, Dt) statistics are all-reduced.  The only
communication per evaluation is that psum, so speed-up is linear in devices.
Multi-host: call jax.distributed.initialize() before building the mesh."""
import jax
import jax.numpy as jnp
from jax import shard_map                      # public since JAX 0.7; keyword is check_vma
from jax.sharding import NamedSharding, PartitionSpec as P

from .stats import Stats, sufficient_statistics


def sufficient_statistics_sharded(theta, spec, model, ind, cfg, ds, mesh):
    n = mesh.devices.size
    if ds.n_batches % n != 0:
        raise ValueError(f"ds.n_batches={ds.n_batches} is not divisible by mesh device count={n}")
    in_specs = (P(), jax.tree.map(lambda _: P("data"), ds))
    out_specs = Stats(*[P()] * len(Stats._fields))          # replicated after the psum

    def local(theta, shard):
        st = sufficient_statistics(theta, spec, model, ind, cfg, shard)
        return jax.tree.map(lambda x: jax.lax.psum(x, "data"), st)

    # JAX 0.10.1: eager (non-jitted) shard_map cannot run the closed_call that
    # jax.checkpoint (inside sufficient_statistics's scan body) introduces --
    # "Eager evaluation of `closed_call` inside a `shard_map` isn't yet
    # supported."  Wrapping the shard_map-decorated function in jax.jit, as
    # the error message directs, fixes it; the brief's snippet omits this.
    f = jax.jit(shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False))
    ds_sharded = jax.tree.map(lambda a: jax.device_put(a, NamedSharding(mesh, P("data"))), ds)
    return f(theta, ds_sharded)
