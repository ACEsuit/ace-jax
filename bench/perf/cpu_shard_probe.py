"""In-process rank parallelism: the dense E / dE/drij / V call with its rows split
over D host devices (`--xla_force_host_platform_device_count=D`, `shard_map`), so
each device runs a 1/D slice of the atoms as its own program, as a
domain-decomposed rank would (docs/dev/cpu-gap-profile.md).

    PYTHONPATH=bench:src taskset -c 0-15 python bench/perf/cpu_shard_probe.py <model> <system> <n> \
        --devices 16 [--reps 10] [--eigen-single]

Times `energy_forces_virial_dense(..., return_edge_grad=True)` (what the skin
step calls; no compaction or force gather) unsharded on the default device and
sharded over D devices (rows split, E and the virial summed with psum), and
checks the sharded E and dE/drij against the unsharded ones.
"""
import argparse
import json
import os
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--devices", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--eigen-single", action="store_true",
                    help="also --xla_cpu_multi_thread_eigen=false")
    a = ap.parse_args()
    f = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={a.devices}"
    if a.eigen_single:
        f += " --xla_cpu_multi_thread_eigen=false"
    os.environ["XLA_FLAGS"] = f.strip()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, PartitionSpec as P

    from ace_jax.calc.point import ACECalculator
    from ace_jax.calc.skin import round_k
    from ace_jax.eval.nlist import dense_from_sparse, sparse_graph
    from scaling.structures import supercell

    calc = ACECalculator(a.model, dtype=jnp.float64)
    m = calc.eval_model
    at = supercell(a.system, a.n)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, calc.cutoff)
    K = round_k(int(np.bincount(np.asarray(g.senders), minlength=a.n).max()))
    d = dense_from_sparse(g, calc.cutoff, max_neighbours=K)
    lut = {int(z): i for i, z in enumerate(calc.meta["elements"])}
    node_z = jnp.asarray([lut[int(z)] for z in at.numbers], jnp.int32)
    idx = jnp.asarray(d.idx, jnp.int32)
    mask = jnp.arange(K)[None, :] < jnp.asarray(d.count)[:, None]
    pad = np.asarray([1.0, 0.0, 0.0]) * float(m.pad_cutoff())
    rij = jnp.asarray(np.where(np.asarray(mask)[..., None], d.rij, pad))
    zi = jnp.broadcast_to(node_z[:, None], idx.shape)
    zj = node_z[idx]
    params, static = eqx.partition(m, eqx.is_array)

    def local(p, rij, zi, zj, idx, mask, nz):
        with jax.default_matmul_precision("highest"):
            return eqx.combine(p, static).energy_forces_virial_dense(rij, zi, zj, idx, mask, nz,
                                                                     return_edge_grad=True)

    def timeit(f, *xs):
        for _ in range(2):
            jax.block_until_ready(f(*xs))
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); jax.block_until_ready(f(*xs)); ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    args = (rij, zi, zj, idx, mask, node_z)
    f1 = jax.jit(local)
    E1, g1, V1 = f1(params, *args)
    t1 = timeit(f1, params, *args)
    mesh = Mesh(np.array(jax.devices()[:a.devices]), ("d",))

    def sharded(p, rij, zi, zj, idx, mask, nz):
        E, gr, V = local(p, rij, zi, zj, idx, mask, nz)
        return jax.lax.psum(E, "d"), gr, jax.lax.psum(V, "d")
    row = P("d")
    fD = jax.jit(jax.shard_map(sharded, mesh=mesh, in_specs=(P(), row, row, row, row, row, row),
                               out_specs=(P(), row, P()), check_vma=False))
    ED, gD, VD = fD(params, *args)
    tD = timeit(fD, params, *args)
    out = json.dumps({"model": os.path.basename(a.model), "system": a.system, "n": a.n,
                      "devices": a.devices, "cpus": sorted(os.sched_getaffinity(0)),
                      "XLA_FLAGS": os.environ["XLA_FLAGS"], "efv_s_1dev": t1, "efv_s_sharded": tD,
                      "speedup": t1 / tD, "us_per_atom_1dev": t1 / a.n * 1e6,
                      "us_per_atom_sharded": tD / a.n * 1e6,
                      "dE": float(abs(ED - E1)), "dg_max": float(jnp.abs(gD - g1).max()),
                      "loadavg": open("/proc/loadavg").read().split()[:3]})
    print(out)
    res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cpu_gap", "shard.jsonl")
    with open(res, "a") as fh:
        fh.write(out + "\n")


if __name__ == "__main__":
    sys.exit(main())
