"""XLA's compiled temp memory of the dense E/F/V call, chain vs DAG product basis.

  python bench/aa_dag/memory.py [--atoms 1000] [--models ...]"""
import argparse
import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from ace_jax.eval import load  # noqa: E402
from ace_jax.eval.aa_dag import with_aa_dag  # noqa: E402
from ace_jax.eval.model import highest_precision, lean  # noqa: E402
from ace_jax.eval.nlist import dense_from_sparse, sparse_graph  # noqa: E402

from run import path_of, structure  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ace/o4d16", "ace/o4d20", "pace/f800", "pace/f3200"])
    ap.add_argument("--atoms", type=int, default=1000)
    a = ap.parse_args(argv)
    at = structure(a.atoms)
    for name in a.models:
        m, meta, *_ = load(path_of(name))
        m = lean(m)
        g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
        d = dense_from_sparse(g, meta["rcut"])
        idx = jnp.asarray(d.idx)
        nz = jnp.zeros(len(at), jnp.int32)
        args = (jnp.asarray(d.rij), jnp.broadcast_to(nz[:, None], idx.shape), nz[idx], idx, jnp.asarray(d.mask), nz)
        for mode in ("none", "greedy"):
            mm = m if mode == "none" else with_aa_dag(m, mode)
            with highest_precision():
                c = jax.jit(lambda m, *x: m.energy_forces_virial_dense(*x)).lower(mm, *args).compile()
            ma = c.memory_analysis()
            print(json.dumps({"model": name, "mode": mode, "n_atoms": len(at), "K": int(idx.shape[1]),
                              "device": jax.devices()[0].platform,
                              "temp_MB": ma.temp_size_in_bytes / 2**20}), flush=True)


if __name__ == "__main__":
    main()
