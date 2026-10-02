"""A/B of ACEModel's product-basis layout (Task 9): the energy+forces+virial call
(`energy_forces_virial_dense`, readout folded) timed for each variant in ONE
process on one device, interleaved over rounds so drift hits all equally.

    PYTHONPATH=bench:src python bench/perf/ab_ace_fm.py <npz> <system> <n> \
        [--dtype float64] [--reps 20] [--rounds 5] [--variants node_major,fm,fm_T]

Variants (all the same model; E, F, V are checked against `node_major`):
  node_major  ACEModel as committed: node-major A (n, n_A) from `pool_a_dense`,
              `_aa` gathering columns, readout einsum against ctilde[:, z]
  fm          the Task 9 candidate: At (n_A, n) by an einsum with the node axis
              last + row select, `_aa` on rows, readout ctilde.T @ AA
  fm_T        `fm` with At = pool_a_dense(...).T (node-major assembly, transposed)
  fm_T_ein    `fm_T` with the node-major readout form (attribution)
  nm_gemm     `node_major` with the GEMM readout (attribution)

Prints one JSON object.  Used by bench/perf/modal_profile.py::ab_fm.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_ace import timeit  # noqa: E402


def variant_classes():
    """The Task 9 candidates as ACEModel subclasses (the feature-major form was
    measured and not adopted, so it lives only here; see docs/dev/perf-ace-profile.md
    section 5).  `node_major` is ACEModel as committed."""
    import jax
    import jax.numpy as jnp

    from ace_jax.eval.model import ACEModel, pool_dense

    hi = jax.lax.Precision.HIGHEST

    class FeatureMajor(ACEModel):
        """At (n_A, n); `_aa` = prod(At[g.T], axis=0), as PACE's
        `_node_energies_t`; readout ctilde.T @ AA then the centre species' row."""
        def _aa_t(self, At):
            return jnp.concatenate([jnp.prod(At[g.T], axis=0) for g in self.aa_specs], axis=0)

        def _pool_a_t_dense(self, Rnl, Y):
            n, K, nr = Rnl.shape
            ny = Y.shape[-1]
            A_full = jnp.einsum("nkr,nky->ryn", Rnl, Y, precision=hi).reshape(nr * ny, n)
            return A_full[self.aspec_r * ny + self.aspec_y]

        def _readout_t(self, AA, Apair, node_z):
            e_all = jnp.matmul(self.ctilde.T, AA, precision=hi)             # (NZ, n)
            e = e_all[node_z, jnp.arange(AA.shape[1])]
            e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
            return e + self.E0[node_z]

        def site_energies_dense(self, rij, zi, zj, mask, node_z):
            n, K = mask.shape
            flat = lambda a: a.reshape(n * K, *a.shape[2:])
            Rnl, Y, Rpair = self._edge_factors_and_pair(flat(rij), flat(zi), flat(zj),
                                                        flat(mask))
            un = lambda a: a.reshape(n, K, -1)
            At = self._pool_a_t_dense(un(Rnl), un(Y))
            return self._readout_t(self._aa_t(At), pool_dense(un(Rpair), mask), node_z)

    class FmTransposed(FeatureMajor):
        """`fm` with At = pool_a_dense(...).T (node-major assembly, transposed)."""
        def _pool_a_t_dense(self, Rnl, Y):
            return self.pool_a_dense(Rnl, Y, None).T

    class FmEinsumReadout(FmTransposed):
        """attribution: `fm_T` with the node-major readout's form (ctilde[:, z]
        gathered, multiply-reduce) instead of the (NZ, n_AA) x (n_AA, n) GEMM."""
        def _readout_t(self, AA, Apair, node_z):
            e = jnp.einsum("ai,ai->i", AA, self.ctilde[:, node_z])
            e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
            return e + self.E0[node_z]

    class NodeMajorGemmReadout(ACEModel):
        """attribution: `node_major` with the GEMM readout, (AA @ ctilde)[i, z_i]."""
        def _readout_folded(self, A, Apair, node_z):
            e_all = jnp.matmul(self._aa(A), self.ctilde, precision=hi)      # (n, NZ)
            e = e_all[jnp.arange(A.shape[0]), node_z]
            e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
            return e + self.E0[node_z]

    return {"node_major": ACEModel, "fm": FeatureMajor, "fm_T": FmTransposed,
            "fm_T_ein": FmEinsumReadout, "nm_gemm": NodeMajorGemmReadout}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--variants", default="node_major,fm,fm_T")
    args = ap.parse_args()

    import dataclasses

    import jax
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    import jax.numpy as jnp
    import numpy as np
    from ase.neighborlist import neighbor_list

    from ace_jax.eval.io import load
    from ace_jax.eval.model import highest_precision
    from ace_jax.eval.nlist import dense_graph
    from scaling.structures import supercell

    dt = getattr(jnp, args.dtype)
    base, meta, _ = load(args.npz, dtype=dt)
    rc = float(meta["rcut"])
    at = supercell(args.system, args.n)
    i = neighbor_list("i", at, rc)
    K = int(np.bincount(i, minlength=len(at)).max())
    g = dense_graph(at.get_positions(), at.cell.array, at.pbc, rc, K)
    z2i = {int(z): k for k, z in enumerate(meta["elements"])}
    node_z = jnp.asarray(np.array([z2i[int(z)] for z in at.get_atomic_numbers()], np.int32))
    idx = jnp.asarray(g.idx, jnp.int32)
    count = jnp.asarray(g.count, jnp.int32)
    rij = jnp.asarray(g.rij, dt)
    zi = jnp.broadcast_to(node_z[:, None], idx.shape)
    zj = node_z[idx]
    mask = jnp.arange(idx.shape[1])[None, :] < count[:, None]
    dargs = (rij, zi, zj, idx, mask, node_z)

    classes = variant_classes()
    names = args.variants.split(",")
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    models = {v: classes[v](**fields) for v in names}
    efv = jax.jit(lambda m, *a: m.energy_forces_virial_dense(*a))

    out = {"npz": os.path.basename(args.npz), "system": args.system, "n": args.n,
           "dtype": args.dtype, "device": str(jax.devices()[0].device_kind),
           "jax": jax.__version__, "K": K, "variants": {}}
    with highest_precision():
        ref = None
        for v in names:
            E, F, V = jax.block_until_ready(efv(models[v], *dargs))
            E, F, V = float(E), np.asarray(F), np.asarray(V)
            if ref is None:
                ref = (E, F, V)
            out["variants"][v] = {
                "dE_rel": abs(E - ref[0]) / abs(ref[0]),
                "dF_max": float(np.max(np.abs(F - ref[1]))),
                "dV_max": float(np.max(np.abs(V - ref[2]))), "rounds_s": []}
        for _ in range(args.rounds):
            for v in names:
                out["variants"][v]["rounds_s"].append(
                    timeit(efv, models[v], *dargs, reps=args.reps))
    for v in names:
        r = out["variants"][v]
        r["median_s"] = statistics.median(r["rounds_s"])
        r["min_s"] = min(r["rounds_s"])
    b = out["variants"][names[0]]["median_s"]
    for v in names:
        out["variants"][v]["vs_" + names[0]] = out["variants"][v]["median_s"] / b - 1.0
    print(json.dumps(out))


if __name__ == "__main__":
    main()
