"""Per-stage GPU profile of the dense E/F/V call for an ACE (.npz) or PACE (.yace)
model, with the SAME kernel-to-stage attribution for both (docs/ace-vs-pace-gap.md §2).

    PYTHONPATH=bench:src python bench/perf/profile_gap.py <model file> <system> <n> \
        --trace DIR --dump DIR [--variant V]   (V: an ace_fast variant, ACE only)

Reuses `profile_ace.attribute` (kernels -> stages by the call chains of their
fused HLO instructions).  For PACE the stage map is swapped, and
`PACEModel._node_energies_t` is replaced by an op-for-op identical version whose
product basis and rho contraction sit in named helpers, so they attribute
separately.  Stages, common to both:

  radial   per-edge radial basis (ACE: transform, envelope, spline Rnl and the
           pair radial; PACE: g_k, the SBessel basis)
  angular  Y_lm
  core     PACE only: the per-edge core repulsion
  pool     A assembly (ACE pool_a_dense + the pair pool; PACE pool_first_dense
           incl. W = crad)
  aa       the product basis
  readout  ACE: AA . ctilde[:, z] + pair + E0; PACE: ctilde_real, rho = ct^T AA
  tail     PACE only: embedding F(rho), core switch, E0
  strain / force / other  as in profile_ace
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PACE_FN_STAGE = {
    "PACEModel.edge_basis_factors": "radial", "PACEModel._edge_core": "core",
    "EdgeSiteModel.pool_first_dense": "pool", "PACEModel.pool_first_weights": "pool",
    "_pace_aa": "aa", "_pace_readout": "readout", "PACEModel.ctilde_real": "readout",
    "PACEModel._energy_tail": "tail", "PACEModel._embedding": "tail",
    "PACEModel.site_energies_dense": "other",
    "EdgeSiteModel.energy_forces_virial_dense.<locals>.total": "strain",
    "EdgeSiteModel.energy_forces_virial_dense": "force"}
PACE_FILE_STAGE = {"harmonics.py": "angular"}
# the prototype's single fused method: attribute what we can by file
FAST_FN_STAGE = {"ACEBlocked.site_energies_dense": "pool+radial", "ACEModel._aa": "aa",
                 "ACEModel._readout_folded": "readout", "ACEModel.pair_radial": "radial",
                 "EdgeSiteModel.energy_forces_virial_dense.<locals>.total": "strain",
                 "EdgeSiteModel.energy_forces_virial_dense": "force"}


def _pace_aa(m, At):
    import jax.numpy as jnp
    return jnp.concatenate([jnp.prod(At[s.T], axis=0) for s in m.aa_specs], axis=0)


def _pace_readout(m, AA, n_nodes, node_z):
    import jax
    import jax.numpy as jnp
    ct = m.ctilde_real().reshape(m.n_aa, -1)
    rho_all = jnp.matmul(ct.T, AA, precision=jax.lax.Precision.HIGHEST)
    return rho_all.reshape(m.nz, m.ndensity, n_nodes)[node_z, :, jnp.arange(n_nodes)]


def _node_energies_t(self, At, cr, d, dcin, segment_ids, n_nodes, node_z):
    AA = _pace_aa(self, At)
    rho = _pace_readout(self, AA, n_nodes, node_z)
    return self._energy_tail(rho, cr, d, dcin, segment_ids, n_nodes, node_z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--trace", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--trace-calls", type=int, default=5)
    a = ap.parse_args()
    if a.dump:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   f" --xla_dump_to={a.dump} --xla_dump_hlo_as_text").strip()
    import jax
    jax.config.update("jax_enable_x64", a.dtype == "float64")
    import jax.numpy as jnp

    import ace_fast
    import profile_ace
    from ace_gap_bench import dense_args
    from ace_jax.eval import pace_model
    from ace_jax.eval.io import load
    from ace_jax.eval.model import highest_precision

    dt = getattr(jnp, a.dtype)
    is_pace = a.path.endswith(".yace")
    if is_pace:
        pace_model.PACEModel._node_energies_t = _node_energies_t
        profile_ace._FN_STAGE.clear(); profile_ace._FN_STAGE.update(PACE_FN_STAGE)
        profile_ace._FILE_STAGE.clear(); profile_ace._FILE_STAGE.update(PACE_FILE_STAGE)
    model, meta, _ = load(a.path, dtype=dt)
    if a.variant != "baseline":
        model = ace_fast.make(a.variant, model)
        profile_ace._FN_STAGE.update(FAST_FN_STAGE)
    _, args = dense_args(meta, a.system, a.n, dt)
    efv = jax.jit(lambda m, *x: m.energy_forces_virial_dense(*x))
    rij, zi, zj, idx, mask, node_z = args
    energy = jax.jit(lambda m, r, b, c, mk, nz: jnp.sum(m.site_energies_dense(r, b, c, mk, nz)))
    out = {"path": os.path.basename(a.path), "variant": a.variant, "n": a.n, "dtype": a.dtype,
           "K": int(mask.shape[1]), "device": str(jax.devices()[0].device_kind)}
    with highest_precision():
        t0 = time.perf_counter()
        jax.block_until_ready(efv(model, *args))
        out["first_call_s"] = time.perf_counter() - t0
        out["efv_s"] = profile_ace.timeit(efv, model, *args, reps=a.reps)
        out["energy_only_s"] = profile_ace.timeit(energy, model, rij, zi, zj, mask, node_z,
                                                  reps=a.reps)
        out["efv_cost"] = profile_ace.cost(efv, model, *args)
        if a.trace:
            with jax.profiler.trace(a.trace, create_perfetto_trace=True):
                for _ in range(a.trace_calls):
                    jax.block_until_ready(efv(model, *args))
            if a.dump:
                out["kernels"] = profile_ace.attribute(a.trace, a.dump, a.trace_calls, top=25)
    try:
        out["peak_bytes"] = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception:                                              # noqa: BLE001
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
