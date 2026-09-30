"""The work the lammps-jax dense bundle does, vs the standalone dense layout.

bench/scaling/run_lammps.py sizes the bundle from the rcut + skin list:
max_atoms = 1.1 x (owned + ghost shell), k_dense = k_max(rcut + skin) + 8,
and `export.lammps.make_energy_fn` evaluates the dense model on ALL max_atoms
rows (ghost rows included) with k_dense slots each.  lammps-jax already drops
skin pairs (r > cutoff) when packing edges, so only ~k(rcut) slots per owned
row are ever live.

    PYTHONPATH=bench:src python bench/perf/lammps_capacity.py pace_Cantor_medium Cantor 8192 65536
"""
import json
import sys

import numpy as np


def main():
    from ace_jax.eval.edge_model import estimate_a_bytes
    from ace_jax.eval.nlist import sparse_graph
    from ace_jax.eval.pace_model import load_yace
    from scaling.run_lammps import capacity
    from scaling.structures import supercell
    import jax
    jax.config.update("jax_enable_x64", True)

    name, system = sys.argv[1], sys.argv[2]
    model, meta, _ = load_yace(f"bench/scaling/models/{name}.yace")
    rc = float(meta["rcut"])
    for n in map(int, sys.argv[3:]):
        at = supercell(system, n)
        cap = capacity(at, rc)
        g = sparse_graph(at.positions, at.cell.array, at.pbc, rc)
        k_cut = int(np.bincount(g.senders, minlength=n).max())
        k_std = -(-k_cut // 4) * 4                       # ACECalculator's _round_k
        bundle_slots = cap["max_atoms"] * cap["k_dense"]
        std_slots = n * k_std
        owned_cut_slots = int(np.ceil(1.1 * n)) * (k_cut + 4)
        out = {"model": name, "n": n, "rcut": rc, **cap, "k_cut": k_cut,
               "bundle_rows": cap["max_atoms"], "bundle_slots": bundle_slots,
               "standalone_slots": std_slots, "owned_rows_cut_slots": owned_cut_slots,
               "bundle_over_standalone": round(bundle_slots / std_slots, 2),
               "bundle_over_owned_cut": round(bundle_slots / owned_cut_slots, 2),
               "est_A_GB_bundle": estimate_a_bytes(model, "dense", cap["max_atoms"],
                                                   cap["max_edges"], cap["k_dense"], 8) / 1e9,
               "est_A_GB_owned_cut": estimate_a_bytes(model, "dense", int(1.1 * n),
                                                      n * k_cut, k_cut + 4, 8) / 1e9}
        print(json.dumps(out))


if __name__ == "__main__":
    main()
