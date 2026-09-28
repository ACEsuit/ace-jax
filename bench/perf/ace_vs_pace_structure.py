"""Structural quantities of the benchmark ACE (.npz) and PACE (.yace) models, side
by side, and the per-node / per-edge work they imply (docs/ace-vs-pace-gap.md §1).

    PYTHONPATH=bench:src uv run python bench/perf/ace_vs_pace_structure.py [--out JSON]

No GPU needed: everything here is read off the loaded model plus the neighbour
count K of the benchmark structure.  The work estimates follow the code paths
the dense force call runs today:

  ACE  (ACEModel.site_energies_dense, folded):
    per edge  spline radial: 4-row gather of coefs[zi, zj] (4 * n_rnl reads), Y_lm
    per node  A_full = sum_k Rnl_k (x) Y_k        n*K*n_rnl*n_Y MACs  (pool_a_dense)
              A = A_full[:, sel]                  n_A columns
              AA product basis                    sum_o (o-1) n_o mults; adjoint o*n_o
              readout AA . ctilde[:, z]           n_AA (gathered ctilde)
  PACE (PACEModel.site_energies_dense, pool-first):
    per edge  g_k (nradbase), Y_lm
    per node  Ag = sum_k onehot(zj) g_k (x) Y_k   n*K*(C*nb)*n_Y MACs
              Agy = Ag @ sel_y                    n*C*nb*n_Y*n_a MACs
              At = W[zi] . Agy                    n*C*nb*n_a MACs
              AA product basis over C*n_a         as above
              rho = ctilde^T AA                   NZ*P*n_AA MACs (all NZ, then select)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS = os.path.join(ROOT, "bench", "scaling", "models")


def k_of(system, rc):
    import numpy as np
    from ase.neighborlist import neighbor_list

    from scaling.structures import supercell
    at = supercell(system, 1024)
    i = neighbor_list("i", at, rc)
    c = np.bincount(i, minlength=len(at))
    return int(c.max()), float(c.mean())


def prod_counts(aa_specs):
    by = {int(s.shape[1]): int(s.shape[0]) for s in aa_specs}
    fwd = sum((o - 1) * n for o, n in by.items())                 # multiplies per node
    gath = sum(o * n for o, n in by.items())                      # gathered factors per node
    return by, fwd, gath


def ace_row(path, system):
    import numpy as np
    from ace_jax.eval.io import load
    m, meta, _ = load(path)
    n_rnl, n_y = m.edge_a_widths()
    nz = int(m.E0.shape[0])
    K, Kmean = k_of(system, float(meta["rcut"]))
    by, fwd, gath = prod_counts(m.aa_specs)
    n_aa = sum(by.values())
    n_a = int(m.aspec_r.shape[0])
    ncoef = int(m.rnl_coefs.shape[-2]) if m.radial_kind == "spline" else None
    # distinct (n, l) and max l actually used by A
    ls = np.floor(np.sqrt(np.asarray(m.aspec_y))).astype(int)
    used_r = np.unique(np.asarray(m.aspec_r))
    nzm = np.abs(np.asarray(m.rnl_coefs)).max(axis=2) > 0          # (zi, zj, r)
    per_edge_nonzero = int(nzm[0].sum(axis=-1).max())               # Rnl columns live on one edge
    used = {"n_rnl_used": int(used_r.size), "n_Y_used": int((ls.max() + 1) ** 2),
            "rnl_nonzero_per_edge": per_edge_nonzero,
            "rnl_used_nonzero_per_edge": int(nzm[0][:, used_r].sum(axis=-1).max()),
            "A_full_per_node": int(n_rnl * n_y),
            "A_full_used_frac": n_a / float(n_rnl * n_y)}
    return {**used,"model": os.path.basename(path), "kind": "ACE", "system": system, "NZ": nz,
            "rcut": float(meta["rcut"]), "K": K, "K_mean": Kmean,
            "radial_kind": m.radial_kind, "ncoef": ncoef, "n_rnl": int(n_rnl),
            "lmax": int(m.lmax), "lmax_used": int(ls.max()), "n_Y": int(n_y),
            "C": 1, "n_A": n_a, "n_A_total": n_a,
            "n_AA": n_aa, "aa_by_order": by, "max_order": max(by),
            "n_B": int(m.WB.shape[0]), "n_pair": int(m.pair_coefs.shape[-1]),
            "prod_mults_per_node": fwd, "prod_gathered_per_node": gath,
            # widths per node
            "pooled_per_node": int(n_rnl * n_y),                  # A_full
            "per_edge_cols": int(n_rnl + n_y),                    # Rnl + Y per slot
            "radial_table_bytes": int(nz * nz * (ncoef or 0) * n_rnl * 8),
            # MACs per node (x K where per-slot)
            "pool_macs_per_node": int(K * n_rnl * n_y),
            "readout_macs_per_node": n_aa,
            "radial_reads_per_node": int(K * 4 * n_rnl)}


def pace_row(path, system):
    from ace_jax.eval.pace_model import load_yace
    from scaling.models import pace_functions_per_element
    m, meta, _ = load_yace(path)
    nz = int(m.E0.shape[0])
    K, Kmean = k_of(system, float(meta["rcut"]))
    nb, ny = m.pool_first_widths()
    by, fwd, gath = prod_counts(m.aa_specs)
    n_a = int(m.n_a_local)
    C = nz
    return {"model": os.path.basename(path), "kind": "PACE", "system": system, "NZ": nz,
            "rcut": float(meta["rcut"]), "K": K, "K_mean": Kmean,
            "nradbase": int(nb), "nradmax": int(m.crad.shape[2]), "lmax": int(m.lmax),
            "n_Y": int(ny), "C": C, "n_A": n_a, "n_A_total": C * n_a,
            "n_AA": int(m.n_aa), "aa_by_order": by, "max_order": max(by),
            "n_B": pace_functions_per_element(path), "ndensity": int(m.ndensity),
            "n_ctilde_terms": int(m.ctilde_complex.shape[0]),
            "prod_mults_per_node": fwd, "prod_gathered_per_node": gath,
            "pooled_per_node": int(C * nb * ny),                  # Ag
            "agy_per_node": int(C * nb * n_a),                    # Agy
            "per_edge_cols": int(C * nb + ny),                    # one-hot g + Y per slot
            "pool_macs_per_node": int(K * C * nb * ny + C * nb * ny * n_a + C * nb * n_a),
            "readout_macs_per_node": int(nz * m.ndensity * m.n_aa)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ace_vs_pace_structure.json"))
    a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    rows = []
    for system in ("SiGe", "Cantor"):
        for size in ("small", "medium", "large"):
            rows.append(ace_row(os.path.join(MODELS, f"ace_{system}_{size}.npz"), system))
            rows.append(pace_row(os.path.join(MODELS, f"pace_{system}_{size}.yace"), system))
            for r in rows[-2:]:
                r["size"] = size
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    cols = ["model", "NZ", "K", "n_B", "n_rnl", "nradbase", "nradmax", "lmax", "n_Y", "C",
            "n_A_total", "n_AA", "max_order", "aa_by_order", "prod_gathered_per_node",
            "pooled_per_node", "pool_macs_per_node", "readout_macs_per_node"]
    for r in rows:
        print(" | ".join(f"{c}={r.get(c)}" for c in cols if c in r))
    print("->", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
