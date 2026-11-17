"""The linear ACE ladder as `.yace` files that `pair_style ye3t` reads (family `ace_cheb`): timing only.

    python -m accuracy_cost.ace_to_ye3t [SIZE ...]            (from bench/; default: SIZES)

ye3t-lammps's `.yace` reader (src/ye3t_yace_model.cpp) is stricter than ML-PACE's: `radbasename: ChebExpCos`
only (no `Tabulated`, so not the ace_tab files), no core repulsion or inner cutoff, exact key sets, and every
m-tuple together with its negation (not pacemaker's half basis).  So each ladder model is written once more:

- product basis: exact (ace_to_yace.products), `full=True`;
- radials: each R_nl(r) r^l the products read, and the rank-1 function (pair radial + order-1 terms), projected
  by least squares on K ChebExpCos functions g_k at decay LAM on [R_FIT, rcut], with the cosine taper narrowed to
  DCUT (the family's default 0.01 A taper is the one thing a polynomial in x(r) times the envelope cannot follow:
  at 0.01 the projection error is 5e-5 of the radial's maximum, at 0.001 it is 4e-11; o3d12).

Stock ML-PACE (product and recursive evaluators), ye3t and ace-jax's PACEModel all read the same file, so the
timing gate is same-file parity.  `register` also records how far the file is from the source `.npz`
(`source_error`: ace-jax on both, on the gate's rattled cells): the projection error, reported, not gated.
"""
import json

import numpy as np

from . import common

SIZES = ("o3d12", "o4d12", "o4d14", "o4d16", "o4d20")
LAM, K, R_FIT, DCUT, DELTA = 1.5, 48, 1.0, 0.001, 0.001
TOP_KEYS = {"elements", "E0", "deltaSplineBins", "embeddings", "bonds", "functions"}
BOND_KEYS = {"nradmax", "lmax", "nradbasemax", "radbasename", "radparameters", "radcoefficients", "prehc",
             "lambdahc", "rcut", "dcut", "rcut_in", "dcut_in", "inner_cutoff_type"}
FUNC_KEYS = {"mu0", "rank", "ndensity", "num_ms_combs", "mus", "ns", "ls", "ms_combs", "ctildes"}


def convert(path, lam=LAM, K=K, r_fit=R_FIT, dcut=DCUT):
    """(tree, info) for pair_style ye3t: ace_to_yace.convert's ChebExpCos form with the full m list, checked."""
    from .ace_to_yace import convert as to_yace
    tree, info = to_yace(path, "chebexpcos", K, r_fit, DELTA, lam=lam, dcut=dcut, full=True)
    check(tree)
    return tree, {**info, "lam": lam, "dcut": dcut}


def check(tree):
    """What ye3t's reader refuses, raised here: key sets, radial family, cutoffs, one linear density, and a
    conjugation-closed m list with equal coefficients on each m-tuple and its negation."""
    if set(tree) != TOP_KEYS:
        raise ValueError(f"top-level keys {sorted(tree)}")
    for b in tree["bonds"].values():
        if set(b) != BOND_KEYS:
            raise ValueError(f"bond keys {sorted(set(b) ^ BOND_KEYS)}")
        if b["radbasename"] != "ChebExpCos" or b["prehc"] != 0 or b["rcut_in"] != 0 or b["dcut_in"] != 0 \
                or not 0 < b["dcut"] < b["rcut"]:
            raise ValueError("ye3t reads ChebExpCos without core repulsion or inner cutoff, 0 < dcut < rcut")
    for e in tree["embeddings"].values():
        if e["ndensity"] != 1 or e["FS_parameters"][1] != 1.0:
            raise ValueError("ye3t reads one linear density")
    for fs in tree["functions"].values():
        for f in fs:
            if set(f) != FUNC_KEYS:
                raise ValueError(f"function keys {sorted(set(f) ^ FUNC_KEYS)}")
            r = f["rank"]
            ms = {tuple(f["ms_combs"][i * r:(i + 1) * r]): c for i, c in enumerate(f["ctildes"])}
            for m, c in ms.items():
                neg = _negated(f, m)
                if neg not in ms or abs(ms[neg] - c) > 1e-12 * max(abs(c), 1e-300):
                    raise ValueError(f"m-tuple {m} of ns {f['ns']} ls {f['ls']} has no equal conjugate partner")


def _negated(f, m):
    """The negated m-tuple in the order `convert` writes it: factors sorted by (n, l, m)."""
    return tuple(t[2] for t in sorted((n, l, -mm) for n, l, mm in zip(f["ns"], f["ls"], m)))


def source_error(npz, yace, cells=None):
    """ace-jax on the `.yace` (PACEModel) against ace-jax on the source `.npz`: max |dE|/atom (eV), max |dF| and
    rms dF (eV/A) over the timing gate's cells."""
    from scaling import parity
    if cells is None:
        from .timing import gate_cells
        cells = gate_cells()
    dE, dF = [], []
    for at in cells:
        E0, F0 = parity._acejax_ef(npz, at)
        E1, F1 = parity._acejax_ef(yace, at)
        dE.append(abs(E1 - E0) / len(at)); dF.append(np.asarray(F1) - np.asarray(F0))
    dF = np.concatenate([d.ravel() for d in dF])
    return {"max_dE_per_atom": float(max(dE)), "max_dF": float(np.abs(dF).max()),
            "rms_dF": float(np.sqrt(np.mean(dF ** 2)))}


def register(size, **kw):
    """runs/ace_cheb/<size>/model.yace + status.json (`source`: the ace fit, the gate's second reference)."""
    from ace_jax.eval.pace_io import dump_tree
    src = common.read_status(common.run_dir("ace", size))
    if not src or src.get("status") != "ok":
        raise RuntimeError(f"ace/{size}: no successful fit")
    from .timing import _local
    npz = _local(src["model"])
    d = common.run_dir("ace_cheb", size); d.mkdir(parents=True, exist_ok=True)
    tree, info = convert(npz, **kw)
    dump_tree(tree, d / "model.yace")
    info["source_error"] = source_error(npz, d / "model.yace")
    (d / "convert.json").write_text(json.dumps(info, indent=1))
    common.write_status(d, family="ace_cheb", size=size, status="ok", model=str(d / "model.yace"),
                        n_params=src.get("n_params"), source=src["model"], split_hash=src.get("split_hash"),
                        convert=info)
    return info


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:] or SIZES:
        print(s, json.dumps(register(s)))
