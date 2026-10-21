"""One scorer for every family: ASE calculators on DATA/test.xyz, so no
fitter's own metrics enter the comparison.  Refuses a run fitted on another
split (its hash is in status.json)."""
import json
import pathlib
import sys

import numpy as np

from . import common
from .common import DATA, RESULTS


class SplitMismatch(Exception):
    pass


def _full(s):
    """ASE calculators' get_stress takes no `voigt` (only Atoms does) and returns Voigt-6 (xx yy zz yz xz xy)."""
    s = np.asarray(s, float)
    if s.shape == (3, 3): return s
    xx, yy, zz, yz, xz, xy = s
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def errors(atoms, calc):
    out = []
    for at in atoms:
        a = at.copy(); a.calc = None
        E, F = calc.get_potential_energy(a), calc.get_forces(a)
        dV = None
        if "virial" in at.info:                     # virial = -stress * volume (ASE sign convention)
            V = -_full(calc.get_stress(a)) * a.get_volume()
            dV = (V - np.reshape(at.info["virial"], (3, 3))) / len(at)
        out.append({"config_type": at.info["config_type"], "n": len(at),
                    "dE": (E - at.info["energy"]) / len(at), "dF": np.asarray(F) - at.arrays["forces"], "dV": dV})
    return out


def _rmse(rows):
    dE = np.array([r["dE"] for r in rows]); dF = np.concatenate([r["dF"].ravel() for r in rows])
    dV = [r["dV"].ravel() for r in rows if r["dV"] is not None]
    V = float(np.sqrt(np.mean(np.concatenate(dV) ** 2))) if dV else None
    return float(np.sqrt(np.mean(dE ** 2))), float(np.sqrt(np.mean(dF ** 2))), V


def summarise(rows):
    E, F, V = _rmse(rows)
    per = {}
    for t in sorted({r["config_type"] for r in rows}):
        sub = [r for r in rows if r["config_type"] == t]
        e, f, v = _rmse(sub)
        per[t] = {"rmse_E": e, "rmse_F": f, "rmse_V": v, "n_frames": len(sub)}
    return {"rmse_E": E, "rmse_F": F, "rmse_V": V, "per_type": per}


def load_calc(family, path, device="cpu"):
    path = str(path)
    if family in ("ace", "ace_lr"):              # ace_lr: lean + spline_tol="auto", as deployed
        from ace_jax.calc.point import ACECalculator
        return ACECalculator(path)
    if family == "ace_gp":
        from ace_jax.calc.gp import GPCalculator
        return GPCalculator.from_file(path)
    if family == "pace":
        from ace_jax.calc.point import ACECalculator         # ace-jax PACEModel; pyace cross-check in Task 6
        return ACECalculator(path)
    if family == "mace":
        from mace.calculators import MACECalculator
        if device != "gpu":            # Modal-trained models keep CUDA tensors: score a CPU copy
            from .fit_mace import cpu_copy
            path = str(cpu_copy(pathlib.Path(path).parent))
        return MACECalculator(model_paths=path, device="cuda" if device == "gpu" else "cpu",
                              default_dtype="float64")
    if family == "nep":                          # NEP_CPU via calorine, float64, E0 added back (fit_nep)
        from .fit_nep import calculator
        return calculator(path)
    if family in ("grace", "grace_fs", "grace_abl"):   # gracemaker's SavedModel (float64), E0 added back (fit_grace)
        from .fit_grace import calculator
        return calculator(path)
    if family in ("gap", "gap_noglue", "gap_E30"):  # noglue: the same loader, the XML just has no Glue term; E30: other sigmas
        import quippy
        return quippy.potential.Potential(param_filename=path)
    raise ValueError(family)


def score(family, size, device="cpu"):
    d = common.run_dir(family, size)
    st = common.read_status(d)
    if not st or st.get("status") != "ok":
        raise RuntimeError(f"{family}/{size}: no successful fit in {d}")
    want = json.loads((DATA / "split.json").read_text())["hash"]
    if st.get("split_hash") != want:
        raise SplitMismatch(f"{family}/{size} was fitted on split {st.get('split_hash')}, test is {want}")
    from .split import read_atoms
    s = summarise(errors(read_atoms(DATA / "test.xyz"), load_calc(family, st["model"], device)))
    row = {"family": family, "size": size, "model": st["model"], "split_hash": want,
           "n_params": st.get("n_params"), "fit_wall_s": st.get("fit_wall_s"), "fit_host": st.get("host"),
           **({"fit_gpu": st["gpu"]} if st.get("gpu") else {}), **s}
    common.upsert_jsonl(RESULTS / "metrics.jsonl", row, ("family", "size"))
    return row


if __name__ == "__main__":
    fam, sizes = sys.argv[1], sys.argv[2:]
    for s in sizes:
        r = score(fam, s)
        print(f"{fam}/{s}: rmse_E {r['rmse_E'] * 1e3:.2f} meV/atom  rmse_F {r['rmse_F']:.4f} eV/A")
