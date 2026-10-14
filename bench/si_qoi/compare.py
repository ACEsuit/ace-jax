"""Compare models' silicon QoIs with the CASTEP (DFT) reference of libAtoms/silicon-testing-framework,
run under a local py3 port of that framework (~/silicon-testing-framework, branch py3; the upstream
repository has no licence, so none of its code is copied here).  GAP18_py3 is GAP-18 (models/GAP)
through the same port: it reproduces the GAP-18 paper's own numbers (arXiv:1805.01568, Fig. 1),
the check that the port keeps the protocol.

    python bench/si_qoi/compare.py MODEL [MODEL ...] [--md out.md] [--json out.json]
"""
import argparse, json, os
import numpy as np

FW = os.path.expanduser("~/silicon-testing-framework")
EV_A2_TO_J_M2 = 16.021766
PHASES = ("beta-Sn", "sh", "hex_diamond", "bc8", "st12", "hcp_short_and_fat", "hcp", "fcc", "bcc")
CASTEP_NAME = {"surface-energy-100-relaxed": "surface-energy", "surface-decohesion-100-unrelaxed": "surface-decohesion",
               "diinterstitial-64-energy": "diinterstitial-64", "stacking-fault-surface": "stacking-fault"}


def load(model, test):
    if model == "CASTEP":
        f = f"{FW}/test-results/model-CASTEP_ASE-test-{CASTEP_NAME.get(test, test)}-properties.json"
    else:
        f = f"{FW}/test-runner/model-{model}-test-{test}-properties.json"
    return json.load(open(f)) if os.path.exists(f) and os.path.getsize(f) else None


def ev_min(d):
    k = [k for k in d if k.endswith("E_vs_V")][0]
    A = np.array(sorted(d[k])); i = int(np.argmin(A[:, 1])); return A[i, 0], A[i, 1]


def qois(model):
    q = {}
    dia = load(model, "bulk_diamond")
    if dia:
        v0, e0 = ev_min(dia)
        for k in ("diamond_a0", "diamond_bulk_modulus", "diamond_c11", "diamond_c12", "diamond_c44"):
            q[k] = dia.get(k)
        for ph in PHASES:
            d = load(model, f"bulk_{ph}")
            if d:
                v, e = ev_min(d)
                q[f"dE_{ph} (meV/at)"] = 1e3 * (e - e0); q[f"V_{ph} (A3/at)"] = v
    for t, keys in (("vacancy-energy", ("vacancy_energy",)),
                    ("interstitial-energy", ("dumbbell_interstitial_energy", "hexagonal_interstitial_energy",
                                             "tetrahedral_interstitial_energy")),
                    ("diinterstitial-64-energy", ("diinterstitial_formation_energy_XT", "diinterstitial_formation_energy_W",
                                                  "diinterstitial_formation_energy_XX3", "diinterstitial_formation_energy_XEX",
                                                  "diinterstitial_formation_energy_EXT", "diinterstitial_formation_energy_TT"))):
        d = load(model, t)
        for k in keys:
            if d and d.get(k) is not None: q[k] = d[k]
    d = load(model, "surface-energy-100-relaxed")
    if d: q["surface_100 (J/m2)"] = d["surface_energy_100_relaxed"] * EV_A2_TO_J_M2
    d = load(model, "surface-decohesion-100-unrelaxed")
    if d:
        q["decohesion_max_stress"] = float(np.max(np.abs(d["surface_decohesion_unrelaxed_stress"])))
        q["_decohesion_E"] = np.asarray(d["surface_decohesion_unrelaxed_energy"], float)
    d = load(model, "stacking-fault-surface")
    if d:
        for k, name in (("glide_stacking_fault_path", "gamma_us glide (J/m2)"), ("shuffle_stacking_fault_path", "gamma_us shuffle (J/m2)")):
            q[name] = float(np.max(np.asarray(d[k], float)[:, 3])) * EV_A2_TO_J_M2   # columns: shift (3), energy / area
    d = load(model, "phonon_diamond")
    if d: q["_phonon_f"] = np.asarray(d["phonon_diamond_frequencies"], float)
    return q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="+"); ap.add_argument("--md"); ap.add_argument("--json")
    a = ap.parse_args()
    ref = qois("CASTEP"); rows = {m: qois(m) for m in a.models}
    for q in rows.values():
        if "_decohesion_E" in q and "_decohesion_E" in ref:
            q["decohesion_E rms vs DFT"] = float(np.sqrt(np.mean((q["_decohesion_E"] - ref["_decohesion_E"]) ** 2)))
        if "_phonon_f" in q and "_phonon_f" in ref and q["_phonon_f"].shape == ref["_phonon_f"].shape:
            q["phonon_f rms vs DFT (THz)"] = float(np.sqrt(np.mean((q["_phonon_f"] - ref["_phonon_f"]) ** 2)))
    keys = [k for k in ref if not k.startswith("_")] + ["decohesion_E rms vs DFT", "phonon_f rms vs DFT (THz)"]
    lines = ["| QoI | DFT | " + " | ".join(a.models) + " |", "|---|---|" + "---|" * len(a.models)]
    for k in keys:
        r = ref.get(k)
        cells = []
        for m in a.models:
            v = rows[m].get(k)
            if v is None: cells.append("-")
            elif r is None or "rms" in k: cells.append(f"{v:.3g}")
            elif k.startswith("dE_") or k.startswith("V_"): cells.append(f"{v:.3g} ({v - r:+.3g})")
            else: cells.append(f"{v:.3g} ({100 * (v - r) / abs(r):+.0f}%)")
        lines.append(f"| {k} | {'' if r is None else f'{r:.3g}'} | " + " | ".join(cells) + " |")
    text = "\n".join(lines); print(text)
    if a.md: open(a.md, "w").write(text + "\n")
    if a.json:
        json.dump({m: {k: v for k, v in q.items() if not k.startswith("_")} for m, q in {"CASTEP": ref, **rows}.items()},
                  open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
