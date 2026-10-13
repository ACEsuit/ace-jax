"""Aggregate QoI score per model: each QoI's error against the CASTEP (DFT) reference divided by a
stated tolerance (1 = at tolerance), then the fraction of QoIs within tolerance and the RMS of the
normalised errors, and their geometric mean (each clipped to [GM_FLOOR, GM_CAP]): a continuous score,
where the fraction within tolerance jumps whenever one borderline QoI crosses 1.  The tolerances are a
choice, printed with every result so they can be argued with.

    python bench/si_qoi/score.py MODEL [MODEL ...] [--json out.json]
"""
import argparse, json, sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare import PHASES, qois  # noqa: E402

# (QoI key, tolerance, "rel" fraction of the DFT value | "abs" in the QoI's units)
TOLERANCES = (
    [(k, 0.10, "rel") for k in ("diamond_bulk_modulus", "diamond_c11", "diamond_c12", "diamond_c44")]
    + [(f"dE_{p} (meV/at)", 10.0, "abs") for p in PHASES]
    + [(k, 0.10, "abs") for k in ("vacancy_energy", "dumbbell_interstitial_energy", "hexagonal_interstitial_energy",
                                  "tetrahedral_interstitial_energy")]
    + [("surface_100 (J/m2)", 0.05, "rel"), ("gamma_us glide (J/m2)", 0.10, "rel"),
       ("gamma_us shuffle (J/m2)", 0.10, "rel"), ("decohesion_max_stress", 0.10, "rel")]
    + [("phonon_f rms vs DFT (THz)", 0.2, "rms")]
)
# Left out of the score (and the heatmap), with the reason.  Compressed ("short and fat") hcp lies outside the
# training data: every model fits the hcp configurations it has (GAP-18 and the GAP refit both ~1 meV/atom on
# train and test), and the QoI then measures extrapolation, where a soft region (GAP refit 60 meV/atom, MACE
# c16L0 1.6 eV/atom) swamps the other 21 QoIs.  compare.py still reports it.
EXCLUDED = {"dE_hcp_short_and_fat (meV/at)": "extrapolation: outside the training data"}
TOLERANCES = [t for t in TOLERANCES if t[0] not in EXCLUDED]


GM_CAP, GM_FLOOR = 10.0, 0.05   # the geometric mean clips each |error|/tolerance: one wild QoI (or an exact one) must not dominate


def normalised_errors(q, ref, signed=False):
    """{QoI: |error| / tolerance} over the QoIs the model has; `signed` keeps the sign of model - DFT
    (the phonon rms has none: it stays positive)."""
    out = {}
    for k, tol, kind in TOLERANCES:
        v = q.get(k)
        if v is None:
            continue
        if kind == "rms":
            out[k] = abs(v) / tol
        elif ref.get(k) is not None:
            err = v - ref[k]
            out[k] = (err if signed else abs(err)) / (tol * abs(ref[k]) if kind == "rel" else tol)
    return out


def score(model, ref):
    q = qois(model)
    ref_q = dict(ref)
    if "_phonon_f" in q and "_phonon_f" in ref_q and q["_phonon_f"].shape == ref_q["_phonon_f"].shape:
        q["phonon_f rms vs DFT (THz)"] = float(np.sqrt(np.mean((q["_phonon_f"] - ref_q["_phonon_f"]) ** 2)))
    ne = normalised_errors(q, ref_q)
    v = np.array(list(ne.values()))
    return {"n_qoi": len(v), "n_total": len(TOLERANCES), "frac_within": float(np.mean(v <= 1.0)) if len(v) else None,
            "rms_norm": float(np.sqrt(np.mean(v ** 2))) if len(v) else None,
            "gm_norm": float(np.exp(np.mean(np.log(np.clip(v, GM_FLOOR, GM_CAP))))) if len(v) else None,
            "normalised": ne, "signed": normalised_errors(q, ref_q, signed=True)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="+"); ap.add_argument("--json")
    a = ap.parse_args()
    ref = qois("CASTEP")
    res = {m: score(m, ref) for m in a.models}
    nan = lambda x: float("nan") if x is None else x
    print(f"{'model':28s} {'QoIs':>6s} {'within tol':>10s} {'rms(err/tol)':>12s} {'gmean(err/tol)':>14s}")
    for m, r in res.items():
        print(f"{m:28s} {r['n_qoi']:3d}/{r['n_total']:<2d} {nan(r['frac_within']):10.2f} {nan(r['rms_norm']):12.2f} {nan(r['gm_norm']):14.2f}")
    print("excluded:", "; ".join(f"{k} ({why})" for k, why in EXCLUDED.items()))
    print("tolerances:", "; ".join(f"{k} {t:g}{'' if kind == 'abs' else (' rel' if kind == 'rel' else ' rms')}" for k, t, kind in TOLERANCES))
    if a.json:
        json.dump({"tolerances": TOLERANCES, "scores": res}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
