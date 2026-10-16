"""Silicon QoIs of the linear ACE order-4 ladder evaluated in float32, against float64 (docs/dev/fp32-results.md).

Variants per size s (silicon-testing-framework py3 port, models/<name>/model.py; share/force_only_opt.py):
  ACE_s         float64, the framework's relaxations (PreconLBFGS, Armijo line search): the plotted QoIs
  ACEode_s      float64, relaxed by ODE12r (SI_TESTS_OPTIMIZER=ode12r: forces only, adaptive step control)
  ACEf32ode_s   float32, ODE12r -- against ACEode_s: the evaluation precision alone
  ACEf32_s      float32, Armijo (most relaxations fail: the line search cannot use float32 energies)
  ACEf32e0_s    float32, Armijo, energies referenced to E0 (does not rescue them)
  ACEna_s, ACEf32na_s   use_armijo=False -- which in ASE's PreconLBFGS falls back to a Wolfe line search,
                         still on energies (float32 fails the same way)

Per QoI the difference in units of its score tolerance (bench/si_qoi/score.py), and each variant's QoI score.

    uv run --no-sync python bench/fp32/qoi_compare.py [--json out.json]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "si_qoi"))
import compare  # noqa: E402
from compare import qois  # noqa: E402
from score import TOLERANCES, score  # noqa: E402

_load = compare.load


def _load_complete(model, test):
    """compare.load, but a properties file whose E(V) curve is empty (an aborted relaxation's partial
    output) counts as missing, as a failed test does."""
    d = _load(model, test)
    if d and any(k.endswith("E_vs_V") and len(v) == 0 for k, v in d.items()):
        return None
    return d


compare.load = _load_complete

SIZES = ("o4d12", "o4d14", "o4d16", "o4d20")
VARIANTS = ("ACE", "ACEode", "ACEf32ode", "ACEf32", "ACEf32e0", "ACEna", "ACEf32na")


def tol_of(k, ref):
    for key, tol, kind in TOLERANCES:
        if key == k:
            return tol * abs(ref[k]) if kind == "rel" else tol
    return None


def phonon_rms(q, ref):
    if "_phonon_f" in q and "_phonon_f" in ref and np.shape(q["_phonon_f"]) == np.shape(ref["_phonon_f"]):
        return float(np.sqrt(np.mean((np.asarray(q["_phonon_f"]) - np.asarray(ref["_phonon_f"])) ** 2)))
    return None


def diff_in_tol(a, b, ref):
    """{QoI: (b - a) / tolerance} over the scored QoIs both have; the phonon rms as the change of the
    rms error against DFT."""
    out = {}
    for k, tol, kind in TOLERANCES:
        if kind == "rms":
            pa, pb = phonon_rms(a, ref), phonon_rms(b, ref)
            if pa is not None and pb is not None:
                out[k] = (pb - pa) / tol
            continue
        if a.get(k) is not None and b.get(k) is not None and ref.get(k) is not None:
            out[k] = (b[k] - a[k]) / tol_of(k, ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    ref = qois("CASTEP")
    res = {}
    for s in SIZES:
        q = {v: qois(f"{v}_{s}") for v in VARIANTS}
        sc = {v: score(f"{v}_{s}", ref) for v in VARIANTS}
        r = {"n_qoi": {v: sc[v]["n_qoi"] for v in VARIANTS},
             "qoi_score": {v: sc[v]["gm_norm"] if sc[v]["n_qoi"] == sc[v]["n_total"] else None for v in VARIANTS},
             "optimiser_effect": diff_in_tol(q["ACE"], q["ACEode"], ref),        # float64: PreconLBFGS vs ODE12r
             "precision_effect": diff_in_tol(q["ACEode"], q["ACEf32ode"], ref)}  # ODE12r: float64 vs float32
        for k in ("optimiser_effect", "precision_effect"):
            d = r[k]
            r[k + "_max"] = max((abs(v), q_) for q_, v in d.items()) if d else None
        res[s] = r
        print(f"== {s}: QoIs computed {r['n_qoi']}; score {({v: None if x is None else round(x, 3) for v, x in r['qoi_score'].items()})}")
        for k in ("optimiser_effect", "precision_effect"):
            d = r[k]
            if d:
                worst = sorted(d.items(), key=lambda kv: -abs(kv[1]))[:4]
                print(f"   {k:17s} max |diff|/tol {max(abs(v) for v in d.values()):.3f}  worst: "
                      + ", ".join(f"{q_} {v:+.3f}" for q_, v in worst))
    # elastic constants with the elastic fits' internal relaxation converged (SI_TESTS_ELASTIC_FMAX=1e-3):
    # ACEt (L-BFGS), ACEodet (ODE12r), ACEf32odet (ODE12r, float32); bulk_diamond only
    keys = ("diamond_bulk_modulus", "diamond_c11", "diamond_c12", "diamond_c44")
    print("\nelastic constants (GPa), default fmax 0.05 vs tight 1e-3; differences in tolerances (10%):")
    for s in SIZES:
        q = {v: qois(f"{v}_{s}") for v in ("ACE", "ACEode", "ACEt", "ACEodet", "ACEf32odet")}
        row = {}
        for k in keys:
            vals = {v: q[v].get(k) for v in q}
            row[k] = vals
            fmt = lambda x: "  -   " if x is None else f"{x:6.1f}"
            def d(a_, b_, vals=vals, k=k):
                if vals[a_] is None or vals[b_] is None:
                    return "  -  "
                return f"{(vals[b_] - vals[a_]) / tol_of(k, ref):+.3f}"
            print(f"  {s} {k:21s} ACE {fmt(vals['ACE'])} ode {fmt(vals['ACEode'])} | tight: lbfgs {fmt(vals['ACEt'])} "
                  f"ode {fmt(vals['ACEodet'])} ode-f32 {fmt(vals['ACEf32odet'])} | lbfgs->ode {d('ACEt', 'ACEodet')} "
                  f"f64->f32 {d('ACEodet', 'ACEf32odet')} default->tight(lbfgs) {d('ACE', 'ACEt')}")
        res.setdefault(s, {})["elastic"] = row
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
