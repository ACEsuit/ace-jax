"""D4 of docs/dev/specs/2026-10-05-gp-discrepancy-design.md: the rev2 coverage table under two noise models.

For each run dir (model.npz, posterior.npz, ard.json, big3*_err.npz):
  - in-distribution coverage on the 300-config bench365 test sample of group_flag_default.py (rng(0)), served
    through ACECalculator(posterior=) and cached as <run>/id300.npz (s, g, cfg);
  - big-cell coverage of the served region (validate_shape: all / crack / tip / edge / screw, cell-weighted,
    90 % cell-bootstrap CI) and Spearman rho(sigma, |dF|) per family;
  - test RMSE (metrics.json) and the noise scales the ARD posterior used.

    uv run python d4_score.py <acegp-data> <run dir> [<run dir> ...] --out d4.md
"""
import argparse
import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from ace_jax.fit.ard import conformal_scores  # noqa: E402
from validate_shape import coverage_table, load_run  # noqa: E402

LIMIT = int(os.environ.get("D4_LIMIT", "0"))    # smoke runs: ID configs (0 = the 300-config sample)


def id300(D, run, log=print):
    idf = f"{run}/id300.npz"
    if os.path.exists(idf):
        return dict(np.load(idf))
    from ase import Atoms
    from ace_jax.calc.point import ACECalculator
    from ace_jax.fit.xyz import read_extxyz
    z = np.load(f"{run}/posterior.npz")
    tab = json.loads(bytes(z["group_table_json"]).decode())
    lam, eps = np.asarray(tab["lam_rms"], float), float(z["eps"])
    frames = read_extxyz(f"{D}/cantor/bench365/test.xyz")
    pick = np.sort(np.random.default_rng(0).choice(len(frames), 300, replace=False))
    if LIMIT:
        pick = pick[:LIMIT]
    if os.path.exists(f"{run}/gp_model.npz"):         # --uq ard-gp: the GP model with its ARD posterior
        from ace_jax.calc.gp import GPCalculator
        calc = GPCalculator.from_file(f"{run}/gp_model.npz", posterior=f"{run}/posterior.npz")
    else:
        calc = ACECalculator(f"{run}/model.npz", posterior=f"{run}/posterior.npz")
    out = {k: [] for k in ("s", "g", "cfg")}
    for n, i in enumerate(pick):
        a0 = frames[i]
        at = Atoms(numbers=a0.numbers, positions=a0.positions, cell=a0.cell, pbc=a0.pbc)
        at.calc = calc
        cv = np.asarray(calc.get_property("forces_cov", at))
        g = np.asarray(calc.get_property("forces_group", at)).astype(int)
        e = np.asarray(a0.arrays["mace_force"]) - at.get_forces()
        out["s"].append(conformal_scores(e, cv / lam[g][:, None, None] ** 2, "aniso", eps))
        out["g"].append(g); out["cfg"].append(np.full(len(g), i))
        if n % 50 == 0:
            log(f"{os.path.basename(run)}: id {n}", flush=True)
    res = {k: np.concatenate(v) for k, v in out.items()}
    if not LIMIT:
        np.savez(idf, **res)
    return res


def id_cov(I, q):
    h = I["s"] <= q[I["g"]]
    return float(np.mean([h[I["cfg"] == c].mean() for c in np.unique(I["cfg"])])), float(h.mean())


def row(D, run):
    R = load_run(run)
    q = np.array([np.inf if x == "inf" else x for x in R["ard"]["groups"]["q"]], float)
    I = id300(D, run)
    idc, ida = id_cov(I, q)
    cov = {r["label"]: r for r in coverage_table(R)}
    A = R["A"]
    rho = {f: spearmanr(A["sd"][A["fam"] == f], A["err"][A["fam"] == f])[0] for f in ("crack", "edge", "screw")}
    m = json.load(open(f"{run}/metrics.json"))
    rm = m.get("test/map", m.get("test", {}))
    h = R["ard"].get("h"); names = R["ard"].get("h_names", [])
    th = json.load(open(f"{run}/theta_map.json")) if os.path.exists(f"{run}/theta_map.json") else None
    sig = {q: float(np.exp(th[f"log_sigma_{q}"])) if th else float("nan") for q in "EFV"}
    return dict(name=R["name"], id_cfg=idc, id_atom=ida, cov=cov, rho=rho, rmse={k: rm[k]["rmse"] for k in "EFV" if k in rm},
                sig=sig, h=dict(zip(names, h or [])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [row(a.data, r) for r in a.runs]
    labs = ("all", "crack", "crack tip", "edge", "screw")
    fmt = lambda c: f"{c['cell']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]" if c and c["cell"] is not None else "n/a"
    L = ["# D4 noise-model sensitivity: rev2 coverage table\n",
         "| run | MAP σ_E / σ_F / σ_V | ID cfg-weighted (atom) | " + " | ".join(labs) + " | ρ crack / edge / screw | test RMSE E (meV/atom) / F (eV/Å) / V |",
         "|---|---|---|" + "---|" * len(labs) + "---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['sig']['E']:.3g} / {r['sig']['F']:.3g} / {r['sig']['V']:.3g} | "
                 f"{r['id_cfg']:.3f} ({r['id_atom']:.3f}) | " + " | ".join(fmt(r["cov"].get(x)) for x in labs)
                 + f" | {r['rho']['crack']:.2f} / {r['rho']['edge']:.2f} / {r['rho']['screw']:.2f} | "
                 + " / ".join(f"{r['rmse'][k]:.4g}" for k in "EFV" if k in r["rmse"]) + " |")
    L.append("\nARD hyperparameters (h):")
    for r in rows:
        L.append(f"- {r['name']}: " + ", ".join(f"{k} {v:+.3f}" for k, v in r["h"].items()))
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
