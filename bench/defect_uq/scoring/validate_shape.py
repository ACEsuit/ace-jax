"""Validation report for the jackknife-shape / conformal-scale force sigma (schema-3 ARD runs).

Per run directory (ard.json, posterior.npz, big3*_err.npz written by modal_bench365.big_errors):
  1. leverage summary (shape.lev_*), 2. global lambda_rms (T_val pooled, configuration-weighted),
  3. Spearman rho(sigma, |e|) per family, 4. coverage of the served region per conformal group, per
  family and per tip band, configuration- and atom-weighted, with 90 % configuration-bootstrap
  intervals (whole cells resampled, B = 1000), 5. crack-tip coverage (r_core <= 10 A),
  6. mean region volume, raw and rescaled to the nominal coverage, 7. log lambda_g against log N_fit
  across f-sweep runs, 8. median v and lambda_g against ell across ell-sweep runs.
Runs that predate forces_q / forces_group (or schema 3) are reported as far as their arrays allow and
marked n/a elsewhere.

    python validate_shape.py --runs DIR[:PAT] [DIR[:PAT] ...] --out report.md [--boot 1000] [--files PAT]

The bootstrap and tip band follow conformal_cv.py (cells = cfg // 3, tip = r_core <= 10 A); that file is
a script that runs on import, so its few lines are mirrored here rather than imported.
"""
import argparse
import fnmatch
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

TIP = 10.0
OPT = ("forces_q", "forces_group", "forces_cov", "dF")
NA = "n/a"


def load_run(d, files="big3*_err.npz"):
    """`files`: comma-separated fnmatch patterns on the err-file basenames (default all big3*_err.npz); use it to
    score only the cells a fold run did not train on.  Run dictionary: name, ard (ard.json or {}), force_shape/eps (posterior.npz), and the per-atom free-atom
    arrays of every big3*_err.npz (cell = (file, cfg // 3)); optional arrays only when present in all files."""
    d = str(d)
    ard = json.load(open(f"{d}/ard.json")) if os.path.exists(f"{d}/ard.json") else {}
    shape, eps = "iso", 1e-3
    if os.path.exists(f"{d}/posterior.npz"):
        z = np.load(f"{d}/posterior.npz")
        if "force_shape" in z.files:
            shape = str(z["force_shape"])
        if "eps" in z.files:
            eps = float(z["eps"])
    pats = files.split(",")
    files = sorted(f for f in glob.glob(f"{d}/big3*_err.npz") if any(fnmatch.fnmatch(os.path.basename(f), q) for q in pats))
    parts = [np.load(f, allow_pickle=True) for f in files]
    A = {}
    if parts:
        common = [k for k in OPT if all(k in p.files for p in parts)]
        for k, dst in (("err", "err"), ("sd", "sd"), ("family", "fam"), ("r_core", "rc")) + tuple((k, k) for k in common):
            if all(k in p.files for p in parts):
                A[dst] = np.concatenate([p[k][~p["fixed"].astype(bool)] for p in parts])
        A["cell"] = np.concatenate([i * 10**6 + (p["cfg"] // 3)[~p["fixed"].astype(bool)]
                                    for i, p in enumerate(parts)])
        if "fam" in A:
            A["fam"] = A["fam"].astype(str)
    return {"dir": d, "name": os.path.basename(d.rstrip("/")), "ard": ard, "force_shape": shape, "eps": eps,
            "A": A}


def _lam_atoms(R):
    A, g = R["A"], R["ard"].get("groups")
    if "forces_group" not in A or not g:
        return None
    return np.asarray(g["lam_rms"], float)[A["forces_group"].astype(int)]


def _hit(R):
    """(hit bool (n,), ratio (n,)) with hit == (ratio <= 1): |e| / forces_q (iso), or the Mahalanobis score of
    the stored error vector over the group's q (aniso, needs dF + forces_cov); None without forces_q."""
    A, g = R["A"], R["ard"].get("groups")
    if "forces_q" not in A:
        return None
    if R["force_shape"] == "aniso" and "forces_cov" in A and "dF" in A and g and "forces_group" in A:
        lam = _lam_atoms(R)
        V = A["forces_cov"].astype(float) / lam[:, None, None] ** 2
        M = V + R["eps"] * (np.trace(V, axis1=1, axis2=2) / 3)[:, None, None] * np.eye(3)
        e = A["dF"].astype(float)
        s = np.sqrt(np.einsum("na,na->n", e, np.linalg.solve(M, e[:, :, None])[:, :, 0]))
        q = np.asarray(g["q"], float)[A["forces_group"].astype(int)]
        r = s / q
    else:
        r = A["err"] / np.maximum(A["forces_q"], 1e-300)
    return r <= 1.0, r


def _masks(R):
    A = R["A"]
    if not A:
        return []
    fam, rc = A.get("fam"), A.get("rc")
    n = len(A["err"])
    out = [("all", np.ones(n, bool))]
    if fam is not None:
        out += [(f, fam == f) for f in ("crack", "edge", "screw") if (fam == f).any()]
        if rc is not None and (fam == "crack").any():
            c = fam == "crack"
            out += [("crack tip", c & (rc <= TIP)), ("crack 10-20", c & (rc > TIP) & (rc <= 20)),
                    ("crack >20", c & (rc > 20))]
    if "forces_group" in A:
        out += [(f"group {g}", A["forces_group"] == g) for g in np.unique(A["forces_group"])]
    return [(lab, m) for lab, m in out if m.any()]


def _cell_stats(hit, cell, m):
    u, inv = np.unique(cell, return_inverse=True)
    n = np.bincount(inv[m], minlength=len(u)).astype(float)
    h = np.bincount(inv[m], weights=hit[m].astype(float), minlength=len(u))
    return h, n


def _est(h, n, W):
    """atom- and cell-weighted coverage for bootstrap weight rows W (b, cells)."""
    ok = n > 0
    atom = (W @ h) / np.maximum(W @ n, 1e-300)
    cell = (W[:, ok] @ (h[ok] / n[ok])) / np.maximum(W[:, ok].sum(1), 1e-300)
    return atom, cell


def coverage_table(R, B=1000, seed=0):
    """Rows {label, n, ncell, atom, cell, ci, ci_atom}: coverage of the served region; None entries when the
    run has no forces_q.  ci / ci_atom are 90 % percentile intervals of the cell- / atom-weighted coverage over
    B resamples of whole cells."""
    rows = []
    h = _hit(R)
    for lab, m in _masks(R):
        row = {"label": lab, "n": int(m.sum()), "ncell": int(len(np.unique(R["A"]["cell"][m]))),
               "atom": None, "cell": None, "ci": None, "ci_atom": None}
        if h is not None:
            hc, nc = _cell_stats(h[0], R["A"]["cell"], m)
            ok = nc > 0                      # resample only the subset's own cells
            hc, nc = hc[ok], nc[ok]
            W = np.random.default_rng(seed).multinomial(len(nc), np.full(len(nc), 1 / len(nc)), B).astype(float)
            a1, c1 = _est(hc, nc, np.ones((1, len(nc))))
            ba, bc = _est(hc, nc, W)
            row.update(atom=float(a1[0]), cell=float(c1[0]),
                       ci=tuple(float(x) for x in np.percentile(bc, [5, 95])),
                       ci_atom=tuple(float(x) for x in np.percentile(ba, [5, 95])))
        rows.append(row)
    return rows


def volumes(R):
    """(mean region volume, mean volume rescaled to the nominal coverage) or None."""
    A, g, h = R["A"], R["ard"].get("groups"), _hit(R)
    if h is None:
        return None
    q = A["forces_q"].astype(float)
    if R["force_shape"] == "aniso" and "forces_cov" in A and g and "forces_group" in A:
        lam = _lam_atoms(R)
        V = A["forces_cov"].astype(float) / lam[:, None, None] ** 2
        M = V + R["eps"] * (np.trace(V, axis1=1, axis2=2) / 3)[:, None, None] * np.eye(3)
        qg = np.asarray(g["q"], float)[A["forces_group"].astype(int)]
        vol = 4 * np.pi / 3 * qg ** 3 * np.sqrt(np.linalg.det(M))
    else:
        vol = 4 * np.pi / 3 * q ** 3
    cov0 = 1.0 - float((g or {}).get("alpha", 0.1))
    c = float(np.quantile(h[1], cov0))
    return float(vol.mean()), float(vol.mean() * c ** 3)


def lambda_rms(R):
    g = R["ard"].get("groups")
    if not g or "n_cfg_val" not in g:
        return None
    w = np.asarray(g["n_cfg_val"], float)
    return float(np.sqrt(np.sum(w * np.asarray(g["lam_rms"], float) ** 2) / w.sum()))


def scalar_lam(R):
    """The #18 scalar lam of a sandwich run (ard.json "lam": kappa_closed_form on the T_val (e^2, v), still
    reported beside the per-group scales) -- the 30-Sep scalar-lambda comparison; None if absent."""
    a = R["ard"]
    lam = a.get("lam")
    return float(lam) if lam is not None and a.get("variance", "sandwich") == "sandwich" else None


def spearman_by_family(R):
    A = R["A"]
    if "fam" not in A or "sd" not in A:
        return {}
    return {f: float(spearmanr(A["sd"][A["fam"] == f], A["err"][A["fam"] == f])[0])
            for f in ("crack", "edge", "screw") if (A["fam"] == f).sum() > 2}


def _f(x, fmt="{:.3f}"):
    if isinstance(x, str):                       # ard.json stores non-finite floats as "inf"/"nan" strings
        try:
            x = float(x)
        except ValueError:
            return x
    return NA if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def _ci(c):
    return NA if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"


def f_sweep(runs):
    """Per conformal group (and pooled): least-squares slope of log lambda_g on log N_fit across runs."""
    pts = [(r["ard"]["transfer"]["N_fit"], r["ard"]["groups"]["lam_rms"], lambda_rms(r)) for r in runs
           if r["ard"].get("transfer", {}).get("N_fit") and r["ard"].get("groups")]
    if len({p[0] for p in pts}) < 2:
        return None
    x = np.log([p[0] for p in pts])
    out = {"pooled": float(np.polyfit(x, np.log([p[2] for p in pts]), 1)[0])}
    G = min(len(p[1]) for p in pts)
    for g in range(G):
        out[f"group {g}"] = float(np.polyfit(x, np.log([p[1][g] for p in pts]), 1)[0])
    return out


def ell_sweep(runs):
    """Rows (ell, median v, lambda_rms pooled, lam per group) sorted by ell; v = (sigma / lambda_g)^2."""
    rows = []
    for r in runs:
        ell = r["ard"].get("shape", {}).get("ell")
        if ell is None:
            continue
        lam = _lam_atoms(r)
        v = None if lam is None or "sd" not in r["A"] else float(np.median((r["A"]["sd"] / lam) ** 2))
        rows.append((float(ell), v, lambda_rms(r), (r["ard"].get("groups") or {}).get("lam_rms")))
    return sorted(rows, key=lambda t: t[0]) if len(rows) > 1 else None


def report(runs, B=1000):
    out = ["# Jackknife shape / conformal scale validation", ""]
    out += ["## Summary", "", "| run | variant | ell | K / K_fit | rank R | lev p50 / p99 / max | n lev~1 | lambda_rms | "
            "lam (scalar) | vol | vol @ nominal |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s = r["ard"].get("shape") or {}
        v = volumes(r)
        out.append("| {} | {} | {} | {} / {} | {} | {} / {} / {} | {} | {} | {} | {} | {} |".format(
            r["name"], s.get("variant", NA), _f(s.get("ell"), "{:g}"), s.get("K", NA), s.get("K_fit", NA),
            s.get("rank_R", NA), _f(s.get("lev_p50")), _f(s.get("lev_p99")), _f(s.get("lev_max")),
            s.get("n_lev_near1", NA), _f(lambda_rms(r), "{:.4f}"), _f(scalar_lam(r), "{:.4f}"),
            _f(v and v[0], "{:.4g}"), _f(v and v[1], "{:.4g}")))
    out += ["", "lambda_rms: T_val pooled, configuration-weighted over the conformal groups "
            "(sqrt(sum n_cfg_val lam_g^2 / sum n_cfg_val)).  lam (scalar): the #18 scalar lam of ard.json "
            "(atom-weighted kappa_closed_form over T_val), the 30-Sep scalar-lambda comparison.", ""]
    out += ["## Spearman rho(sigma, |e|) per family", "", "| run | crack | edge | screw |", "|---|---|---|---|"]
    for r in runs:
        rho = spearman_by_family(r)
        out.append(f"| {r['name']} | " + " | ".join(_f(rho.get(f)) for f in ("crack", "edge", "screw")) + " |")
    out += ["", f"## Coverage of the served region (nominal 1 - alpha; 90 % CI over whole cells, B = {B})", ""]
    for r in runs:
        out += [f"### {r['name']} ({r['force_shape']})", "",
                "| subset | atoms | cells | cell-weighted | CI | atom-weighted | CI |", "|---|---|---|---|---|---|---|"]
        for t in coverage_table(r, B):
            out.append(f"| {t['label']} | {t['n']} | {t['ncell']} | {_f(t['cell'])} | {_ci(t['ci'])} | "
                       f"{_f(t['atom'])} | {_ci(t['ci_atom'])} |")
        out.append("")
    fs = f_sweep(runs)
    out += ["## f sweep: slope of log lambda_g on log N_fit", ""]
    if fs:
        out += ["| quantity | slope |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in fs.items()]
    else:
        out.append("n/a (needs >= 2 runs with different N_fit)")
    es = ell_sweep(runs)
    out += ["", "## ell sweep: median v and lambda_g against ell", ""]
    if es:
        out += ["| ell | median v | lambda_rms | lambda_g |", "|---|---|---|---|"]
        out += [f"| {e:g} | {_f(v, '{:.4g}')} | {_f(lm, '{:.4f}')} | {_f_list(lg)} |" for e, v, lm, lg in es]
    else:
        out.append("n/a (needs >= 2 runs with a shape.ell)")
    return "\n".join(out) + "\n"


def _f_list(x):
    return NA if x is None else ", ".join(f"{v:.3f}" for v in x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs; DIR:PAT[,PAT] restricts that run to err files whose basename matches (fnmatch), "
                         "e.g. a fold run scored on the cells it did not train on")
    ap.add_argument("--files", default="big3*_err.npz", help="default err-file pattern(s) for runs without :PAT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    runs = []
    for spec in a.runs:
        d, _, pat = spec.partition(":")
        runs.append(load_run(d, pat or a.files))
    md = report(runs, a.boot)
    open(a.out, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
