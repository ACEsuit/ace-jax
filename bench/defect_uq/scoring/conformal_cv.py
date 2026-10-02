"""Leave-one-realisation-out conformal calibration of the ARD force sigma at crack tips.

Pool: labelled in-distribution test + small-cell held-out atoms, plus the labelled v3 crack cells of
every realisation except the held-out one.  Target: the held-out realisation's crack cells (and, for
fold r in {0, 1}, its dislocation cells).  Methods at alpha = 0.1:
  A split     one global quantile
  B weighted  Tibshirani et al. 2019, logistic density ratio on whitened 2-body descriptors
  M Mondrian  per-group quantile, groups = local-distortion band x (coordination == 12)
Reported per fold and pooled: coverage for the crack (all, tip <= 10 A, 10-20 A, > 20 A) and the
dislocations; the spread across folds is the benchmark's noise.

    python conformal_cv.py <sites.npz> <ard_run_dir> <v3 part> [<v3 part> ...]
    each part = <err.npz>:<desc.npz>:<xyz>:<first realisation>.  The v3 main file (first = 0) holds 5 cells
    per realisation (3 cracks, edge, screw); the extra-crack files hold 3 cells (cracks) per realisation.
"""
import sys

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.optimize import minimize

ALPHA = 0.10
NN = 3.6502 / np.sqrt(2)
S = np.load(sys.argv[1], allow_pickle=True)
run = sys.argv[2]
parts = [p.split(":") for p in sys.argv[3:]]


def score(e2, s2):
    return np.sqrt(e2 / (s2 / 3))


def shell_strain(a):
    i, j, d = neighbor_list("ijd", a, 3.2)
    out = np.full(len(a), np.nan); o = np.lexsort((d, i)); i, d = i[o], d[o]
    st = np.searchsorted(i, np.arange(len(a)))
    for n in range(len(a)):
        dd = d[st[n]:st[n] + 12]
        if len(dd) == 12:
            out[n] = np.sqrt(np.mean((dd / NN - 1) ** 2))
    return out


def coord(a):
    i, _ = neighbor_list("ij", a, 3.1)
    return np.bincount(i, minlength=len(a))


# ---- pool (in-distribution + small-cell OOD)
P = {"s": [], "X": [], "Z": [], "st": [], "cn": []}
for split in ("test", "ood"):
    p = np.load(f"{run}/pred_{split}_map.npz")
    ats = read(f"{sys.argv[1].rsplit('/spikes', 1)[0]}/cantor/bench365/{split}.xyz", ":")
    P["s"].append(score(np.sum((p["F"] - p["F_mean"]) ** 2, 1), p["F_var"].sum(1)))
    P["X"].append(S[f"X_{split}"]); P["Z"].append(S[f"Z_{split}"])
    P["st"].append(np.concatenate([shell_strain(a) for a in ats])); P["cn"].append(np.concatenate([coord(a) for a in ats]))
P = {k: np.concatenate(v) for k, v in P.items()}
# ---- v3 big cells (all parts)
B = {k: [] for k in ("s", "X", "Z", "st", "cn", "fam", "rc", "real")}
for err_f, desc_f, xyz_f, roff in parts:
    C, Dd, cells = np.load(err_f, allow_pickle=True), np.load(desc_f), read(xyz_f, ":")
    free = ~C["fixed"].astype(bool)
    real = int(roff) + (C["cfg"] // 3) // (5 if int(roff) == 0 else 3)   # v3 main file: 5 cells/realisation
    B["s"].append(score(C["err"] ** 2, C["sd"] ** 2)[free]); B["X"].append(Dd["X"][free]); B["Z"].append(Dd["Z"][free])
    B["st"].append(np.concatenate([shell_strain(a) for a in cells])[free])
    B["cn"].append(np.concatenate([coord(a) for a in cells])[free])
    B["fam"].append(C["family"].astype(str)[free]); B["rc"].append(C["r_core"][free]); B["real"].append(real[free])
B = {k: np.concatenate(v) for k, v in B.items()}
NB = int(S["n_B"])
PAIR = np.concatenate([np.flatnonzero(S["order"] == 1), np.arange(NB, S["X_test"].shape[1])])
edges = np.nanpercentile(P["st"], [50, 90, 99])


def group(st, c):
    return np.searchsorted(edges, np.nan_to_num(st, nan=1.0)) * 2 + (c != 12)


def qfin(x):
    n = len(x)
    return np.quantile(x, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)) if n else np.inf


def whitener(X):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    _, sv, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    var = sv ** 2 / len(X); r = int(np.sum(var > 1e-10 * var[0]))
    W = Vt[:r].T / np.sqrt(var[:r])
    return lambda Y: ((Y - mu) / sd) @ W


def logistic(Xa, Xb, lam=1e-2):
    X = np.concatenate([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    wt = np.where(y == 1, len(Xa) / len(Xb), 1.0); X1 = np.hstack([X, np.ones((len(X), 1))])

    def f(b):
        z = X1 @ b
        return ((wt * (np.logaddexp(0, z) - y * z)).sum() / len(X) + lam * b[:-1] @ b[:-1] / 2,
                X1.T @ (wt * (1 / (1 + np.exp(-z)) - y)) / len(X) + lam * np.r_[b[:-1], 0])

    b = minimize(f, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    return lambda Y: np.hstack([Y, np.ones((len(Y), 1))]) @ b


def weighted_q(sc, wc, wt):
    o = np.argsort(sc); ss, cw = sc[o], np.cumsum(wc[o])
    idx = np.searchsorted(cw, (1 - ALPHA) * (cw[-1] + wt))
    q = np.full(len(wt), np.inf); ok = idx < len(ss); q[ok] = ss[idx[ok]]
    return q


reals = np.unique(B["real"])
groups = {"crack all": lambda f, r: f == "crack", "crack tip <=10": lambda f, r: (f == "crack") & (r <= 10),
          "crack 10-20": lambda f, r: (f == "crack") & (r > 10) & (r <= 20), "crack >20": lambda f, r: (f == "crack") & (r > 20),
          "edge": lambda f, r: f == "edge", "screw": lambda f, r: f == "screw"}
cov = {m: {g: [] for g in groups} for m in "ABM"}
hits = {m: {g: [0, 0] for g in groups} for m in "ABM"}
for r in reals:
    cal = (B["real"] != r) & (B["fam"] == "crack")
    tg = B["real"] == r
    s_cal = np.r_[P["s"], B["s"][cal]]; X_cal = np.r_[P["X"], B["X"][cal]]; Z_cal = np.r_[P["Z"], B["Z"][cal]]
    g_cal = np.r_[group(P["st"], P["cn"]), group(B["st"][cal], B["cn"][cal])]
    q = {m: np.empty(tg.sum()) for m in "ABM"}
    q["A"][:] = qfin(s_cal)
    qg = np.array([qfin(s_cal[g_cal == k]) for k in range(8)])
    q["M"] = qg[group(B["st"][tg], B["cn"][tg])]
    Zt, Xt = B["Z"][tg], B["X"][tg]
    for z in np.unique(Z_cal):
        mc, mt = Z_cal == z, Zt == z
        wh = whitener(X_cal[mc][:, PAIR].astype(np.float64))
        Pc, Pt = wh(X_cal[mc][:, PAIR].astype(np.float64)), wh(Xt[mt][:, PAIR].astype(np.float64))
        lo = logistic(Pc, Pt)
        q["B"][mt] = weighted_q(s_cal[mc], np.exp(np.clip(lo(Pc), -30, 30)), np.exp(np.clip(lo(Pt), -30, 30)))
    st, fam, rc = B["s"][tg], B["fam"][tg], B["rc"][tg]
    line = [f"fold realisation {r}:"]
    for g, fn in groups.items():
        m = fn(fam, rc)
        if m.sum() == 0:
            continue
        for k in "ABM":
            c = np.mean(st[m] <= q[k][m]); cov[k][g].append(c)
            hits[k][g][0] += int(np.sum(st[m] <= q[k][m])); hits[k][g][1] += int(m.sum())
        line.append(f"{g} A/B/M {cov['A'][g][-1]:.3f}/{cov['B'][g][-1]:.3f}/{cov['M'][g][-1]:.3f}")
    print("  ".join(line), flush=True)
print(f"\nleave-one-realisation-out over {len(reals)} realisations (alpha {ALPHA}): pooled coverage [fold min, max]")
for g in groups:
    if not hits["A"][g][1]:
        continue
    print(f"  {g:15s} " + "   ".join(f"{k} {hits[k][g][0] / hits[k][g][1]:.3f} [{min(cov[k][g]):.3f},{max(cov[k][g]):.3f}]"
                                         for k in "ABM") + f"   (folds {len(cov['A'][g])})")
