"""Within-cell ranking of local force errors (for adaptive refinement / active learning).

Scores, all on the ARD run's own errors:
  ard      tempered ARD posterior sigma (epistemic, kappa-scaled)
  pair     kNN distance (k=5) in the whitened 2-body descriptor columns, per species (the
           spike's best big-cell ranker)
  combo    sigma^2 = a * sigma_ard^2 + b * d_pair^c, (a, b, c) fitted by Gaussian NLL on the
           calibration half of the in-distribution TEST split only -- never on OOD / big cells
Metrics per family: Spearman rho(score, |dF|); the oracle rho a perfect sigma of the same spread
would reach given one chi^2_3 draw per atom; recall of the top-1 % error atoms in the top 10 %
by score; rms-z / cov90 for the calibrated variances (ard, combo).

    python rank_local.py <sites.npz> <ard_run_dir>
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, spearmanr

S = np.load(sys.argv[1], allow_pickle=True)
run = sys.argv[2]
pt, po, B = (np.load(f"{run}/{f}") for f in ("pred_test_map.npz", "pred_ood_map.npz", "big_err.npz"))
free = ~S["fixed_big"]
e2 = {"test": np.sum((pt["F"] - pt["F_mean"]) ** 2, 1), "ood": np.sum((po["F"] - po["F_mean"]) ** 2, 1),
      "big": B["err"][free] ** 2}
s2 = {"test": pt["F_var"].sum(1), "ood": po["F_var"].sum(1), "big": B["sd"][free] ** 2}
fam = {"test": np.where(S["fam_test"] == "bulk", "test bulk", "test defect"), "ood": S["fam_ood"],
       "big": np.array([f"big {f}" for f in S["fam_big"][free]])}
cfg = {"test": S["cfg_test"], "ood": S["cfg_ood"], "big": S["cfg_big"][free]}
X = {"test": S["X_test"], "ood": S["X_ood"], "big": S["X_big"][free]}
Z = {"test": S["Z_test"], "ood": S["Z_ood"], "big": S["Z_big"][free]}
NB = int(S["n_B"])
PAIR = np.concatenate([np.flatnonzero(S["order"] == 1), np.arange(NB, S["X_train"].shape[1])])


def knn_pair():
    out = {k: np.full(len(X[k]), np.nan) for k in X}
    Xtr, Ztr = S["X_train"], S["Z_train"]
    rng = np.random.default_rng(0)
    for z in np.unique(Ztr):
        T = Xtr[Ztr == z][:, PAIR].astype(np.float64)
        Ts = T[rng.choice(len(T), min(40000, len(T)), replace=False)]
        mu, sd = Ts.mean(0), Ts.std(0) + 1e-9
        _, sv, Vt = np.linalg.svd((Ts - mu) / sd, full_matrices=False)
        var = sv ** 2 / len(Ts)
        r = int(np.sum(var > 1e-12 * var[0]))
        W = Vt[:r].T / np.sqrt(var[:r])
        ref = ((T - mu) / sd) @ W
        t2 = np.sum(ref * ref, 1)
        for k in X:
            m = Z[k] == z
            Q = ((X[k][m][:, PAIR].astype(np.float64) - mu) / sd) @ W
            d = np.empty(len(Q))
            for i in range(0, len(Q), 4096):
                q = Q[i:i + 4096]
                d2 = np.maximum(np.sum(q * q, 1)[:, None] + t2[None, :] - 2 * q @ ref.T, 0)
                d[i:i + 4096] = np.sqrt(np.partition(d2, 5, axis=1)[:, :5]).mean(1)
            out[k][m] = d
    return out


dp = knn_pair()
rng = np.random.default_rng(0)
u = np.unique(cfg["test"])
cal = np.isin(cfg["test"], rng.permutation(u)[: len(u) // 2])


def combo_var(p, k):
    return np.exp(p[0]) * s2[k] + np.exp(p[1]) * dp[k] ** np.exp(p[2])


def nll(v, e):
    return np.mean(e / (2 * v / 3) + 1.5 * np.log(2 * np.pi * v / 3))


r = minimize(lambda p: nll(combo_var(p, "test")[cal], e2["test"][cal]), np.array([0.0, -3.0, 0.0]),
             method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
p = r.x
v_combo = {k: combo_var(p, k) for k in X}
frac_pair = {k: float(np.median(np.exp(p[1]) * dp[k] ** np.exp(p[2]) / v_combo[k])) for k in X}
print(f"combo: a={np.exp(p[0]):.3f}  b={np.exp(p[1]):.4g}  c={np.exp(p[2]):.3f}   "
      f"median pair share of variance: " + ", ".join(f"{k} {v:.2f}" for k, v in frac_pair.items()))


def row(score, e, v=None):
    top = e >= np.percentile(e, 99)
    sel = score >= np.percentile(score, 90)
    out = {"rho": float(spearmanr(score, e)[0]), "recall1@10": float((top & sel).sum() / top.sum())}
    if v is not None:
        z2 = e / (v / 3)
        out.update(rms_z=float(np.sqrt(np.mean(z2) / 3)), cov90=float(np.mean(z2 <= chi2.ppf(0.9, 3))))
    return out


res = {"combo_params": {"a": float(np.exp(p[0])), "b": float(np.exp(p[1])), "c": float(np.exp(p[2]))}, "families": {}}
print(f"{'family':14s} {'oracle':>6s} | {'rho ard':>7s} {'pair':>5s} {'combo':>5s} | {'rec ard':>7s} {'pair':>5s} {'combo':>5s} |"
      f" {'rmsz ard':>8s} {'combo':>5s} | {'cov90 ard':>9s} {'combo':>5s}")
for k in ("test", "ood", "big"):
    for f in dict.fromkeys(fam[k]):
        m = fam[k] == f
        if k == "test":
            m = m & ~cal
        eo = s2[k][m] * rng.chisquare(3, m.sum()) / 3                     # oracle: sigma exactly right
        a, pr, c = row(s2[k][m], e2[k][m], s2[k][m]), row(dp[k][m], e2[k][m]), row(v_combo[k][m], e2[k][m], v_combo[k][m])
        res["families"][f] = {"oracle_rho": float(spearmanr(s2[k][m], eo)[0]), "ard": a, "pair": pr, "combo": c}
        print(f"{f:14s} {res['families'][f]['oracle_rho']:6.2f} | {a['rho']:7.2f} {pr['rho']:5.2f} {c['rho']:5.2f} | "
              f"{a['recall1@10']:7.2f} {pr['recall1@10']:5.2f} {c['recall1@10']:5.2f} | {a['rms_z']:8.2f} {c['rms_z']:5.2f} | "
              f"{a['cov90']:9.2f} {c['cov90']:5.2f}")
json.dump(res, open(f"{run}/rank_local.json", "w"), indent=1)
