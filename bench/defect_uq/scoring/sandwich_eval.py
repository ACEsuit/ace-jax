"""Score the sandwich spike (modal/sandwich_spike.py) against the tempered ARD sigma, on the ARD run's
own force errors.  Per variant: RAW calibration (the sandwich needs no kappa if it is right), the
calibration after one scale fitted on the in-distribution test calibration half (as eval_ard.py's
diagnostic), and the scale-free ranking metrics: Spearman rho(sigma^2, |dF|^2), recall of the
top-1 % error atoms in the top 10 % by sigma, AUROC(sigma: family vs test).

    python sandwich_eval.py <sites.npz> <ard_run_dir>
"""
import json
import sys

import numpy as np
from scipy.stats import chi2, spearmanr

S = np.load(sys.argv[1], allow_pickle=True)
run = sys.argv[2]
Z = np.load(f"{run}/sandwich.npz")
kappa = json.load(open(f"{run}/ard.json"))["kappa"]
pt, po, B = (np.load(f"{run}/{f}") for f in ("pred_test_map.npz", "pred_ood_map.npz", "big_err.npz"))
free = ~S["fixed_big"]
e2 = {"test": np.sum((pt["F"] - pt["F_mean"]) ** 2, 1), "ood": np.sum((po["F"] - po["F_mean"]) ** 2, 1),
      "big": B["err"][free] ** 2}
sel = {"test": slice(None), "ood": slice(None), "big": free}
fam = {"test": np.where(S["fam_test"] == "bulk", "test bulk", "test defect"), "ood": S["fam_ood"],
       "big": np.array([f"big {f}" for f in S["fam_big"][free]])}
V = {"ard": {k: kappa ** 2 * Z[f"s2_ard_{k}"][sel[k]] for k in e2}}
for v in ("hc0", "hc3", "pops", "atom", "cfg"):
    V[v] = {k: Z[f"s2_{v}_{k}"][sel[k]] for k in e2}
print("leverage quantiles (1,5,25,50,75,95,99 %):", np.round(Z["lev_quantiles"], 6).tolist(),
      " mean rho^2:", float(Z["mean_rho2"]), " kappa^2:", round(kappa ** 2, 2))

rng = np.random.default_rng(0)
u = np.unique(S["cfg_test"])
cal = np.isin(S["cfg_test"], rng.permutation(u)[: len(u) // 2])


def auroc(neg, pos):
    r = np.concatenate([neg, pos]).argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def metrics(v, e, neg=None):
    z2 = e / (v / 3)
    top, hi = e >= np.percentile(e, 99), v >= np.percentile(v, 90)
    out = {"rms_z": float(np.sqrt(np.mean(z2) / 3)), "cov90": float(np.mean(z2 <= chi2.ppf(0.9, 3))),
           "rho": float(spearmanr(v, e)[0]), "rec": float((top & hi).sum() / top.sum())}
    if neg is not None:
        out["auroc"] = auroc(neg, v)
    return out


res = {}
for name, vv in V.items():
    scale = float(np.mean(e2["test"][cal] / (vv["test"][cal] / 3)) / 3)            # NLL-optimal factor on test-cal
    res[name] = {"scale": scale, "families": {}}
    for k in ("test", "ood", "big"):
        for f in dict.fromkeys(fam[k]):
            m = fam[k] == f
            if k == "test":
                m = m & ~cal
            neg = None if k == "test" else vv["test"][~cal]
            raw = metrics(vv[k][m], e2[k][m], neg)
            raw["rms_z_scaled"] = float(np.sqrt(np.mean(e2[k][m] / (scale * vv[k][m] / 3)) / 3))
            res[name]["families"][f] = raw
json.dump(res, open(f"{run}/sandwich_eval.json", "w"), indent=1)

fams = list(res["ard"]["families"])
for metric in ("rms_z", "rms_z_scaled", "rho", "rec", "auroc"):
    print(f"\n{metric:12s} " + " ".join(f"{f.replace('test ', 't.').replace('big ', 'B.'):>9s}" for f in fams))
    for name, r in res.items():
        tag = f"{name}" + (f" x{r['scale']:.2f}" if metric == "rms_z_scaled" else "")
        print(f"{tag:12s} " + " ".join(f"{r['families'][f].get(metric, float('nan')):9.2f}" for f in fams))
