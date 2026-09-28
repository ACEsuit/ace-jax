"""Evaluate bayes_joint.py's posteriors (blr / ardG / ardJ / ardJm), each on its OWN mean's errors:
kappa fitted by NLL on the calibration half of the in-distribution test configs (as calibrate.py),
then per family rms-z, 90 % coverage, NLL, AUROC(sigma: family vs test), rho(sigma, |dF|).

    python eval_joint.py <sites.npz> <pops_run_dir> <joint.npz>
"""
import json
import sys

import numpy as np
from scipy.stats import chi2, spearmanr

S = np.load(sys.argv[1], allow_pickle=True)
pt, po = np.load(f"{sys.argv[2]}/pred_test_map.npz"), np.load(f"{sys.argv[2]}/pred_ood_map.npz")
Z = np.load(sys.argv[3])
lab = {"test": pt["F"], "ood": po["F"]}
rng = np.random.default_rng(0)
cfg_t = S["cfg_test"]
cal = np.isin(cfg_t, rng.permutation(np.unique(cfg_t))[: len(np.unique(cfg_t)) // 2])
fam = {"test": np.where(S["fam_test"] == "bulk", "test bulk", "test defect"), "ood": S["fam_ood"]}


def auroc(neg, pos):
    r = np.concatenate([neg, pos]).argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


res = {}
for m in ("blr", "ardG", "ardJ", "ardJm"):
    e2 = {k: np.sum((lab[k] - Z[f"{k}/F_{m}"]) ** 2, 1) for k in ("test", "ood")}
    s2 = {k: Z[f"{k}/sdF_{m}"] ** 2 for k in ("test", "ood")}
    k2 = np.mean(e2["test"][cal] / (s2["test"][cal] / 3.0)) / 3.0          # NLL-optimal kappa^2 (closed form)
    row = {"kappa": float(np.sqrt(k2)), "test_F_rmse": float(np.sqrt(np.mean(e2["test"][~cal]) / 3))}
    for k in ("test", "ood"):
        for f in dict.fromkeys(fam[k]):
            msk = (fam[k] == f) & (~cal if k == "test" else True)
            z2 = e2[k][msk] / (k2 * s2[k][msk] / 3.0)
            v = k2 * s2[k][msk] / 3.0
            row[f] = {"rms_z": float(np.sqrt(np.mean(z2) / 3)), "cov90": float(np.mean(z2 <= chi2.ppf(0.9, 3))),
                      "nll": float(np.mean(e2[k][msk] / (2 * v) + 1.5 * np.log(2 * np.pi * v))),
                      "rho": float(spearmanr(s2[k][msk], e2[k][msk])[0])}
            if k == "ood":
                row[f]["auroc"] = auroc(s2["test"], s2[k][msk])
    res[m] = row
json.dump(res, open(sys.argv[3].replace(".npz", "_eval.json"), "w"), indent=1)
fams = ["test bulk", "test defect"] + list(dict.fromkeys(S["fam_ood"]))
for metric in ("rms_z", "cov90", "nll", "rho", "auroc"):
    print(f"\n{metric:6s} | " + " | ".join(fams))
    for m, row in res.items():
        print(f"{m:6s} k={row['kappa']:.2f} | " + " | ".join(f"{row[f].get(metric, float('nan')):.2f}" for f in fams))
print("\ntest F rmse (own mean):", {m: round(r["test_F_rmse"], 4) for m, r in res.items()})
