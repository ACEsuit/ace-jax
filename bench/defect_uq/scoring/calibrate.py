"""Step 2: calibrated per-atom force uncertainty on bench365.

Per atom, the force error dF (3-vector) is modelled as N(0, (s^2 / 3) I_3), s the per-atom sigma
(sqrt of the summed component variances), so |dF|^2 / (s^2 / 3) ~ chi^2_3.  Scale parameters are
fitted ONLY on a random half (by config) of the in-distribution test split, by Gaussian NLL, and
evaluated on the other half and on every held-out family.

  pops        s = sigma_POPS                                (as fitted; no parameters)
  pops_s      s = a * sigma_POPS                            (1 parameter)
  temp_blr    s = k * sigma_BLR       (tempered posterior, fitted Gamma prior)
  temp_ard    s = k * sigma_ARD       (tempered posterior, ARD per body order)
  pops+ard    s^2 = a^2 sigma_POPS^2 + k^2 sigma_ARD^2      (misspecification + tempered epistemic)

    python calibrate.py <sites.npz> <pops_run_dir> <bayes.npz>
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

sites, pops_dir, bayes = sys.argv[1], sys.argv[2], sys.argv[3]
S = np.load(sites, allow_pickle=True)
Z = np.load(bayes)
pt, po = np.load(f"{pops_dir}/pred_test_map.npz"), np.load(f"{pops_dir}/pred_ood_map.npz")
e2 = {"test": np.sum((pt["F"] - pt["F_mean"]) ** 2, 1), "ood": np.sum((po["F"] - po["F_mean"]) ** 2, 1)}
sig = {"pops": {"test": pt["F_var"].sum(1), "ood": po["F_var"].sum(1)},
       "blr": {k: Z[f"{k}/sdF_blr"] ** 2 for k in ("test", "ood")},
       "ard": {k: Z[f"{k}/sdF_ardG"] ** 2 for k in ("test", "ood")}}      # variances s^2

cfg_t = S["cfg_test"]
rng = np.random.default_rng(0)
cal_cfg = rng.permutation(np.unique(cfg_t))[: len(np.unique(cfg_t)) // 2]
cal = np.isin(cfg_t, cal_cfg)
fam_t = np.where(S["fam_test"] == "bulk", "test bulk", "test defect")

MODELS = {
    "pops": (0, lambda p, k: sig["pops"][k]),
    "pops_s": (1, lambda p, k: np.exp(2 * p[0]) * sig["pops"][k]),
    "temp_blr": (1, lambda p, k: np.exp(2 * p[0]) * sig["blr"][k]),
    "temp_ard": (1, lambda p, k: np.exp(2 * p[0]) * sig["ard"][k]),
    "pops+ard": (2, lambda p, k: np.exp(2 * p[0]) * sig["pops"][k] + np.exp(2 * p[1]) * sig["ard"][k]),
}


def nll(s2, e2_):
    v = s2 / 3.0
    return np.mean(e2_ / (2 * v) + 1.5 * np.log(2 * np.pi * v))


def metrics(s2, e2_):
    z2 = e2_ / (s2 / 3.0)                                # ~ chi^2_3 if calibrated
    out = {"rms_z": float(np.sqrt(np.mean(z2) / 3.0)), "nll": float(nll(s2, e2_))}
    for lvl in (0.5, 0.9, 0.95):
        out[f"cov{int(lvl * 100)}"] = float(np.mean(z2 <= chi2.ppf(lvl, 3)))
    return out


rows = {}
for name, (npar, f) in MODELS.items():
    if npar:
        r = minimize(lambda p, f=f: nll(f(p, "test")[cal], e2["test"][cal]), np.zeros(npar), method="Nelder-Mead")
        p = r.x
    else:
        p = np.zeros(0)
    res = {"params": np.exp(p).tolist()}
    s2t, s2o = f(p, "test"), f(p, "ood")
    for fam in ("test bulk", "test defect"):
        m = (fam_t == fam) & ~cal
        res[fam] = metrics(s2t[m], e2["test"][m])
    for fam in dict.fromkeys(S["fam_ood"]):
        m = S["fam_ood"] == fam
        res[fam] = metrics(s2o[m], e2["ood"][m])
    rows[name] = res

json.dump(rows, open(bayes.replace(".npz", "_calibration.json"), "w"), indent=1)
fams = ["test bulk", "test defect"] + list(dict.fromkeys(S["fam_ood"]))
for metric in ("rms_z", "cov90", "nll"):
    print(f"\n{metric} (ideal {'1.00' if metric == 'rms_z' else '0.90' if metric == 'cov90' else 'lower'}) | " + " | ".join(fams))
    for name, res in rows.items():
        print(f"{name:10s} {str(np.round(res['params'], 3)):>18s} | " + " | ".join(f"{res[f][metric]:.2f}" for f in fams))
