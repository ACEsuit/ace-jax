"""Acceptance scoring for the ace-jax `--uq ard` implementation (feat/ard-uq) on bench365.

Uses the pipeline's OWN tempered sigma (kappa from its internal 20 % train hold-out; F_var is
already kappa^2-tempered, and noise_F is NOT added).  Per family: rms-z, 90 % coverage, NLL,
Spearman rho(sigma, |dF|), AUROC(sigma: family vs in-distribution test).  Big cells (cracks,
dislocations): the same, on the free (non-fixed) atoms, plus a near-core subset.

Atoms in one configuration share their environment and are not independent (the big cells are
only 30 configs: 18 crack, 6 edge, 6 screw), so every metric carries a 95 % block-bootstrap
interval that resamples CONFIGURATIONS (AUROC: test configs and family configs resampled jointly).

As a diagnostic only, also reports kappa re-fitted prototype-style on half the test configs
(eval_joint.py), to separate "sigma ranks well" from "the train hold-out kappa transfers".

    python eval_ard.py <sites.npz> <ard_run_dir>
"""
import json
import sys

import numpy as np
from scipy.stats import chi2, spearmanr

S = np.load(sys.argv[1], allow_pickle=True)
run = sys.argv[2]
pt, po = np.load(f"{run}/pred_test_map.npz"), np.load(f"{run}/pred_ood_map.npz")
ard = json.load(open(f"{run}/ard.json"))
kappa = float(ard["kappa"])
e2 = {"test": np.sum((pt["F"] - pt["F_mean"]) ** 2, 1), "ood": np.sum((po["F"] - po["F_mean"]) ** 2, 1)}
s2 = {"test": pt["F_var"].sum(1), "ood": po["F_var"].sum(1)}          # tempered: kappa^2 * s_untempered^2
fam = {"test": np.where(S["fam_test"] == "bulk", "test bulk", "test defect"), "ood": S["fam_ood"]}
assert len(e2["test"]) == len(S["fam_test"]) and len(e2["ood"]) == len(S["fam_ood"]), "atom order mismatch"

try:
    B = np.load(f"{run}/big_err.npz")
    free = ~S["fixed_big"]
    e2["big"], s2["big"] = B["err"][free] ** 2, B["sd"][free] ** 2
    fb = S["fam_big"][free]
    near = S["rcore_big"][free] <= 10.0                                # within 10 A of crack tip / core
    cfg_big = S["cfg_big"][free]
    fam["big"] = np.array([f"big {f}" for f in fb])
    fam["near"] = np.array([f"big {f} (core<=10A)" for f in fb[near]])
    e2["near"], s2["near"] = e2["big"][near], s2["big"][near]
    cfgs = {"big": cfg_big, "near": cfg_big[near]}
except (FileNotFoundError, KeyError):
    cfgs = {}
    print("no big_err.npz with sd -- big cells skipped")


def auroc(neg, pos):
    r = np.concatenate([neg, pos]).argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def metrics(e2_, s2_, k2=1.0):
    v = k2 * s2_ / 3.0
    z2 = e2_ / v
    return {"n": int(len(e2_)), "rms_z": float(np.sqrt(np.mean(z2) / 3)), "cov90": float(np.mean(z2 <= chi2.ppf(0.9, 3))),
            "nll": float(np.mean(e2_ / (2 * v) + 1.5 * np.log(2 * np.pi * v))),
            "rho": float(spearmanr(s2_, e2_)[0]), "rmse_F": float(np.sqrt(np.mean(e2_) / 3))}


cfgs.update(test=S["cfg_test"], ood=S["cfg_ood"])
NBOOT = 500


def _groups(cfg):
    u, inv = np.unique(cfg, return_inverse=True)
    return u, inv


def boot(e2_, s2_, cfg, k2=1.0, neg=None):
    """95 % block-bootstrap CI over configurations for rms_z, cov90, nll (+ auroc vs neg=(s2, cfg))."""
    rng_b = np.random.default_rng(1)
    u, inv = _groups(cfg)
    v = k2 * s2_ / 3.0
    z2 = e2_ / v
    per = np.stack([np.bincount(inv, w, len(u)) for w in
                    (np.ones_like(z2), z2, (z2 <= chi2.ppf(0.9, 3)).astype(float),
                     e2_ / (2 * v) + 1.5 * np.log(2 * np.pi * v))])
    W = rng_b.multinomial(len(u), np.full(len(u), 1 / len(u)), NBOOT).astype(float)   # (NBOOT, ncfg)
    tot = W @ per.T                                                                     # (NBOOT, 4)
    out = {"rms_z": np.sqrt(tot[:, 1] / tot[:, 0] / 3), "cov90": tot[:, 2] / tot[:, 0], "nll": tot[:, 3] / tot[:, 0]}
    if neg is not None:
        ns2, ncfg = neg
        nu, ninv = _groups(ncfg)
        pos_ix = [np.flatnonzero(inv == c) for c in range(len(u))]
        neg_ix = [np.flatnonzero(ninv == c) for c in range(len(nu))]
        a = []
        for _ in range(min(NBOOT, 200)):
            pc = rng_b.integers(0, len(u), len(u))
            nc = rng_b.integers(0, len(nu), len(nu))
            a.append(auroc(ns2[np.concatenate([neg_ix[c] for c in nc])], s2_[np.concatenate([pos_ix[c] for c in pc])]))
        out["auroc"] = np.array(a)
    return {k: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] for k, x in out.items()}


# prototype-style diagnostic kappa: refit on a random half (by config) of the test split
rng = np.random.default_rng(0)
cfg_t = S["cfg_test"]
cal = np.isin(cfg_t, rng.permutation(np.unique(cfg_t))[: len(np.unique(cfg_t)) // 2])
k2_test = float(np.mean(e2["test"][cal] / (s2["test"][cal] / 3.0)) / 3.0)   # multiplies the already-tempered s2

res = {"kappa_train_holdout": kappa, "kappa_test_refit_factor": float(np.sqrt(k2_test)),
       "ard": {k: ard[k] for k in ard if k not in ("h_std",)}, "families": {}, "families_test_refit": {}}
for split in [k for k in ("test", "ood", "big", "near") if k in e2]:
    for f in dict.fromkeys(fam[split]):
        m = fam[split] == f
        if split == "test":
            m = m & ~cal                                               # hold the diagnostic's cal half out
        row = metrics(e2[split][m], s2[split][m])
        neg = None
        if f not in ("test bulk", "test defect"):
            row["auroc"] = auroc(s2["test"][~cal], s2[split][m])
            neg = (s2["test"][~cal], S["cfg_test"][~cal])
        row["n_cfg"] = int(len(np.unique(cfgs[split][m])))
        row["ci95"] = boot(e2[split][m], s2[split][m], cfgs[split][m], neg=neg)
        res["families"][f] = row
        res["families_test_refit"][f] = metrics(e2[split][m], s2[split][m], k2_test)

json.dump(res, open(f"{run}/ard_acceptance.json", "w"), indent=1)
print(f"kappa (train hold-out) = {kappa:.3f};  test-refit factor on top = {np.sqrt(k2_test):.3f}")


def ci(r, k):
    lo, hi = r["ci95"].get(k, [float("nan")] * 2)
    return f"{r.get(k, float('nan')):.2f} [{lo:.2f},{hi:.2f}]"


print(f"{'family':26s} {'cfg':>4s} {'atoms':>7s} {'rmseF':>6s} {'rms_z [95% CI]':>17s} {'cov90 [95% CI]':>17s} "
      f"{'rho':>5s} {'auroc [95% CI]':>17s} | rms_z cov90 (test-refit kappa)")
for f, r in res["families"].items():
    q = res["families_test_refit"][f]
    print(f"{f:26s} {r['n_cfg']:4d} {r['n']:7d} {r['rmse_F']:6.3f} {ci(r, 'rms_z'):>17s} {ci(r, 'cov90'):>17s} "
          f"{r['rho']:5.2f} {ci(r, 'auroc'):>17s} | {q['rms_z']:5.2f} {q['cov90']:5.2f}")
