"""Conformal calibration of the ARD force sigma under covariate shift (big cells as the target).

Our labels are a deterministic function of the structure, so train -> big-cell is a pure covariate
shift (P(y|x) fixed; only which environments occur changes).  Calibration pool: the labelled
in-distribution test atoms plus the small-cell held-out families (ard_sw2 predictions).  Target:
the v3 big-cell atoms, whose structures (not labels) are known at prediction time.  Per-atom
nonconformity score s = |dF| / (sigma / sqrt 3) (chi_3 if sigma were a calibrated Gaussian scale).

  A  split conformal: one quantile of the pool scores.
  B  weighted conformal (Tibshirani, Barber, Candes, Ramdas 2019): w(x) = P(target|x)/P(pool|x) from a
     logistic classifier on whitened 2-body descriptors; per target atom the (1-alpha) quantile of
     sum_i w_i delta(s_i) + w(x) delta(+inf), normalised.
  C  localized (kNN) conformal: per target atom, the quantile of its k nearest pool atoms' scores
     (same species, same whitened 2-body space), with the 1/(k+1) mass at +inf.
Coverage is P(s <= q(x)) on the target atoms (their MACE errors are used only for scoring).  Atoms in
a cell are correlated, so the per-atom guarantee is approximate; n_eff is reported.

    python conformal_shift.py <sites.npz> <ard_run_dir> <big3_desc.npz> [cal_realisation]

cal_realisation (0 or 1): add that species realisation's labelled v3 CRACK cells to the pool and
score only the other realisation's cells (target-regime calibration data under the same weighting).
"""
import sys

import numpy as np
from scipy.optimize import minimize

ALPHA, K = 0.10, 200
S = np.load(sys.argv[1], allow_pickle=True)
run = sys.argv[2]
Dsc = np.load(sys.argv[3])
pt, po = np.load(f"{run}/pred_test_map.npz"), np.load(f"{run}/pred_ood_map.npz")
C = np.load(f"{run}/big3_err.npz", allow_pickle=True)


def score(e2, s2):
    return np.sqrt(e2 / (s2 / 3))


pool_s = np.concatenate([score(np.sum((p["F"] - p["F_mean"]) ** 2, 1), p["F_var"].sum(1)) for p in (pt, po)])
pool_X = np.concatenate([S["X_test"], S["X_ood"]])
pool_Z = np.concatenate([S["Z_test"], S["Z_ood"]])
pool_cfg = np.concatenate([S["cfg_test"], 10 ** 6 + S["cfg_ood"]])
free = ~C["fixed"].astype(bool)
assert len(Dsc["X"]) == len(C["err"]), "big3 descriptor / error order mismatch"
real = (C["cfg"] // 3 >= 5).astype(int)                  # cells 0-4: realisation 0, 5-9: realisation 1
famall = C["family"].astype(str)
sc_all = score(C["err"] ** 2, C["sd"] ** 2)
CAL_REAL = int(sys.argv[4]) if len(sys.argv) > 4 else None
if CAL_REAL is not None:
    add = free & (real == CAL_REAL) & (famall == "crack")
    pool_s = np.concatenate([pool_s, sc_all[add]])
    pool_X = np.concatenate([pool_X, Dsc["X"][add]])
    pool_Z = np.concatenate([pool_Z, Dsc["Z"][add]])
    pool_cfg = np.concatenate([pool_cfg, 2 * 10 ** 6 + C["cfg"][add] // 3])
    free = free & (real != CAL_REAL)
    print(f"pool += realisation {CAL_REAL} crack cells: {add.sum()} labelled atoms; target = realisation {1 - CAL_REAL}")
tg_s = sc_all[free]
tg_X, tg_Z = Dsc["X"][free], Dsc["Z"][free]
fam, rc = famall[free], C["r_core"][free]
NB = int(S["n_B"])
PAIR = np.concatenate([np.flatnonzero(S["order"] == 1), np.arange(NB, S["X_train"].shape[1])])


def whitener(X):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    _, sv, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    var = sv ** 2 / len(X)
    r = int(np.sum(var > 1e-10 * var[0]))
    W = Vt[:r].T / np.sqrt(var[:r])
    return lambda Y: ((Y - mu) / sd) @ W


def logistic(Xa, Xb, lam=1e-2):
    """L2 logistic regression, label 1 = Xb; returns log-odds function (class-balanced)."""
    X = np.concatenate([Xa, Xb]); y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    wt = np.where(y == 1, len(Xa) / len(Xb), 1.0)
    Xa1 = np.hstack([X, np.ones((len(X), 1))])

    def f(b):
        z = Xa1 @ b
        l = np.logaddexp(0, z) - y * z
        g = Xa1.T @ (wt * (1 / (1 + np.exp(-z)) - y))
        return (wt * l).sum() / len(X) + lam * b[:-1] @ b[:-1] / 2, g / len(X) + lam * np.r_[b[:-1], 0]

    b = minimize(f, np.zeros(Xa1.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    return lambda Y: np.hstack([Y, np.ones((len(Y), 1))]) @ b


def weighted_q(scores, w_cal, w_test):
    """Tibshirani et al. weighted quantile per test point (inf when the mass at +inf is needed)."""
    o = np.argsort(scores); s_sorted, cw = scores[o], np.cumsum(w_cal[o])
    tot = cw[-1] + w_test
    idx = np.searchsorted(cw, (1 - ALPHA) * tot, side="left")
    q = np.full(len(w_test), np.inf)
    ok = idx < len(s_sorted)
    q[ok] = s_sorted[idx[ok]]
    return q


def knn_q(Pc, sc, Pt, k=K, chunk=2048):
    t2 = np.sum(Pc * Pc, 1)
    q, dk = np.empty(len(Pt)), np.empty(len(Pt))
    level = min(np.ceil((k + 1) * (1 - ALPHA)) / k, 1.0)
    for i in range(0, len(Pt), chunk):
        p = Pt[i:i + chunk]
        d2 = np.maximum(np.sum(p * p, 1)[:, None] + t2[None, :] - 2 * p @ Pc.T, 0)
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        q[i:i + chunk] = np.quantile(sc[nn], level, axis=1)
        dk[i:i + chunk] = np.sqrt(np.take_along_axis(d2, nn, 1).max(1))
    return q, dk


res = {m: np.empty(len(tg_s)) for m in ("A", "B", "C")}
neff, logw = {}, np.empty(len(tg_s))
dk_all = np.empty(len(tg_s)); dk_self = []
# A: global split conformal (finite-sample level)
n = len(pool_s)
qA = np.quantile(pool_s, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0))
res["A"][:] = qA
insample = {}
for z in np.unique(pool_Z):
    mc, mt = pool_Z == z, tg_Z == z
    wh = whitener(pool_X[mc][:, PAIR].astype(np.float64))
    Pc, Pt = wh(pool_X[mc][:, PAIR].astype(np.float64)), wh(tg_X[mt][:, PAIR].astype(np.float64))
    # B: density-ratio weights from a classifier (pool vs target); w = P(t|x)/P(c|x) (balanced)
    lo = logistic(Pc, Pt)
    wc, wt = np.exp(np.clip(lo(Pc), -30, 30)), np.exp(np.clip(lo(Pt), -30, 30))
    res["B"][mt] = weighted_q(pool_s[mc], wc, wt)
    neff[int(z)] = float(wc.sum() ** 2 / (wc ** 2).sum())
    logw[mt] = lo(Pt)
    # C: localized kNN conformal
    res["C"][mt], dk_all[mt] = knn_q(Pc, pool_s[mc], Pt)
    # C sanity in distribution: one half of the pool (by config) as calibration, the other as test
    cfgz = pool_cfg[mc]; u = np.unique(cfgz)
    half = np.isin(cfgz, np.random.default_rng(int(z)).permutation(u)[: len(u) // 2])
    qh, dh = knn_q(Pc[half], pool_s[mc][half], Pc[~half])
    insample.setdefault("cov", []).append(pool_s[mc][~half] <= qh); dk_self.append(dh)

print(f"alpha = {ALPHA}; pool atoms {n}; target free atoms {len(tg_s)}; kNN k = {K}")
print(f"A split conformal q = {qA:.3f}  (calibrated Gaussian: 2.500)")
print(f"C in-distribution sanity (pool half -> other half): coverage {np.mean(np.concatenate(insample['cov'])):.3f}")
print("B classifier weights: Kish n_eff of the pool per species:", {k: round(v) for k, v in neff.items()},
      f"(pool per species ~{n // len(neff)})")
print(f"kNN radius (k-th neighbour distance): in-distribution median {np.median(np.concatenate(dk_self)):.2f}, "
      f"targets median {np.median(dk_all):.2f}")
groups = [("crack all", fam == "crack"), ("crack tip <=10 A", (fam == "crack") & (rc <= 10)),
          ("crack 10-20 A", (fam == "crack") & (rc > 10) & (rc <= 20)), ("crack >20 A", (fam == "crack") & (rc > 20)),
          ("edge all", fam == "edge"), ("edge core <=10 A", (fam == "edge") & (rc <= 10)),
          ("screw all", fam == "screw"), ("screw core <=10 A", (fam == "screw") & (rc <= 10))]
print(f"\n{'group':20s} {'n':>6s} | coverage A / B / C | median q: B, C | frac inf B / C | mean log-odds")
for lab, m in groups:
    cov = [np.mean(tg_s[m] <= res[k][m]) for k in "ABC"]
    print(f"{lab:20s} {m.sum():6d} | {cov[0]:.3f} / {cov[1]:.3f} / {cov[2]:.3f} | "
          f"{np.median(res['B'][m]):.2f}, {np.median(res['C'][m]):.2f} | "
          f"{np.mean(np.isinf(res['B'][m])):.3f} / {np.mean(np.isinf(res['C'][m])):.3f} | {np.mean(logw[m]):+.2f}")
np.savez(f"{run}/conformal_shift{'' if CAL_REAL is None else f'_cal{CAL_REAL}'}.npz", qA=qA, qB=res["B"], qC=res["C"], score=tg_s, family=fam, r_core=rc,
         logw=logw, dk=dk_all)
