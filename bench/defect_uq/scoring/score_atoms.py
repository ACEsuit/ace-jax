"""Per-atom uncertainty scores on the bench365 defect benchmark, against per-atom force errors of
the refitted models.  Hyperparameters are chosen ONLY on the in-distribution test split (bulk +
defect test families); the OOD families (combinations, strained surfaces) and the big cells
(cracks, dislocations) are held out.

Each candidate is FIT ONCE per species on the training sites and applied to all query sets
(test / ood / big).  Rows are written to atom_scores.json as they finish.

    python score_atoms.py <pops_run_dir> [<gp_run_dir>]
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).parent
S = dict(np.load(HERE / "sites.npz", allow_pickle=True))
ORDER, NB = S["order"], int(S["n_B"])
SETS = ("test", "ood", "big")


# ---------------- errors and the models' own sigma ----------------

def run_arrays(run):
    run = pathlib.Path(run)
    t, o = np.load(run / "pred_test_map.npz"), np.load(run / "pred_ood_map.npz")
    err = {k: np.linalg.norm(d["F"] - d["F_mean"], axis=1) for k, d in (("test", t), ("ood", o))}
    sig = {k: np.sqrt(np.maximum(d["F_var"], 0).sum(1)) for k, d in (("test", t), ("ood", o))}
    if (run / "big_err.npz").exists():
        err["big"] = np.load(run / "big_err.npz")["err"]
    return err, sig


# ---------------- scorers: fit(T) -> score(Q) ----------------

def _pca(T, sub=40000, seed=0):
    rng = np.random.default_rng(seed)
    Ts = T[rng.choice(len(T), min(sub, len(T)), replace=False)].astype(np.float64)
    mu, sd = Ts.mean(0), Ts.std(0) + 1e-9
    _, sv, Vt = np.linalg.svd((Ts - mu) / sd, full_matrices=False)
    return mu, sd, Vt, sv ** 2 / len(Ts)


def _knn(T, Q, k=5, chunk=4096):
    t2 = np.sum(T * T, 1); out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d2 = np.maximum(np.sum(q * q, 1)[:, None] + t2[None, :] - 2 * q @ T.T, 0)
        out[i:i + chunk] = np.sqrt(np.partition(d2, k, axis=1)[:, :k]).mean(1)
    return out


def fit_t2q(T, rank):
    """Hotelling T^2 in the top-`rank` PCA subspace + residual Q outside it (squared
    reconstruction error), each normalised by its training median."""
    mu, sd, Vt, var = _pca(T)
    V = Vt[:rank].T
    def parts(X):
        Z = (X.astype(np.float64) - mu) / sd
        P = Z @ V
        return np.sum(P ** 2 / var[:rank], 1), np.maximum(np.sum(Z ** 2, 1) - np.sum(P ** 2, 1), 0)
    t2m, qm = (np.median(a) for a in parts(T[np.random.default_rng(1).choice(len(T), min(20000, len(T)), replace=False)]))
    def score(Q):
        t2, q = parts(Q)
        return t2 / t2m + q / qm
    return score


def fit_knn_white(T, rank, cols=None, k=5):
    Tc = T if cols is None else T[:, cols]
    mu, sd, Vt, var = _pca(Tc)
    r = min(rank, int(np.sum(var > 1e-12 * var[0])))
    W = Vt[:r].T / np.sqrt(var[:r])
    ref = (((Tc.astype(np.float64) - mu) / sd) @ W)
    return lambda Q: _knn(ref, (((Q if cols is None else Q[:, cols]).astype(np.float64) - mu) / sd) @ W, k=k)


def fit_leverage(T, ridge):
    """Descriptor-space leverage q (T^T T + ridge tr/L)^-1 q^T (uncentred): the site-energy analogue
    of the ACE D-optimality extrapolation grade."""
    Td = T.astype(np.float64)
    G = Td.T @ Td
    G += ridge * np.trace(G) / len(G) * np.eye(len(G))
    Ginv = np.linalg.inv(G)
    def score(Q, chunk=8192):
        out = np.empty(len(Q))
        for i in range(0, len(Q), chunk):
            q = Q[i:i + chunk].astype(np.float64)
            out[i:i + chunk] = np.sum((q @ Ginv) * q, 1)
        return out
    return score


def fit_dadapy(T, rank, sub=20000):
    """-log density from DADApy's k*-NN estimator, out-of-sample, in the centred whitened top-`rank`
    PCA space of a training subsample; also the 2NN intrinsic dimension of that space."""
    from dadapy import Data
    rng = np.random.default_rng(2)
    Ts = T[rng.choice(len(T), min(sub, len(T)), replace=False)]
    mu, sd, Vt, var = _pca(Ts)
    W = Vt[:rank].T / np.sqrt(var[:rank])
    dd = Data(((Ts.astype(np.float64) - mu) / sd) @ W, verbose=False)
    dd.compute_distances(maxk=100)
    ide = float(dd.compute_id_2NN()[0])
    def score(Q):
        log_den, _ = dd.return_interpolated_density_kstarNN(((Q.astype(np.float64) - mu) / sd) @ W)
        return -np.asarray(log_den)
    score.ide = ide
    return score


PAIR_COLS = np.concatenate([np.flatnonzero(ORDER == 1), np.arange(NB, S["X_train"].shape[1])])
CANDIDATES = {
    **{f"T2+Q rank {r}": (lambda T, r=r: fit_t2q(T, r)) for r in (16, 32, 64, 128)},
    **{f"kNN whitened rank {r}": (lambda T, r=r: fit_knn_white(T, r)) for r in (32, 64, 128)},
    "kNN whitened pair (spike)": lambda T: fit_knn_white(T, 10 ** 6, cols=PAIR_COLS),
    **{f"leverage ridge {g:g}": (lambda T, g=g: fit_leverage(T, g)) for g in (1e-8, 1e-6, 1e-4)},
    **{f"DADApy -log rho k*NN rank {r}": (lambda T, r=r: fit_dadapy(T, r)) for r in (16, 32)},
}


def apply_per_species(fit):
    out = {k: np.full(len(S[f"X_{k}"]), np.nan) for k in SETS}
    extra = []
    Xtr, Ztr = S["X_train"], S["Z_train"]
    for z in np.unique(Ztr):
        sc = fit(Xtr[Ztr == z])
        extra.append(getattr(sc, "ide", np.nan))
        for k in SETS:
            m = S[f"Z_{k}"] == z
            out[k][m] = sc(S[f"X_{k}"][m])
    return out, extra


# ---------------- metrics ----------------

def auroc(neg, pos):
    r = np.concatenate([neg, pos]).argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def recall_top(score, err, q_err=0.95, q_score=0.90):
    return float(np.mean((score >= np.quantile(score, q_score))[err >= np.quantile(err, q_err)]))


def report(name, sc, err):
    row = {"score": name, "rho_test": float(spearmanr(sc["test"], err["test"])[0]),
           "recall_test": recall_top(sc["test"], err["test"])}
    for k in ("ood", "big"):
        if sc.get(k) is None or k not in err:
            continue
        fam, free = S[f"fam_{k}"], ~S[f"fixed_{k}"]
        for f in dict.fromkeys(fam):
            m = (fam == f) & free
            row[f"rho_{f}"] = float(spearmanr(sc[k][m], err[k][m])[0])
            row[f"auroc_{f}"] = auroc(sc["test"], sc[k][m])
            row[f"recall_{f}"] = recall_top(sc[k][m], err[k][m])
            if k == "big":
                rc = S["rcore_big"]
                core, far = m & (rc < 10), m & (rc > 25)
                row[f"core_vs_far_{f}"] = auroc(sc[k][far], sc[k][core])
    return row


def main():
    err, sig_pops = run_arrays(sys.argv[1])
    rows = [report("POPS sigma_F", {"test": sig_pops["test"], "ood": sig_pops["ood"]}, err)]
    if len(sys.argv) > 2 and not sys.argv[2].startswith("-"):
        sig_gp = run_arrays(sys.argv[2])[1]
        rows.append(report("GP sigma_F", {"test": sig_gp["test"], "ood": sig_gp["ood"]}, err))
    dest = HERE / "atom_scores.json"
    if "--extra" in sys.argv:                  # precomputed per-atom scores: keys "<set>/<name>"
        z = np.load(sys.argv[sys.argv.index("--extra") + 1])
        names = sorted({k.split("/", 1)[1] for k in z.files if k.startswith("test/")})
        for nm in names:
            if nm.startswith(("F_K", "varF_K")):
                continue
            sc = {k: z[f"{k}/{nm}"] for k in SETS if f"{k}/{nm}" in z.files}
            rows.append(report(f"[bayes] {nm}", sc, err))
        dest = HERE / "atom_scores_bayes.json"
        json.dump(rows, open(dest, "w"), indent=1)
    json.dump(rows, open(dest, "w"), indent=1)
    only = [a for a in sys.argv[3:] if not a.startswith("-") and not a.endswith(".npz")]
    for name, fit in CANDIDATES.items():
        if only and not any(o in name for o in only):
            continue
        t = time.time()
        try:
            sc, extra = apply_per_species(fit)
            if not np.all(np.isnan(extra)):
                name += f" (2NN ID {np.nanmean(extra):.1f})"
            rows.append(report(name, sc, err))
        except Exception as e:                                   # one failing candidate never loses the rest
            rows.append({"score": name, "error": repr(e)})
        json.dump(rows, open(dest, "w"), indent=1)
        print(f"{name}: {time.time() - t:.0f}s", flush=True)
    allk = list(dict.fromkeys(k for r in rows for k in r if k not in ("score", "error")))
    keys = ["rho_test", "recall_test"] + [k for k in allk if k.startswith("auroc_")] + \
           [k for k in allk if k.startswith("rho_") and k != "rho_test"] + [k for k in allk if k.startswith("core_vs_far")]
    print("\n" + " | ".join(["score"] + keys))
    for r in rows:
        print(" | ".join([r["score"]] + [f"{r.get(k, float('nan')):.2f}" for k in keys]) +
              (f"  ERROR {r['error']}" if "error" in r else ""))


if __name__ == "__main__":
    main()
