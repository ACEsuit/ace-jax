"""Covariate-shift support diagnostic (math rev. 2): per species, PCA of the site descriptor,
a grouped-CV L2 logistic density ratio (target vs calibration), weighted conformal quantile."""
import numpy as np


def fit_pca(X_by_species, var=0.99, cap=64):
    out = {}
    for z, X in X_by_species.items():
        X = np.asarray(X, float)
        mu, sd = X.mean(0), X.std(0) + 1e-12
        _, s, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        k = min(int(np.searchsorted(e, var) + 1), cap, len(s))
        out[int(z)] = (mu, sd, Vt[:k].T / (s[:k] / np.sqrt(len(X))))       # whitened components
    return out


def _proj(pca, z, X):
    mu, sd, W = pca[int(z)]
    return ((np.asarray(X, float) - mu) / sd) @ W


def build_support(pca, X, Z, scores, cfg, max_atoms, seed):
    rng = np.random.default_rng(seed)
    ref = {"pca": pca}
    for z in np.unique(Z):
        m = np.flatnonzero(Z == z)
        cs = rng.permutation(np.unique(cfg[m]))
        keep, n = [], 0
        for c in cs:
            mc = m[cfg[m] == c]
            if n + len(mc) > max_atoms and keep:
                break
            keep.append(mc); n += len(mc)
        m = np.concatenate(keep)
        _, inv, cnt = np.unique(cfg[m], return_inverse=True, return_counts=True)
        ref[int(z)] = {"Xc": _proj(pca, z, X[m]), "s": np.asarray(scores, float)[m],
                       "m": 1.0 / cnt[inv], "g": inv}
    return ref


def _logistic(X, y, w, l2, iters=200):
    """Weighted L2 logistic regression by Newton's method; returns (beta, b0)."""
    n, p = X.shape
    Xa = np.c_[X, np.ones(n)]
    beta = np.zeros(p + 1)
    reg = np.r_[np.full(p, l2), 0.0]
    for _ in range(iters):
        t = 1 / (1 + np.exp(-Xa @ beta))
        gr = Xa.T @ (w * (t - y)) + reg * beta
        H = (Xa * (w * t * (1 - t))[:, None]).T @ Xa + np.diag(reg)
        step = np.linalg.solve(H, gr)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def _ratio(Xc, gc, Xt, seed):
    """log density ratio log p_t/p_c at calibration and target points, L2 chosen by 5-fold CV grouped by
    calibration configuration (target atoms split at random)."""
    rng = np.random.default_rng(seed)
    X = np.r_[Xc, Xt]
    y = np.r_[np.zeros(len(Xc)), np.ones(len(Xt))]
    w = np.r_[np.full(len(Xc), 0.5 / len(Xc)), np.full(len(Xt), 0.5 / len(Xt))] * len(X)
    ug = np.unique(gc)
    fold_c = dict(zip(ug, rng.integers(0, 5, len(ug))))
    fold = np.r_[np.array([fold_c[g] for g in gc]), rng.integers(0, 5, len(Xt))]
    best, l2b = np.inf, 1e-2
    for l2 in (1e-3, 1e-2, 1e-1, 1.0):
        nll = 0.0
        for f in range(5):
            tr, te = fold != f, fold == f
            if not te.any():
                continue
            beta = _logistic(X[tr], y[tr], w[tr], l2 * tr.sum())
            t = np.clip(1 / (1 + np.exp(-np.c_[X[te], np.ones(te.sum())] @ beta)), 1e-12, 1 - 1e-12)
            nll -= np.sum(w[te] * (y[te] * np.log(t) + (1 - y[te]) * np.log(1 - t)))
        if nll < best:
            best, l2b = nll, l2
    beta = _logistic(X, y, w, l2b * len(X))
    f = np.c_[X, np.ones(len(X))] @ beta
    return f[:len(Xc)], f[len(Xc):]


def support_check(ref, X_t, Z_t, alpha, seed=0):
    n = len(Z_t)
    ok, q, neff = np.zeros(n, bool), np.full(n, np.inf), {}
    for z in np.unique(Z_t):
        mt = np.flatnonzero(Z_t == z)
        r = ref.get(int(z))
        if r is None:
            neff[int(z)] = 0.0
            continue
        Xt = _proj(ref["pca"], z, X_t[mt])
        lc, lt = _ratio(r["Xc"], r["g"], Xt, seed)
        sh = max(lc.max(), lt.max())
        pc = r["m"] * np.exp(lc - sh)
        neff[int(z)] = float(pc.sum() ** 2 / np.sum(pc ** 2))
        o = np.argsort(r["s"])
        cs = np.cumsum(pc[o])
        for j, i in enumerate(mt):
            pt = np.exp(lt[j] - sh)
            k = int(np.searchsorted(cs / (cs[-1] + pt), 1 - alpha - 1e-15))
            if k < len(o):
                ok[i], q[i] = True, r["s"][o][k]
    return {"support_ok": ok, "support_q": q, "n_eff": neff}


def flatten_support(ref, dtype=np.float32):
    """The nested reference as a flat dict of arrays: support_pca_{z}_{mu,sd,W}, support_{z}_{Xc,s,m,g}."""
    out = {}
    for z, (mu, sd, W) in ref["pca"].items():
        for k, v in (("mu", mu), ("sd", sd), ("W", W)):
            out[f"support_pca_{z}_{k}"] = np.asarray(v, dtype)
    for z, r in ref.items():
        if z == "pca":
            continue
        for k in ("Xc", "s", "m"):
            out[f"support_{z}_{k}"] = np.asarray(r[k], dtype)
        out[f"support_{z}_g"] = np.asarray(r["g"], np.int64)
    return out


def unflatten_support(flat):
    """Inverse of flatten_support; flat keys have the leading 'support_' stripped.  float64 on return."""
    ref = {"pca": {}}
    for k, v in flat.items():
        p = k.split("_")
        if p[0] == "pca":
            ref["pca"].setdefault(int(p[1]), {})[p[2]] = np.asarray(v, np.float64)
        else:
            ref.setdefault(int(p[0]), {})[p[1]] = np.asarray(v) if p[1] == "g" else np.asarray(v, np.float64)
    ref["pca"] = {z: (d["mu"], d["sd"], d["W"]) for z, d in ref["pca"].items()}
    return ref
