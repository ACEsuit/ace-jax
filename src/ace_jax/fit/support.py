"""Covariate-shift support diagnostic (math rev. 2): per species, PCA of the site descriptor,
a grouped-CV L2 logistic density ratio (target vs calibration), weighted conformal quantile."""
import warnings

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


def _fit_subset(g, cap, rng):
    """Indices of whole configurations (random order) totalling <= cap atoms, for the classifier fit."""
    if len(g) <= cap:
        return np.arange(len(g))
    keep, n = [], 0
    for c in rng.permutation(np.unique(g)):
        mc = np.flatnonzero(g == c)
        if n + len(mc) > cap and keep:
            break
        keep.append(mc); n += len(mc)
    return np.sort(np.concatenate(keep))


def build_support(pca, X, Z, scores, cfg, max_atoms, seed, fit_max=10000):
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
                       "m": 1.0 / cnt[inv], "g": inv, "f": _fit_subset(inv, fit_max, rng)}
    return ref


def _sigmoid(f):
    return 1 / (1 + np.exp(-np.clip(f, -30.0, 30.0)))


def _logistic(X, y, w, l2, beta0=None, iters=30, max_step=10.0):
    """Weighted L2 logistic regression by damped Newton; returns (beta, converged).  Logits are clipped
    to +-30 inside the sigmoid and the step norm is capped, so separable data cannot overflow."""
    n, p = X.shape
    Xa = np.c_[X, np.ones(n)]
    beta = np.zeros(p + 1) if beta0 is None else np.array(beta0, float)
    reg = np.r_[np.full(p, l2), 1e-8]
    for _ in range(iters):
        t = _sigmoid(Xa @ beta)
        gr = Xa.T @ (w * (t - y)) + reg * beta
        H = (Xa * (w * t * (1 - t) + 1e-12)[:, None]).T @ Xa + np.diag(reg)
        try:
            step = np.linalg.solve(H, gr)
        except np.linalg.LinAlgError:
            return beta, False
        nrm = np.linalg.norm(step)
        if not np.isfinite(nrm):
            return beta, False
        if nrm > max_step:
            step *= max_step / nrm
        beta = beta - step
        if np.abs(step).max() < 1e-8:
            return beta, bool(np.isfinite(beta).all())
    return beta, False


def _ratio(Xc, gc, Xt, seed, fit_idx=None):
    """log density ratio log p_t/p_c at calibration (all) and target points, L2 chosen by 5-fold CV grouped by
    calibration configuration (target atoms split at random).  The classifier is fitted on the calibration
    subset fit_idx (default all) and the target; its logits are then evaluated on every Xc.
    Returns (logit_c, logit_t, converged)."""
    rng = np.random.default_rng(seed)
    fi = np.arange(len(Xc)) if fit_idx is None else np.asarray(fit_idx)
    Xf, gf = Xc[fi], gc[fi]
    X = np.r_[Xf, Xt]
    y = np.r_[np.zeros(len(Xf)), np.ones(len(Xt))]
    w = np.r_[np.full(len(Xf), 0.5 / len(Xf)), np.full(len(Xt), 0.5 / len(Xt))] * len(X)
    ug = np.unique(gf)
    fold_c = dict(zip(ug, rng.integers(0, 5, len(ug))))
    fold = np.r_[np.array([fold_c[g] for g in gf]), rng.integers(0, 5, len(Xt))]
    grid = (1e-3, 1e-2, 1e-1, 1.0)
    nll = np.zeros(len(grid))
    for f in range(5):
        tr, te = fold != f, fold == f
        if not te.any() or not tr.any():
            continue
        Xte = np.c_[X[te], np.ones(te.sum())]
        beta = None                                   # warm start across the l2 grid
        for j, l2 in enumerate(grid):
            beta, _ = _logistic(X[tr], y[tr], w[tr], l2 * tr.sum(), beta)
            t = np.clip(_sigmoid(Xte @ beta), 1e-12, 1 - 1e-12)
            nll[j] -= np.sum(w[te] * (y[te] * np.log(t) + (1 - y[te]) * np.log(1 - t)))
    l2b = grid[int(np.argmin(nll))]
    beta, ok = _logistic(X, y, w, l2b * len(X))
    f = np.clip(np.c_[Xc, np.ones(len(Xc))] @ beta, -30.0, 30.0)
    ft = np.clip(np.c_[Xt, np.ones(len(Xt))] @ beta, -30.0, 30.0)
    return f, ft, ok


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
        lc, lt, conv = _ratio(r["Xc"], r["g"], Xt, seed, r.get("f"))
        if not conv:
            warnings.warn(f"support_check: density-ratio classifier did not converge for species {int(z)}; "
                          f"its atoms are flagged unsupported", RuntimeWarning, stacklevel=2)
            neff[int(z)] = 0.0
            continue
        sh = max(lc.max(), lt.max())
        pc = r["m"] * np.exp(lc - sh)
        den = float(np.sum(pc ** 2))
        if not np.isfinite(den) or den <= 0 or pc.sum() <= 0:
            neff[int(z)] = 0.0
            continue
        neff[int(z)] = float(pc.sum() ** 2 / den)
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
        if "f" in r:
            out[f"support_{z}_f"] = np.asarray(r["f"], np.int64)
    return out


def unflatten_support(flat):
    """Inverse of flatten_support; flat keys have the leading 'support_' stripped.  float64 on return."""
    ref = {"pca": {}}
    for k, v in flat.items():
        p = k.split("_")
        if p[0] == "pca":
            ref["pca"].setdefault(int(p[1]), {})[p[2]] = np.asarray(v, np.float64)
        else:
            ref.setdefault(int(p[0]), {})[p[1]] = np.asarray(v) if p[1] in ("g", "f") else np.asarray(v, np.float64)
    ref["pca"] = {z: (d["mu"], d["sd"], d["W"]) for z, d in ref["pca"].items()}
    return ref


def extend_support(ref, keep, X, Z, scores, cfg, max_atoms, seed, fit_max=10000):
    """Rebuild a support reference on (kept stored points) + new atoms under the SAME per-species PCA.

    keep[z]: boolean mask over ref[z]'s stored points to retain (None: all, absent species: all).
    X (n, D) raw descriptors, Z, scores, cfg of the new atoms.  Per species the pooled configurations are
    capped at max_atoms atoms (whole configurations, random order), masses re-derived as 1/(atoms per
    configuration), and the classifier-fit subset redrawn.  Scores are rounded through float32, as a
    saved reference holds them."""
    rng = np.random.default_rng(seed)
    out = {"pca": ref["pca"]}
    Z, cfg = np.asarray(Z), np.asarray(cfg)
    for z in sorted(k for k in ref if k != "pca"):
        r = ref[z]
        k = np.ones(len(r["s"]), bool) if keep.get(z) is None else np.asarray(keep[z], bool)
        m = np.flatnonzero(Z == z)
        Xc = np.r_[r["Xc"][k], _proj(ref["pca"], z, np.asarray(X)[m])]
        s = np.r_[r["s"][k], np.asarray(scores, np.float32).astype(float)[m]]
        g_old = np.asarray(r["g"])[k]
        _, g_new = np.unique(cfg[m], return_inverse=True)
        g = np.r_[g_old, g_new + (g_old.max() + 1 if len(g_old) else 0)].astype(np.int64)
        if len(s) == 0:
            continue
        sel, n = [], 0
        for c in rng.permutation(np.unique(g)):
            mc = np.flatnonzero(g == c)
            if n + len(mc) > max_atoms and sel:
                break
            sel.append(mc); n += len(mc)
        sel = np.sort(np.concatenate(sel))
        _, inv, cnt = np.unique(g[sel], return_inverse=True, return_counts=True)
        out[z] = {"Xc": Xc[sel], "s": s[sel], "m": 1.0 / cnt[inv], "g": inv, "f": _fit_subset(inv, fit_max, rng)}
    return out
