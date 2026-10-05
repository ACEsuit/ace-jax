"""Covariate-shift support diagnostic (math rev. 2): per species, PCA of the site descriptor,
a grouped-CV L2 logistic density ratio (target vs calibration), weighted conformal quantile.

Features (`ard_support_features`): "raw" is the compact descriptor phi = [B | A_pair] itself; "normalised"
is phi_hat = phi/(|phi| + eps) plus log(|phi| + eps) and log(|phi_b| + eps) per body order b, the log-norm
channels bypassing the PCA (standardised, one coordinate each).  An atom losing its neighbours has phi -> 0,
which sits inside a raw training cloud reaching towards 0 (the CALM stretching test, arXiv:2609.40060);
its log-norms leave the training range instead.  The reference records its features as ref["features"]
({"kind", "body"}); a reference without it (written before) is raw."""
import warnings

import numpy as np

FEATURE_KINDS = ("raw", "normalised")
_EPS = 1e-12


def support_body(meta):
    """Body order of each compact-descriptor column [B | A_pair]: nu + 1 for B (len(nnll) = nu), 2 for pair."""
    return np.r_[[len(x) + 1 for x in meta["nnll"]], np.full(int(meta["n_pair"]), 2)].astype(np.int64)


def n_pass(features):
    """Trailing feature columns that bypass the PCA (the log-norm channels): 0 for raw."""
    if features is None or features["kind"] == "raw":
        return 0
    return 1 + len(np.unique(features["body"]))


def support_features(X, features):
    """Support features of raw descriptors X (n, D) under `features` (None = raw)."""
    X = np.asarray(X, float)
    if features is None or features["kind"] == "raw":
        return X
    if features["kind"] != "normalised":
        raise ValueError(f"unknown support features {features['kind']!r}; expected one of {FEATURE_KINDS}")
    body = np.asarray(features["body"])
    if len(body) != X.shape[1]:
        raise ValueError(f"support features: {X.shape[1]} descriptor columns but {len(body)} body orders")
    nrm = np.linalg.norm(X, axis=1)
    blk = [np.linalg.norm(X[:, body == b], axis=1) for b in np.unique(body)]
    return np.column_stack([X / (nrm + _EPS)[:, None], np.log(np.column_stack([nrm] + blk) + _EPS)])


def _species(ref):
    return [k for k in ref if isinstance(k, int)]


def fit_pca(X_by_species, var=0.99, cap=64, n_pass=0):
    """Per species (mu, sd, W): standardise, then whitened principal components (var explained, <= cap) of
    all but the last n_pass columns; those pass through standardised (identity block of W)."""
    out = {}
    for z, X in X_by_species.items():
        X = np.asarray(X, float)
        mu, sd = X.mean(0), X.std(0) + 1e-12
        D = X.shape[1] - n_pass
        _, s, Vt = np.linalg.svd(((X - mu) / sd)[:, :D], full_matrices=False)
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        k = min(int(np.searchsorted(e, var) + 1), cap, len(s))
        W = np.zeros((D + n_pass, k + n_pass))
        W[:D, :k] = Vt[:k].T / (s[:k] / np.sqrt(len(X)))                # whitened components
        W[D:, k:] = np.eye(n_pass)
        out[int(z)] = (mu, sd, W)
    return out


def explained_variance(X_by_species, pca, n_pass=0):
    """Fraction of the standardised PCA-block variance the kept components explain, per species."""
    out = {}
    for z, X in X_by_species.items():
        mu, sd, W = pca[int(z)]
        Y = ((np.asarray(X, float) - mu) / sd)[:, :W.shape[0] - n_pass]
        k = W.shape[1] - n_pass
        _, s, _ = np.linalg.svd(Y, full_matrices=False)
        out[int(z)] = float(np.sum(s[:k] ** 2) / np.sum(s ** 2))
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


def build_support(pca, X, Z, scores, cfg, max_atoms, seed, fit_max=10000, grp=None, features=None):
    """X: RAW descriptors (n, D); `features` ({"kind", "body"}, None = raw) maps them to the space pca was
    fitted in, and is stored as ref["features"] for support_check / extend_support."""
    rng = np.random.default_rng(seed)
    ref = {"pca": pca}
    if features is not None:
        ref["features"] = features
    X = support_features(X, features)
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
        if grp is not None:                          # conformal group of each stored point (for calibrate)
            ref[int(z)]["grp"] = np.asarray(grp)[m].astype(np.int64)
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
        Xt = _proj(ref["pca"], z, support_features(X_t[mt], ref.get("features")))
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
    """The nested reference as a flat dict of arrays: support_pca_{z}_{mu,sd,W}, support_{z}_{Xc,s,m,g},
    support_feat_{kind,body}."""
    out = {}
    for z, (mu, sd, W) in ref["pca"].items():
        for k, v in (("mu", mu), ("sd", sd), ("W", W)):
            out[f"support_pca_{z}_{k}"] = np.asarray(v, dtype)
    if "features" in ref:
        out["support_feat_kind"] = np.array(str(ref["features"]["kind"]))
        out["support_feat_body"] = np.asarray(ref["features"]["body"], np.int64)
    for z in _species(ref):
        r = ref[z]
        for k in ("Xc", "s", "m"):
            out[f"support_{z}_{k}"] = np.asarray(r[k], dtype)
        out[f"support_{z}_g"] = np.asarray(r["g"], np.int64)
        if "grp" in r:
            out[f"support_{z}_grp"] = np.asarray(r["grp"], np.int64)
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
        elif p[0] == "feat":
            ref.setdefault("features", {})[p[1]] = str(v) if p[1] == "kind" else np.asarray(v, np.int64)
        else:
            ref.setdefault(int(p[0]), {})[p[1]] = np.asarray(v) if p[1] in ("g", "f", "grp") else np.asarray(v, np.float64)
    ref["pca"] = {z: (d["mu"], d["sd"], d["W"]) for z, d in ref["pca"].items()}
    return ref


def extend_support(ref, keep, X, Z, scores, cfg, max_atoms, seed, grp, fit_max=10000, log=print):
    """Rebuild a support reference on (kept stored points) + new atoms under the SAME per-species PCA.

    keep[z]: boolean mask over ref[z]'s stored points to retain (None/absent: all).
    X (n, D) raw descriptors, Z, scores, cfg, grp (conformal group) of the new atoms.  Per species the pooled
    configurations are capped at max_atoms atoms (whole configurations, random order), masses re-derived as
    1/(atoms per configuration), and the classifier-fit subset redrawn.  Scores are rounded through float32,
    as a saved reference holds them.  A species of the new atoms with no stored reference (no PCA) is not
    added -- support_check reports its atoms out of support -- and a species left with no points is
    dropped; both are noted through log."""
    rng = np.random.default_rng(seed)
    out = {"pca": ref["pca"]}
    if "features" in ref:
        out["features"] = ref["features"]
    X = support_features(X, ref.get("features"))
    Z, cfg, grp = np.asarray(Z), np.asarray(cfg), np.asarray(grp)
    stored = set(_species(ref))
    for z in sorted(set(np.unique(Z).tolist()) - stored):
        log(f"note: support: species index {z} ({int(np.sum(Z == z))} atoms in the calibration set) has no "
            f"stored support reference; its atoms are not added and stay out of support")
    for z in sorted(stored):
        r = ref[z]
        k = np.ones(len(r["s"]), bool) if keep.get(z) is None else np.asarray(keep[z], bool)
        m = np.flatnonzero(Z == z)
        Xc = np.r_[r["Xc"][k], _proj(ref["pca"], z, X[m])]
        s = np.r_[r["s"][k], np.asarray(scores, np.float32).astype(float)[m]]
        gr = np.r_[np.asarray(r["grp"])[k], grp[m]].astype(np.int64)
        g_old = np.asarray(r["g"])[k]
        _, g_new = np.unique(cfg[m], return_inverse=True)
        g = np.r_[g_old, g_new + (g_old.max() + 1 if len(g_old) else 0)].astype(np.int64)
        if len(s) == 0:
            log(f"note: support: species index {z} has no points left after this recalibration; dropped from "
                f"the support reference (its atoms are out of support)")
            continue
        sel, n = [], 0
        for c in rng.permutation(np.unique(g)):
            mc = np.flatnonzero(g == c)
            if n + len(mc) > max_atoms and sel:
                break
            sel.append(mc); n += len(mc)
        sel = np.sort(np.concatenate(sel))
        _, inv, cnt = np.unique(g[sel], return_inverse=True, return_counts=True)
        out[z] = {"Xc": Xc[sel], "s": s[sel], "m": 1.0 / cnt[inv], "g": inv, "grp": gr[sel],
                  "f": _fit_subset(inv, fit_max, rng)}
    return out


def keep_for_mode(ref, mode, u_cfg, n_min):
    """Boolean keep mask per species over the stored support points: replace -> none; append -> all;
    per-group -> points whose conformal group has fewer than n_min U configurations (u_cfg[g])."""
    return {z: (np.zeros(len(r["s"]), bool) if mode == "replace" else
                np.ones(len(r["s"]), bool) if mode == "append" else
                np.asarray(u_cfg)[np.asarray(r["grp"])] < n_min)
            for z, r in ((z, ref[z]) for z in _species(ref))}


def recalibrate_support(ref, mode, u_cfg, n_min, X, Z, scores, cfg, grp, max_atoms=50000, seed=0):
    """The support reference on the pooled calibration atoms (stored points kept per `keep_for_mode`, plus
    U); None, with a notice, when the stored reference has no per-point group labels."""
    if any("grp" not in ref[z] for z in _species(ref)):
        print("note: support reference dropped: it stores no per-point conformal groups (refit with --uq ard)")
        return None
    return extend_support(ref, keep_for_mode(ref, mode, u_cfg, n_min), X, Z, scores, cfg, max_atoms, seed, grp)
