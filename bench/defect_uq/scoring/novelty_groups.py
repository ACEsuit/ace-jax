"""Feature-space novelty as a Mondrian variable, evaluated offline (plan: docs/dev/force-uq-vs-calm.md, Step 3, H3).

A CALM-like grade (arXiv:2609.40060) per species, fitted on TRAINING atoms only and label-free (so the
exchangeability argument for the groups is unchanged):
- features: the normalised support features (phi/|phi|, log-norm channels) in the whitened space of a
  `support_rebuild.py` normalised reference;
- per-species k-means (scipy kmeans2, ++ init), K from {1, 2, 4, 8}: the smallest K whose next doubling
  lowers the inertia by less than `--elbow` (0.1) of the K = 1 inertia; clusters with fewer than D + 1
  members are merged into the nearest centroid;
- sigma = Mahalanobis distance to the nearest cluster (its covariance + 1e-6 I, pseudo-inverse), and
  gamma = sigma / theta with theta = median + 3 x 1.4826 MAD of that cluster's training sigma.

Then, for the served posterior (scores = the aniso conformal score s = sqrt(e^T (V + eps v/3 I)^-1 e) with
V = forces_cov / lam_g^2, so s <= q_g is the guarantee the fit states), per gamma bin {<= 1, 1-2, > 2}:
(a) coverage of the geometric groups as served; (b) coverage when q is recomputed per gamma bin
("novelty"); (b) bins below effective_n_min configurations take the all-groups pooled q; (c) per (distortion group x
gamma bin), falling back to the distortion group's own q where the
crossed group has fewer than effective_n_min configurations ("distortion+novelty").  q in (b)/(c) is the same
configuration-weighted pooled quantile (`conformal._pooled_q`, `config_weights`) the fit uses, on the
posterior's own calibration atoms (`cal_*`).

    python novelty_groups.py --posterior P --rebuild DIR (support_rebuild.py build output)
        --big RUN/big3_err.npz:big3_desc.npz [--big ...] --out novelty.md
"""
import argparse
import pathlib
import sys

import numpy as np

BINS = (1.0, 2.0)
KS = (1, 2, 4, 8)


def kmeans_elbow(F, ks=KS, elbow=0.1, seed=0):
    """(centroids, labels) with K chosen by the doubling rule in the module docstring."""
    from scipy.cluster.vq import kmeans2
    fits, J = {}, {}
    for k in ks:
        if k > len(F):
            break
        c, lab = kmeans2(F, k, minit="++", seed=seed + k)
        fits[k] = (c, lab)
        J[k] = float(np.sum((F - c[lab]) ** 2))
    ks = sorted(J)
    for k, k2 in zip(ks, ks[1:]):
        if (J[k] - J[k2]) / max(J[ks[0]], 1e-300) < elbow:
            return fits[k]
    return fits[ks[-1]]


def merge_small(F, c, lab, min_size):
    """Merge clusters with fewer than min_size members into the nearest other centroid, smallest first."""
    c, lab = c.copy(), lab.copy()
    while True:
        ids, cnt = np.unique(lab, return_counts=True)
        small = ids[cnt < min_size]
        if len(small) == 0 or len(ids) == 1:
            break
        k = small[np.argmin(cnt[cnt < min_size])]
        others = ids[ids != k]
        j = others[np.argmin(np.linalg.norm(c[others] - c[k], axis=1))]
        lab[lab == k] = j
        c[j] = F[lab == j].mean(0)
    ids = np.unique(lab)
    remap = {int(i): n for n, i in enumerate(ids)}
    return c[ids], np.array([remap[int(i)] for i in lab])


def fit_novelty(F, elbow=0.1, seed=0):
    """Clusters of one species' training features F (n, D): [(mu, Pinv, theta)]."""
    c, lab = kmeans_elbow(F, elbow=elbow, seed=seed)
    c, lab = merge_small(F, c, lab, F.shape[1] + 1)
    cl = []
    for k in range(len(c)):
        Fk = F[lab == k]
        mu = Fk.mean(0)
        P = np.linalg.pinv(np.cov(Fk.T, bias=True).reshape(F.shape[1], F.shape[1]) + 1e-6 * np.eye(F.shape[1]))
        sig = np.sqrt(np.maximum(np.einsum("na,ab,nb->n", Fk - mu, P, Fk - mu), 0.0))
        med = np.median(sig)
        cl.append((mu, P, med + 3 * 1.4826 * np.median(np.abs(sig - med))))
    return cl


def grade(cl, F):
    """gamma (n,): sigma / theta of the nearest cluster in Mahalanobis distance."""
    g = np.full(len(F), np.inf)
    s_best = np.full(len(F), np.inf)
    for mu, P, th in cl:
        s = np.sqrt(np.maximum(np.einsum("na,ab,nb->n", F - mu, P, F - mu), 0.0))
        b = s < s_best
        s_best[b], g[b] = s[b], s[b] / th
    return g


def gamma_of(nov, ref, X, Z):
    from ace_jax.fit.support import _proj, support_features
    F = support_features(X, ref.get("features"))
    out = np.full(len(Z), np.inf)
    for z in np.unique(Z):
        if int(z) in nov:
            m = Z == z
            out[m] = grade(nov[int(z)], _proj(ref["pca"], z, F[m]))
    return out


def binned(gm):
    return np.searchsorted(np.asarray(BINS), gm, side="left")    # 0: <= 1, 1: (1, 2], 2: > 2


def quantiles(s, grp, cfg, G, alpha, n_min, fallback):
    """Per group the configuration-weighted pooled q (conformal.group_scales' rule); groups below
    effective_n_min configurations take fallback[k]."""
    from ace_jax.fit.conformal import _pooled_q, config_weights, effective_n_min
    w = config_weights(grp, cfg, G)
    q, ncfg = np.array(fallback, float), np.zeros(G, int)
    for k in range(G):
        m = grp == k
        ncfg[k] = len(np.unique(cfg[m]))
        if ncfg[k] >= effective_n_min(n_min, alpha):
            q[k] = _pooled_q(s[m], w[m], alpha, ncfg[k])
    return q, ncfg


def load_big(spec, post):
    """(s, g, gamma-input X, Z, family, cell) of the free atoms of one big_errors file + its descriptors."""
    from ace_jax.fit.ard import conformal_scores
    err, desc = spec.split(":")
    e = np.load(err, allow_pickle=True)
    d = np.load(desc)
    if len(d["Z"]) != len(e["err"]):
        raise ValueError(f"{desc}: {len(d['Z'])} rows, {err}: {len(e['err'])}")
    free = ~e["fixed"].astype(bool)
    lam = np.asarray(post.group_table["lam_rms"], float)[e["forces_group"].astype(int)]
    V = np.asarray(e["forces_cov"], float) / lam[:, None, None] ** 2
    s = conformal_scores(np.asarray(e["dF"], float), V, post.force_shape, post.eps)
    fam = e["family"].astype(str)
    tag = pathlib.Path(err).stem
    return (s[free], e["forces_group"].astype(int)[free], d["X"][free].astype(float), d["Z"].astype(int)[free],
            fam[free], np.char.add(tag + ":", e["cfg"].astype(str))[free])


def main(argv=None):
    import jax
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from support_rebuild import load_ref

    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.conformal import effective_n_min
    from ace_jax.fit.support import _proj, support_features
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--posterior", required=True)
    p.add_argument("--rebuild", required=True, help="support_rebuild.py build output (descriptors.npz, "
                                                     "support_normalised.npz)")
    p.add_argument("--big", action="append", default=[], help="big_errors npz:descriptor npz (same row order)")
    p.add_argument("--elbow", type=float, default=0.1)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    post = ARDPosterior.load(a.posterior)
    alpha, n_min = float(post.group_table["alpha"]), int(post.group_table["n_min"])
    q_served = np.asarray(post.group_table["q"], float)
    G0 = len(q_served)
    reb = pathlib.Path(a.rebuild)
    D = np.load(reb / "descriptors.npz")
    ref = load_ref(reb / "support_normalised.npz")
    nov, lines = {}, []
    for k in D.files:
        if k.startswith("X_pool_"):
            z = int(k.split("_")[-1])
            F = _proj(ref["pca"], z, support_features(D[k].astype(float), ref["features"]))
            nov[z] = fit_novelty(F, a.elbow)
            gt = grade(nov[z], F)
            lines.append(f"species {z}: {len(F)} training atoms, D = {F.shape[1]}, K = {len(nov[z])}, "
                         f"training gamma > 1: {np.mean(gt > 1):.3f}")
    cal = post.cal
    s_c, g_c, cfg_c = np.asarray(cal["scores"], float), np.asarray(cal["groups"], int), np.asarray(cal["cfg"])
    gm_c = gamma_of(nov, ref, D["X_cal"].astype(float), D["Z_cal"].astype(int))
    b_c = binned(gm_c)
    nb = len(BINS) + 1
    q_all, _ = quantiles(s_c, np.zeros(len(s_c), int), cfg_c, 1, alpha, 1, [np.inf])     # all groups pooled
    q_nov, n_nov = quantiles(s_c, b_c, cfg_c, nb, alpha, n_min, np.full(nb, q_all[0]))
    q_x, n_x = quantiles(s_c, g_c * nb + b_c, cfg_c, G0 * nb, alpha, n_min, np.repeat(q_served, nb))
    lines += ["", f"Calibration atoms: {len(s_c)} ({len(np.unique(cfg_c))} configs); fraction per gamma bin "
              + ", ".join(f"{lab}: {np.mean(b_c == i):.3f}" for i, lab in enumerate(("<=1", "1-2", ">2"))),
              f"novelty q per bin: {np.array2string(q_nov, precision=3)} (configs {n_nov.tolist()}); served q: "
              f"{np.array2string(q_served, precision=3)}",
              f"crossed groups with their own q (>= effective_n_min configs): "
              f"{int(np.sum(n_x >= effective_n_min(n_min, alpha)))} of {G0 * nb}", ""]
    sets = {"calibration (in-sample)": (s_c, g_c, None, None, np.full(len(s_c), "cal"), cfg_c.astype(str), gm_c)}
    for spec in a.big:
        s, g, X, Z, fam, cell = load_big(spec, post)
        sets[pathlib.Path(spec.split(":")[0]).stem] = (s, g, X, Z, fam, cell, gamma_of(nov, ref, X, Z))
    lines += ["| set | family | gamma bin | atoms | configs | coverage: served groups | novelty | "
              "distortion+novelty |", "|---|---|---|---|---|---|---|---|"]
    for name, (s, g, _, _, fam, cell, gm) in sets.items():
        b = binned(gm)
        cov = {"served": s <= q_served[g], "novelty": s <= q_nov[b], "cross": s <= q_x[g * nb + b]}
        for f in sorted(set(fam)):
            for i, lab in [(None, "all"), (0, "<= 1"), (1, "1-2"), (2, "> 2")]:
                m = (fam == f) & (True if i is None else b == i)
                if m.sum() == 0:
                    continue
                lines.append(f"| {name} | {f} | {lab} | {int(m.sum())} | {len(np.unique(cell[m]))} | "
                             + " | ".join(f"{np.mean(c[m]):.3f}" for c in cov.values()) + " |")
    md = "\n".join(lines) + "\n"
    pathlib.Path(a.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
