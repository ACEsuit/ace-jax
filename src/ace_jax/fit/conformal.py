"""Mondrian groups, stratified split and the two configuration-weighted per-group scales of the
ARD force sigma (math rev. 2)."""
import dataclasses

import numpy as np
from scipy.stats import chi


def chi3_ppf(p):
    return float(chi.ppf(p, 3))


def shell_reference(dists, bins=400):
    """r1: first minimum of the radial density after its first peak; 1.25 x the first peak if none."""
    d = np.asarray(dists, float)
    h, e = np.histogram(d, bins=bins, range=(0.0, d.max()))
    c = 0.5 * (e[1:] + e[:-1])
    g = np.convolve(h / np.maximum(c ** 2, 1e-12), np.ones(5) / 5, mode="same")
    p = int(np.argmax(g > 0.2 * g.max()))
    while p + 1 < len(g) and g[p + 1] >= g[p]:
        p += 1
    m = p
    while m + 1 < len(g) and g[m + 1] <= g[m]:
        m += 1
    return float(c[m] if (m + 1 < len(g) and g[m] < 0.5 * g[p]) else 1.25 * c[p])


def shell_features(batch, r1):
    """Per node: z = #{r_ij < r1}, d = std/mean of those r_ij (NaN if z < 2), from the batch's neighbour list."""
    r = np.linalg.norm(np.asarray(batch.rij), axis=-1)
    m = np.asarray(batch.nbr_mask) & (r < r1)
    z = m.sum(1)
    mean = np.where(m, r, 0.0).sum(1) / np.maximum(z, 1)
    var = np.where(m, (r - mean[:, None]) ** 2, 0.0).sum(1) / np.maximum(z, 1)
    return z.astype(np.int64), np.where(z >= 2, np.sqrt(var) / np.maximum(mean, 1e-12), np.nan)


def band_edges(d, qs=(0.5, 0.9, 0.99)):
    d = np.asarray(d, float)
    return np.quantile(d[np.isfinite(d)], qs)


def n_groups(edges):
    return 2 * (len(edges) + 1)


def assign_groups(z, d, z_star, edges):
    band = np.searchsorted(np.asarray(edges, float), np.nan_to_num(np.asarray(d, float), nan=np.inf), side="right")
    return (band * 2 + (np.asarray(z) != z_star)).astype(np.int64)


def config_strata(groups_per_cfg):
    """Stratum of each configuration: its most extreme group, max over its atoms of band*2 + flag (0 if empty)."""
    return np.array([int(np.max(g)) if len(g) else 0 for g in groups_per_cfg])


def stratified_split(strata, f, seed):
    """(fit_idx, val_idx): within each stratum a fraction f to val, with >= 1 config on each side.
    Strata with < 2 configs merge into the next less extreme populated stratum first."""
    s = np.asarray(strata).copy()
    for k in sorted(set(s.tolist()), reverse=True):
        lower = [j for j in sorted(set(s.tolist()), reverse=True) if j < k]
        if np.sum(s == k) < 2 and lower:
            s[s == k] = lower[0]
    rng = np.random.default_rng(seed)
    val = []
    for k in sorted(set(s.tolist())):
        idx = rng.permutation(np.flatnonzero(s == k))
        if len(idx) < 2:
            continue
        nv = min(max(1, int(round(f * len(idx)))), len(idx) - 1)
        val.extend(idx[:nv].tolist())
    val = np.sort(np.asarray(val, int))
    return np.setdiff1d(np.arange(len(s)), val), val


def _pooled_q(s, w, alpha):
    if len(s) == 0:
        return np.inf
    o = np.argsort(s)
    cw = np.cumsum(w[o]) / (w.sum() + 1.0)
    k = int(np.searchsorted(cw, 1 - alpha - 1e-15))
    return float(s[o][k]) if k < len(s) else np.inf


@dataclasses.dataclass(frozen=True)
class GroupTable:
    alpha: float
    n_min: int
    lam_rms: np.ndarray
    q: np.ndarray
    r: np.ndarray
    n_cfg: np.ndarray
    n_cfg_val: np.ndarray
    n_cfg_cal: np.ndarray
    n_atoms: np.ndarray
    merged: list
    sources: list = dataclasses.field(default_factory=list)

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        return cls(float(d["alpha"]), int(d["n_min"]),
                   *(np.asarray(d[k], float) for k in ("lam_rms", "q", "r")),
                   *(np.asarray(d[k], int) for k in ("n_cfg", "n_cfg_val", "n_cfg_cal", "n_atoms")),
                   [list(m) for m in d["merged"]], list(d.get("sources", [])))


def group_scales(scores, groups, cfg, G, alpha, n_min, src=None):
    """Configuration-weighted per-group lam_rms and pooled-CDF q (atom weights 1/n_{c,g}).  Groups below
    n_min configurations take the nearest qualifying band's values (same z-flag, then the other flag,
    then all groups pooled).  src per atom: 0 = T_val, >= 1 = a calibrate set (composition counts only)."""
    s, g, c = np.asarray(scores, float), np.asarray(groups), np.asarray(cfg)
    src = np.zeros(len(s), int) if src is None else np.asarray(src)
    _, inv, cnt = np.unique(c * G + g, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]

    def stats(m):
        nc = len(np.unique(c[m]))
        if nc == 0:
            return np.nan, np.inf, 0
        return float(np.sqrt(np.sum(w[m] * s[m] ** 2) / (3 * nc))), _pooled_q(s[m], w[m], alpha), nc

    lam, q, ncfg = np.full(G, np.nan), np.full(G, np.inf), np.zeros(G, int)
    nval, ncal, nat = np.zeros(G, int), np.zeros(G, int), np.zeros(G, int)
    for k in range(G):
        m = g == k
        lam[k], q[k], ncfg[k] = stats(m)
        nval[k] = len(np.unique(c[m & (src == 0)]))
        ncal[k] = len(np.unique(c[m & (src > 0)]))
        nat[k] = int(m.sum())
    ok = ncfg >= n_min
    lam_all, q_all, _ = stats(np.ones(len(s), bool))
    lam_s, q_s, merged = lam.copy(), q.copy(), []
    for k in np.flatnonzero(~ok):
        band, flag = divmod(int(k), 2)
        cand = [(abs(b - band), 0, b * 2 + flag) for b in range(G // 2) if ok[b * 2 + flag]]
        cand += [(abs(b - band), 1, b * 2 + 1 - flag) for b in range(G // 2) if ok[b * 2 + 1 - flag]]
        cand.sort(key=lambda t: (t[1], t[0]))
        k_src = cand[0][2] if cand else -1
        lam_s[k], q_s[k] = (lam[k_src], q[k_src]) if k_src >= 0 else (lam_all, q_all)
        merged.append([int(k), int(k_src)])
    r = q_s / (lam_s * chi3_ppf(1 - alpha))
    return GroupTable(float(alpha), int(n_min), lam_s, q_s, r, ncfg, nval, ncal, nat, merged)
