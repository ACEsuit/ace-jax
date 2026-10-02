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
    """The p50 / p90 / p99 quantiles of the finite d, tie-safe: tied quantiles collapse (np.unique) and an
    edge at the minimum d is dropped, since many identical d (perfect-lattice atoms) would otherwise give
    empty bands (band k is edges[k-1] <= d < edges[k]).  Fewer edges, fewer groups; continuous d unchanged."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    e = np.unique(np.quantile(d, qs))
    return e[e > d.min()]


def n_groups(edges):
    return 2 * (len(edges) + 1)


def assign_groups(z, d, z_star, edges):
    """band(d) * 2 + [z != z*], band k holding edges[k-1] <= d < edges[k]; NaN d goes in the top band."""
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


def effective_n_min(n_min, alpha):
    """The merge threshold: n_min, raised to the fewest configurations whose pooled CDF (with its +inf test
    point of one configuration's weight) can reach 1 - alpha, ceil((1 - alpha) / alpha); a smaller pool has
    q = inf, in a per-group pool and in the all-groups fallback alike."""
    return max(int(n_min), int(np.ceil((1 - alpha) / alpha - 1e-9)))


def _pooled_q(s, w, alpha, n_cfg):
    """inf{t : F(t) >= 1 - alpha} for the pooled CDF F(t) = sum_{s_i <= t} w_i / (W (1 + 1/n_cfg)), W = sum w:
    the +inf test point of the pool (Dunn, Wasserman & Ramdas) weighs W / n_cfg, one configuration's mean
    total weight.  For a single group W = n_cfg and this is sum_{s_i <= t} w_i / (W + 1); for the all-groups
    pool, where a configuration spans several groups, it keeps the test point at one configuration's share, so
    q is finite iff n_cfg >= ceil((1 - alpha) / alpha)."""
    if len(s) == 0:
        return np.inf
    o = np.argsort(s)
    W = w.sum()
    cw = np.cumsum(w[o]) / (W + W / n_cfg)
    k = int(np.searchsorted(cw, 1 - alpha - 1e-15))
    return float(s[o][k]) if k < len(s) else np.inf


def json_safe(obj):
    """obj with every non-finite float replaced by the string "inf" / "-inf" / "nan", so json.dumps writes
    strict JSON (no bare Infinity / NaN tokens); `restore_table` and np.asarray(..., float) read them back."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return "nan" if np.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


_TABLE_FLOATS = ("lam_rms", "q", "r")


def restore_table(d):
    """A group-table dict read from JSON with its float columns as floats: "inf"/"nan" strings (json_safe),
    None, and the legacy bare Infinity token (already a float) all map back to inf / nan."""
    if d is None:
        return None
    out = dict(d)
    for k in _TABLE_FLOATS:
        if k in out:
            out[k] = [np.nan if x is None else float(x) for x in out[k]]
    return out


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
        d = restore_table(d)
        return cls(float(d["alpha"]), int(d["n_min"]),
                   *(np.asarray(d[k], float) for k in ("lam_rms", "q", "r")),
                   *(np.asarray(d[k], int) for k in ("n_cfg", "n_cfg_val", "n_cfg_cal", "n_atoms")),
                   [list(m) for m in d["merged"]], list(d.get("sources", [])))


def config_weights(groups, cfg, G):
    """Atom weights 1/n_{c,g}: each (configuration, group) cell weighs one in total."""
    _, inv, cnt = np.unique(np.asarray(cfg) * G + np.asarray(groups), return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def _weighted_lam(s, w):
    return float(np.sqrt(np.sum(w * s ** 2) / (3 * np.sum(w))))


def pooled_lam_rms(scores, groups, cfg, G):
    """The configuration-weighted rms scale pooled over every (c, g): sum w s^2 / (3 sum w), w = 1/n_{c,g}
    -- group_scales' all-groups lam_rms."""
    s = np.asarray(scores, float)
    return _weighted_lam(s, config_weights(groups, cfg, G)) if len(s) else np.nan


BETA_MAX = 0.5        # bias-dominated limit: lam^2 ~ N when e^2 is flat and v ~ 1/N (the (1 - f)^-1/2 bound)


def transfer_fixed(method, lam1, N, N_fit):
    """report["transfer"] entries of a fixed exponent: "sqrt" beta = 1/2, "none" beta = 0 (no second fit)."""
    beta = {"sqrt": BETA_MAX, "none": 0.0}[method]
    return {"method": method, "N_fit2": None, "lam1": float(lam1), "lam2": None, "beta_raw": None,
            "beta": beta, "factor": float((N / N_fit) ** beta) if beta else 1.0}


def transfer_exponent(lam1, lam2, N, N_fit, N_fit2, log=print):
    """The per-fit transfer exponent from two hold-out posteriors scored on the same T_val: lam1 from P_fit
    (N_fit configs), lam2 from P_fit2 (N_fit2 < N_fit): beta_raw = log(lam1/lam2) / log(N_fit/N_fit2),
    beta = clip(beta_raw, 0, 1/2), factor t = (N/N_fit)^beta carries the hold-out scale to the served
    posterior on all N.  An undefined beta_raw (N_fit2 = N_fit, a non-finite or non-positive scale) falls
    back to the conservative (bias-dominated) bound beta = 1/2; a clip or that fallback is logged as a
    WARNING.  beta = 0 is the variance-dominated limit (e^2 and v both ~ 1/N)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = float(np.log(lam1 / lam2) / np.log(N_fit / N_fit2)) if N_fit2 < N_fit else np.nan
    if not np.isfinite(raw):
        beta = BETA_MAX
        log(f"WARNING: ARD transfer: exponent undefined (lam_fit {lam1:.4g} with N {N_fit}, lam_fit2 "
            f"{lam2:.4g} with N {N_fit2}); using the bias-dominated bound beta = {BETA_MAX}")
    else:
        beta = float(np.clip(raw, 0.0, BETA_MAX))
    factor = float((N / N_fit) ** beta)
    log(f"ARD transfer: lam_fit {lam1:.4g} (N {N_fit}), lam_fit2 {lam2:.4g} (N {N_fit2}), "
        f"beta_raw {raw:.3g} -> beta {beta:.3g}, factor {factor:.4g}")
    if np.isfinite(raw) and raw != beta:
        why = ""
        if raw < 0 and lam2 >= lam1 * (1 + 1e-3):
            why = (f": lam_fit2 {lam2:.4g} >= lam_fit {lam1:.4g}, the hold-out scale shrinks with more data "
                   f"(expected to grow or stay flat: noise in the two estimates?)")
        log(f"WARNING: ARD transfer: beta_raw {raw:.3g} outside [0, {BETA_MAX}], clipped to {beta:.3g}{why}")
    return {"method": "exponent", "N_fit2": int(N_fit2), "lam1": float(lam1), "lam2": float(lam2),
            "beta_raw": raw, "beta": beta, "factor": factor}


def group_scales(scores, groups, cfg, G, alpha, n_min, src=None, log=None):
    """Configuration-weighted per-group lam_rms and pooled-CDF q (atom weights 1/n_{c,g}).  Groups below
    effective_n_min(n_min, alpha) configurations take the nearest qualifying band's values (same z-flag,
    then the other flag, then all groups pooled).  lam_rms^2 = sum w s^2 / (3 sum w): sum w is n_cfg for
    one group, and the total weight (not n_cfg) for the all-groups pool, where a configuration counts once
    per group it spans; its CDF puts the +inf test point at weight sum w / n_cfg, so every pool needs
    effective_n_min configurations for a finite q.  If even the all-groups pool has fewer, q stays inf (that
    coverage is not attainable from this many configurations) and a WARNING goes to log.
    src per atom: 0 = T_val, >= 1 = a calibrate set (composition counts only)."""
    s, g, c = np.asarray(scores, float), np.asarray(groups), np.asarray(cfg)
    src = np.zeros(len(s), int) if src is None else np.asarray(src)
    w = config_weights(g, c, G)

    def stats(m):
        nc = len(np.unique(c[m]))
        if nc == 0:
            return np.nan, np.inf, 0
        return _weighted_lam(s[m], w[m]), _pooled_q(s[m], w[m], alpha, nc), nc

    lam, q, ncfg = np.full(G, np.nan), np.full(G, np.inf), np.zeros(G, int)
    nval, ncal, nat = np.zeros(G, int), np.zeros(G, int), np.zeros(G, int)
    for k in range(G):
        m = g == k
        lam[k], q[k], ncfg[k] = stats(m)
        nval[k] = len(np.unique(c[m & (src == 0)]))
        ncal[k] = len(np.unique(c[m & (src > 0)]))
        nat[k] = int(m.sum())
    n_eff = effective_n_min(n_min, alpha)
    ok = ncfg >= n_eff
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
    if log is not None and not np.isfinite(q_s).all():
        need = effective_n_min(1, alpha)
        log(f"WARNING: ARD conformal: coverage {1 - alpha:.6g} needs at least {need} configurations in a pool "
            f"for a finite q, but the calibration set has "
            f"{len(np.unique(c))} configurations: forces_q is infinite for "
            f"{int(np.sum(~np.isfinite(q_s)))} of {G} groups (lower the coverage or add configurations)")
    r = q_s / (lam_s * chi3_ppf(1 - alpha))
    return GroupTable(float(alpha), int(n_min), lam_s, q_s, r, ncfg, nval, ncal, nat, merged)
