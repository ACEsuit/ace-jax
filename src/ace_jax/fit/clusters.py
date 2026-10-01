"""Sandwich cluster layout (math rev. 2): a configuration is one cluster unless it spans > ell in some
direction; then its force rows split into spatial blocks of side ~ell and its E+V rows form one more."""
import numpy as np


def _widths_and_frac(cfg):
    cell, pos, pbc = np.asarray(cfg.cell, float), np.asarray(cfg.positions, float), np.asarray(cfg.pbc, bool)
    w, frac = np.zeros(3), np.zeros((len(pos), 3))
    vol = abs(np.linalg.det(cell)) if np.all(np.linalg.norm(cell, axis=1) > 0) else 0.0
    inv = np.linalg.inv(cell) if vol > 0 else None
    for a in range(3):
        cr = np.cross(cell[(a + 1) % 3], cell[(a + 2) % 3])
        ncr = np.linalg.norm(cr)
        if pbc[a] and inv is not None and ncr > 1e-12:
            w[a] = vol / ncr
            frac[:, a] = (pos @ inv)[:, a] % 1.0
        elif len(pos):
            # bounding box: along the cell-face normal when usable, else the Cartesian axis
            p = pos @ (cr / ncr) if ncr > 1e-12 else pos[:, a]
            w[a] = p.max() - p.min()
            frac[:, a] = (p - p.min()) / max(w[a], 1e-12)
    return w, frac


def config_blocks(cfg, ell):
    """Block id per atom, or None when the configuration is a single cluster."""
    if not np.isfinite(ell):
        return None
    w, frac = _widths_and_frac(cfg)
    n = np.maximum(1, np.floor(w / ell)).astype(int)
    if np.all(n == 1):
        return None
    idx = np.minimum(np.floor(frac * n).astype(int), n - 1)
    _, ids = np.unique(np.ravel_multi_index(idx.T, n), return_inverse=True)
    return ids.astype(np.int64)


def row_clusters(ds, configs, ell):
    """Per batch {"E": (C,), "F": (Ncap,), "V": (C,)} int64 cluster ids (-1 = padding), and K."""
    if np.isfinite(ell):
        if configs is None:
            raise ValueError("row_clusters: configs is required when ell is finite")
        n_live = int(np.asarray(ds.cfg_mask).sum())
        if len(configs) != n_live:
            raise ValueError(f"row_clusters: {len(configs)} configs but {n_live} live configurations in ds")
    out, k, gc = [], 0, 0
    for i in range(ds.n_batches):
        cm = np.asarray(ds.cfg_mask[i])
        ncfg = np.asarray(ds.node_cfg[i])
        C, N = len(cm), len(ncfg)
        E, F, V = np.full(C, -1, np.int64), np.full(N, -1, np.int64), np.full(C, -1, np.int64)
        for c in np.flatnonzero(cm):
            nodes = np.flatnonzero(ncfg == c)
            blk = config_blocks(configs[gc], ell) if np.isfinite(ell) else None
            if blk is None:
                E[c] = V[c] = k
                F[nodes] = k
                k += 1
            else:
                F[nodes] = k + blk
                k += int(blk.max()) + 1
                E[c] = V[c] = k
                k += 1
            gc += 1
        out.append({"E": E, "F": F, "V": V})
    return out, k
