"""theta-independent, once-per-dataset quantities: compact site descriptors,
the per-component descriptor scale, and the inducing set (farthest-point
sampling per species, GAP-style)."""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..eval import highest_precision
from .data import flat_edges
from .summary import site_summary


@dataclass(frozen=True)
class GPConfig:
    r0: float
    rcut: float
    n_B: int
    n_pair: int
    NZ: int
    C: int                 # configs per batch
    p: int = 10            # power-mean exponent of the summary s
    node_chunk: int = 32   # Task 7 scan chunk

    @property
    def len_basis(self):
        return (self.n_B + self.n_pair) * self.NZ

    @property
    def D(self):
        return self.n_B + self.n_pair


class Inducing(NamedTuple):
    XM: jnp.ndarray     # (M, d) feature-mapped inducing coordinates U_M
    SM: jnp.ndarray     # (M,)
    ZM: jnp.ndarray     # (M,) species index
    scale: jnp.ndarray  # (D,)  per-component descriptor scale (isotropic map)
    Pmap: jnp.ndarray   # (D, d) residual feature projection (diag(scale) if isotropic)
    warp: str           # feature warp ("none" | "sqrt"); static
    embed: jnp.ndarray  # (NZ, de) unit-normalized species embedding (eye = block-diagonal)


def principal_frame(R, d):
    """Uncentred principal-frame reduction of the rows of R (n, D) to d <= rank
    channels: returns (coordinates U S (n, d), frame V (D, d)), so coordinates =
    R @ V and, for d >= rank, their Gram equals R R^T exactly.  The same reduction
    as the species embedding (ACEpotentials ``_pca_reduce``; run.py's
    Python embedded-model authoring).  Uncentred on purpose: dot products
    (the cosine kernel's inputs) are what the frame preserves.  d is capped at the
    numerical rank (tolerance as ``_pca_reduce``)."""
    import numpy as _np
    R = _np.asarray(R, float)
    U, sv, Vt = _np.linalg.svd(R, full_matrices=False)
    tol = max(R.shape) * _np.finfo(float).eps * (sv[0] if sv.size else 0.0)
    k = min(int(d), int(_np.sum(sv > tol)))
    return U[:, :k] * sv[:k], Vt[:k].T


def _frame_from_rows(X, mask, scale, d, chunk=4096):
    """principal_frame's V for the scaled live rows of X without forming them:
    eigh of the streamed uncentred second moment (D, D)."""
    import numpy as _np
    Xl = _np.asarray(X)[_np.asarray(mask)] * _np.asarray(scale)
    C = _np.zeros((Xl.shape[1], Xl.shape[1]))
    for i in range(0, len(Xl), chunk):
        C += Xl[i:i + chunk].T @ Xl[i:i + chunk]
    w, V = _np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    sv = _np.sqrt(_np.maximum(w, 0.0))
    tol = max(Xl.shape) * _np.finfo(float).eps * sv[0]
    k = min(int(d), int(_np.sum(sv > tol)))
    return V[:, :k]


def build_pmap(cfg, scale, density=None, d=None, X=None, mask=None):
    """Residual feature projection Pmap (D, d).  density=None -> isotropic
    (diag(scale), d = D).  density="pair" -> select the n_pair ACE pair-density
    channels (the last n_pair compact components), scaled, d = n_pair.
    density="pca" -> diag(scale) V_d, V_d the top-d uncentred principal frame of
    the scaled live training-site descriptors X (nb, Ncap, D) under mask: a
    low-rank view of the FULL descriptor (many-body included), so the kernel is
    not limited to pair densities yet stays d-wide for the host cache."""
    import numpy as _np
    D = cfg.D
    if density is None:
        return _np.diag(_np.asarray(scale))
    if density == "pair":
        cols = _np.arange(cfg.n_B, cfg.n_B + cfg.n_pair)
        P = _np.zeros((D, len(cols)))
        P[cols, _np.arange(len(cols))] = _np.asarray(scale)[cols]
        return P
    if density == "pca":
        if X is None or mask is None or d is None:
            raise ValueError("density='pca' needs the training descriptors X, their node mask and d")
        return _np.asarray(scale)[:, None] * _frame_from_rows(X, mask, scale, d)
    raise ValueError(f"unknown density {density!r}")


def site_features(model, cfg, ds):
    def one(rij, nbr, nmask, node_z):
        n = nbr.shape[0]
        r, send, recv, m = flat_edges(rij, nbr, nmask)
        X = model.compact_basis(r, node_z[send], node_z[recv], send, n, m)
        S = site_summary(r, send, n, cfg.r0, cfg.rcut, cfg.p, m)
        return X, S
    with highest_precision():
        # sequential over batches: vmap would materialise every batch's edge
        # features at once (hundreds of GB at 6890 descriptors per site)
        return jax.lax.map(lambda t: one(*t), (ds.rij, ds.nbr, ds.nbr_mask, ds.node_z))


def descriptor_scale(X, node_mask):
    Xl = X.reshape(-1, X.shape[-1])[node_mask.reshape(-1)]
    std = jnp.std(Xl, axis=0)
    return 1.0 / (std + 1e-6 * std.max() + 1e-12)


def farthest_point(X, m, start=0):
    """Greedy farthest-point sampling; stops early once every remaining point
    coincides with a chosen one (never returns duplicates, possibly fewer than m)."""
    X = np.asarray(X)
    if m <= 0 or len(X) == 0:
        return np.zeros(0, int)
    idx = [int(start)]
    d = np.sum((X - X[start]) ** 2, axis=1)
    for _ in range(1, min(m, len(X))):
        if d.max() == 0:
            break
        nxt = int(np.argmax(d))
        idx.append(nxt)
        d = np.minimum(d, np.sum((X - X[nxt]) ** 2, axis=1))
    return np.asarray(idx)


def select_inducing(X, S, Z, node_mask, m_per_species, scale, Pmap=None, warp="none", embed=None, nz=None, de=None):
    X = np.asarray(X).reshape(-1, X.shape[-1]); S = np.asarray(S).reshape(-1)
    Z = np.asarray(Z).reshape(-1); live = np.asarray(node_mask).reshape(-1)
    scale = np.asarray(scale)
    Pmap = np.diag(scale) if Pmap is None else np.asarray(Pmap)   # isotropic default
    xs, ss, zs = [], [], []
    # An environment with no live edges (isolated atom) has an identically zero
    # descriptor and delta(s) = 0: it would make K_MM singular, so it is never a candidate.
    nonzero = np.any(X != 0, axis=1)
    for z in np.unique(Z[live]):
        pool = np.flatnonzero(live & (Z == z) & nonzero)
        m = m_per_species if isinstance(m_per_species, int) else m_per_species[int(z)]
        pick = pool[farthest_point(X[pool] * scale, m)]     # FPS in scaled B-space
        xs.append(X[pick]); ss.append(S[pick]); zs.append(Z[pick])
    from .embedding import species_onehot
    from .feature import apply as _apply
    Xpick = np.concatenate(xs) if xs else np.zeros((0, X.shape[-1]))
    UM = np.asarray(_apply(jnp.asarray(Xpick), jnp.asarray(Pmap), warp))
    if embed is None:
        # Default one-hot species embedding. Size it to the full model species count `nz`
        # (e.g. len(meta["elements"])) when given: the kernel gathers embed[z] by centre
        # species at PREDICTION too, and a test-only species with index >= width would be
        # silently clamped by JAX's gather to the last row (spurious cross-covariance). Any
        # width >= the species actually present reproduces (z==zm) bit-for-bit. Falls back to
        # the present-species count when nz is None (correct only if prediction sees no unseen
        # species); guards an all-masked batch.
        zlive = np.asarray(Z).reshape(-1)[np.asarray(node_mask).reshape(-1)]
        present = int(zlive.max()) + 1 if zlive.size else 1
        NZ = present if nz is None else max(int(nz), present)
        embed = species_onehot(NZ)
    else:
        embed = jnp.asarray(embed, jnp.float64)
        if de is not None and embed.shape[1] != int(de):
            raise ValueError(f"embed width {embed.shape[1]} != de {de}")
    return Inducing(jnp.asarray(UM), jnp.asarray(np.concatenate(ss) if ss else np.zeros(0)),
                    jnp.asarray(np.concatenate(zs) if zs else np.zeros(0), jnp.int32),
                    jnp.asarray(scale), jnp.asarray(Pmap), warp, embed=embed)
