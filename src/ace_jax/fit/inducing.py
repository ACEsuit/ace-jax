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


def build_pmap(cfg, scale, density=None):
    """Residual feature projection Pmap (D, d).  density=None -> isotropic
    (diag(scale), d = D).  density="pair" -> select the n_pair ACE pair-density
    channels (the last n_pair compact components), scaled, d = n_pair."""
    import numpy as _np
    D = cfg.D
    if density is None:
        return _np.diag(_np.asarray(scale))
    if density == "pair":
        cols = _np.arange(cfg.n_B, cfg.n_B + cfg.n_pair)
        P = _np.zeros((D, len(cols)))
        P[cols, _np.arange(len(cols))] = _np.asarray(scale)[cols]
        return P
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


def select_inducing(X, S, Z, node_mask, m_per_species, scale, Pmap=None, warp="none", embed=None):
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
        NZ = int(np.asarray(Z).reshape(-1)[np.asarray(node_mask).reshape(-1)].max()) + 1
        embed = species_onehot(NZ)
    else:
        embed = jnp.asarray(embed, jnp.float64)
    return Inducing(jnp.asarray(UM), jnp.asarray(np.concatenate(ss) if ss else np.zeros(0)),
                    jnp.asarray(np.concatenate(zs) if zs else np.zeros(0), jnp.int32),
                    jnp.asarray(scale), jnp.asarray(Pmap), warp, embed=embed)
