"""Per-species Finnis-Sinclair density columns for the VarPro rung.

For each centre species z a single learned density rho_i = eta_z . B_pair_i over
the n_pair ACE pair channels, FS-embedded u_i = ssqrt(rho_i) (the smooth signed
square root, feature.py), contributes a species-blocked column to the linear
design (site energy d_z * u_i).  eta (NZ, n_pair) is the VarPro parameter; given
eta the fit is the ordinary M = 0 linear solve over [c | d].

Rows are analytic from the precomputed edge Jacobian J = dB/dr (linear_rows), so
d/deta is cheap -- no autodiff through compact_basis.  Force/virial assembly
mirrors rows.residual_rows and is FD-checked against autodiff of the energy rows.
"""
import jax
import jax.numpy as jnp

from .data import VOIGT, flat_edges
from .rows import Rows, _voigt

EPS = 1e-6


def _ssqrt(rho):
    return rho * (rho * rho + EPS) ** -0.25


def _dssqrt(rho):
    q = rho * rho + EPS
    return q ** -1.25 * (0.5 * rho * rho + EPS)


def density_rows(eta, cfg, batch, X, J):
    """NZ per-species density columns from the full descriptor X and edge
    Jacobian J -- slices the pair block and calls density_rows_pair."""
    Ncap, K = batch.nbr.shape
    nB, nP = cfg.n_B, cfg.n_pair
    Bpair = X[:, nB:nB + nP]
    Jpair = J.reshape(Ncap, K, cfg.D, 3)[:, :, nB:nB + nP, :]
    return density_rows_pair(eta, cfg, batch, Bpair, Jpair)


def density_rows_pair(eta, cfg, batch, Bpair, Jpair):
    """NZ per-species density columns from the pre-sliced pair block Bpair
    (Ncap, n_pair) and its edge Jacobian Jpair (Ncap, K, n_pair, 3) -- the only
    parts of X, J the density needs, so a caller can cache these (~13 MB/batch)
    rather than the full J (~0.6 GB/batch).  Returns Rows E (C, NZ), F (Ncap, 3,
    NZ), V (C, 6, NZ)."""
    Ncap, K = batch.nbr.shape
    C = batch.y_E.shape[0]
    NZ = cfg.NZ
    z = batch.node_z
    rho = jnp.sum(eta[z] * Bpair, axis=1)                      # (Ncap,)
    u = _ssqrt(rho)
    g = _dssqrt(rho)[:, None] * eta[z]                         # (Ncap, nP) = du_i / dB_pair_i
    onehot = jax.nn.one_hot(z, NZ) * batch.node_mask[:, None]  # (Ncap, NZ) species column selector

    E = jax.ops.segment_sum(u[:, None] * onehot, batch.node_cfg, num_segments=C + 1)[:C]

    T = jnp.einsum("np,nkpa->nka", g, Jpair)                   # (Ncap, K, 3)
    T = jnp.where(batch.nbr_mask[:, :, None], T, 0.0)

    # forces (Ncap, 3, NZ): -dE/dr_i = + sum_k T (sender), -dE/dr_j = -T (receiver);
    # both land in the SENDER's species column.
    F = jnp.zeros((Ncap, 3, NZ)).at[jnp.arange(Ncap)].add(T.sum(1)[:, :, None] * onehot[:, None, :])
    colhot = jax.nn.one_hot(jnp.repeat(z, K), NZ)              # (E, NZ): sender species per edge
    contrib = -T.reshape(Ncap * K, 3)[:, :, None] * colhot[:, None, :]
    F = F.at[batch.nbr.reshape(-1)].add(contrib)

    # virials (C, 6, NZ): V_ab = -1/2 (T_a r_b + T_b r_a) summed per config, species-blocked
    rij = batch.rij.reshape(Ncap * K, 3)
    Tf = T.reshape(Ncap * K, 3)
    ecfg = batch.node_cfg[jnp.repeat(jnp.arange(Ncap), K)]
    V = jnp.zeros((C + 1, 6, NZ))
    for v, (a, b) in enumerate(VOIGT):
        val = -0.5 * (Tf[:, a] * rij[:, b] + Tf[:, b] * rij[:, a])      # (E,)
        V = V.at[:, v].add(jax.ops.segment_sum(val[:, None] * colhot, ecfg, num_segments=C + 1))
    return Rows(E, F, V[:C])
