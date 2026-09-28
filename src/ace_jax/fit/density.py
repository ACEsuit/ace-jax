"""Per-species Finnis-Sinclair density columns for the VarPro rung.

For each centre species z a single learned density rho_i = eta_z . B_pair_i over
the n_pair ACE pair channels, FS-embedded u_i = ssqrt(rho_i) (the smooth signed
square root, feature.py), contributes a species-blocked column to the linear
design (site energy d_z * u_i).  eta (NZ, n_pair) is the VarPro parameter; given
eta the fit is the ordinary M = 0 linear solve over [c | d].

`density_rows_masked` generalises this to P >= 0 densities over an arbitrary
masked span of the full compact basis (eta (P, NZ, D), mask (D,) 0/1), used by
`fit.radial_density`'s joint radial + density VarPro to widen the linear design
by P * NZ columns.

Rows are analytic from the precomputed edge Jacobian J = dB/dr (linear_rows), so
d/deta is cheap -- no autodiff through compact_basis.  Force/virial assembly
mirrors rows.residual_rows and is FD-checked against autodiff of the energy rows.
"""
import jax
import jax.numpy as jnp

from .data import VOIGT
from .rows import Rows, linear_rows

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


def density_rows_masked(eta, mask, cfg, batch, X, J):
    """P * NZ density columns over the masked compact basis: column p * NZ + z is
    species z's ssqrt(eta[p, z] . (mask * X_i)).  eta (P, NZ, D), mask (D,) 0/1,
    X (Ncap, D) and J (Ncap * K, D, 3) as linear_rows returns them.  Each p is
    density_rows_pair over the full width with eta masked (its algebra is
    width-agnostic), so P = 1 with the pair mask reproduces density_rows."""
    Ncap, K = batch.nbr.shape
    P = eta.shape[0]
    if P == 0:
        C = batch.y_E.shape[0]
        return Rows(jnp.zeros((C, 0)), jnp.zeros((Ncap, 3, 0)), jnp.zeros((C, 6, 0)))
    Jr = J.reshape(Ncap, K, cfg.D, 3)
    rows = [density_rows_pair(eta[p] * mask, cfg, batch, X, Jr) for p in range(P)]
    return Rows(*(jnp.concatenate([getattr(r, k) for r in rows], axis=-1) for k in "EFV"))


def density_mask(cfg, span):
    """(D,) 0/1 span of the density over the compact basis [B | pair]."""
    if span == "full":
        return jnp.ones(cfg.D)
    if span == "pair":
        return jnp.concatenate([jnp.zeros(cfg.n_B), jnp.ones(cfg.n_pair)])
    raise ValueError(f"density span must be 'full' or 'pair', got {span!r}")


def compact_gamma(gamma, cfg):
    """The species-blocked prior diagonal (len_basis,) regrouped per species onto
    the compact basis, (NZ, D), in the layout of rows._place."""
    g, nB, nP, NZ = jnp.asarray(gamma), cfg.n_B, cfg.n_pair, cfg.NZ
    return jnp.stack([jnp.concatenate([g[z * nB:(z + 1) * nB], g[NZ * nB + z * nP:NZ * nB + (z + 1) * nP]])
                      for z in range(NZ)])


def density_gamma(gamma, P, NZ):
    """Prior diagonal of the widened readout [c | d]: Gamma, then the geometric
    mean of Gamma for each of the P * NZ density coefficients."""
    g = jnp.asarray(gamma)
    return jnp.concatenate([g, jnp.full(P * NZ, jnp.exp(jnp.mean(jnp.log(g))))])


def batch_density_stats(model, eta, mask, cfg, batch):
    from .stats import Stats, _linear_type_stats
    r, X, J = linear_rows(model, cfg, batch)
    d = density_rows_masked(eta, mask, cfg, batch, X, J)
    E, F, V = (jnp.concatenate([getattr(r, k), getattr(d, k)], -1) for k in "EFV")
    Dt = E.shape[-1]
    sE = _linear_type_stats(E, batch.y_E, batch.w_E)
    sF = _linear_type_stats(F.reshape(-1, Dt), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3))
    sV = _linear_type_stats(V.reshape(-1, Dt), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6))
    return Stats(*(s[i] for i in range(5) for s in (sE, sF, sV)))


def linear_density_statistics(model, eta, mask, cfg, ds):
    """Streamed statistics of the widened design [linear | P * NZ density columns],
    Dt = len_basis + P * NZ, same layout and checkpointed scan as
    stats.linear_statistics.  P = 0 IS linear_statistics (bit-identical)."""
    from .stats import Stats, linear_statistics
    if eta.shape[0] == 0:
        return linear_statistics(model, cfg, ds)
    Dt = cfg.len_basis + eta.shape[0] * cfg.NZ
    z2, z1, z0 = jnp.zeros((Dt, Dt)), jnp.zeros(Dt), jnp.zeros(())
    zero = Stats(z2, z2, z2, z1, z1, z1, z0, z0, z0, z0, z0, z0, z0, z0, z0)
    f = jax.checkpoint(lambda b: batch_density_stats(model, eta, mask, cfg, b))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(b)), None), zero, ds)[0]
