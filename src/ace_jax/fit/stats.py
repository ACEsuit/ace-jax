"""Sufficient statistics of the data, per observation type, streamed over
batches.  Memory is one batch's rows plus the (Dt, Dt) accumulators; the scan's
reverse pass recomputes each batch under jax.checkpoint, which is the two-pass
gradient algorithm of the spec with no hand-written VJP."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .rows import batch_rows


class Stats(NamedTuple):
    G_E: jnp.ndarray; G_F: jnp.ndarray; G_V: jnp.ndarray
    b_E: jnp.ndarray; b_F: jnp.ndarray; b_V: jnp.ndarray
    yy_E: jnp.ndarray; yy_F: jnp.ndarray; yy_V: jnp.ndarray
    n_E: jnp.ndarray; n_F: jnp.ndarray; n_V: jnp.ndarray
    logw_E: jnp.ndarray; logw_F: jnp.ndarray; logw_V: jnp.ndarray


def _type_stats(Phi, y, w):
    """Phi (n, Dt), y (n,), w (n,) structural weights (0 = absent)."""
    Pw = Phi * w[:, None]
    yw = y * w
    live = w > 0
    return (Pw.T @ Pw, Pw.T @ yw, yw @ yw, live.sum(),
            jnp.sum(jnp.where(live, 2.0 * jnp.log(jnp.where(live, w, 1.0)), 0.0)))


def batch_stats(theta, spec, model, ind, cfg, batch):
    r = batch_rows(theta, spec, model, ind, cfg, batch)
    Dt = r.E.shape[-1]
    E = _type_stats(r.E, batch.y_E, batch.w_E)
    F = _type_stats(r.F.reshape(-1, Dt), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3))
    V = _type_stats(r.V.reshape(-1, Dt), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6))
    return Stats(E[0], F[0], V[0], E[1], F[1], V[1], E[2], F[2], V[2],
                 E[3], F[3], V[3], E[4], F[4], V[4])


def sufficient_statistics(theta, spec, model, ind, cfg, ds):
    Dt = cfg.len_basis + ind.XM.shape[0]
    z2, z1, z0 = jnp.zeros((Dt, Dt)), jnp.zeros(Dt), jnp.zeros(())
    zero = Stats(z2, z2, z2, z1, z1, z1, z0, z0, z0, z0, z0, z0, z0, z0, z0)
    f = jax.checkpoint(lambda th, b: batch_stats(th, spec, model, ind, cfg, b))

    def body(carry, batch):
        return jax.tree.map(jnp.add, carry, f(theta, batch)), None

    return jax.lax.scan(body, zero, ds)[0]


# ---------------------------------------------------------------------------
# theta-split statistics.  The linear block G_BB is theta-INDEPENDENT (the ACE
# features do not move with the hyperparameters; sigma_c enters as the prior,
# not the Gram), so it is streamed ONCE and cached, and only the M residual
# columns k_theta(B, B_M) are recomputed per evaluation.  For the Cantor basis
# (L = 6890, M = 500) this cuts the per-gradient Gram cost ~10x -- the expensive
# 6890^2 block leaves the inner loop.  assemble(linear, residual(theta)) equals
# sufficient_statistics(theta) to f64 roundoff (tests/test_gp_stats.py).

class ResidualStats(NamedTuple):
    GBM_E: jnp.ndarray; GBM_F: jnp.ndarray; GBM_V: jnp.ndarray   # (L, M)
    GMM_E: jnp.ndarray; GMM_F: jnp.ndarray; GMM_V: jnp.ndarray   # (M, M)
    bM_E: jnp.ndarray;  bM_F: jnp.ndarray;  bM_V: jnp.ndarray    # (M,)


def _linear_type_stats(Phi_B, y, w):
    """G_BB, b_B, yy, n, logw for one observation type from the linear rows."""
    Pw = Phi_B * w[:, None]
    yw = y * w
    live = w > 0
    return (Pw.T @ Pw, Pw.T @ yw, yw @ yw, live.sum(),
            jnp.sum(jnp.where(live, 2.0 * jnp.log(jnp.where(live, w, 1.0)), 0.0)))


def batch_linear_stats(model, cfg, batch):
    from .rows import linear_rows
    r, _, _ = linear_rows(model, cfg, batch)
    L = r.E.shape[-1]
    E = _linear_type_stats(r.E, batch.y_E, batch.w_E)
    F = _linear_type_stats(r.F.reshape(-1, L), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3))
    V = _linear_type_stats(r.V.reshape(-1, L), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6))
    return Stats(E[0], F[0], V[0], E[1], F[1], V[1], E[2], F[2], V[2],
                 E[3], F[3], V[3], E[4], F[4], V[4])


def linear_statistics(model, cfg, ds):
    """theta-independent linear statistics: Stats with Dt = len_basis."""
    L = cfg.len_basis
    z2, z1, z0 = jnp.zeros((L, L)), jnp.zeros(L), jnp.zeros(())
    zero = Stats(z2, z2, z2, z1, z1, z1, z0, z0, z0, z0, z0, z0, z0, z0, z0)
    f = jax.checkpoint(lambda b: batch_linear_stats(model, cfg, b))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(b)), None), zero, ds)[0]


def _residual_type_stats(Phi_B, Phi_M, y, w):
    """G_BM, G_MM, b_M for one type; the cross block reuses the (theta-indep)
    linear rows but is theta-dependent through Phi_M."""
    Bw, Mw, yw = Phi_B * w[:, None], Phi_M * w[:, None], y * w
    return Bw.T @ Mw, Mw.T @ Mw, Mw.T @ yw


def batch_residual_stats(theta, spec, model, ind, cfg, batch):
    r = batch_rows(theta, spec, model, ind, cfg, batch)
    L, Dt = cfg.len_basis, r.E.shape[-1]
    sl = lambda A: (A[..., :L], A[..., L:])
    parts = []
    for R, y, w, reps in ((r.E, batch.y_E, batch.w_E, 1),
                          (r.F, batch.y_F, batch.w_F, 3),
                          (r.V, batch.y_V, batch.w_V, 6)):
        B, Mrows = sl(R.reshape(-1, Dt))
        parts.append(_residual_type_stats(B, Mrows, y.reshape(-1), jnp.repeat(w, reps)))
    return ResidualStats(parts[0][0], parts[1][0], parts[2][0],
                         parts[0][1], parts[1][1], parts[2][1],
                         parts[0][2], parts[1][2], parts[2][2])


def residual_statistics(theta, spec, model, ind, cfg, ds):
    L, M = cfg.len_basis, ind.XM.shape[0]
    zBM, zMM, zM = jnp.zeros((L, M)), jnp.zeros((M, M)), jnp.zeros(M)
    zero = ResidualStats(zBM, zBM, zBM, zMM, zMM, zMM, zM, zM, zM)
    f = jax.checkpoint(lambda th, b: batch_residual_stats(th, spec, model, ind, cfg, b))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(theta, b)), None), zero, ds)[0]


def assemble_statistics(lin, res):
    """Full Stats (Dt = L + M) from the cached linear part and a residual part."""
    def blk(GBB, GBM, GMM):
        return jnp.block([[GBB, GBM], [GBM.T, GMM]])
    G_E = blk(lin.G_E, res.GBM_E, res.GMM_E)
    G_F = blk(lin.G_F, res.GBM_F, res.GMM_F)
    G_V = blk(lin.G_V, res.GBM_V, res.GMM_V)
    cat = jnp.concatenate
    return Stats(G_E, G_F, G_V,
                 cat([lin.b_E, res.bM_E]), cat([lin.b_F, res.bM_F]), cat([lin.b_V, res.bM_V]),
                 lin.yy_E, lin.yy_F, lin.yy_V, lin.n_E, lin.n_F, lin.n_V,
                 lin.logw_E, lin.logw_F, lin.logw_V)


# ---------------------------------------------------------------------------
# Per-(quantity, TYPE) statistics.  These mirror the single-type functions above
# but split each quantity's rows by their config-type index and accumulate one
# Stats block PER TYPE (a leading n_types axis on every field).  The split is a
# weight MASK (w * (type == t)) -- the structural weights carry NO sigma, so each
# per-(q,t) Gram stays theta-INDEPENDENT and cacheable exactly like the linear
# statistics above.  The per-type noise sigma_{q,t} enters only in the objective
# (objective.combine), NOT the Gram, which is what preserves the streamed
# theta-independent Gram caching.  With a single type these reduce (up to the
# leading size-1 axis) to the functions above, so the fit is bit-identical.

def _linear_type_stats_typed(Phi_B, y, w, tidx, n_types):
    comps = [_linear_type_stats(Phi_B, y, jnp.where(tidx == t, w, 0.0))
             for t in range(n_types)]
    return tuple(jnp.stack([c[k] for c in comps]) for k in range(5))


def batch_linear_stats_typed(model, cfg, batch, n_types):
    from .rows import linear_rows
    r, _, _ = linear_rows(model, cfg, batch)
    L = r.E.shape[-1]
    E = _linear_type_stats_typed(r.E, batch.y_E, batch.w_E, batch.cfg_type, n_types)
    F = _linear_type_stats_typed(r.F.reshape(-1, L), batch.y_F.reshape(-1),
                                 jnp.repeat(batch.w_F, 3), jnp.repeat(batch.node_type, 3), n_types)
    V = _linear_type_stats_typed(r.V.reshape(-1, L), batch.y_V.reshape(-1),
                                 jnp.repeat(batch.w_V, 6), jnp.repeat(batch.cfg_type, 6), n_types)
    return Stats(E[0], F[0], V[0], E[1], F[1], V[1], E[2], F[2], V[2],
                 E[3], F[3], V[3], E[4], F[4], V[4])


def linear_statistics_typed(model, cfg, ds, n_types):
    """theta-independent linear statistics, one Stats block per type
    (leading n_types axis on every field)."""
    L = cfg.len_basis
    z2, z1, z0 = jnp.zeros((n_types, L, L)), jnp.zeros((n_types, L)), jnp.zeros((n_types,))
    zero = Stats(z2, z2, z2, z1, z1, z1, z0, z0, z0, z0, z0, z0, z0, z0, z0)
    f = jax.checkpoint(lambda b: batch_linear_stats_typed(model, cfg, b, n_types))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(b)), None), zero, ds)[0]


def _residual_type_stats_typed(Phi_B, Phi_M, y, w, tidx, n_types):
    comps = [_residual_type_stats(Phi_B, Phi_M, y, jnp.where(tidx == t, w, 0.0))
             for t in range(n_types)]
    return tuple(jnp.stack([c[k] for c in comps]) for k in range(3))


def batch_residual_stats_typed(theta, spec, model, ind, cfg, batch, n_types):
    r = batch_rows(theta, spec, model, ind, cfg, batch)
    L, Dt = cfg.len_basis, r.E.shape[-1]
    sl = lambda A: (A[..., :L], A[..., L:])
    parts = []
    for R, y, w, tidx, reps in (
            (r.E, batch.y_E, batch.w_E, batch.cfg_type, 1),
            (r.F, batch.y_F, batch.w_F, jnp.repeat(batch.node_type, 3), 3),
            (r.V, batch.y_V, batch.w_V, jnp.repeat(batch.cfg_type, 6), 6)):
        B, Mrows = sl(R.reshape(-1, Dt))
        parts.append(_residual_type_stats_typed(B, Mrows, y.reshape(-1),
                                                jnp.repeat(w, reps), tidx, n_types))
    return ResidualStats(parts[0][0], parts[1][0], parts[2][0],
                         parts[0][1], parts[1][1], parts[2][1],
                         parts[0][2], parts[1][2], parts[2][2])


def residual_statistics_typed(theta, spec, model, ind, cfg, ds, n_types):
    L, M = cfg.len_basis, ind.XM.shape[0]
    zBM, zMM, zM = (jnp.zeros((n_types, L, M)), jnp.zeros((n_types, M, M)),
                    jnp.zeros((n_types, M)))
    zero = ResidualStats(zBM, zBM, zBM, zMM, zMM, zMM, zM, zM, zM)
    f = jax.checkpoint(lambda th, b: batch_residual_stats_typed(th, spec, model, ind, cfg, b, n_types))
    return jax.lax.scan(lambda c, b: (jax.tree.map(jnp.add, c, f(theta, b)), None), zero, ds)[0]


def assemble_statistics_typed(lin, res):
    """Per-type full Stats: vmap assemble_statistics over the leading type axis."""
    return jax.vmap(assemble_statistics)(lin, res)
