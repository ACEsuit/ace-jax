"""Feature rows Phi = [linear block | residual block] for one padded batch.

Linear block: energy row = sum of site descriptors; force/virial rows from the
edge Jacobian, species-blocked exactly as ACEpotentials.site_descriptors so the
rows equal ACEfit's design matrix (tests/test_gp_rows.py).

Residual block: energy row = sum_i k(x_i, x_m); force/virial rows from
dk/dx . dX/dr + dk/ds . ds/dr, scanned over node chunks so the (chunk, M, D)
kernel-gradient tensor -- 8 GB for a whole batch at target scale -- never
exists at once."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .data import VOIGT, flat_edges
from .kernels import grad_k_rows, k_rows
from .summary import summary_edge_jacobian


class Rows(NamedTuple):
    E: jnp.ndarray   # (C, Dt)
    F: jnp.ndarray   # (Ncap, 3, Dt)
    V: jnp.ndarray   # (C, 6, Dt)


def _place(dst, src, z, cfg):
    """Add compact (..., D) rows into species block z of a (..., len_basis) array,
    in the layout of ACEModel.site_descriptors (B blocks first, then pair blocks)."""
    nB, nP, NZ = cfg.n_B, cfg.n_pair, cfg.NZ
    dst = dst.at[..., z * nB:(z + 1) * nB].add(src[..., :nB])
    return dst.at[..., NZ * nB + z * nP: NZ * nB + (z + 1) * nP].add(src[..., nB:])


def _voigt(T, rij):
    """T (E, ..., 3) derivative w.r.t. rij (E, 3) -> the 6 Voigt virial
    contributions (E, ..., 6): V_ab = -1/2 (T_a r_b + T_b r_a)."""
    cols = [-0.5 * (T[..., a] * rij[:, b].reshape((-1,) + (1,) * (T.ndim - 2))
                    + T[..., b] * rij[:, a].reshape((-1,) + (1,) * (T.ndim - 2)))
            for a, b in VOIGT]
    return jnp.stack(cols, axis=-1)


def linear_rows(model, cfg, batch):
    Ncap, K = batch.nbr.shape
    C = batch.y_E.shape[0]          # configs per batch is a property of the batch, not of cfg
    rij, send, recv, mask = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    zi, zj = batch.node_z[send], batch.node_z[recv]
    # dense per-node contraction: the sparse edge_jacobian gathers dB/dA onto
    # every edge, 14 GB at the Cantor basis size (n_B 1348, 18k edges/batch)
    X, J = model.edge_jacobian_dense(batch.rij, jnp.broadcast_to(batch.node_z[:, None], (Ncap, K)),
                                     batch.node_z[batch.nbr], batch.nbr_mask)      # (Ncap, D), (E, D, 3)
    seg = lambda a, ids, n: jax.ops.segment_sum(a, ids, num_segments=n)
    L = cfg.len_basis
    E = jnp.zeros((Ncap, L))
    F = jnp.zeros((Ncap, 3, L))
    V = jnp.zeros((C + 1, 6, L))
    edge_cfg = batch.node_cfg[send]
    for z in range(cfg.NZ):
        E = _place(E, jnp.where((batch.node_z == z)[:, None], X, 0.0), z, cfg)
        Jz = jnp.where((zi == z)[:, None, None], J, 0.0)
        dEdr = seg(Jz, recv, Ncap) - seg(Jz, send, Ncap)             # (Ncap, D, 3)
        F = _place(F, -jnp.swapaxes(dEdr, 1, 2), z, cfg)
        Vz = seg(_voigt(Jz, rij), edge_cfg, C + 1)                   # (C+1, D, 6)
        V = _place(V, jnp.swapaxes(Vz, 1, 2), z, cfg)
    E = seg(E, batch.node_cfg, C + 1)[:C]
    return Rows(E, F, V[:C]), X, J


def residual_inputs(ind, cfg, batch, X):
    """The residual GP's site inputs for one batch: feature-mapped coordinates
    U = phi_res(X) (Ncap, d), summaries s (Ncap,) and the summary edge Jacobian
    Js (E, 3).  Shared by residual_rows and predict's DTC term."""
    from .feature import apply
    Ncap = batch.nbr.shape[0]
    rij, send, _, mask = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    s, Js = summary_edge_jacobian(rij, send, Ncap, cfg.r0, cfg.rcut, cfg.p, mask)
    return apply(X, ind.Pmap, ind.warp), s, Js


def residual_rows(theta, spec, ind, cfg, batch, X, J):
    from .feature import apply, dwarp
    Ncap, K = batch.nbr.shape
    C, c = batch.y_E.shape[0], cfg.node_chunk
    M, d = ind.XM.shape                 # d = feature dim (D for isotropic, n_pair for density)
    Dfull = cfg.D
    if M == 0:                      # BLR limit: no residual block (static shape, legal under jit)
        return Rows(jnp.zeros((C, 0)), jnp.zeros((Ncap, 3, 0)), jnp.zeros((C, 6, 0)))
    rij, send, _, mask = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    U, s, Js = residual_inputs(ind, cfg, batch, X)                    # (Ncap, d)
    # feature edge Jacobian JU (Ncap, K, d, 3) = dwarp(X@Pmap) * (Pmap^T J);
    # folds ind.scale, the projection and the warp into one object.
    U0 = X @ ind.Pmap
    dw = dwarp(U0, ind.warp)                                          # (Ncap, d)
    Jr = J.reshape(Ncap, K, Dfull, 3)
    JU = dw[:, None, :, None] * jnp.einsum("nkDa,Dq->nkqa", Jr, ind.Pmap)     # (Ncap, K, d, 3)
    Kx = k_rows(theta, spec, U, s, batch.node_z, ind.XM, ind.SM, ind.ZM, ind.embed)   # (Ncap, M)
    E = jax.ops.segment_sum(Kx, batch.node_cfg, num_segments=C + 1)[:C]

    def step(carry, ch):
        F, V = carry
        Uc, sc, zc, JUc, Jsc, rc, nc, mc, nodes, cfgc = ch
        dKx, dKs = grad_k_rows(theta, spec, Uc, sc, zc, ind.XM, ind.SM, ind.ZM, ind.embed)  # (c,M,d), (c,M)
        T = (jnp.einsum("ckqa,cmq->ckam", JUc, dKx)
             + jnp.einsum("cka,cm->ckam", Jsc, dKs))                          # (c, K, 3, M)
        T = jnp.where(mc[:, :, None, None], T, 0.0)
        F = F.at[nodes].add(T.sum(1))                                          # -dE/dr_i = +sum_k T
        F = F.at[nc.reshape(-1)].add(-T.reshape(c * K, 3, M))                  # -dE/dr_j = -T
        Vv = _voigt(jnp.swapaxes(T, 2, 3).reshape(c * K, M, 3), rc.reshape(c * K, 3))  # (cK, M, 6)
        V = V.at[jnp.repeat(cfgc, K)].add(jnp.swapaxes(Vv, 1, 2))
        return (F, V), None

    n_chunks = Ncap // c
    chunk = lambda a: a.reshape((n_chunks, c) + a.shape[1:])
    chunks = (chunk(U), chunk(s), chunk(batch.node_z), chunk(JU),
              chunk(Js.reshape(Ncap, K, 3)), chunk(batch.rij), chunk(batch.nbr),
              chunk(batch.nbr_mask), chunk(jnp.arange(Ncap)), chunk(batch.node_cfg))
    (F, V), _ = jax.lax.scan(step, (jnp.zeros((Ncap, 3, M)), jnp.zeros((C + 1, 6, M))), chunks)
    return Rows(E, F, V[:C])


def batch_rows(theta, spec, model, ind, cfg, batch):
    lin, X, J = linear_rows(model, cfg, batch)
    res = residual_rows(theta, spec, ind, cfg, batch, X, J)
    return Rows(jnp.concatenate([lin.E, res.E], 1), jnp.concatenate([lin.F, res.F], 2),
                jnp.concatenate([lin.V, res.V], 2))
