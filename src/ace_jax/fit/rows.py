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
    zi = batch.node_z[send]
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


def linear_rows_chunked(model, cfg, batch, node_chunk=256):
    """`linear_rows(model, cfg, batch)[0]` computed a chunk of centre nodes at a time.

    `linear_rows` materialises the edge Jacobian J (Ncap*K, D, 3) of the whole batch.  For one
    cell of more than ~2.8k atoms at the production Cantor basis (D = 3007, K ~ 85) that tensor
    and the GEMM building it exceed 2^31 elements, and XLA's int32-indexed GEMM autotuning fails.
    Here each chunk's J (node_chunk*K, D, 3) is built, scattered into E/F/V and dropped; the rows
    are identical (tests/test_rows_chunked.py).  Returns Rows only (no X, J)."""
    Ncap, K = batch.nbr.shape
    C = batch.y_E.shape[0]
    L = cfg.len_basis
    nc = int(min(node_chunk, Ncap))
    n_chunks = -(-Ncap // nc)
    pad = n_chunks * nc - Ncap
    Np = Ncap + pad
    rij = jnp.concatenate([batch.rij, jnp.broadcast_to(batch.rij[-1:], (pad, K, 3))])
    nbr = jnp.concatenate([batch.nbr, jnp.zeros((pad, K), batch.nbr.dtype)])
    msk = jnp.concatenate([batch.nbr_mask, jnp.zeros((pad, K), bool)])
    node_z = jnp.concatenate([batch.node_z, jnp.zeros(pad, batch.node_z.dtype)])
    node_cfg = jnp.concatenate([batch.node_cfg, jnp.full(pad, C, batch.node_cfg.dtype)])
    seg = lambda a, ids, n: jax.ops.segment_sum(a, ids, num_segments=n)
    local = jnp.repeat(jnp.arange(nc), K)

    def body(i, acc):
        Enodes, F, V = acc
        s0 = i * nc
        r_c = jax.lax.dynamic_slice_in_dim(rij, s0, nc)
        nb_c = jax.lax.dynamic_slice_in_dim(nbr, s0, nc)
        m_c = jax.lax.dynamic_slice_in_dim(msk, s0, nc)
        z_c = jax.lax.dynamic_slice_in_dim(node_z, s0, nc)
        X, J = model.edge_jacobian_dense(r_c, jnp.broadcast_to(z_c[:, None], (nc, K)), node_z[nb_c], m_c)
        send, recv = s0 + local, nb_c.reshape(-1)
        zi, r_flat, edge_cfg = z_c[local], r_c.reshape(-1, 3), node_cfg[s0 + local]
        E_c = jnp.zeros((nc, L))
        for z in range(cfg.NZ):
            E_c = _place(E_c, jnp.where((z_c == z)[:, None], X, 0.0), z, cfg)
            Jz = jnp.where((zi == z)[:, None, None], J, 0.0)
            dEdr = seg(Jz, recv, Np) - seg(Jz, send, Np)                      # (Np, D, 3)
            F = _place(F, -jnp.swapaxes(dEdr, 1, 2), z, cfg)
            V = _place(V, jnp.swapaxes(seg(_voigt(Jz, r_flat), edge_cfg, C + 1), 1, 2), z, cfg)
        Enodes = jax.lax.dynamic_update_slice_in_dim(Enodes, E_c, s0, 0)
        return Enodes, F, V

    init = (jnp.zeros((Np, L)), jnp.zeros((Np, 3, L)), jnp.zeros((C + 1, 6, L)))
    Enodes, F, V = jax.lax.fori_loop(0, n_chunks, body, init)
    E = seg(Enodes, node_cfg, C + 1)[:C]
    return Rows(E, F[:Ncap], V[:C])


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
    """Residual rows from the full compact descriptors X (Ncap, D) and their
    edge Jacobian J (E, D, 3): projects onto the feature map and defers to
    residual_rows_from_inputs."""
    Ncap, K = batch.nbr.shape
    if ind.XM.shape[0] == 0:        # BLR limit: no residual block (static shape, legal under jit)
        C = batch.y_E.shape[0]
        return Rows(jnp.zeros((C, 0)), jnp.zeros((Ncap, 3, 0)), jnp.zeros((C, 6, 0)))
    U0 = X @ ind.Pmap
    JU0 = jnp.einsum("nkDa,Dq->nkqa", J.reshape(Ncap, K, cfg.D, 3), ind.Pmap)   # (Ncap, K, d, 3)
    return residual_rows_from_inputs(theta, spec, ind, cfg, batch, U0, JU0)


def pair_feature_inputs(model, ind, cfg, batch):
    """(U0, JU0) for a pair-density feature map (Pmap zero outside the n_pair pair
    rows) from the pair radials alone -- no many-body basis.  Equals
    (X @ Pmap, Pmap^T J) of the full path (tests/test_gp_hostcache.py)."""
    Ncap, K = batch.nbr.shape
    zi = jnp.broadcast_to(batch.node_z[:, None], (Ncap, K))
    Xp, dRp = model.pair_features_dense(batch.rij, zi, batch.node_z[batch.nbr], batch.nbr_mask)
    P = ind.Pmap[cfg.n_B:]                                            # (n_pair, d)
    return Xp @ P, jnp.einsum("nkpa,pq->nkqa", dRp.reshape(Ncap, K, -1, 3), P)


def residual_rows_from_inputs(theta, spec, ind, cfg, batch, U0, JU0):
    """Residual rows from the projected feature coordinates U0 = X @ Pmap
    (Ncap, d) and their edge Jacobian JU0 = Pmap^T J (Ncap, K, d, 3), pre-warp."""
    from .feature import dwarp, warp_u
    Ncap, K = batch.nbr.shape
    C, c = batch.y_E.shape[0], cfg.node_chunk
    M, d = ind.XM.shape                 # d = feature dim (D for isotropic, n_pair for density)
    if M == 0:
        return Rows(jnp.zeros((C, 0)), jnp.zeros((Ncap, 3, 0)), jnp.zeros((C, 6, 0)))
    rij, send, _, mask = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    s, Js = summary_edge_jacobian(rij, send, Ncap, cfg.r0, cfg.rcut, cfg.p, mask)
    U = warp_u(U0, ind.warp)                                          # (Ncap, d)
    # feature edge Jacobian JU (Ncap, K, d, 3) = dwarp(U0) * (Pmap^T J);
    # folds ind.scale, the projection and the warp into one object.
    JU = dwarp(U0, ind.warp)[:, None, :, None] * JU0
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
