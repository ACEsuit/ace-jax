"""The jackknife of the E/F/V fit: PRESS scores of its row clusters and the committee form of the force
shape, on `ace_jax.uq.jackknife` (whose names are re-exported here)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..uq.jackknife import (ATOM_CHUNK, _MU_FLOOR, _solve_sym, atom_chunk, atom_shape, press_scores_rows,  # noqa: F401
                            shape_factor, truncate_shape_factor)


def _batch_rows(r, bt, ids, sig):
    """Whitened rows Psi (n, L), targets y~ (n,), cluster id (n,), row-block id (n,) of one batch's live rows,
    sorted (stably) by cluster id.  r: the batch's linear rows (.E, .F, .V).
    Row blocks (for mode="block"): an energy row, an atom's 3 force rows, a config's 6 virial rows."""
    C, L = r.E.shape
    N = r.F.shape[0]
    P = np.concatenate([np.asarray(r.E), np.asarray(r.F).reshape(-1, L), np.asarray(r.V).reshape(-1, L)])
    y = np.concatenate([np.asarray(bt.y_E), np.asarray(bt.y_F).reshape(-1), np.asarray(bt.y_V).reshape(-1)])
    w = np.concatenate([np.asarray(bt.w_E) / sig[0], np.repeat(np.asarray(bt.w_F), 3) / sig[1],
                        np.repeat(np.asarray(bt.w_V), 6) / sig[2]])
    k = np.concatenate([np.asarray(ids["E"]), np.repeat(np.asarray(ids["F"]), 3), np.repeat(np.asarray(ids["V"]), 6)])
    blk = np.concatenate([np.arange(C), C + np.repeat(np.arange(N), 3), C + N + np.repeat(np.arange(C), 6)])
    live = np.flatnonzero((k >= 0) & (w > 0))
    live = live[np.argsort(k[live], kind="stable")]
    return P[live] * w[live, None], y[live] * w[live], k[live], blk[live]


def press_scores(post, prob, ds, clusters, K, sig, mode="exact", rows_fn=None):
    """`uq.jackknife.press_scores_rows` of the E/F/V rows of ds, whitened at the noise scales sig (E, F, V),
    with clusters[i] the cluster ids of batch i's rows (`uq.clusters.row_clusters`):
    G (L, K) with g~_k = Psi_k^T (I - H_kk)^-1 rho_k (rho_k = y~_k - Psi_k c), and lambda_max(H_kk).

    mode: "exact" (n_k x n_k solve of I - W^T W; push-through when n_k > L), "pushthrough" (always the
    L x L form: S_k = L (I - W W^T) L^T, so g~_k = L (I - W W^T)^-1 W rho_k / dinv), or "block"
    (block-diagonal approximation of I - H_kk over E rows, per-atom F triples and per-config V sextets).
    Only the stored Cholesky factor is used (S is never formed).  Clusters are processed batch by batch,
    so memory scales with one batch's rows; a cluster must not span batches (row_clusters guarantees it).
    rows_fn (ard.rows_fn_for): the joint rows [B | k(B, B_M)] of an ard-gp posterior; default the linear
    rows.  The prior root R0 (post.prior_root) stands in for D: W = L^-1 (Psi_k R0^-1)^T and, push-through,
    g~_k = R0^T L z (M = 0: the diagonal D, evaluated as before)."""
    from .rows import chunked_rows_fn
    rows = chunked_rows_fn(prob.model, prob.cfg) if rows_fn is None else rows_fn

    def batches():
        for i in range(ds.n_batches):
            bt = jax.tree.map(lambda a, i=i: a[i], ds)
            yield _batch_rows(rows(bt), bt, clusters[i], sig)
    return press_scores_rows(post, batches(), K, mode)


def committee_coefs(R, dinv, cfg):
    """C = D^-1 R split into the compact per-species blocks of the design layout (`rows._place`):
    (CB (NZ, n_B, r), CP (NZ, n_pair, r)).  Rows of R past the readout (joint-E0 columns) are dropped:
    they are zero on every force row."""
    NZ, nB, nP = cfg.NZ, cfg.n_B, cfg.n_pair
    L0 = (nB + nP) * NZ
    C = jnp.asarray(dinv, jnp.float64)[:L0, None] * jnp.asarray(R, jnp.float64)[:L0]
    return C[:NZ * nB].reshape(NZ, nB, -1), C[NZ * nB:].reshape(NZ, nP, -1)


COMMITTEE_NODE_CHUNK = 64     # centre nodes per pass of committee_forces (bounds (nc, n_B, n_A) and (nc, n_B, r))


def _committee_body(model, batch, CB, CP, nc):
    Ncap, K = batch.nbr.shape
    r = CB.shape[-1]
    n_chunks = -(-Ncap // nc)
    pad = n_chunks * nc - Ncap
    Np = Ncap + pad
    rij = jnp.concatenate([batch.rij, jnp.broadcast_to(batch.rij[-1:], (pad, K, 3))])
    nbr = jnp.concatenate([batch.nbr, jnp.zeros((pad, K), batch.nbr.dtype)])
    msk = jnp.concatenate([batch.nbr_mask, jnp.zeros((pad, K), bool)])
    node_z = jnp.concatenate([batch.node_z, jnp.zeros(pad, batch.node_z.dtype)])
    sl = lambda a, s0: jax.lax.dynamic_slice_in_dim(a, s0, nc)

    def body(i, P):
        s0 = i * nc
        r_c, nb_c, m_c, z_c = sl(rij, s0), sl(nbr, s0), sl(msk, s0), sl(node_z, s0)
        T = model.committee_edge_grad_dense(r_c, jnp.broadcast_to(z_c[:, None], (nc, K)), node_z[nb_c], m_c,
                                            z_c, CB, CP)                              # (nc, K, r, 3)
        P = P - jax.ops.segment_sum(T.reshape(nc * K, r, 3), nb_c.reshape(-1), num_segments=Np)   # receivers
        return jax.lax.dynamic_update_slice_in_dim(P, sl(P, s0) + T.sum(1), s0, 0)               # senders

    P = jax.lax.fori_loop(0, n_chunks, body, jnp.zeros((Np, r, 3)))
    return jnp.swapaxes(P[:Ncap], 1, 2)


_committee_jit = eqx.filter_jit(_committee_body)


def committee_forces(model, cfg, batch, CB, CP, node_chunk=None):
    """P (Ncap, 3, r): the forces of the r-output linear model (CB, CP) on every node of one batch, i.e.
    phi_a^T C per atom without the design rows phi (Ncap, 3, L) or the edge Jacobian (E, D, 3)."""
    nc = int(min(node_chunk or COMMITTEE_NODE_CHUNK, batch.nbr.shape[0]))
    return _committee_jit(model, batch, jnp.asarray(CB), jnp.asarray(CP), nc)


def committee_shape(model, cfg, batch, R, dinv, node_chunk=None):
    """The shape V (Ncap, 3, 3) = P P^T of `atom_shape(R, dinv, rows.F)` from `committee_forces` with
    C = D^-1 R: equal to roundoff (tests/test_shape_committee.py), with memory O(Ncap r) not O(Ncap L)."""
    CB, CP = committee_coefs(R, dinv, cfg)
    P = committee_forces(model, cfg, batch, CB, CP, node_chunk)
    return np.asarray(jnp.einsum("nar,nbr->nab", P, P))
