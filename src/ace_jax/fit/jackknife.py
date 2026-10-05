"""Exact centred delete-one-cluster (PRESS / CR3) jackknife shape (math rev. 2).

Coordinates: S = D^-1 A D^-1 = L L^T (post.chol), D^-1 = diag(post.dinv).  For a cluster k with whitened
rows Psi_k (n_k, L) and residuals rho_k = y~_k - Psi_k c, the PRESS score is
g~_k = Psi_k^T (I - H_kk)^-1 rho_k with H_kk = Psi_k A^-1 Psi_k^T = W^T W, W = L^-1 (Psi_k D^-1)^T, so that
A^-1 g~_k = c - c_(-k) exactly (h fixed)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, solve_triangular


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


_MU_FLOOR = 1e-12     # eigenvalues of I - H_kk below this (leverage 1 within roundoff) are clamped to it


def _solve_sym(M, b):
    """M^-1 b for a symmetric positive definite M = I - H_kk (cond ~ 1/(1 - lambda_max)), and
    lambda_max(H_kk) = 1 - mu_min.  mu <= _MU_FLOOR (lambda_max = 1 within roundoff, where 1/mu would be
    ~1e16 garbage or of the wrong sign) is clamped to _MU_FLOOR; the returned leverage is unclamped, so
    the stage counts such clusters in n_lev_near1 / n_mu_clamped."""
    mu, U = np.linalg.eigh(M)
    return U @ ((U.T @ b) / np.maximum(mu, _MU_FLOOR)), float(1.0 - mu.min())


def press_scores(post, prob, ds, clusters, K, sig, mode="exact"):
    """G (L, K) with g~_k = Psi_k^T (I - H_kk)^-1 rho_k (rho_k = y~_k - Psi_k c), and lambda_max(H_kk).

    mode: "exact" (n_k x n_k solve of I - W^T W; push-through when n_k > L), "pushthrough" (always the
    L x L form: S_k = L (I - W W^T) L^T, so g~_k = L (I - W W^T)^-1 W rho_k / dinv), or "block"
    (block-diagonal approximation of I - H_kk over E rows, per-atom F triples and per-config V sextets).
    Only the stored Cholesky factor is used (S is never formed).  Clusters are processed batch by batch,
    so memory scales with one batch's rows; a cluster must not span batches (row_clusters guarantees it)."""
    from .rows import chunked_rows_fn
    if mode not in ("exact", "pushthrough", "block"):
        raise ValueError(f"press_scores: unknown mode {mode!r}")
    Lj = jnp.asarray(post.chol, jnp.float64)
    dinv = np.asarray(post.dinv, np.float64)
    c = np.asarray(post.mean, np.float64)
    L = len(c)
    rows = chunked_rows_fn(prob.model, prob.cfg)
    G, lev = np.zeros((L, K)), np.zeros(K)
    done = np.zeros(K, bool)
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        P, y, k, blk = _batch_rows(rows(bt), bt, clusters[i], sig)
        ids, starts = np.unique(k, return_index=True)
        if done[ids].any():
            raise ValueError(f"press_scores: cluster {int(ids[done[ids]][0])} spans batches")
        done[ids] = True
        for kk, a, b in zip(ids, starts, np.r_[starts[1:], len(k)]):
            Pk, bk = P[a:b], blk[a:b]
            rho = y[a:b] - Pk @ c
            W = np.asarray(solve_triangular(Lj, jnp.asarray((Pk * dinv[None, :]).T), lower=True))  # (L, n_k)
            if mode == "pushthrough" or (mode == "exact" and len(Pk) > L):
                z, lev[kk] = _solve_sym(np.eye(L) - W @ W.T, W @ rho)
                G[:, kk] = np.asarray(Lj @ jnp.asarray(z)) / dinv
            elif mode == "block":
                H = W.T @ W
                lev[kk] = float(np.linalg.eigvalsh(H).max())
                zr = np.empty_like(rho)
                for q in np.unique(bk):
                    m = bk == q
                    zr[m] = np.linalg.solve(np.eye(int(m.sum())) - H[np.ix_(m, m)], rho[m])
                G[:, kk] = Pk.T @ zr
            else:
                zr, lev[kk] = _solve_sym(np.eye(len(Pk)) - W.T @ W, rho)
                G[:, kk] = Pk.T @ zr
    return G, lev


def shape_factor(post, G, tau=1.0):
    """R with R R^T = Q~ Q~^T, Q~ = S^-1 D^-1 (G - g-bar).  R = Q~ when K <= L and tau = 1, else
    U_r Sigma_r of the thin SVD with r the smallest rank holding >= tau of sum sigma^2.  Centred scores
    with no spread (K = 1, or all zero) give a single zero column, (L, 1), for every tau."""
    Lc = jnp.asarray(post.chol, jnp.float64)
    G = jnp.asarray(G, jnp.float64)
    Qt = cho_solve((Lc, True), (G - G.mean(1, keepdims=True)) * jnp.asarray(post.dinv, jnp.float64)[:, None])
    if Qt.shape[1] <= Qt.shape[0] and tau >= 1.0:
        return Qt
    U, s, _ = jnp.linalg.svd(Qt, full_matrices=False)
    e = np.cumsum(np.asarray(s) ** 2)
    if e[-1] == 0:                                   # no spread (K = 1, or all-zero scores): one zero column
        return jnp.zeros((Qt.shape[0], 1), jnp.float64)
    r = int(np.searchsorted(e / e[-1], min(tau, 1.0) - 1e-12) + 1)
    return U[:, :r] * s[None, :r]


ATOM_CHUNK = 256            # atoms per device pass in atom_shape (bounds the (chunk, 3, max(L, r)) temporaries)


def atom_chunk(n_cols, chunk=None):
    """Atoms per pass: `chunk` if given, else ATOM_CHUNK capped so a (chunk, 3, n_cols) float64 array is <= 1 GiB."""
    if chunk is not None:
        return max(1, int(chunk))
    return max(1, min(ATOM_CHUNK, (1 << 30) // (3 * 8 * max(int(n_cols), 1))))


def atom_shape(R, dinv, Frows, chunk=None):
    """V (N, 3, 3): V_ab = (R^T u_a) . (R^T u_b), u_a = D^-1 phi_a^T, from force rows (N, 3, L).
    Evaluated over chunks of atoms (device arrays stay on device) and concatenated on the host."""
    R = jnp.asarray(R, jnp.float64)
    dinv = jnp.asarray(dinv, jnp.float64)
    N = Frows.shape[0]
    c = atom_chunk(max(R.shape), chunk)
    out = np.empty((N, 3, 3))
    for i in range(0, N, c):
        U = jnp.asarray(Frows[i:i + c], jnp.float64) * dinv[None, None, :]
        Pr = U @ R
        out[i:i + c] = np.asarray(jnp.einsum("nar,nbr->nab", Pr, Pr))
    return out


def truncate_shape_factor(R, tau):
    """R (L, r) -> U_k Sigma_k of its thin SVD, the same R R^T to rank k: the smallest k holding >= tau of
    sum sigma^2 (tau a float <= 1), or k = tau (an int).  tau = 1.0 returns R unchanged."""
    if not isinstance(tau, (int, np.integer)) and tau >= 1.0:
        return R
    U, s, _ = np.linalg.svd(np.asarray(R, np.float64), full_matrices=False)
    if isinstance(tau, (int, np.integer)):
        k = int(min(max(tau, 1), len(s)))
    else:
        e = np.cumsum(s ** 2)
        k = 1 if e[-1] == 0 else int(np.searchsorted(e / e[-1], tau - 1e-12) + 1)
    return U[:, :k] * s[None, :k]


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
