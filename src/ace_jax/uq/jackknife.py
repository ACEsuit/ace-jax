"""Exact centred delete-one-cluster (PRESS / CR3) jackknife shape (math rev. 2).

Coordinates: S = D^-1 A D^-1 = L L^T (post.chol), D^-1 = diag(post.dinv).  For a cluster k with whitened
rows Psi_k (n_k, L) and residuals rho_k = y~_k - Psi_k c, the PRESS score is
g~_k = Psi_k^T (I - H_kk)^-1 rho_k with H_kk = Psi_k A^-1 Psi_k^T = W^T W, W = L^-1 (Psi_k D^-1)^T, so that
A^-1 g~_k = c - c_(-k) exactly (h fixed)."""
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, solve_triangular


_MU_FLOOR = 1e-12     # eigenvalues of I - H_kk below this (leverage 1 within roundoff) are clamped to it


def _solve_sym(M, b):
    """M^-1 b for a symmetric positive definite M = I - H_kk (cond ~ 1/(1 - lambda_max)), and
    lambda_max(H_kk) = 1 - mu_min.  mu <= _MU_FLOOR (lambda_max = 1 within roundoff, where 1/mu would be
    ~1e16 garbage or of the wrong sign) is clamped to _MU_FLOOR; the returned leverage is unclamped, so
    the stage counts such clusters in n_lev_near1 / n_mu_clamped."""
    mu, U = np.linalg.eigh(M)
    return U @ ((U.T @ b) / np.maximum(mu, _MU_FLOOR)), float(1.0 - mu.min())


def shape_factor(post, G, tau=1.0):
    """R with R R^T = Q~ Q~^T, Q~ = S^-1 D^-1 (G - g-bar).  R = Q~ when K <= L and tau = 1, else
    U_r Sigma_r of the thin SVD with r the smallest rank holding >= tau of sum sigma^2.  Centred scores
    with no spread (K = 1, or all zero) give a single zero column, (L, 1), for every tau."""
    Lc = jnp.asarray(post.chol, jnp.float64)
    G = jnp.asarray(G, jnp.float64)
    Qt = cho_solve((Lc, True), post.prior_root.dual(G - G.mean(1, keepdims=True)))
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
    Evaluated over chunks of atoms (device arrays stay on device) and concatenated on the host.
    dinv: the diagonal D^-1 (L,), or a prior_root.PriorRoot (ard-gp: u_a = R0^-T phi_a^T)."""
    from ..fit.linear.prior_root import PriorRoot
    R = jnp.asarray(R, jnp.float64)
    root = dinv if isinstance(dinv, PriorRoot) else None
    if root is None:
        dinv = jnp.asarray(dinv, jnp.float64)
    N = Frows.shape[0]
    c = atom_chunk(max(R.shape), chunk)
    out = np.empty((N, 3, 3))
    for i in range(0, N, c):
        F = jnp.asarray(Frows[i:i + c], jnp.float64)
        U = root.rows(F) if root is not None else F * dinv[None, None, :]
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


def press_scores_rows(post, batches, K, mode="exact"):
    """G (L, K) with g~_k = Psi_k^T (I - H_kk)^-1 rho_k (rho_k = y~_k - Psi_k c), and lambda_max(H_kk), for
    the K clusters of any rows.  batches: an iterable of (Psi (n, L), y~ (n,), cluster (n,), block (n,)), the
    whitened rows and targets of the live rows of one batch (weights and noise scales applied), sorted by
    cluster id; block ids group rows for mode "block" (a potential fit: an energy row, an atom's 3 force
    rows, a configuration's 6 virial rows).  A cluster must not span batches.

    mode: "exact" (n_k x n_k solve of I - W^T W; push-through when n_k > L), "pushthrough" (always the
    L x L form: S_k = L (I - W W^T) L^T, so g~_k = L (I - W W^T)^-1 W rho_k / dinv), or "block"
    (block-diagonal approximation of I - H_kk over the row blocks).  Only the stored Cholesky factor is used
    (S is never formed); memory scales with one batch's rows.  The prior root R0 (post.prior_root) stands in
    for D: W = L^-1 (Psi_k R0^-1)^T and, push-through, g~_k = R0^T L z (M = 0: the diagonal D)."""
    if mode not in ("exact", "pushthrough", "block"):
        raise ValueError(f"press_scores: unknown mode {mode!r}")
    Lj = jnp.asarray(post.chol, jnp.float64)
    root = post.prior_root
    c = np.asarray(post.mean, np.float64)
    L = len(c)
    G, lev = np.zeros((L, K)), np.zeros(K)
    done = np.zeros(K, bool)
    for P, y, k, blk in batches:
        ids, starts = np.unique(k, return_index=True)
        if done[ids].any():
            raise ValueError(f"press_scores: cluster {int(ids[done[ids]][0])} spans batches")
        done[ids] = True
        for kk, a, b in zip(ids, starts, np.r_[starts[1:], len(k)]):
            Pk, bk = P[a:b], blk[a:b]
            rho = y[a:b] - Pk @ c
            W = np.asarray(solve_triangular(Lj, jnp.asarray(root.rows(Pk)).T, lower=True))         # (L, n_k)
            if mode == "pushthrough" or (mode == "exact" and len(Pk) > L):
                z, lev[kk] = _solve_sym(np.eye(L) - W @ W.T, W @ rho)
                G[:, kk] = np.asarray(root.lift(Lj @ jnp.asarray(z)))
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
