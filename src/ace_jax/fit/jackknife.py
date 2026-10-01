"""Exact centred delete-one-cluster (PRESS / CR3) jackknife shape (math rev. 2).

Coordinates: S = D^-1 A D^-1 = L L^T (post.chol), D^-1 = diag(post.dinv).  For a cluster k with whitened
rows Psi_k (n_k, L) and residuals rho_k = y~_k - Psi_k c, the PRESS score is
g~_k = Psi_k^T (I - H_kk)^-1 rho_k with H_kk = Psi_k A^-1 Psi_k^T = W^T W, W = L^-1 (Psi_k D^-1)^T, so that
A^-1 g~_k = c - c_(-k) exactly (h fixed)."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, solve_triangular


def _batch_rows(prob, bt, ids, sig):
    """Whitened rows Psi (n, L), targets y~ (n,), cluster id (n,), row-block id (n,) of one batch's live rows.
    Row blocks (for mode="block"): an energy row, an atom's 3 force rows, a config's 6 virial rows."""
    from .rows import linear_rows
    r = linear_rows(prob.model, prob.cfg, bt)[0]
    C, L = r.E.shape
    N = r.F.shape[0]
    P = np.concatenate([np.asarray(r.E), np.asarray(r.F).reshape(-1, L), np.asarray(r.V).reshape(-1, L)])
    y = np.concatenate([np.asarray(bt.y_E), np.asarray(bt.y_F).reshape(-1), np.asarray(bt.y_V).reshape(-1)])
    w = np.concatenate([np.asarray(bt.w_E) / sig[0], np.repeat(np.asarray(bt.w_F), 3) / sig[1],
                        np.repeat(np.asarray(bt.w_V), 6) / sig[2]])
    k = np.concatenate([ids["E"], np.repeat(ids["F"], 3), np.repeat(ids["V"], 6)])
    blk = np.concatenate([np.arange(C), C + np.repeat(np.arange(N), 3), C + N + np.repeat(np.arange(C), 6)])
    live = (k >= 0) & (w > 0)
    return P[live] * w[live, None], y[live] * w[live], k[live], blk[live]


def _collect(prob, ds, clusters, sig):
    """All live whitened rows of ds, sorted (stably) by cluster id."""
    P, y, k, b = [], [], [], []
    off = 0
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a, i=i: a[i], ds)
        p, q, kk, bb = _batch_rows(prob, bt, clusters[i], sig)
        P.append(p); y.append(q); k.append(kk); b.append(bb + off)
        off += int(bb.max()) + 1 if len(bb) else 0
    P, y, k, b = map(np.concatenate, (P, y, k, b))
    o = np.argsort(k, kind="stable")
    return P[o], y[o], k[o], b[o]


def press_scores(post, prob, ds, clusters, K, sig, mode="exact"):
    """G (L, K) with g~_k = Psi_k^T (I - H_kk)^-1 rho_k (rho_k = y~_k - Psi_k c), and lambda_max(H_kk).

    mode: "exact" (n_k x n_k solve; push-through when n_k > L), "pushthrough" (always the L x L
    downdate S_k = S - Ps^T Ps, g~_k = (S S_k^-1 Ps^T rho) / dinv), or "block" (block-diagonal
    approximation of I - H_kk over E rows, per-atom F triples and per-config V sextets)."""
    if mode not in ("exact", "pushthrough", "block"):
        raise ValueError(f"press_scores: unknown mode {mode!r}")
    Lc = np.asarray(post.chol, np.float64)
    Lj = jnp.asarray(Lc)
    dinv = np.asarray(post.dinv, np.float64)
    c = np.asarray(post.mean, np.float64)
    L = len(c)
    S = Lc @ Lc.T
    P, y, k, blk = _collect(prob, ds, clusters, sig)
    starts = np.searchsorted(k, np.arange(K + 1))
    G, lev = np.zeros((L, K)), np.zeros(K)
    for kk in range(K):
        sl = slice(starts[kk], starts[kk + 1])
        Pk, bk = P[sl], blk[sl]
        if len(Pk) == 0:
            continue
        rho = y[sl] - Pk @ c
        Ps = Pk * dinv[None, :]
        W = np.asarray(solve_triangular(Lj, jnp.asarray(Ps.T), lower=True))       # (L, n_k)
        small = len(Pk) <= L
        lev[kk] = float(np.linalg.eigvalsh(W.T @ W if small else W @ W.T).max())  # same nonzero spectrum
        if mode == "pushthrough" or (mode == "exact" and not small):
            x = np.linalg.solve(S - Ps.T @ Ps, Ps.T @ rho)
            G[:, kk] = (S @ x) / dinv
        elif mode == "block":
            H = W.T @ W
            z = np.empty_like(rho)
            for b in np.unique(bk):
                m = bk == b
                z[m] = np.linalg.solve(np.eye(int(m.sum())) - H[np.ix_(m, m)], rho[m])
            G[:, kk] = Pk.T @ z
        else:
            G[:, kk] = Pk.T @ np.linalg.solve(np.eye(len(Pk)) - W.T @ W, rho)
    return G, lev


def shape_factor(post, G, tau=1.0):
    """R with R R^T = Q~ Q~^T, Q~ = S^-1 D^-1 (G - g-bar).  R = Q~ when K <= L and tau = 1, else
    U_r Sigma_r of the thin SVD with r the smallest rank holding >= tau of sum sigma^2."""
    Lc = jnp.asarray(post.chol, jnp.float64)
    G = jnp.asarray(G, jnp.float64)
    Qt = cho_solve((Lc, True), (G - G.mean(1, keepdims=True)) * jnp.asarray(post.dinv, jnp.float64)[:, None])
    if Qt.shape[1] <= Qt.shape[0] and tau >= 1.0:
        return Qt
    U, s, _ = jnp.linalg.svd(Qt, full_matrices=False)
    e = np.cumsum(np.asarray(s) ** 2)
    r = int(np.searchsorted(e / e[-1], min(tau, 1.0) - 1e-12) + 1)
    return U[:, :r] * s[None, :r]


def atom_shape(R, dinv, Frows):
    """V (N, 3, 3): V_ab = (R^T u_a) . (R^T u_b), u_a = D^-1 phi_a^T, from force rows (N, 3, L)."""
    U = jnp.asarray(Frows, jnp.float64) * jnp.asarray(dinv, jnp.float64)[None, None, :]
    Pr = U @ jnp.asarray(R, jnp.float64)
    return np.asarray(jnp.einsum("nar,nbr->nab", Pr, Pr))
