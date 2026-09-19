"""Leave-one-CONFIGURATION-out pseudo-likelihood by block leverages.  Solves
the manuscript's 'LOO with forces' complication: a configuration has one
energy but many force rows, so the unit left out is the configuration's whole
row block, and (I - H_aa)^-1 on that block is what the closed form needs."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, solve_triangular

from .objective import _chol_S, combine
from .rows import batch_rows


def config_row_index(ds):
    nb, C = ds.y_E.shape
    Ncap = ds.node_z.shape[1]
    n_rows = C + 3 * Ncap + 6 * C
    R = 1 + 3 * int(ds.n_atoms.max()) + 6
    out = np.full((nb, C, R), n_rows, np.int32)          # n_rows = the appended zero row
    node_cfg = np.asarray(ds.node_cfg)
    for b in range(nb):
        for c in range(C):
            nodes = np.flatnonzero(node_cfg[b] == c)
            idx = np.concatenate([[c], C + (3 * nodes[:, None] + np.arange(3)).reshape(-1),
                                  C + 3 * Ncap + 6 * c + np.arange(6)])
            out[b, c, :len(idx)] = idx
    return out


def _batch_rows_weighted(theta, prob, batch):
    r = batch_rows(theta, prob.spec, prob.model, prob.ind, prob.cfg, batch)
    Dt = r.E.shape[-1]
    inv = {t: jnp.exp(-getattr(theta, f"log_sigma_{t}")) for t in "EFV"}
    sw = jnp.concatenate([batch.w_E * inv["E"], jnp.repeat(batch.w_F, 3) * inv["F"],
                          jnp.repeat(batch.w_V, 6) * inv["V"]])                       # sqrt(tau)
    Phi = jnp.concatenate([r.E, r.F.reshape(-1, Dt), r.V.reshape(-1, Dt)])
    y = jnp.concatenate([batch.y_E, batch.y_F.reshape(-1), batch.y_V.reshape(-1)])
    Phi_t = jnp.concatenate([Phi * sw[:, None], jnp.zeros((1, Dt))])
    y_t = jnp.concatenate([y * sw, jnp.zeros(1)])
    return Phi_t, y_t


def loo_objective(theta, st, prob, ds, cfg_rows):
    L, b, yy, logtau, N, _ = _chol_S(theta, st, prob)
    mu = cho_solve((L, True), b)

    def per_batch(theta, batch, rows_b):
        Phi_t, y_t = _batch_rows_weighted(theta, prob, batch)

        def per_config(idx):
            P = Phi_t[idx]                                            # (R, Dt)
            A = solve_triangular(L, P.T, lower=True)                  # (Dt, R)
            Q = jnp.eye(idx.shape[0]) - A.T @ A
            g = y_t[idx] - P @ mu
            Lq = jnp.linalg.cholesky(Q)
            u = solve_triangular(Lq, g, lower=True)
            return -0.5 * u @ u + jnp.sum(jnp.log(jnp.diag(Lq)))

        return jnp.sum(jax.vmap(per_config)(rows_b))

    f = jax.checkpoint(per_batch)
    total = jax.lax.scan(lambda c, x: (c + f(theta, x[0], x[1]), None), 0.0, (ds, cfg_rows))[0]
    return total + 0.5 * logtau - 0.5 * N * jnp.log(2.0 * jnp.pi)
