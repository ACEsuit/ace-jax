"""Leave-one-block-out of a Bayesian linear model in closed form, by block leverages.

A block is a set of rows left out together (a configuration's energy, force and virial rows; the
atoms or bonds of one structure).  With whitened rows P (m, k) of the block, the posterior
precision A = L L^T of all the rows and the block's residual g = y - P mean, the residual of the
fit without the block is Q^-1 g and its predictive covariance Q^-1, Q = I - P A^-1 P^T (the block
of I - H, H the hat matrix): the residuals of the delete-one-cluster (PRESS / CR3) jackknife of
`uq.jackknife`.  No model, no E/F/V: `fit.gp.loo` is the potential fit's objective on `block_loo`."""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from .objective import posterior_qr


def block_loo(L, P, g):
    """(u, Lq) of one block: Lq Lq^T = Q = I - P A^-1 P^T and u = Lq^-1 g (g: (m,) or (m, q)), so
    u^T u = g^T Q^-1 g, the left-out residual is Lq^-T u and log det Q = 2 sum log diag Lq.
    Rows of zeros (padding) change nothing.  A block of leverage 1 within roundoff gives NaN."""
    W = solve_triangular(L, P.T, lower=True)                          # (k, m)
    Lq = jnp.linalg.cholesky(jnp.eye(P.shape[0], dtype=P.dtype) - W.T @ W)
    return solve_triangular(Lq, g, lower=True), Lq


@jax.jit
def loo_terms(lin, factor, P, Y):
    """The leave-one-block-out sums of the diagonal-prior model at prior precisions `lin` and unit
    noise, for every right-hand side of `factor` (a `stats.Factor` of all the rows, c (k, q)).
    P (K, m, k), Y (K, m, q): the rows and targets of the K blocks, padded with zero rows; any
    rows with the block's Gram and moments do, e.g. its own triangular factor and Q^T y (a part
    of y outside the block's column space has no leverage: add its square to quad and press).
    Returns (quad (q,): sum g^T Q^-1 g; half the sum of log det Q; press (q,): the sum of squared
    left-out residuals; mean (k, q)).  The log predictive density of the blocks left out, at
    noise sigma, is -quad / (2 sigma^2) + half log det - (n / 2) log(2 pi sigma^2); it is largest
    at sigma^2 = quad / n."""
    mean, L = posterior_qr(lin, [factor], [0.0])

    def one(x):
        u, Lq = block_loo(L, x[0], x[1] - x[0] @ mean)
        e = solve_triangular(Lq.T, u, lower=False)
        return (u * u).sum(0), jnp.log(jnp.diag(Lq)).sum(), (e * e).sum(0)

    quad, hld, press = jax.lax.map(one, (P, Y), batch_size=16)         # batched: one block at a time is slow on a GPU
    return quad.sum(0), hld.sum(), press.sum(0), mean
