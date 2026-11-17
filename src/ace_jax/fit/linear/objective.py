"""The evidence and posterior of a Bayesian linear model from its streamed statistics.

The model: weighted rows of several quantities, quantity q with noise scale sigma_q, and a Gaussian prior
on the coefficients with precision Lambda.  With G = sum_q G_q / sigma_q^2, b = sum_q b_q / sigma_q^2 and
S = G + Lambda,

    log p(y) = -1/2 (y'y - b' S^-1 b) - 1/2 (log|S| - log|Lambda| - log|tau|) - N/2 log 2 pi,

log|tau| = sum_q (logw_q - n_q log sigma_q^2) the log-determinant of the noise precision and N the live
rows.  Two forms: from the Gram statistics by Cholesky (`combine`, `log_evidence`, `posterior_cholesky`),
and from the QR statistics with a diagonal prior by a QR of the stacked factors (`qr_system`,
`log_evidence_qr`, `posterior_qr`), at kappa(Phi) and not its square.  Statistics are sequences of
`stats.Gram` / `stats.Factor`, one per quantity; noise scales are their log sigma_q in the same order.
With k right-hand sides (b, c with a second axis) the evidence is the sum over them (shared noise and
prior) and the posterior mean has one column each.

`hoist` and `jit_lik` pass a likelihood's statistics into its jit as arguments."""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular


def combine(grams, log_sigmas):
    """The noise-scaled Gram G, right-hand side b, y'y, log|noise precision| and row count of the
    quantities `grams` at noise scales exp(log_sigmas)."""
    s2 = [jnp.exp(2.0 * ls) for ls in log_sigmas]
    G = sum(g.G / v for g, v in zip(grams, s2))
    b = sum(g.b / v for g, v in zip(grams, s2))
    yy = sum(g.yy / v for g, v in zip(grams, s2))
    logtau = sum(g.logw - g.n * jnp.log(v) for g, v in zip(grams, s2))
    N = sum(g.n for g in grams)
    return G, b, yy, logtau, N


def _quad(v):
    """v'v, summed over right-hand sides."""
    return v @ v if v.ndim == 1 else jnp.sum(v * v)


def _rhs(v):
    """Number of right-hand sides of a (L,) or (L, k) array."""
    return 1 if v.ndim == 1 else v.shape[1]


def log_evidence(L, b, yy, logtau, N, logdet_Lam):
    """The evidence from the lower Cholesky factor L of S = G + Lambda and `combine`'s b, yy, logtau, N."""
    v = solve_triangular(L, b, lower=True)
    logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    if b.ndim == 1:
        return (-0.5 * (yy - v @ v) - 0.5 * (logdet_S - logdet_Lam - logtau)
                - 0.5 * N * jnp.log(2.0 * jnp.pi))
    k = b.shape[1]
    return (-0.5 * (jnp.sum(yy) - _quad(v)) - 0.5 * k * (logdet_S - logdet_Lam - logtau)
            - 0.5 * k * N * jnp.log(2.0 * jnp.pi))


def posterior_cholesky(L, b):
    """(mean, L) from the lower Cholesky factor of S."""
    return cho_solve((L, True), b), L


# ---------------------------------------------------------------------------
# QR form, diagonal prior `lin` (L,).  G + Lambda = A^T A with
#     A = [ diag(sqrt(lin)) ; R_q / sigma_q ... ]
# and b = A^T [0 ; c_q / sigma_q], so a QR of A, A = Q R, gives the Cholesky factor R^T and
# v = L^-1 b = Q^T [0 ; c_q / sigma_q] at kappa(A) = kappa(Phi), not kappa(Phi)^2.  An evidence fit that
# nearly interpolates its data (noise at its floor, prior near flat) drives kappa(G + Lambda) past 1/eps:
# there the Cholesky evidence is NaN, or finite but wrong by percent, and this form is not.  The prior rows
# go first: Householder QR is accurate for rows of widely varying norm when the large ones lead (Powell &
# Reid), and a pinned column's precision can be 1e16.

def _stacked(lin, factors, log_sigmas):
    s = [jnp.exp(ls) for ls in log_sigmas]
    A = jnp.concatenate([jnp.diag(jnp.sqrt(lin))] + [f.R / v for f, v in zip(factors, s)], 0)
    c0 = factors[0].c
    y = jnp.concatenate([jnp.zeros_like(lin) if c0.ndim == 1 else jnp.zeros((lin.shape[0], c0.shape[1]), lin.dtype)]
                        + [f.c / v for f, v in zip(factors, s)])
    return A, y


def qr_system(lin, factors, log_sigmas):
    """(R_S, v): the upper factor of S (positive diagonal, R_S^T R_S = G + diag(lin)) and v = R_S^-T b."""
    A, y = _stacked(lin, factors, log_sigmas)
    Q, R = jnp.linalg.qr(A, mode="reduced")
    sg = jnp.where(jnp.diag(R) < 0, -1.0, 1.0)              # Q -> Q diag(sg): a positive diagonal
    return R * sg[:, None], (Q.T @ y) * (sg if y.ndim == 1 else sg[:, None])


def qr_residual_system(lin, factors, log_sigmas):
    """(R_S, residual^2) with the residual from the factors alone: the reduced system's QR takes its
    target as an extra column (last diagonal = its residual norm), plus the carried rho_q^2 / sigma_q^2.
    No y'y - v'v, which cancels when the factors came from float32 rows.  One right-hand side."""
    s = [jnp.exp(ls) for ls in log_sigmas]
    A, y = _stacked(lin, factors, log_sigmas)
    Rt = jnp.linalg.qr(jnp.concatenate([A, y[:, None]], 1), mode="r")
    L = lin.shape[0]
    resid = Rt[L, L] ** 2 + sum(f.rho ** 2 / v ** 2 for f, v in zip(factors, s))
    return Rt[:L, :L], resid


def log_evidence_qr(lin, factors, log_sigmas):
    """The evidence of the diagonal-prior model from its QR statistics.  Factors that carry rho (float32
    rows) take the residual from the factors (`qr_residual_system`)."""
    s2 = [jnp.exp(2.0 * ls) for ls in log_sigmas]
    if factors[0].rho is not None:
        R, resid = qr_residual_system(lin, factors, log_sigmas)
        logtau = sum(f.logw - f.n * jnp.log(v) for f, v in zip(factors, s2))
        N = sum(f.n for f in factors)
        logdet_S = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(R))))
        return -0.5 * resid - 0.5 * (logdet_S - jnp.sum(jnp.log(lin)) - logtau) - 0.5 * N * jnp.log(2.0 * jnp.pi)
    R, v = qr_system(lin, factors, log_sigmas)
    yy = sum(f.yy / v2 for f, v2 in zip(factors, s2))
    logtau = sum(f.logw - f.n * jnp.log(v2) for f, v2 in zip(factors, s2))
    N = sum(f.n for f in factors)
    logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diag(R)))
    if v.ndim == 1:
        return (-0.5 * (yy - v @ v) - 0.5 * (logdet_S - jnp.sum(jnp.log(lin)) - logtau)
                - 0.5 * N * jnp.log(2.0 * jnp.pi))
    k = _rhs(v)
    return (-0.5 * (jnp.sum(yy) - _quad(v)) - 0.5 * k * (logdet_S - jnp.sum(jnp.log(lin)) - logtau)
            - 0.5 * k * N * jnp.log(2.0 * jnp.pi))


def posterior_qr(lin, factors, log_sigmas):
    """(mean, L) from the same QR as `log_evidence_qr`: L = R_S^T, lower with a positive diagonal,
    L L^T = G + diag(lin)."""
    R, v = qr_system(lin, factors, log_sigmas)
    return solve_triangular(R, v, lower=False), R.T


def hoist(pure, consts):
    """pure(a, consts) as a function of a, carrying (pure, consts) so that a jit tracing it (`jit_lik`)
    takes consts as an argument.  A jit that closes over them instead embeds them in the executable as
    HLO constants: a second copy of the QR statistics (43 GB at L = 42,324), and past 2^32 elements a
    lowering error (R_F at L = 71,440: "Expected 5104245136 elements in LiteralProto repeated field").
    A plain closure: a jitted PjitFunction cannot carry attributes."""
    def f(a):
        return pure(a, consts)
    f.pure, f.consts = pure, consts
    return f


def jit_lik(lik, body):
    """jax.jit(lambda *x: body(lik, *x)), with a `hoist`ed lik's statistics passed in rather than captured
    (any other lik is captured as before).  The ladder rungs past MAP still close over lik."""
    if not hasattr(lik, "pure"):
        return jax.jit(lambda *x: body(lik, *x))
    pure = lik.pure
    jf = jax.jit(lambda c, *x: body(lambda a: pure(a, c), *x))
    return lambda *x: jf(lik.consts, *x)
