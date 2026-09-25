"""Log marginal likelihood of the hybrid model at fixed hyperparameters, from
the streamed sufficient statistics.  At fixed theta the model is Bayesian
linear regression over Phi = [B | k_theta(B, B_M)] with prior precision
Lambda = blkdiag(Gamma^2 / sigma_c^2, K_MM) and noise precision
tau_i = (w_i / sigma_type(i))^2."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from .hypers import from_array, log_prior
from .kernels import K_MM
from .stats import sufficient_statistics


class Problem(NamedTuple):
    spec: object
    model: object
    ind: object
    cfg: object
    gamma: jnp.ndarray    # (len_basis,)
    prior: object


def prior_precision(theta, prob):
    L = prob.cfg.len_basis
    M = prob.ind.XM.shape[0]
    sc2 = jnp.exp(2.0 * theta.log_sigma_c)
    lin = prob.gamma ** 2 / sc2
    Kmm = K_MM(theta, prob.spec, prob.ind.XM, prob.ind.SM, prob.ind.ZM, prob.ind.embed)
    Lam = jnp.zeros((L + M, L + M)).at[jnp.arange(L), jnp.arange(L)].set(lin).at[L:, L:].set(Kmm)
    logdet = jnp.sum(jnp.log(lin)) + (2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(Kmm))))
                                      if M > 0 else 0.0)
    return Lam, logdet


def combine(theta, st):
    s2 = {t: jnp.exp(2.0 * getattr(theta, f"log_sigma_{t}")) for t in "EFV"}
    G = sum(getattr(st, f"G_{t}") / s2[t] for t in "EFV")
    b = sum(getattr(st, f"b_{t}") / s2[t] for t in "EFV")
    yy = sum(getattr(st, f"yy_{t}") / s2[t] for t in "EFV")
    logtau = sum(getattr(st, f"logw_{t}") - getattr(st, f"n_{t}") * jnp.log(s2[t]) for t in "EFV")
    N = sum(getattr(st, f"n_{t}") for t in "EFV")
    return G, b, yy, logtau, N


def _chol_S(theta, st, prob):
    G, b, yy, logtau, N = combine(theta, st)
    Lam, logdet_Lam = prior_precision(theta, prob)
    L = jnp.linalg.cholesky(G + Lam)
    return L, b, yy, logtau, N, logdet_Lam


def log_marginal_likelihood(theta, st, prob):
    L, b, yy, logtau, N, logdet_Lam = _chol_S(theta, st, prob)
    v = solve_triangular(L, b, lower=True)
    logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return (-0.5 * (yy - v @ v) - 0.5 * (logdet_S - logdet_Lam - logtau)
            - 0.5 * N * jnp.log(2.0 * jnp.pi))


def posterior(theta, st, prob):
    L, b, *_ = _chol_S(theta, st, prob)
    return cho_solve((L, True), b), L


def _stats_fn(mesh):
    """sufficient_statistics, or its shard_map'd multi-device counterpart when
    a mesh is given (Task 15)."""
    if mesh is None:
        return sufficient_statistics
    from .sharding import sufficient_statistics_sharded

    def f(theta, spec, model, ind, cfg, ds):
        return sufficient_statistics_sharded(theta, spec, model, ind, cfg, ds, mesh)
    return f


def make_lml(prob, ds, mesh=None, cache_linear=True):
    """cache_linear: stream the theta-independent linear Gram G_BB once and
    recompute only the M residual columns per call (10x for a large basis;
    stats.py 'theta-split statistics').  mesh disables it (sharded path)."""
    if cache_linear and mesh is None:
        from .stats import assemble_statistics, linear_statistics, residual_statistics
        lin = jax.jit(lambda: linear_statistics(prob.model, prob.cfg, ds))()
        jax.block_until_ready(lin)

        @jax.jit
        def f(a):
            theta = from_array(a)
            res = residual_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
            return log_marginal_likelihood(theta, assemble_statistics(lin, res), prob)
        return f
    stats = _stats_fn(mesh)

    @jax.jit
    def f(a):
        theta = from_array(a)
        st = stats(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        return log_marginal_likelihood(theta, st, prob)
    return f


def make_log_density(prob, ds, objective="lml", mesh=None, cache_linear=True):
    stats = _stats_fn(mesh)
    if objective == "lml":
        lik = make_lml(prob, ds, mesh, cache_linear)
    elif objective == "loo":
        from .loo import config_row_index, loo_objective
        rows = jnp.asarray(config_row_index(ds))

        @jax.jit
        def lik(a):
            theta = from_array(a)
            st = stats(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
            return loo_objective(theta, st, prob, ds, rows)
    else:
        raise ValueError(f"objective must be 'lml' or 'loo', got {objective!r}")

    def f(a):                       # plain closure: a jitted PjitFunction cannot carry attributes
        return lik(a) + log_prior(from_array(a), prob.prior)
    f.likelihood = lik
    return f
