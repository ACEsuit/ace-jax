"""Log marginal likelihood of the hybrid model at fixed hyperparameters, from
the streamed sufficient statistics.  At fixed theta the model is Bayesian
linear regression over Phi = [B | k_theta(B, B_M)] with prior precision
Lambda = blkdiag(Gamma^2 / sigma_c^2, K_MM) and noise precision
tau_i = (w_i / sigma_type(i))^2."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from .embedding import normalize_rows
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


def combine(theta, st, log_ratios=None):
    """Assemble the noise-scaled Gram G, right-hand side b, y'y, log|noise
    precision| and observation count from the sufficient statistics.

    log_ratios=None is the single-type path (st fields have NO leading axis):
    each quantity's block is scaled by 1/sigma_q^2, exactly as before.

    log_ratios (n_types, 3) columns (E, F, V) is the per-config-type path (st
    fields carry a leading n_types axis).  The effective noise for quantity q,
    type t is sigma_{q,t} = sigma_q * exp(log_ratios[t, q]), so each per-(q,t)
    block is scaled by 1/sigma_{q,t}^2 = (1/sigma_q^2) * exp(-2 log_ratios[t, q]).
    The default type (row 0) is pinned to log_ratios[0]=0 upstream, so its scale
    is exactly 1/sigma_q^2 and the single-type limit is bit-identical."""
    s2 = {t: jnp.exp(2.0 * getattr(theta, f"log_sigma_{t}")) for t in "EFV"}
    if log_ratios is None:
        G = sum(getattr(st, f"G_{t}") / s2[t] for t in "EFV")
        b = sum(getattr(st, f"b_{t}") / s2[t] for t in "EFV")
        yy = sum(getattr(st, f"yy_{t}") / s2[t] for t in "EFV")
        logtau = sum(getattr(st, f"logw_{t}") - getattr(st, f"n_{t}") * jnp.log(s2[t]) for t in "EFV")
        N = sum(getattr(st, f"n_{t}") for t in "EFV")
        return G, b, yy, logtau, N
    n_types = log_ratios.shape[0]
    G = b = yy = logtau = N = 0.0
    for col, q in enumerate("EFV"):
        Gq, bq = getattr(st, f"G_{q}"), getattr(st, f"b_{q}")
        yyq, nq, logwq = getattr(st, f"yy_{q}"), getattr(st, f"n_{q}"), getattr(st, f"logw_{q}")
        logs2 = jnp.log(s2[q])
        for t in range(n_types):
            inv = jnp.exp(-2.0 * log_ratios[t, col]) / s2[q]     # 1 / sigma_{q,t}^2
            G = G + Gq[t] * inv
            b = b + bq[t] * inv
            yy = yy + yyq[t] * inv
            logtau = logtau + logwq[t] - nq[t] * (logs2 + 2.0 * log_ratios[t, col])
            N = N + nq[t]
    return G, b, yy, logtau, N


def _chol_S(theta, st, prob, log_ratios=None):
    G, b, yy, logtau, N = combine(theta, st, log_ratios)
    Lam, logdet_Lam = prior_precision(theta, prob)
    L = jnp.linalg.cholesky(G + Lam)
    return L, b, yy, logtau, N, logdet_Lam


def log_marginal_likelihood(theta, st, prob, log_ratios=None):
    L, b, yy, logtau, N, logdet_Lam = _chol_S(theta, st, prob, log_ratios)
    v = solve_triangular(L, b, lower=True)
    logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return (-0.5 * (yy - v @ v) - 0.5 * (logdet_S - logdet_Lam - logtau)
            - 0.5 * N * jnp.log(2.0 * jnp.pi))


def posterior(theta, st, prob, log_ratios=None):
    L, b, *_ = _chol_S(theta, st, prob, log_ratios)
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


def _sigma_type_decode(a, n_hypers, n_types):
    """Split the LML vector into (theta, log_ratios).  a = [hypers | free rows],
    the free rows being log_ratios[1:] flattened; row 0 is pinned to 0."""
    theta = from_array(a[:n_hypers])
    free = a[n_hypers:].reshape(n_types - 1, 3)
    log_ratios = jnp.concatenate([jnp.zeros((1, 3)), free], axis=0)
    return theta, log_ratios


def make_lml(prob, ds, mesh=None, cache_linear=True, n_types=1, n_free_ratios=0):
    """cache_linear: stream the theta-independent linear Gram G_BB once and
    recompute only the M residual columns per call (10x for a large basis;
    stats.py 'theta-split statistics').  mesh disables it (sharded path).

    n_free_ratios > 0 selects the per-config-type noise path: the LML vector is
    [10 hypers | (n_types-1)*3 free log-ratios], decoded by _sigma_type_decode
    (default type row pinned to 0).  The per-(quantity, type) Gram is streamed
    ONCE and cached exactly like the single-type linear Gram; only the noise
    scaling 1/sigma_{q,t}^2 depends on the ratios, so caching is preserved."""
    if n_free_ratios:
        from .hypers import Hypers
        from .stats import (assemble_statistics_typed, linear_statistics_typed,
                            residual_statistics_typed)
        n_hypers = len(Hypers._fields)
        lin = jax.jit(lambda: linear_statistics_typed(prob.model, prob.cfg, ds, n_types))()
        jax.block_until_ready(lin)

        @jax.jit
        def f(a):
            theta, log_ratios = _sigma_type_decode(a, n_hypers, n_types)
            res = residual_statistics_typed(theta, prob.spec, prob.model, prob.ind,
                                            prob.cfg, ds, n_types)
            st = assemble_statistics_typed(lin, res)
            return log_marginal_likelihood(theta, st, prob, log_ratios)
        return f
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


def make_lml_embed(prob, ds):
    """LML as a differentiable function (E, a) -> scalar for the outer VarOpt over
    the species embedding E. E enters the GP kernel via ind.embed = normalize_rows(E)
    (K_MM + the M residual columns); the E-independent linear Gram is cached once.
    Bypasses make_lml's jitted, E-frozen closure so jax.grad(., E) is available."""
    from .stats import assemble_statistics, linear_statistics, residual_statistics
    lin = jax.jit(lambda: linear_statistics(prob.model, prob.cfg, ds))()
    jax.block_until_ready(lin)

    def lml_E(E, a):
        theta = from_array(a)
        ind_E = prob.ind._replace(embed=normalize_rows(E))
        res = residual_statistics(theta, prob.spec, prob.model, ind_E, prob.cfg, ds)
        st = assemble_statistics(lin, res)
        return log_marginal_likelihood(theta, st, prob._replace(ind=ind_E))
    return lml_E
