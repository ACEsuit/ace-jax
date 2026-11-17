"""Log marginal likelihood of the hybrid model at fixed hyperparameters, from
the streamed sufficient statistics.  At fixed theta the model is Bayesian
linear regression over Phi = [B | k_theta(B, B_M)] with prior precision
Lambda = blkdiag(Gamma^2 / sigma_c^2, K_MM) and noise precision
tau_i = (w_i / sigma_type(i))^2."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from .hypers import from_array, log_prior
from .gp.kernels import K_MM
from .linear import objective as _lin
from .linear.objective import hoist, jit_lik  # noqa: F401
from .rows import QUANTITIES
from .linear.stats import Factor, Gram
from .stats import sufficient_statistics


class Problem(NamedTuple):
    spec: object
    model: object
    ind: object
    cfg: object
    gamma: jnp.ndarray    # (len_basis,), or wider when extra linear columns are appended
    prior: object
    e0_prec: object = None  # (NZ,) fixed prior precision of the joint-E0 columns (cfg.e0_cols)
    lml_solver: str = "cholesky"  # "qr": the linear arm's LML and posterior from QRStats (make_lml)
    row_dtype: str = "float64"    # "float32": QR statistics from float32 rows (stats.linear_qr_statistics)


def linear_prior_diag(theta, prob):
    """Prior precision of the linear columns: gamma^2 / sigma_c^2 for the readout, then
    the fixed e0_prec for the joint-E0 columns (independent of sigma_c)."""
    lin = prob.gamma ** 2 / jnp.exp(2.0 * theta.log_sigma_c)
    e0p = getattr(prob, "e0_prec", None)
    return lin if e0p is None else jnp.concatenate([lin, jnp.asarray(e0p, lin.dtype)])


def prior_precision(theta, prob):
    lin = linear_prior_diag(theta, prob)
    L = lin.shape[0]
    M = prob.ind.XM.shape[0]
    Kmm = K_MM(theta, prob.spec, prob.ind.XM, prob.ind.SM, prob.ind.ZM, prob.ind.embed)
    Lam = jnp.zeros((L + M, L + M)).at[jnp.arange(L), jnp.arange(L)].set(lin).at[L:, L:].set(Kmm)
    logdet = jnp.sum(jnp.log(lin)) + (2.0 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(Kmm))))
                                      if M > 0 else 0.0)
    return Lam, logdet


def log_sigmas(theta):
    """The noise scales of the E, F and V rows in `rows.QUANTITIES` order: theta's log_sigma_E, _F, _V."""
    return [getattr(theta, f"log_sigma_{q}") for q in QUANTITIES]


def _grams(st):
    """st's quantities as `linear.stats.Gram`s, read by field name (Stats, QRStats, or any object with them)."""
    return tuple(Gram(getattr(st, f"G_{q}"), getattr(st, f"b_{q}"), getattr(st, f"yy_{q}"), getattr(st, f"n_{q}"),
                      getattr(st, f"logw_{q}")) for q in QUANTITIES)


def _factors(qs):
    """qs's quantities as `linear.stats.Factor`s, read by field name."""
    return tuple(Factor(getattr(qs, f"R_{q}"), getattr(qs, f"c_{q}"), getattr(qs, f"yy_{q}"), getattr(qs, f"n_{q}"),
                        getattr(qs, f"logw_{q}"), getattr(qs, f"rho_{q}", None)) for q in QUANTITIES)


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
    if log_ratios is None:
        return _lin.combine(_grams(st), log_sigmas(theta))
    s2 = {t: jnp.exp(2.0 * getattr(theta, f"log_sigma_{t}")) for t in "EFV"}
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
    return _lin.log_evidence(L, b, yy, logtau, N, logdet_Lam)


def posterior(theta, st, prob, log_ratios=None):
    L, b, *_ = _chol_S(theta, st, prob, log_ratios)
    return cho_solve((L, True), b), L


# ---------------------------------------------------------------------------
# QR form (linear arm, M = 0): `linear.objective`'s, with the prior precision linear_prior_diag and the
# noise scales of theta.  qs: QRStats.

def _qr_S(theta, qs, prob):
    lin = linear_prior_diag(theta, prob)
    return (*_lin.qr_system(lin, _factors(qs), log_sigmas(theta)), lin)


def _qr_resid(theta, qs, prob):
    """(R_S, residual^2, lin) with the residual from the factors alone (`linear.objective.qr_residual_system`)."""
    lin = linear_prior_diag(theta, prob)
    return (*_lin.qr_residual_system(lin, _factors(qs), log_sigmas(theta)), lin)


def log_marginal_likelihood_qr(theta, qs, prob):
    """log_marginal_likelihood of the linear arm (M = 0) from QRStats, by QR (`linear.objective.log_evidence_qr`).
    With QRStats.rho_q (float32 rows) the residual comes from the factors."""
    return _lin.log_evidence_qr(linear_prior_diag(theta, prob), _factors(qs), log_sigmas(theta))


def posterior_from_qr(theta, qs, prob):
    """posterior's (mu, L) from the same QR as log_marginal_likelihood_qr: L = R^T, lower
    with a positive diagonal, L L^T = G + Lambda."""
    return _lin.posterior_qr(linear_prior_diag(theta, prob), _factors(qs), log_sigmas(theta))


def uses_qr(prob):
    """Whether prob's LML and posterior take the QR form: lml_solver 'qr' on the linear arm."""
    return getattr(prob, "lml_solver", "cholesky") in ("qr", "svd") and prob.ind.XM.shape[0] == 0


def _stats_fn(mesh):
    """sufficient_statistics, or its shard_map'd multi-device counterpart when
    a mesh is given (Task 15)."""
    if mesh is None:
        return sufficient_statistics
    from .gp.sharding import sufficient_statistics_sharded

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


QR_EVAL = "auto"    # the QR LML in make_lml: "host" (lml_host.HostQREvidence), "dense" (a jitted QR of the stacked
                    # matrix, autodiff through it) or "auto": host when the statistics are on the CPU


def _host_qr_eval(qs):
    if QR_EVAL not in ("auto", "host", "dense"):
        raise ValueError(f"QR_EVAL must be 'auto', 'host' or 'dense', got {QR_EVAL!r}")
    if QR_EVAL != "auto":
        return QR_EVAL == "host"
    devs = getattr(qs.R_F, "devices", None)
    return all(d.platform == "cpu" for d in devs()) if devs is not None else True


def make_lml(prob, ds, mesh=None, cache_linear=True, n_types=1, n_free_ratios=0, shared_noise=False):
    """shared_noise: the caller wraps the result in paramset.tied_likelihood (cfg.noise "shared"), so the host
    QR evidence may merge the three quantities' factors once (lml_host.HostQREvidence shared).

    cache_linear: stream the theta-independent linear Gram G_BB once and
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
    if uses_qr(prob) and mesh is None:
        from .stats import linear_qr_statistics
        qs = linear_qr_statistics(prob.model, prob.cfg, ds,      # a host loop on the CPU: not jitted
                                  row_dtype=getattr(prob, "row_dtype", "float64"))
        jax.block_until_ready(qs)

        if _host_qr_eval(qs):     # the structured host evaluation (fit/lml_host.py): same evidence, ~10x faster
            from .linear.lml_host import HostQREvidence
            ev = HostQREvidence(qs, prob, shared=shared_noise)   # shared: only under paramset.tied_likelihood

            def f(a):
                return ev.lml(from_array(a))
            f.host_evidence = ev
        else:
            f = hoist(jax.jit(lambda a, qs: log_marginal_likelihood_qr(from_array(a), qs, prob)), qs)
        f.qr_stats = qs            # the statistics the posterior must share with the LML
        return f
    if cache_linear and mesh is None:
        from .stats import assemble_statistics, linear_statistics, residual_statistics
        lin = jax.jit(lambda: linear_statistics(prob.model, prob.cfg, ds))()
        jax.block_until_ready(lin)

        @jax.jit
        def lml(a):
            theta = from_array(a)
            res = residual_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
            return log_marginal_likelihood(theta, assemble_statistics(lin, res), prob)

        def f(a):                  # a plain closure, as above, to carry the cached Gram
            return lml(a)
        f.lin_stats = lin          # the predictions' linear statistics: no second linear pass
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
        from .gp.loo import config_row_index, loo_objective
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
