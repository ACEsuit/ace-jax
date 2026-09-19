"""The four inference rungs off one numpyro model.  Sampling the log values
with Normal priors is the log-normal hyperprior; `lml` is the streamed marginal
likelihood (objective.make_lml), differentiable through scan + checkpoint."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import (AutoDelta, AutoLaplaceApproximation,
                                     AutoMultivariateNormal)
from numpyro.infer.initialization import init_to_value

from .hypers import Hypers, from_array, to_array

FIELDS = Hypers._fields


def numpyro_model(lml, prior):
    def model():
        th = [numpyro.sample(f, dist.Normal(float(m), float(s)))
              for f, m, s in zip(FIELDS, prior.mu, prior.sigma)]
        numpyro.factor("loglik", lml(jnp.stack(th)))
    return model


def _init(prior, init):
    h = prior.mu if init is None else init
    return {f: jnp.asarray(v, jnp.float64) for f, v in zip(FIELDS, h)}


def _stack(samples):
    a = np.stack([np.asarray(samples[f]) for f in FIELDS], axis=-1)
    return a.reshape(-1, len(FIELDS))


def _svi(model, guide, steps, lr, seed):
    # Adam on a deterministic objective oscillates at the lr scale, so decay it
    # to 0.1% over the run; the MAP then converges rather than jitters.
    svi = SVI(model, guide, optax.adam(optax.linear_schedule(lr, 1e-3 * lr, steps)), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed), steps, progress_bar=False)
    return res.params


def run_map(lml, prior, *, steps=500, lr=0.02, seed=0, init=None):
    model = numpyro_model(lml, prior)
    guide = AutoDelta(model, init_loc_fn=init_to_value(values=_init(prior, init)))
    params = _svi(model, guide, steps, lr, seed)
    med = guide.median(params)
    return Hypers(*[float(med[f]) for f in FIELDS])


def run_laplace(lml, prior, *, n_draws=100, steps=500, lr=0.02, seed=0, init=None):
    model = numpyro_model(lml, prior)
    guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=_init(prior, init)))
    params = _svi(model, guide, steps, lr, seed)
    draws = guide.sample_posterior(jax.random.PRNGKey(seed + 1), params, sample_shape=(n_draws,))
    med = guide.median(params)
    return _stack(draws), Hypers(*[float(med[f]) for f in FIELDS])


def run_vi(lml, prior, *, n_draws=100, steps=2000, lr=0.01, seed=0, init=None):
    model = numpyro_model(lml, prior)
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=_init(prior, init)))
    params = _svi(model, guide, steps, lr, seed)
    draws = guide.sample_posterior(jax.random.PRNGKey(seed + 1), params, sample_shape=(n_draws,))
    return _stack(draws), params


def run_nuts(lml, prior, *, num_warmup=500, num_samples=500, num_chains=4, seed=0, init=None):
    model = numpyro_model(lml, prior)
    kernel = NUTS(model, init_strategy=init_to_value(values=_init(prior, init)))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    samples = mcmc.get_samples(group_by_chain=True)
    from numpyro.diagnostics import effective_sample_size, gelman_rubin
    summary = {"r_hat": {f: float(gelman_rubin(np.asarray(samples[f]))) for f in FIELDS},
               "ess": {f: float(effective_sample_size(np.asarray(samples[f]))) for f in FIELDS},
               "divergences": int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))}
    return _stack(mcmc.get_samples()), summary


def run_laplace_fd(lml, prior, theta_map, *, n_draws=100, eps=1e-3, seed=0, floor=1e-6):
    """Laplace approximation with the Hessian from central finite differences
    of the exact gradient of the log posterior (2 x 10 gradient evaluations).

    `jax.hessian` (what AutoLaplaceApproximation uses) is forward-over-reverse
    and materialises (Dt, Dt, 10) intermediates through the Cholesky of the
    posterior precision -- 3.8 GB at Dt = 6890 -- and through the streamed
    scan for M > 0.  The gradient is exact and cheap, and the log posterior is
    smooth in the log hyperparameters, so central differences with eps = 1e-3
    give the Hessian to ~1e-6 relative.  Negative-curvature directions
    (unidentified hyperparameters) are floored at `floor` x the largest
    eigenvalue and reported in `info`.

    Returns (draws (n_draws, 10) in log space, info dict).
    """
    from .hypers import log_prior
    logpost = jax.jit(lambda a: lml(a) + log_prior(from_array(a), prior))
    grad = jax.jit(jax.grad(logpost))
    x0 = np.asarray(to_array(theta_map), float)
    n = x0.size
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        H[:, i] = (np.asarray(grad(jnp.asarray(x0 + e))) - np.asarray(grad(jnp.asarray(x0 - e)))) / (2 * eps)
    H = -0.5 * (H + H.T)                                   # precision of the Gaussian approximation
    w, V = np.linalg.eigh(H)
    n_floored = int(np.sum(w < floor * w.max()))
    w = np.maximum(w, floor * w.max())
    cov = (V / w) @ V.T
    rng = np.random.default_rng(seed + 1)
    draws = rng.multivariate_normal(x0, cov, size=n_draws)
    info = {"eigenvalues": w.tolist(), "n_floored": n_floored, "std": np.sqrt(np.diag(cov)).tolist(),
            "fields": list(FIELDS)}
    return draws, info
