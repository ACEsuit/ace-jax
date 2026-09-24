"""Inference rungs over one shared log-density.  Sampling the log values with
Normal priors is the log-normal hyperprior; `lml` is the streamed marginal
likelihood (objective.make_lml), differentiable through scan + checkpoint.

Framework split (deliberate -- see the "keep the documented mix" decision).
Every rung consumes the SAME log-density; numpyro and blackjax are just two
inference backends over it:

  * run_map / run_laplace / run_vi / run_nuts -- numpyro.  Its batteries-included
    autoguides (AutoDelta, AutoLaplaceApproximation, AutoMultivariateNormal) and
    adaptive NUTS give validated posteriors with no hand-rolled adaptation, and
    the calibration results rest on them.
  * run_pathfinder -- blackjax.  numpyro has no Pathfinder, so this one rung uses
    blackjax.vi.pathfinder; it takes theta_map (a numpyro run_map result) as its
    L-BFGS start, so the two backends compose cleanly at the log-density.
  * run_laplace_fd -- backend-free: a finite-difference Hessian at the MAP,
    independent of both.

So the numpyro/blackjax mix is intra-ladder by necessity (Pathfinder), not an
accident; do NOT "unify" by dropping Pathfinder or rewriting the numpyro rungs
without re-validating calibration."""
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


def run_map_vec(lml, mu, sigma, init, *, steps=500, lr=0.02, seed=0):
    """MAP over an arbitrary-length LML vector with independent Normal priors
    N(mu[i], sigma[i]) per entry.  The n-hyper `run_map` above is left untouched
    (its calibration rests on the named FIELDS sites); this generalisation drives
    the extended [hypers | sigma_type] vector for the per-config-type fit."""
    mu, sigma, init = np.asarray(mu, float), np.asarray(sigma, float), np.asarray(init, float)
    n = mu.size

    def model():
        th = [numpyro.sample(f"p{i}", dist.Normal(float(mu[i]), float(sigma[i]))) for i in range(n)]
        numpyro.factor("loglik", lml(jnp.stack(th)))
    guide = AutoDelta(model, init_loc_fn=init_to_value(
        values={f"p{i}": jnp.asarray(init[i], jnp.float64) for i in range(n)}))
    params = _svi(model, guide, steps, lr, seed)
    med = guide.median(params)
    return jnp.stack([jnp.asarray(med[f"p{i}"], jnp.float64) for i in range(n)])


def run_map_ps(ps, prob, ds, *, steps=500, lr=0.02, seed=0):
    """Inner MAP over a ParamSet's LML blocks, returning the updated ParamSet.

    The all-LML case holds the embedding FIXED (materialise it once), so this is
    the existing flat path: build `make_lml` ONCE on the materialised problem
    (its theta-independent linear Gram is cached inside), then let `run_map`
    optimise `lml(theta_array)` over the flat LML vector -- which is exactly
    `ps.lml_vector()` (Task-2 `from_hypers` stores the 'hypers' block as
    `to_array(hypers)`).  A materialised embed is threaded through the kernel via
    `ind.embed` with its rows normalised (`normalize_rows(E)`), as the kernel's
    coregionalization expects -- the embed block carries the RAW E -- so a raw
    and a pre-normalised E give the same inner MAP."""
    from .objective import make_lml
    from .embedding import normalize_rows
    h, embed = ps.materialise()
    if embed is not None:
        embed = normalize_rows(embed)
    prob_m = prob if embed is None else prob._replace(ind=prob.ind._replace(embed=embed))
    try:
        st_block = ps.block("sigma_type")
    except StopIteration:
        st_block = None
    if st_block is None:                     # existing all-hypers path (unchanged)
        lml = make_lml(prob_m, ds)
        x_star = run_map(lml, ps.block("hypers").prior, steps=steps, lr=lr, seed=seed,
                         init=ps.lml_vector())
        return ps.set_lml_vector(to_array(x_star))
    # per-config-type path: optimise [hypers | free log-ratios] jointly.  The LML
    # vector already concatenates the hypers block and the sigma_type block, so
    # its layout matches make_lml's _sigma_type_decode (hypers first, ratios last).
    n_types, n_free = st_block.value.shape[0] + 1, st_block.value.size
    lml = make_lml(prob_m, ds, n_types=n_types, n_free_ratios=n_free)
    hp = ps.block("hypers").prior
    st_mu, st_sig = st_block.prior
    mu = jnp.concatenate([to_array(hp.mu), jnp.asarray(st_mu)])
    sigma = jnp.concatenate([to_array(hp.sigma), jnp.asarray(st_sig)])
    x_star = run_map_vec(lml, mu, sigma, init=ps.lml_vector(), steps=steps, lr=lr, seed=seed)
    return ps.set_lml_vector(x_star)


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


def run_nuts(lml, prior, *, num_warmup=500, num_samples=500, num_chains=4, seed=0, init=None,
             max_tree_depth=None, target_accept_prob=None):
    model = numpyro_model(lml, prior)
    kw = {}
    if max_tree_depth is not None:
        kw["max_tree_depth"] = max_tree_depth
    if target_accept_prob is not None:
        kw["target_accept_prob"] = target_accept_prob
    kernel = NUTS(model, init_strategy=init_to_value(values=_init(prior, init)), **kw)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    samples = mcmc.get_samples(group_by_chain=True)
    from numpyro.diagnostics import effective_sample_size, gelman_rubin
    # R-hat (Gelman-Rubin) is undefined for a single chain; report NaN then.
    r_hat = {f: (float(gelman_rubin(np.asarray(samples[f]))) if num_chains >= 2 else float("nan"))
             for f in FIELDS}
    summary = {"r_hat": r_hat,
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


def run_pathfinder(lml, prior, theta_map, *, n_draws=100, seed=0,
                   num_samples=16, maxiter=15):
    """Pathfinder VI (blackjax): a Gaussian approximation built along the L-BFGS
    optimisation path from theta_map, then sampled.  No MCMC loop (cheap, like
    Laplace) but often a better Gaussian than the at-mode Hessian when the
    posterior is skewed -- the recommended VI rung.  Returns (draws (n_draws,
    10) in log space, info).

    Memory: blackjax `approximate` vmaps the (Dt-dim) log-density objective over
    `num_samples` (ELBO estimate per L-BFGS iterate, to pick the best point on
    the path) nested inside a vmap over the path (`maxiter+1` iterates).  Peak is
    ~ c . num_samples . (maxiter+1) . Dt^2 -- linear in both vmap counts and
    quadratic in the parameter dimension Dt = len_basis + M (the objective's
    Cholesky).  So the blackjax defaults (num_samples=200, maxiter=30) blow up at
    scale -- 1.12 TiB at Cantor M=500 (Dt=2450).  The defaults here (16, 15) suit
    moderate problems (Dt<=350 -> ~17 GB) but STILL OOM a 20 GB GPU at Cantor
    M=500 (~68 GB): the Dt^2 term dominates, so LARGE problems must pass small
    values -- ns=4, maxiter=10 -> 11.6 GB, ns=2, maxiter=8 -> 5.2 GB at Dt=2450
    (measured).  num_samples only sets ELBO-estimation noise for path selection
    (NOT the posterior draw count -- that is n_draws, resampled below), so small
    values cost little; maxiter just caps the L-BFGS steps from a MAP start."""
    import blackjax
    from .hypers import log_prior
    logpost = jax.jit(lambda a: lml(a) + log_prior(from_array(a), prior))
    x0 = jnp.asarray(to_array(theta_map))
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    state, _ = blackjax.vi.pathfinder.approximate(
        k1, logpost, x0, num_samples=num_samples, maxiter=maxiter)
    draws, _ = blackjax.vi.pathfinder.sample(k2, state, n_draws)
    draws = np.asarray(draws)
    return draws, {"std": draws.std(0).tolist(), "fields": list(FIELDS),
                   "num_samples": num_samples, "maxiter": maxiter}
