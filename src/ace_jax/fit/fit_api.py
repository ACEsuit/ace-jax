"""Thin, test-facing fitting helpers over the ParamSet + ladder machinery."""
import jax.numpy as jnp


def fit_linear_with_sigma_type(bundle, *, steps=300, lr=0.05, seed=0):
    """Run an LML MAP fit with the per-config-type noise block on a synthetic
    two-type problem and return exp(log_ratios)[:, 0] -- the ENERGY-column noise
    ratios per config-type (type 0 pinned to 1).

    `bundle` is (prob, ds, n_types) with ds carrying `cfg_type`/`node_type`.
    """
    from .ladder import run_map_ps
    from .paramset import from_hypers
    prob, ds, n_types = bundle
    ps = from_hypers(prob.prior.mu, prob.prior, n_types=n_types)
    ps_out = run_map_ps(ps, prob, ds, steps=steps, lr=lr, seed=seed)
    log_ratios = ps_out.sigma_type_ratios()          # (n_types, 3), columns E,F,V
    return jnp.exp(log_ratios)[:, 0]
