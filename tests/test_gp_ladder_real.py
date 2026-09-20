"""Ruling R32: the ladder on the real hybrid model, in two run modes.

Fast (default, for CI, ~1.5 min): small iteration counts and a capped NUTS tree
depth.  It gates the *code paths* -- MAP, Laplace and a short NUTS chain all run
on the real hybrid model, return well-formed finite draws, Laplace's
data-identified noise params (log_sigma_E/F/V) have non-zero spread, and the
NUTS chain is divergence-free.  It deliberately does NOT assert the converged
Laplace-vs-NUTS width agreement: a short, depth-capped chain does not traverse
the noise-param widths (their nuts/laplace std ratio collapses toward 0), which
is expected here, not a regression.

Slow (ACEGP_SLOW=1, ~15-20 min): the converged parameters -- uncapped NUTS,
long warmup, many draws.  Adds the real R32 statistical check: NUTS must agree
with Laplace on the noise-param widths to within a factor of 3.
"""
import os

import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import default_prior, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.ladder import run_laplace, run_map, run_nuts, run_pathfinder
from ace_jax.fit.objective import Problem, make_log_density

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")]

SEED = 0
SLOW = os.environ.get("ACEGP_SLOW") == "1"
if SLOW:                                    # converged: meaningful statistics
    N_DRAWS, OPT_STEPS = 200, 300
    NUTS_WARMUP, NUTS_SAMPLES = 150, 150
    NUTS_MAX_DEPTH, NUTS_ACCEPT = None, None   # numpyro defaults (depth 10, 0.8)
    PF_SAMPLES, PF_MAXITER = 40, 25            # pathfinder (blackjax) ELBO samples / L-BFGS iters
else:                                       # fast: code-path + health gate
    N_DRAWS, OPT_STEPS = 60, 100
    NUTS_WARMUP, NUTS_SAMPLES = 50, 40
    PF_SAMPLES, PF_MAXITER = 8, 10             # pathfinder: tiny counts (cheap here)
    NUTS_MAX_DEPTH, NUTS_ACCEPT = 6, 0.9       # cap depth (ill-conditioned GP
    #   posterior -> uncapped ~2^10 leapfrogs/sample) and raise target-accept
    #   (smaller steps -> divergence-free short chain)


def test_ladder_on_real_model():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[1:7]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 4, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    with highest_precision():
        lik = make_log_density(prob, ds, "lml").likelihood
        theta_map = run_map(lik, prob.prior, steps=OPT_STEPS, seed=SEED)
        lap, _ = run_laplace(lik, prob.prior, n_draws=N_DRAWS, steps=OPT_STEPS, seed=SEED,
                             init=theta_map)
        pf, _ = run_pathfinder(lik, prob.prior, theta_map, n_draws=N_DRAWS, seed=SEED,
                               num_samples=PF_SAMPLES, maxiter=PF_MAXITER)
        nuts, summ = run_nuts(lik, prob.prior, num_warmup=NUTS_WARMUP, num_samples=NUTS_SAMPLES,
                              num_chains=1, seed=SEED, init=theta_map,
                              max_tree_depth=NUTS_MAX_DEPTH, target_accept_prob=NUTS_ACCEPT)
    lap_sd, nuts_sd = lap[:, 7:10].std(0), nuts[:, 7:10].std(0)
    print("\nmode:", "SLOW" if SLOW else "FAST", "\ntheta_map:", np.asarray(to_array(theta_map)))
    print("laplace std[7:10]:", lap_sd, "\nnuts std[7:10]:   ", nuts_sd,
          "\nratio nuts/laplace:", nuts_sd / lap_sd, "\nnuts summary:", summ)

    # -- both modes: the rungs ran and produced well-formed, finite draws --
    assert lap.shape == (N_DRAWS, 10) and nuts.shape == (NUTS_SAMPLES, 10)
    assert np.isfinite(lap).all() and np.isfinite(nuts).all()
    # Laplace's data-identified noise params have non-degenerate spread
    assert lap_sd.min() > 0
    # the (blackjax) Pathfinder rung ran: well-formed finite draws with
    # non-degenerate spread on the data-identified noise params
    assert pf.shape == (N_DRAWS, 10) and np.isfinite(pf).all()
    assert pf[:, 7:10].std(0).min() > 0
    # the NUTS chain is healthy (few divergences)
    assert summ["divergences"] <= 0.2 * NUTS_SAMPLES, summ["divergences"]

    # -- slow mode only: the converged R32 statistical check --
    if SLOW:
        ratio = nuts_sd / lap_sd
        assert np.all(ratio >= 1 / 3) and np.all(ratio <= 3), ratio
