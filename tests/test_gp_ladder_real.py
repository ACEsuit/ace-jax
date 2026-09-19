"""Ruling R32: the ladder exercised on the real hybrid model (opt-in, slow).

MAP + Laplace + one short NUTS chain on 6 Si_tiny configs with M = 4.  The
noise parameters log_sigma_E/F/V are identified by data, so the Laplace draws
must have non-zero spread there and NUTS must agree with Laplace on their
widths within a factor of 3.  Run with ACEGP_SLOW=1 (~10-20 min on CPU)."""
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
from ace_jax.fit.ladder import run_laplace, run_map, run_nuts
from ace_jax.fit.objective import Problem, make_log_density

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = [pytest.mark.skipif(os.environ.get("ACEGP_SLOW") != "1", reason="set ACEGP_SLOW=1"),
              pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")]


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
        theta_map = run_map(lik, prob.prior, steps=300)
        lap, _ = run_laplace(lik, prob.prior, n_draws=200, steps=300, init=theta_map)
        nuts, summ = run_nuts(lik, prob.prior, num_warmup=150, num_samples=150, num_chains=1,
                              init=theta_map)
    lap_sd, nuts_sd = lap[:, 7:10].std(0), nuts[:, 7:10].std(0)
    print("\ntheta_map:", np.asarray(to_array(theta_map)))
    print("laplace std[7:10]:", lap_sd, "\nnuts std[7:10]:   ", nuts_sd,
          "\nratio nuts/laplace:", nuts_sd / lap_sd, "\nnuts summary:", summ)
    assert lap.shape == (200, 10) and nuts.shape == (150, 10)
    assert lap_sd.min() > 0
    assert summ["divergences"] <= 0.1 * 150
    ratio = nuts_sd / lap_sd
    assert np.all(ratio >= 1 / 3) and np.all(ratio <= 3)
