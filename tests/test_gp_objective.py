"""BLR limit (no inducing points) against a ridge solve on ACEfit's exported
design matrix, and the marginal likelihood against a dense Gaussian log-density."""
import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem, log_marginal_likelihood, make_log_density, posterior
from ace_jax.fit.stats import sufficient_statistics

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
DESIGN = FIXTURE_DIR / "si_tiny_design.npz"
pytestmark = pytest.mark.skipif(not (XYZ.exists() and DESIGN.exists()), reason="missing GP fixtures")
NCFG = 6


@pytest.fixture(scope="module")
def blr():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:NCFG]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))  # M = 0
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    theta = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
                   log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
                   log_sigma_F=np.log(0.01), log_sigma_V=np.log(0.01))
    d = np.load(DESIGN)
    # rows per config are [E, F (3 n_atoms), V (6, only if the config has a virial)];
    # configs[0] here is an isolated atom with no virial rows, so summing a flat "+6"
    # over-counts by 6 -- compute the row count from the configs themselves instead.
    nrows = sum(1 + 3 * len(c.numbers) + 6 * (c.virial is not None) for c in configs)
    A, Y, W, gamma = d["A"][:nrows], d["Y"][:nrows], d["W"][:nrows], d["gamma"]
    return prob, ds, theta, A, Y, W, gamma


def test_posterior_mean_is_ridge_solution(blr):
    prob, ds, theta, A, Y, W, gamma = blr
    with highest_precision():
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        mu, L = posterior(theta, st, prob)
    sn, sc = 0.01, 0.3
    Aw, Yw = A * W[:, None], Y * W
    ref = np.linalg.solve(Aw.T @ Aw / sn**2 + np.diag(gamma**2) / sc**2, Aw.T @ Yw / sn**2)
    assert np.abs(np.asarray(mu) - ref).max() < 1e-8 * np.abs(ref).max()


def test_log_marginal_likelihood_matches_dense_gaussian(blr):
    prob, ds, theta, A, Y, W, gamma = blr
    with highest_precision():
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        lml = float(log_marginal_likelihood(theta, st, prob))
    sn, sc = 0.01, 0.3
    # density of the UNWEIGHTED observations: the structural weights are part of
    # the noise model, tau_i = (w_i / sigma)^2, so noise covariance is sn^2 / W^2
    K = A @ np.diag((sc / gamma) ** 2) @ A.T + np.diag(sn**2 / W**2)
    sign, logdet = np.linalg.slogdet(K)
    ref = -0.5 * Y @ np.linalg.solve(K, Y) - 0.5 * logdet - 0.5 * len(Y) * np.log(2 * np.pi)
    assert abs(lml - ref) < 1e-6 * abs(ref)


def test_log_density_is_jittable_and_differentiable(blr):
    prob, ds, theta, *_ = blr
    f = make_log_density(prob, ds)
    a = to_array(theta)
    with highest_precision():
        v, g = jax.value_and_grad(f)(a)
    assert np.isfinite(float(v)) and bool(jnp.all(jnp.isfinite(g)))
    assert float(g[7]) != 0.0        # log_sigma_E moves the likelihood
