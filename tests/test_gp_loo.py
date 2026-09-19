"""Block-leverage LOO equals explicit leave-one-configuration-out refits."""
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
from ace_jax.fit.loo import config_row_index, loo_objective
from ace_jax.fit.objective import Problem, combine, make_log_density, posterior, prior_precision
from ace_jax.fit.rows import batch_rows
from ace_jax.fit.stats import sufficient_statistics

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


def test_loo_matches_explicit_refits():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:4]
    E0 = np.asarray(z["E0"])
    ds = build_dataset(configs, meta, E0, 2)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=2)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 4, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("matern32", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    theta = Hypers(log_ell=np.log(1.0), log_A=np.log(0.05), log_alpha=0.0, log_r0=np.log(2.35),
                   log_eps=np.log(0.3), log_rho=np.log(4.0), log_sigma_c=np.log(0.3),
                   log_sigma_E=np.log(0.01), log_sigma_F=np.log(0.05), log_sigma_V=np.log(0.05))
    with highest_precision():
        st = sufficient_statistics(theta, prob.spec, model, ind, cfg, ds)
        got = float(loo_objective(theta, st, prob, ds, jnp.asarray(config_row_index(ds))))
        # explicit refits
        ref = 0.0
        s2 = {t: np.exp(2 * getattr(theta, f"log_sigma_{t}")) for t in "EFV"}
        for a in range(4):
            rest = build_dataset([c for i, c in enumerate(configs) if i != a], meta, E0, 3)
            st_a = sufficient_statistics(theta, prob.spec, model, ind, cfg, rest)
            mu, L = posterior(theta, st_a, prob)
            # config 0 is an isolated atom: alone it has no neighbours, so k_cap would be
            # inferred as 0; borrow the full dataset's k_cap (padding width only)
            one = build_dataset([configs[a]], meta, E0, 1, k_cap=ds.nbr.shape[-1])
            b = jax.tree.map(lambda x: x[0], one)
            r = batch_rows(theta, prob.spec, model, ind, cfg, b)
            Dt = r.E.shape[-1]
            n = len(configs[a].numbers)
            live = np.asarray(b.node_mask)
            Phi = np.concatenate([np.asarray(r.E[:1]), np.asarray(r.F)[live].reshape(3 * n, Dt),
                                  np.asarray(r.V[0])])
            y = np.concatenate([np.asarray(b.y_E[:1]), np.asarray(b.y_F)[live].reshape(-1), np.asarray(b.y_V[0])])
            w = np.concatenate([np.asarray(b.w_E[:1]), np.repeat(np.asarray(b.w_F)[live], 3), np.repeat(np.asarray(b.w_V[:1]), 6)])
            # the density is over OBSERVED rows only (config 0 has no virial: w_V = 0,
            # tau = 0); an absent row would put an infinite noise variance on the diagonal
            keep = w > 0
            Phi, y, w = Phi[keep], y[keep], w[keep]
            noise = np.concatenate([[s2["E"]], np.full(3 * n, s2["F"]), np.full(6, s2["V"])])[keep] / w**2
            Sinv_Phi = np.linalg.solve(np.asarray(L) @ np.asarray(L).T, Phi.T)
            cov = Phi @ Sinv_Phi + np.diag(noise)
            res = y - Phi @ np.asarray(mu)
            sign, logdet = np.linalg.slogdet(cov)
            ref += -0.5 * res @ np.linalg.solve(cov, res) - 0.5 * logdet - 0.5 * len(y) * np.log(2 * np.pi)
    assert abs(got - ref) < 1e-6 * abs(ref)
    # Task 14 differentiates the LOO log-density in theta: check the gradient is finite
    with highest_precision():
        g = jax.grad(make_log_density(prob, ds, "loo"))(to_array(theta))
    assert bool(jnp.all(jnp.isfinite(g)))
