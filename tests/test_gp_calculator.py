import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.calc.gp import GPCalculator, fit_posteriors
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem
from ace_jax.fit.predict import predict_mixture

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


def test_calculator_agrees_with_predict_mixture():
    from ase import Atoms
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    E0 = np.asarray(z["E0"])
    train = build_dataset(configs[:6], meta, E0, 3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, train)
    ind = select_inducing(X, S, train.node_z, train.node_mask, 6, descriptor_scale(X, train.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    theta = Hypers(log_ell=np.log(0.8), log_A=np.log(0.05), log_alpha=0.0, log_r0=np.log(2.35),
                   log_eps=np.log(0.3), log_rho=np.log(4.0), log_sigma_c=np.log(0.3),
                   log_sigma_E=np.log(1e-3), log_sigma_F=np.log(0.02), log_sigma_V=np.log(0.02))
    a = np.asarray(to_array(theta))
    draws = np.stack([a, a + np.array([0.2] + [0.0] * 9)])
    with highest_precision():
        fitted = fit_posteriors(prob, train, draws)
        c = configs[7]
        atoms = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc)
        atoms.calc = GPCalculator(fitted, meta)
        E, F, s = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        E_std = atoms.calc.get_property("energy_std", atoms)
        test = build_dataset([c], meta, E0, 1)
        p = predict_mixture(draws, prob, train, test)
    assert abs(E - p.E_mean[0]) < 1e-8 and np.abs(F - p.F_mean).max() < 1e-8
    assert abs(E_std - np.sqrt(p.E_var[0])) < 1e-8
    assert np.allclose(s, -p.V_mean[0] / atoms.get_volume())
