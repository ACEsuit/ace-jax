import json

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR


def _cfg(**kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=30, ntest=8, batch=4, r0=2.35, arm="linear", uq="ard",
                opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False, predict_stats="recompute")
    return FitConfig(**{**base, **kw})


def test_config_validates_ard():
    _cfg().validate()
    with pytest.raises(ValueError, match="linear"):
        _cfg(arm="gp", m_per_species=6).validate()
    with pytest.raises(ValueError, match="ard_mode"):
        _cfg(ard_mode="both").validate()
    with pytest.raises(ValueError, match="ard_val_frac"):
        _cfg(ard_val_frac=1.0).validate()


@pytest.fixture(scope="module")
def ard_fit():
    from ace_jax.fit.pipeline import fit, load_fit_data
    cfg = _cfg().validate()
    return fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")), log=lambda *a: None)


def test_fit_ard_predicts_with_tempered_force_variance(ard_fit):
    from ace_jax.fit.ard import predict_ard
    res = ard_fit
    assert res.ard is not None and res.ard.posterior.kappa > 0
    p = res.preds.arrays["test/map"]
    again = predict_ard(res.ard.posterior, res.built.prob, res.data.ds_test)
    np.testing.assert_allclose(p["F_var"], np.asarray(again.F_var), rtol=1e-10)
    np.testing.assert_allclose(p["F_mean"], np.asarray(again.F_mean), rtol=1e-10, atol=1e-12)


def test_write_outputs_saves_posterior_and_report(ard_fit, tmp_path):
    from ace_jax import ACECalculator
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.pipeline import write_outputs
    from ase import Atoms
    write_outputs(ard_fit, tmp_path, layout=("cli",), argv={})
    post = ARDPosterior.load(tmp_path / "posterior.npz")
    rep = json.load(open(tmp_path / "ard.json"))
    assert post.kappa == pytest.approx(ard_fit.ard.posterior.kappa) and rep["mode"] == "joint"
    calc = ACECalculator(str(tmp_path / "model.npz"))                  # model.npz = the ARD mean
    E = []
    for c in ard_fit.data.test:
        at = Atoms(numbers=c.numbers, positions=c.positions, cell=c.cell, pbc=c.pbc); at.calc = calc
        E.append(at.get_potential_energy())
    np.testing.assert_allclose(E, ard_fit.preds.arrays["test/map"]["E_mean"], rtol=1e-9, atol=1e-9)


def test_sequential_mode_runs(tmp_path):
    from ace_jax.fit.pipeline import fit, load_fit_data
    cfg = _cfg(ard_mode="sequential").validate()
    res = fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")), log=lambda *a: None)
    assert res.ard.report["mode"] == "sequential" and len(res.ard.report["h"]) == len(res.ard.report["groups"])


def test_joint_ard_noise_comes_from_the_ard_refit(ard_fit):
    """Joint mode fits its own sigma_q: pred noise_* must be exp(2 h_q) of the ARD refit, not the MAP's."""
    h = np.asarray(ard_fit.ard.posterior.h)
    assert ard_fit.ard.report["h_names"][:3] == ["log_sigma_E", "log_sigma_F", "log_sigma_V"]
    p = ard_fit.preds.arrays["test/map"]
    np.testing.assert_allclose(p["noise_F"], np.exp(2 * h[1]), rtol=1e-12)
    np.testing.assert_allclose(p["noise_E"], np.exp(2 * h[0]) * p["nat"], rtol=1e-12)
    np.testing.assert_allclose(p["noise_V"], np.exp(2 * h[2]) * p["nat"], rtol=1e-12)
