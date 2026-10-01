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
    assert _cfg().validate().ard_variance == "sandwich"
    with pytest.raises(ValueError, match="ard_variance"):
        _cfg(ard_variance="nope").validate()
    with pytest.raises(ValueError, match="linear"):
        _cfg(arm="gp", m_per_species=6).validate()
    with pytest.raises(ValueError, match="ard_mode"):
        _cfg(ard_mode="both").validate()
    with pytest.raises(ValueError, match="ard_val_frac"):
        _cfg(ard_val_frac=1.0).validate()
    c = _cfg().validate()
    assert (c.ard_force_shape, c.ard_coverage, c.ard_groups, c.ard_cluster_size, c.ard_press, c.ard_n_min,
            c.ard_support, c._shape_variant, c._score_source) == ("iso", 0.9, "distortion", 3.0, "exact", 20,
                                                                  True, "press", "fit")
    for field, bad in (("ard_force_shape", "x"), ("ard_groups", "x"), ("ard_press", "x"),
                       ("_shape_variant", "x"), ("_score_source", "x"), ("ard_coverage", 1.0),
                       ("ard_n_min", 0), ("ard_cluster_size", 0.0)):
        with pytest.raises(ValueError, match=field):
            _cfg(**{field: bad}).validate()


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
    assert res.ard.report["mode"] == "sequential" and len(res.ard.report["h"]) == len(res.ard.report["body_groups"])


def test_joint_ard_noise_comes_from_the_ard_refit(ard_fit):
    """Joint mode fits its own sigma_q: pred noise_* must be exp(2 h_q) of the ARD refit, not the MAP's."""
    h = np.asarray(ard_fit.ard.posterior.h)
    assert ard_fit.ard.report["h_names"][:3] == ["log_sigma_E", "log_sigma_F", "log_sigma_V"]
    p = ard_fit.preds.arrays["test/map"]
    np.testing.assert_allclose(p["noise_F"], np.exp(2 * h[1]), rtol=1e-12)
    np.testing.assert_allclose(p["noise_E"], np.exp(2 * h[0]) * p["nat"], rtol=1e-12)
    np.testing.assert_allclose(p["noise_V"], np.exp(2 * h[2]) * p["nat"], rtol=1e-12)


def test_fit_hands_the_cached_statistics_to_the_ard_stage(monkeypatch):
    """I1: fit() passes the objective's cached linear statistics to run_ard_stage (joint mode)."""
    from ace_jax.fit import ard
    from ace_jax.fit.pipeline import fit, load_fit_data
    seen = []
    real = ard.run_ard_stage

    def spy(*a, **k):
        seen.append(k.get("full_stats"))
        return real(*a, **k)

    monkeypatch.setattr(ard, "run_ard_stage", spy)
    cfg = _cfg().validate()
    fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")), log=lambda *a: None)
    assert len(seen) == 1 and seen[0] is not None and hasattr(seen[0], "G_F")


def test_ard_checkpoint_survives_a_failure_in_prediction(tmp_path, monkeypatch):
    """I2: the stock checkpoint writer writes posterior.npz and ard.json when the "ard" stage fires."""
    import ace_jax.fit.pipeline.run as R
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.outputs import checkpoint_writer

    def boom(*a, **k):
        raise RuntimeError("OOM in prediction")

    monkeypatch.setattr(R, "predict_splits", boom)
    cfg = _cfg().validate()
    with pytest.raises(RuntimeError, match="OOM"):
        R.fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")), log=lambda *a: None,
              on_stage=checkpoint_writer(tmp_path))
    assert (tmp_path / "posterior.npz").exists() and (tmp_path / "ard.json").exists()
    assert ARDPosterior.load(tmp_path / "posterior.npz").kappa > 0
    assert json.load(open(tmp_path / "ard.json"))["mode"] == "joint"
    # the ARD-mean model file too, so big-cell forces_std can run before prediction finishes
    from ace_jax import ACECalculator
    calc = ACECalculator(str(tmp_path / "model.npz"), posterior=str(tmp_path / "posterior.npz"))
    assert calc.posterior is not None                    # the mean-vs-coefficients guard passed


def test_bench_run_driver_passes_the_ard_flags_through(tmp_path):
    """bench/acegp_cantor/run.py exposes --ard-variance/--ard-mode/--ard-val-frac and hands them to
    FitConfig (it used to omit them, so --uq ard silently took the defaults)."""
    import json, subprocess, sys
    from conftest import FIXTURE_DIR, ROOT
    r = subprocess.run(
        [sys.executable, str(ROOT / "bench/acegp_cantor/run.py"), "--model", str(FIXTURE_DIR / "si_fitted.npz"),
         "--data", str(FIXTURE_DIR / "si_tiny_train.xyz"), "--energy-key", "dft_energy", "--force-key",
         "dft_force", "--virial-key", "dft_virial", "--r0", "2.35", "--ntrain", "16", "--test-start", "16",
         "--ntest", "6", "--batch", "4", "--no-predict-train", "--arm", "linear", "--rungs", "map",
         "--map-steps", "5", "--uq", "ard", "--ard-variance", "kappa", "--ard-mode", "sequential",
         "--ard-val-frac", "0.25", "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads(next(tmp_path.rglob("ard.json")).read_text())
    assert rep["variance"] == "kappa" and rep["mode"] == "sequential" and rep["transfer"]["f"] == 0.25
    assert rep["n_val_configs"] == rep["split"]["n_val"] >= 3               # stratified: ~0.25 of 16 configs
