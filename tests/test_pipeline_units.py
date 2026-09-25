import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_predict_uses_explicit_stats(tiny_linear_problem):
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.predict import predict_mixture
    from ace_jax.fit.stats import sufficient_statistics
    prob, ds = tiny_linear_problem
    theta = prob.prior.mu
    calls = []
    def stats(th):
        calls.append(1)
        return sufficient_statistics(th, prob.spec, prob.model, prob.ind, prob.cfg, ds)
    d = np.asarray(to_array(theta))[None]
    ref = predict_mixture(d, prob, ds, ds)
    got = predict_mixture(d, prob, ds, ds, stats=stats)
    assert calls == [1]
    for f in ref._fields:
        assert np.allclose(np.asarray(getattr(got, f)), np.asarray(getattr(ref, f)), rtol=1e-12)


def test_split_configs_matches_run_py_permutation():
    from ace_jax.fit.pipeline import split_configs
    cfgs = list(range(30))
    tr, te, perm = split_configs(cfgs, 10, 5, test_start=20, seed=3)
    ref = np.random.default_rng(3).permutation(30)
    assert tr == [cfgs[i] for i in ref[:10]] and te == [cfgs[i] for i in ref[20:25]]
    assert np.array_equal(perm, ref)


def test_fitconfig_rejects_bad_combinations():
    from ace_jax.fit.pipeline import FitConfig
    with pytest.raises(ValueError, match="arm linear"):
        FitConfig(model="m.npz", arm="gp", uq="pops").validate()
    with pytest.raises(ValueError, match="host-cache"):
        FitConfig(model="m.npz", arm="linear", lml="host-cache").validate()
    with pytest.raises(ValueError, match="host-cache"):
        FitConfig(model="m.npz", arm="gp", density="none", lml="host-cache").validate()
    with pytest.raises(ValueError, match="host-cache"):
        FitConfig(model="m.npz", arm="gp", density="pca", lml="host-cache", rungs=("map", "laplace")).validate()
    with pytest.raises(ValueError, match="host-cache"):
        FitConfig(model="m.npz", arm="gp", density="pca", lml="host-cache", rungs=("map",),
                  opt="adam").validate()
    with pytest.raises(ValueError, match="devices"):
        FitConfig(model="m.npz", arm="gp", density="pca", lml="host-cache", rungs=("map",),
                  devices=2).validate()
    FitConfig(model="m.npz", arm="gp", density="pca", lml="host-cache", rungs=("map",)).validate()
    with pytest.raises(ValueError, match="map_restarts"):
        FitConfig(model="m.npz", map_restarts=0).validate()


def test_load_fit_data_file_split_and_lsq_e0():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy",
                    force_key="dft_force", virial_key="dft_virial", ntrain=16, ntest=6,
                    test_start=16, batch=4, e0="lsq")
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    assert len(d.train) == 16 and len(d.test) == 6 and d.ds_ood is None
    counts = np.array([[np.sum(c.numbers == 14)] for c in d.train], float)
    E0ref, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in d.train]), rcond=None)
    assert np.allclose(d.E0, E0ref)
