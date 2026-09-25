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


def test_build_problem_linear_has_no_inducing_and_gp_has_m():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy",
                force_key="dft_force", virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35)
    for arm, m in (("linear", 0), ("gp", 6)):
        cfg = FitConfig(**base, arm=arm, m_per_species=6, density="pair")
        b = build_problem(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")))
        assert b.prob.ind.XM.shape[0] == m


def test_objective_device_and_hostcache_agree_on_the_fixture():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35, arm="gp",
                m_per_species=6, density="pair", rungs=("map",))
    vals = []
    for lml in ("device", "host-cache"):
        cfg = FitConfig(**base, lml=lml).validate()
        d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
        o = make_objective(cfg, d, build_problem(cfg, d))
        v, g = o.vg(to_array(o.prior_mu))
        vals.append((float(v), np.asarray(g)))
    assert np.isclose(vals[0][0], vals[1][0], rtol=1e-7)
    assert np.allclose(vals[0][1], vals[1][1], rtol=1e-5, atol=1e-6 * np.abs(vals[0][1]).max())


def test_fit_map_restarts_one_equals_single_lbfgs():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35, arm="linear",
                rungs=("map",), map_steps=8)
    thetas = []
    for n in (1, 2):
        cfg = FitConfig(**base, map_restarts=n)
        d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
        b = build_problem(cfg, d)
        r = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a, **k: None)
        thetas.append(r)
    assert len(thetas[0].restarts) == 1 and len(thetas[1].restarts) == 2
    assert thetas[1].restarts[0]["x"] == thetas[0].restarts[0]["x"]      # start 0 identical


def test_rungs_map_is_theta_and_unknown_rung_raises():
    from ace_jax.fit.hypers import default_prior, to_array
    from ace_jax.fit.pipeline import FitConfig
    from ace_jax.fit.pipeline.rungs import run_rungs
    th = default_prior(2.35).mu
    r = run_rungs(FitConfig(model="m", rungs=("map",)), None, None, th, log=lambda *a: None)
    assert np.array_equal(r.draws["map"], np.asarray(to_array(th))[None])
    with pytest.raises(ValueError, match="unknown rung"):
        run_rungs(FitConfig(model="m", rungs=("map", "hmc")), None, None, th, log=lambda *a: None)


def test_metrics_drop_configs_without_labels():
    from ace_jax.fit.pipeline.predict import label_metrics
    nat = np.array([1, 2, 2])
    E = np.array([np.nan, -1.0, -2.0]); Em = np.array([5.0, -1.1, -2.1]); Ev = np.full(3, 0.01)
    F = np.zeros((5, 3)); Fm = np.zeros((5, 3)) + 0.1; Fv = np.full((5, 3), 0.01)
    V = np.full((3, 6), np.nan); V[1:] = 0.0
    Vm = np.zeros((3, 6)); Vv = np.full((3, 6), 0.01)
    m = label_metrics(E, Em, Ev, F, Fm, Fv, V, Vm, Vv, nat)
    assert np.isfinite(m["E"]["rmse"]) and np.isfinite(m["V"]["rmse"])
    assert np.isclose(m["E"]["rmse"], 1e3 * np.sqrt(np.mean(((E[1:] - Em[1:]) / nat[1:]) ** 2)))


def test_pops_ridge_selection_uses_the_fit_subset_statistics(tiny_linear_problem, monkeypatch):
    import ace_jax.fit.predict as P
    prob, ds = tiny_linear_problem
    seen = []
    real = P.sufficient_statistics
    def spy(theta, spec, model, ind, cfg, dsx):
        seen.append(int(dsx.n_batches)); return real(theta, spec, model, ind, cfg, dsx)
    monkeypatch.setattr(P, "sufficient_statistics", spy)
    fit_ds = jax.tree.map(lambda a: a[:1], ds)
    P.select_pops_ridge(prob.prior.mu, prob, fit_ds, ds, [1e-3])
    assert seen and set(seen) == {1}            # the fit subset (1 batch), never the full set


def test_fit_end_to_end_writes_run_layout(tmp_path):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, write_outputs
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                    virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35, arm="linear",
                    rungs=("map",), map_steps=5, predict_train=False)
    res = fit(cfg.validate(), load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")),
              log=lambda *a: None)
    write_outputs(res, tmp_path, layout=("run", "cli"), argv={"arm": "linear"})
    for n in ("theta_map.json", "metrics.json", "metrics.csv", "pred_test_map.npz", "draws_map.npy",
              "map_restarts.json", "timings.json", "config.json", "split_perm.npy"):
        assert (tmp_path / n).exists(), n


def test_cli_weights_accepts_dict_or_factor_list():
    from ace_jax.cli import _parse_weights
    w, f = _parse_weights('{"default": {"E": 30, "F": 1, "V": 1}}')
    assert w == {"default": {"E": 30, "F": 1, "V": 1}} and f is None
    w, f = _parse_weights('[{"Structural": {}}]')
    assert w is None and [type(x).__name__ for x in f] == ["Structural"]
    with pytest.raises(ValueError, match="weights"):
        _parse_weights('"nonsense"')


def test_cli_rejects_train_and_data_together(tmp_path):
    from ace_jax.cli import main
    with pytest.raises(SystemExit):
        main(["fit", "--model", "m.npz", "--train", "a.xyz", "--data", "b.xyz", "--r0", "2.35",
              "--out", str(tmp_path)])
