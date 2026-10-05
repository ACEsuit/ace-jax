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


# ---------------------------------------------------------------------------
# Final-review fixes (Important 1-6)
# ---------------------------------------------------------------------------
def _fx_cfg(**kw):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=12, ntest=4, batch=4, r0=2.35, arm="linear",
                rungs=("map",), map_steps=3, predict_train=False)
    base.update(kw)
    return FitConfig(**base)


def test_cli_keeps_the_ladder_pathfinder_defaults():
    from ace_jax.cli import _fit_config, _parser
    a = _parser().parse_args(["fit", "--model", "m.npz", "--train", "t.xyz", "--r0", "2.35", "--out", "o"])
    cfg = _fit_config(a)
    assert (cfg.pf_samples, cfg.pf_maxiter) == (16, 15)          # run_pathfinder's own defaults


def test_cli_writes_ood_metrics(tmp_path):
    import csv
    from conftest import FIXTURE_DIR
    from ase.io import read, write
    from ace_jax.cli import main
    cfgs = read(FIXTURE_DIR / "si_tiny_train.xyz", ":")
    tr, te, od = tmp_path / "tr.xyz", tmp_path / "te.xyz", tmp_path / "od.xyz"
    write(tr, cfgs[:12]); write(te, cfgs[12:16]); write(od, cfgs[20:24])
    main(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--train", str(tr), "--test", str(te),
          "--ood", str(od), "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key",
          "dft_virial", "--configs-per-batch", "4", "--m-per-species", "0", "--rungs", "map",
          "--map-steps", "3", "--r0", "2.35", "--out", str(tmp_path / "out")])
    rows = list(csv.DictReader(open(tmp_path / "out" / "metrics_ood.csv")))
    assert {r["quantity"] for r in rows} == {"E", "F", "V"} and {r["rung"] for r in rows} == {"map"}
    # the fitted linear model is saved as an ordinary ACE npz that eval reads
    assert main(["eval", "--model", str(tmp_path / "out" / "model.npz"), "--data", str(te),
                 "--energy-key", "dft_energy", "--force-key", "dft_force"]) == 0


def test_map_and_rung_outputs_survive_a_failure_in_prediction(tmp_path, monkeypatch):
    from conftest import FIXTURE_DIR
    import ace_jax.fit.pipeline.run as R
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.outputs import checkpoint_writer
    cfg = _fx_cfg()
    def boom(*a, **k):
        raise RuntimeError("OOM in prediction")
    monkeypatch.setattr(R, "predict_splits", boom)
    with pytest.raises(RuntimeError, match="OOM"):
        R.fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")), log=lambda *a: None,
              on_stage=checkpoint_writer(tmp_path))
    for n in ("split_perm.npy", "theta_map.json", "map_restarts.json", "draws_map.npy"):
        assert (tmp_path / n).exists(), n


def test_pops_reuses_the_objectives_linear_statistics(monkeypatch):
    from conftest import FIXTURE_DIR
    import ace_jax.fit.pipeline.objective as O
    import ace_jax.fit.stats as ST
    from ace_jax.fit.pipeline import fit, load_fit_data
    calls = []
    real = ST.linear_statistics
    def spy(*a, **k):
        calls.append(1); return real(*a, **k)
    monkeypatch.setattr(ST, "linear_statistics", spy)
    monkeypatch.setattr(O, "linear_statistics", spy)
    counts = {}
    for uq in ("blr", "pops"):
        calls.clear()
        cfg = _fx_cfg(uq=uq, pops_ridge="blr", pops_env_nf=10)
        fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")), log=lambda *a: None)
        counts[uq] = len(calls)
    assert counts["pops"] == counts["blr"]                        # no extra ACE pass before POPS


def test_split_configs_rejects_a_file_too_small_for_the_split():
    from ace_jax.fit.pipeline import split_configs
    with pytest.raises(ValueError, match="ntrain"):
        split_configs(list(range(10)), 800, 200)
    with pytest.raises(ValueError, match="test"):
        split_configs(list(range(10)), 8, 5)


def test_map_restarts_needs_lbfgs():
    from ace_jax.fit.pipeline import FitConfig
    with pytest.raises(ValueError, match="map_restarts"):
        FitConfig(model="m.npz", opt="adam", map_restarts=3).validate()
    with pytest.raises(ValueError, match="map_restarts"):
        FitConfig(model="m.npz", sigma_type=True, map_restarts=3).validate()


def test_default_ladder_is_map_only():
    """Laplace compiles a Hessian through the whole LML (>30 min on 40 Si configs on
    a laptop CPU), so hyperparameter draws are opt-in: --rungs map,laplace."""
    from ace_jax.cli import _fit_config, _parser
    from ace_jax.fit.pipeline import FitConfig
    a = _parser().parse_args(["fit", "--model", "m.npz", "--train", "a.xyz", "--r0", "2.35", "--out", "o"])
    assert tuple(_fit_config(a).rungs) == ("map",)
    assert tuple(FitConfig(model="m.npz").rungs) == ("map",)


def test_pops_fit_with_host_rows_matches_device_rows():
    """FitConfig.pops_rows="host" (rows evaluated once, cached in host RAM) gives the
    same POPS fit as "device" (rows re-evaluated per pass), to the Gram's summation
    order."""
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=16, ntest=6, batch=4, r0=2.35, arm="linear", uq="pops",
                opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False,
                pops_ridge_grid=(1e-3, 1e-5, 1e-7),
                # unpolished: the converged MAP has a smaller sigma_E, so its ridge-1e-7 POPS solve is
                # ill-conditioned enough that host vs device summation order moves F_mean ~2x past
                # this test's tolerance; the rows paths, not the MAP, are under test
                map_polish="off")
    out = {}
    for rows in ("device", "host"):
        cfg = FitConfig(**base, pops_rows=rows).validate()
        out[rows] = fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz")),
                        log=lambda *a: None).preds
    assert out["host"].pops["ridge"] == out["device"].pops["ridge"]
    a, b = out["device"].arrays["test/map"], out["host"].arrays["test/map"]
    # agree to the problem's solver accuracy (the BLR mean is 1e-5 from QR here)
    for k in ("E_mean", "F_mean", "E_var", "F_var", "V_var"):
        assert np.allclose(a[k], b[k], rtol=1e-5, atol=1e-7), k
    with pytest.raises(ValueError, match="pops_rows"):
        FitConfig(**base, pops_rows="gpu").validate()


def test_fitting_a_yace_model_raises_a_clear_error():
    """A .yace (PACEModel) is evaluate-only; fitting one used to die with
    "TypeError: 'PACESpec' object is not subscriptable" deep in load_fit_data."""
    import pathlib

    from ace_jax.fit.pipeline import FitConfig
    from ace_jax.fit.pipeline.data import load_fit_data
    fix = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"
    cfg = FitConfig(model=str(fix / "gesi_sbessel.yace"), arm="linear")
    with pytest.raises(ValueError, match=r"\.yace.*cannot be fitted"):
        load_fit_data(cfg, train=str(pathlib.Path(__file__).parent.parent / "fixtures" / "si_tiny_train.xyz"))


def _cfgs(spec):
    """Synthetic Configs: (numbers, energy, cell edge or None for non-periodic)."""
    from ace_jax.fit.data import Config
    out = []
    for nums, E, a in spec:
        n = len(nums)
        pos = np.c_[np.arange(n) * 2.3, np.zeros(n), np.zeros(n)]
        cell, pbc = (np.zeros((3, 3)), np.zeros(3, bool)) if a is None else (np.eye(3) * a, np.ones(3, bool))
        out.append(Config(pos, np.asarray(nums), cell, pbc, E, None, None, 1.0, 1.0, 1.0))
    return out


def test_lsq_e0_takes_isolated_atom_energies_exactly():
    """An isolated atom's energy is E0 alone: least squares must not compromise it
    against the bulk.  Species without one are fitted to the rest, after subtracting."""
    from ace_jax.fit.pipeline.data import lsq_e0
    cs = _cfgs([([6], -5.0, None),                 # isolated C, non-periodic
                ([14], -2.0, 12.0),                # isolated Si: periodic, but no image within rcut
                ([14, 14], -11.0, 4.6), ([14, 6], -14.0, 4.6), ([14, 14, 6], -20.0, 6.9)])
    E0 = lsq_e0(cs, [14, 6], rcut=5.5, log=lambda *a: None)
    assert E0.tolist() == [-2.0, -5.0]
    # one isolated species only: C fixed, Si by least squares on the residual
    cs2 = [c for c in cs if not (len(c.numbers) == 1 and c.numbers[0] == 14)]
    E0 = lsq_e0(cs2, [14, 6], rcut=5.5, log=lambda *a: None)
    counts = np.array([[2.0], [1.0], [2.0]]); r = np.array([-11.0, -14.0 + 5.0, -20.0 + 5.0])
    assert E0[1] == -5.0 and np.isclose(E0[0], np.linalg.lstsq(counts, r, rcond=None)[0][0])


def test_lsq_e0_single_atom_with_close_images_is_not_isolated():
    from ace_jax.fit.pipeline.data import lsq_e0
    cs = _cfgs([([14], -4.0, 2.7), ([14, 14], -9.0, 4.6)])     # 2.7 A cell: images within rcut -> bulk
    E0 = lsq_e0(cs, [14], rcut=5.5, log=lambda *a: None)
    assert np.isclose(E0[0], np.linalg.lstsq([[1.0], [2.0]], [-4.0, -9.0], rcond=None)[0][0])


def test_fit_reports_the_log_evidence_for_both_optimisers():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=16, ntest=6, batch=4, r0=2.35, arm="linear",
                m_per_species=0, rungs=("map",), map_steps=5, predict_train=False)
    for opt in ("adam", "lbfgs"):
        cfg = FitConfig(opt=opt, **base)
        d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lambda *a: None)
        res = fit(cfg, d, log=lambda *a: None)
        b = build_problem(cfg, d)
        ref = float(make_objective(cfg, d, b).lik(to_array(res.theta)))
        assert np.isfinite(res.map.log_evidence) and abs(res.map.log_evidence - ref) <= 1e-8 * abs(ref), opt


def test_fix_rho_pins_a_numeric_rho_under_lbfgs():
    # fit_map once wrote the pin into a read-only view of the prior mean
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                    virial_key="dft_virial", ntrain=16, ntest=6, batch=4, r0=2.35, arm="linear",
                    m_per_species=0, rungs=("map",), opt="lbfgs", map_steps=5, predict_train=False, fix_rho="0.5")
    res = fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lambda *a: None),
              log=lambda *a: None)
    assert abs(np.exp(res.theta.log_rho) - 0.5) < 1e-12


def _lstsq_fit(predict_train=False):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                    virial_key="dft_virial", ntrain=16, ntest=6, batch=4, r0=2.35, arm="linear",
                    m_per_species=0, rungs=("map",), predict_train=predict_train, solver="lstsq")
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lambda *a: None)
    return cfg, d, fit(cfg, d, log=lambda *a: None)


def test_lstsq_solver_is_plain_weighted_least_squares():
    from ace_jax.eval import highest_precision
    from ace_jax.fit.pipeline.export import linear_model_arrays
    from ace_jax.fit.pipeline.lstsq import lstsq_theta
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.solve import stacked_design
    cfg, d, res = _lstsq_fit()
    b = build_problem(cfg, d)
    with highest_precision():
        Phi, y = stacked_design(b.prob, d.ds_train, lstsq_theta(b.prob))
    Dt, L0 = Phi.shape[1], b.prob.cfg.len_readout
    keep = np.r_[np.arange(Phi.shape[0] - Dt), Phi.shape[0] - Dt + np.arange(L0, Dt)]   # data + the E0 prior rows
    ref = np.linalg.lstsq(np.asarray(Phi)[keep], np.asarray(y)[keep], rcond=None)[0]    # no readout prior
    np.testing.assert_allclose(res.readout, ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max())
    arr = linear_model_arrays(res)                       # the saved model carries that readout
    n = b.prob.cfg.n_B
    np.testing.assert_array_equal(arr["WB"][:, 0], np.asarray(res.readout)[:n])
    assert res.map.log_evidence is None and set(res.rungs.draws) == {"lstsq"}
    assert abs(arr["E0"][0] - d.E0[0]) < 3.0          # E0 keeps its 1 eV prior: no arbitrary offset


def test_lstsq_predictions_use_the_readout_with_zero_variance():
    from ace_jax.fit.rows import linear_rows
    import jax
    cfg, d, res = _lstsq_fit(predict_train=True)
    a = res.preds.arrays["train/lstsq"]
    assert np.all(a["E_var"] == 0) and np.all(a["F_var"] == 0)
    # energies straight from the design rows and the readout (+ E0), per config
    Em = []
    for i in range(d.ds_train.n_batches):
        bt = jax.tree.map(lambda x, i=i: x[i], d.ds_train)
        r = linear_rows(res.built.prob.model, res.built.prob.cfg, bt)[0]
        E0 = np.asarray(res.built.prob.model.E0)[np.asarray(bt.node_z)] * np.asarray(bt.node_mask)
        e0c = np.array([E0[np.asarray(bt.node_cfg) == c].sum() for c in range(r.E.shape[0])])
        Em.append((np.asarray(r.E @ res.readout) + e0c)[np.asarray(bt.cfg_mask) > 0])
    np.testing.assert_allclose(a["E_mean"], np.concatenate(Em), rtol=1e-9, atol=1e-9)
    assert np.isfinite(res.preds.metrics["test/lstsq"]["E"]["rmse"])


def test_lstsq_solver_validation():
    from ace_jax.fit.pipeline import FitConfig
    with pytest.raises(ValueError, match="solver"):
        FitConfig(model="m.npz", solver="qr").validate()
    with pytest.raises(ValueError, match="lstsq"):
        FitConfig(model="m.npz", solver="lstsq", arm="gp").validate()


def test_summarise_without_any_uq_still_reports_the_error():
    from ace_jax.fit.metrics import summarise
    m = summarise([1.0, 2.0], [1.5, 2.0], [0.0, 0.0])     # a fixed readout: no predictive variance
    assert abs(m["rmse"] - np.sqrt(0.125)) < 1e-15 and abs(m["mae"] - 0.25) < 1e-15
    assert np.isnan(m["coverage"]) and m["n_dropped"] == 2


def test_fit_refuses_float32():
    # a float32 fit is silently poor (and its MD crawls): the caller must enable float64
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                    virial_key="dft_virial", ntrain=8, ntest=4, batch=4, r0=2.35, arm="linear",
                    m_per_species=0, rungs=("map",))
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lambda *a: None)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64"):
            fit(cfg, d, log=lambda *a: None)
    finally:
        jax.config.update("jax_enable_x64", True)
