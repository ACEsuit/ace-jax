import pytest
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision


def _dense_rows(prob, ds):
    """Weighted live rows (w*phi, w*y) and the quantity index (0 E, 1 F, 2 V) of every observation."""
    from ace_jax.fit.rows import linear_rows
    rows, ys, qs = [], [], []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        r = linear_rows(prob.model, prob.cfg, b)[0]
        L = r.E.shape[-1]
        for k, (Phi, y, w) in enumerate(((r.E, b.y_E, b.w_E),
                                         (r.F.reshape(-1, L), b.y_F.reshape(-1), jnp.repeat(b.w_F, 3)),
                                         (r.V.reshape(-1, L), b.y_V.reshape(-1), jnp.repeat(b.w_V, 6)))):
            w = np.asarray(w); live = w > 0
            rows.append(np.asarray(Phi)[live] * w[live, None]); ys.append(np.asarray(y)[live] * w[live])
            qs.append(np.full(live.sum(), k))
    return np.concatenate(rows), np.concatenate(ys), np.concatenate(qs)


def _dense_logev(Phi, y, q, h, gamma, gidx, groups_n):
    s2 = np.exp(2 * h[:3])[q]
    lam = gamma ** 2 * np.exp(h[3:3 + groups_n])[gidx]
    Cov = (Phi / lam) @ Phi.T + np.diag(s2)
    _, ld = np.linalg.slogdet(Cov)
    return -0.5 * y @ np.linalg.solve(Cov, y) - 0.5 * ld


def test_body_order_columns_layout():
    from ace_jax.fit.ard import body_order_columns
    from ace_jax.fit.inducing import GPConfig
    meta = {"nnll": [[(1, 0)], [(1, 0), (1, 1)], [(1, 0), (1, 1), (2, 1)]]}
    g = GPConfig(r0=2.35, rcut=5.0, n_B=3, n_pair=2, NZ=2, C=1)
    col = body_order_columns(meta, g)
    # species blocks of B first (orders 1,2,3 -> body 2,3,4), then pair blocks (body 2)
    assert col.tolist() == [2, 3, 4, 2, 3, 4, 2, 2, 2, 2]


def test_evidence_matches_dense_marginal_likelihood(tiny_linear_problem):
    """Differences of log p between two h equal those of the explicit Gaussian marginal likelihood
    N(y_w | 0, Phi_w Lambda^-1 Phi_w^T + diag(sigma_q^2)) (constants cancel)."""
    from ace_jax.fit.ard import ARDEvidence, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    with highest_precision():
        st = ard_statistics(theta, prob, ds, "joint")
        meta = {"nnll": [[None] * o for o in np.asarray(_orders(prob))]}
        ev = ARDEvidence(st, np.asarray(prob.gamma), body_order_columns(meta, prob.cfg))
        h1 = ev.h0(theta); h2 = h1 + np.r_[0.1, -0.2, 0.05, [0.3, -0.4, 0.2][: len(ev.groups)]]
        v1, v2 = ev.value_and_grad(h1)[0], ev.value_and_grad(h2)[0]
        Phi, y, q = _dense_rows(prob, ds)
        gidx = np.searchsorted(np.asarray(ev.groups), body_order_columns(meta, prob.cfg))
        d1 = _dense_logev(Phi, y, q, h1, np.asarray(prob.gamma), gidx, len(ev.groups))
        d2 = _dense_logev(Phi, y, q, h2, np.asarray(prob.gamma), gidx, len(ev.groups))
    assert abs((v2 - v1) - (d2 - d1)) < 1e-6 * max(1.0, abs(d2 - d1))


def _orders(prob):
    """Correlation order of each B column of the tiny problem's model (all body orders present)."""
    import json
    from conftest import FIXTURE_DIR
    z = np.load(FIXTURE_DIR / "si_fitted.npz")
    return [len(x) for x in json.loads(bytes(z["meta_json"]).decode())["nnll"]]


def test_gradient_matches_finite_differences(tiny_linear_problem):
    from ace_jax.fit.ard import ARDEvidence, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    with highest_precision():
        st = ard_statistics(theta, prob, ds, "joint")
        meta = {"nnll": [[None] * o for o in _orders(prob)]}
        ev = ARDEvidence(st, np.asarray(prob.gamma), body_order_columns(meta, prob.cfg))
        h = ev.h0(theta); v, g = ev.value_and_grad(h)
        # log p is a ~1e-6 residue of two ~1e8 terms (1/2 yy_E/sigma_E^2 vs 1/2 b^T A^-1 b at
        # sigma_E = 1e-3), so the value carries ~2e-7 of f64 roundoff: a 1e-5 step is noise-dominated
        # (FD error 1e-2).  At e = 3e-3 the worst of 20 directions is 5e-5 relative.
        d = np.random.default_rng(0).standard_normal(len(h)); d /= np.linalg.norm(d); e = 3e-3
        fd = (ev.value_and_grad(h + e * d)[0] - ev.value_and_grad(h - e * d)[0]) / (2 * e)
    assert abs(fd - g @ d) < 3e-4 * max(1.0, abs(fd))


def test_joint_fit_beats_sequential_beats_start(tiny_linear_problem):
    from ace_jax.fit.ard import ARDEvidence, ard_statistics, body_order_columns, fit_ard
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)]}
    with highest_precision():
        evJ = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                          body_order_columns(meta, prob.cfg))
        evS = ARDEvidence(ard_statistics(theta, prob, ds, "sequential"), np.asarray(prob.gamma),
                          body_order_columns(meta, prob.cfg))
        h0J, h0S = evJ.h0(theta), evS.h0(theta)
        hS, vS, _ = fit_ard(evS, h0S)
        hJ, vJ, info = fit_ard(evJ, h0J)
        v0 = evJ.value_and_grad(h0J)[0]
        # the sequential optimum, embedded in the joint parameterisation, is a feasible joint point
        vS_in_J = evJ.value_and_grad(np.concatenate([h0J[:3], hS]))[0]
    assert vS_in_J >= v0 - 1e-6
    assert vJ >= vS_in_J - 1e-4
    assert np.all(hJ >= evJ.lower - 1e-12) and np.all(hJ <= evJ.upper + 1e-12)


def test_posterior_variance_matches_blr_path_at_gamma_prior(tiny_linear_problem):
    """With a_k = log(1/sigma_c^2) for every group and sigma_q at theta, the ARD posterior is the
    existing BLR posterior: same mean, same row variances."""
    from ace_jax.fit.ard import ARDEvidence, ard_posterior, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.objective import posterior
    from ace_jax.fit.predict import _rows_mean_var
    from ace_jax.fit.rows import linear_rows
    from ace_jax.fit.stats import sufficient_statistics
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}
    with highest_precision():
        ev = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                         body_order_columns(meta, prob.cfg))
        post = ard_posterior(ev, ev.h0(theta), 1.0, meta)
        mu, Lc = posterior(theta, sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds), prob)
        b = jax.tree.map(lambda a: a[0], ds)
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, b)[0].F).reshape(-1, prob.cfg.len_basis)
        m_ref, v_ref = _rows_mean_var(jnp.asarray(Fr), mu, Lc)
    np.testing.assert_allclose(Fr @ post.mean, np.asarray(m_ref), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(post.var_rows(Fr), np.asarray(v_ref), rtol=1e-6, atol=1e-14)


def test_posterior_save_load_float32(tiny_linear_problem, tmp_path):
    from ace_jax.fit.ard import ARDEvidence, ARDPosterior, ard_posterior, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.rows import linear_rows
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}
    with highest_precision():
        ev = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                         body_order_columns(meta, prob.cfg))
        post = ard_posterior(ev, ev.h0(theta), 2.5, meta)
        post.save(tmp_path / "posterior.npz")
        back = ARDPosterior.load(tmp_path / "posterior.npz")
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
    a, b = post.forces_std(Fr), back.forces_std(Fr)
    assert back.kappa == 2.5 and back.meta["n_B"] == prob.cfg.n_B
    np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-12)


def test_laplace_reports_interior_psd(tiny_linear_problem):
    from ace_jax.fit.ard import ARDEvidence, ard_statistics, body_order_columns, fit_ard, laplace_hypers
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)]}
    with highest_precision():
        ev = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                         body_order_columns(meta, prob.cfg))
        h, _, _ = fit_ard(ev, ev.h0(theta))
        rep = laplace_hypers(ev, h)
    assert np.all(np.linalg.eigvalsh(rep["cov"]) >= -1e-12)
    assert np.all(np.isfinite(rep["std"]))


def test_fit_ard_info_reports_success_and_message(tiny_linear_problem):
    from ace_jax.fit.ard import ARDEvidence, ard_statistics, body_order_columns, fit_ard
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)]}
    with highest_precision():
        ev = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                         body_order_columns(meta, prob.cfg))
        _, _, info = fit_ard(ev, ev.h0(theta))
    assert isinstance(info["success"], bool) and isinstance(info["message"], str) and info["message"]


def test_ard_fit_warnings_flag_failure_and_bounds():
    from ace_jax.fit.ard import _ard_fit_warnings
    ok = {"success": True, "message": "CONVERGENCE", "at_bound": [False, False]}
    assert _ard_fit_warnings("full", ok, ["a", "b"]) == []
    bad = {"success": False, "message": "ABNORMAL", "at_bound": [False, True]}
    w = _ard_fit_warnings("full", bad, ["a", "b"])
    assert len(w) == 2 and "ABNORMAL" in w[0] and "b" in w[1]


def test_joint_statistics_are_linear_only_with_inducing_points(tiny_linear_problem):
    """Joint ARD statistics are the (L, L) linear Gram even for a hybrid problem with inducing
    points (M > 0): the residual columns never enter the ARD posterior."""
    from ace_jax.fit.ard import ard_statistics
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import descriptor_scale, select_inducing, site_features
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    with highest_precision():
        X, S = site_features(prob.model, prob.cfg, ds)
        ind = select_inducing(X, S, ds.node_z, ds.node_mask, 2, descriptor_scale(X, ds.node_mask))
        assert ind.XM.shape[0] > 0
        st0 = ard_statistics(theta, prob, ds, "joint")
        st1 = ard_statistics(theta, prob._replace(ind=ind), ds, "joint")
    L = prob.cfg.len_basis
    for G0, G1 in zip(st0.G, st1.G):
        assert G1.shape == (L, L)
        np.testing.assert_allclose(np.asarray(G1), np.asarray(G0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(st1.yy, st0.yy, rtol=1e-12)


def test_kappa_closed_form_minimises_nll():
    from scipy.optimize import minimize_scalar
    from ace_jax.fit.ard import kappa_closed_form
    rng = np.random.default_rng(1)
    s2 = rng.uniform(0.01, 1.0, 500)
    e2 = (3.2 ** 2) * s2 / 3 * rng.chisquare(3, 500)          # true kappa 3.2
    k = kappa_closed_form(e2, s2)
    nll = lambda lk: np.mean(e2 / (2 * np.exp(2 * lk) * s2 / 3) + 1.5 * np.log(np.exp(2 * lk) * s2 / 3))
    k_bf = float(np.exp(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x))
    assert abs(k - k_bf) < 1e-4 * k_bf and 2.9 < k < 3.5


def _pipe_cfg(**kw):
    """A uq='ard' linear-arm FitConfig for the ARD stage tests."""
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=30, ntest=8, batch=4, r0=2.35, arm="linear", uq="ard",
                opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False)
    return FitConfig(**{**base, **kw})


def test_ard_stage_fits_kappa_and_refits_on_all_training_data():
    from conftest import FIXTURE_DIR
    from ace_jax.fit.ard import predict_ard, run_ard_stage
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        pred = predict_ard(res.posterior, b.prob, d.ds_test)
        pred1 = predict_ard(res.posterior._replace(kappa=1.0), b.prob, d.ds_test)
    rep = res.report
    assert res.posterior.kappa > 0 and np.isfinite(res.posterior.kappa)
    assert rep["n_val_atoms"] > 0 and rep["n_fit_configs"] + rep["n_val_configs"] == len(d.train)
    assert abs(rep["val_rms_z_tempered"] - 1.0) < 1e-6          # kappa closed form on the val set
    assert rep["logev_full"] >= rep["logev_full_start"] - 1e-6
    assert np.isfinite(rep["val_nll_tempered"]) and rep["val_nll_tempered"] <= rep["val_nll_untempered"] + 1e-9
    # F_var tempered by kappa^2; E_var / V_var untempered; means independent of kappa
    k2 = res.posterior.kappa ** 2
    np.testing.assert_allclose(pred.F_var, k2 * pred1.F_var, rtol=1e-12)
    np.testing.assert_allclose(pred.E_var, pred1.E_var, rtol=1e-12)
    np.testing.assert_allclose(pred.V_var, pred1.V_var, rtol=1e-12)
    np.testing.assert_allclose(pred.F_mean, pred1.F_mean, rtol=1e-12)
    assert pred.F_mean.shape == (sum(len(c.numbers) for c in d.test), 3)
    assert pred.E_mean.shape == (len(d.test),) and np.all(pred.F_var >= 0)


def test_ard_stage_rejects_val_split_without_forces():
    from ace_jax.fit.ard import run_ard_stage
    from ace_jax.fit.pipeline import load_fit_data
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _pipe_cfg(ard_val_frac=0.2).validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    # strip every force label: the validation split then has nothing to fit kappa on
    d = d._replace(train=[c._replace(forces=None, w_F=0.0) for c in d.train])
    with pytest.raises(ValueError, match="ard_val_frac"):
        run_ard_stage(cfg, d, build_problem(cfg, d), None, log=lambda *a: None)


def test_ard_stage_logs_warning_on_failed_or_bounded_fit(monkeypatch):
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    real = ard.fit_ard

    def failing(ev, h0, cond_max=1e14, **kw):
        h, v, info = real(ev, h0, cond_max, **kw)
        return h, v, {**info, "success": False, "message": "ABNORMAL_TERMINATION_IN_LNSRCH",
                      "at_bound": [True] + [False] * (len(h) - 1)}

    monkeypatch.setattr(ard, "fit_ard", failing)
    lines = []
    with highest_precision():
        ard.run_ard_stage(cfg, d, b, default_prior(2.35).mu, log=lines.append)
    warns = [s for s in lines if "WARNING" in s]
    assert any("ABNORMAL" in s for s in warns) and any("log_sigma_E" in s for s in warns)
