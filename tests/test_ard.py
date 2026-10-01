import pytest
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision
from conftest import _orders


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
                opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False, ard_variance="kappa")
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


def test_ard_stage_reuses_cached_full_statistics(monkeypatch):
    """I1: the joint full refit reuses the objective's cached linear statistics (spec 3: "the cached
    M, b where available") -- same h, kappa and posterior as the recompute, one fewer statistics pass."""
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard, stats
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    calls = []
    real = stats.linear_statistics
    monkeypatch.setattr(stats, "linear_statistics", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    with highest_precision():
        obj = make_objective(cfg, d, b)
        theta = fit_map(cfg, d, b, obj, log=lambda *a: None).theta
        # the cached statistics are the joint ARD statistics (unweighted per-quantity G/b/yy/n): the
        # same function on the same data, jitted vs not -- equal up to summation order (~1e-15 of scale)
        joint = ard.ard_statistics(theta, b.prob, d.ds_train, "joint")
        for q, (G, bv, yy, n) in enumerate(zip(joint.G, joint.b, joint.yy, joint.n)):
            G, bv = np.asarray(G), np.asarray(bv)
            np.testing.assert_allclose(np.asarray(getattr(obj.lin, f"G_{'EFV'[q]}")), G, rtol=0,
                                       atol=1e-14 * np.abs(G).max())
            np.testing.assert_allclose(np.asarray(getattr(obj.lin, f"b_{'EFV'[q]}")), bv, rtol=0,
                                       atol=1e-14 * np.abs(bv).max())
            assert float(getattr(obj.lin, f"yy_{'EFV'[q]}")) == pytest.approx(yy, rel=1e-13)
            assert float(getattr(obj.lin, f"n_{'EFV'[q]}")) == n
        calls.clear()
        ref = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        n_recompute = len(calls)
        calls.clear()
        got = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None, full_stats=obj.lin)
        bc = ard.body_order_columns(d.meta, b.prob.cfg)
        ev_ref = ard.ARDEvidence(joint, np.asarray(b.prob.gamma), bc)
        ev_got = ard.ARDEvidence(ard.joint_ard_stats(obj.lin), np.asarray(b.prob.gamma), bc)
        v_ref, g_ref = ev_ref.value_and_grad(ref.posterior.h)
        v_got, g_got = ev_got.value_and_grad(ref.posterior.h)
        p_ref = ard.predict_ard(ref.posterior, b.prob, d.ds_test)
        p_got = ard.predict_ard(got.posterior, b.prob, d.ds_test)
    assert n_recompute == 2 and len(calls) == 1          # the subset only: the full refit reused the cache
    # the ~1e-15 summation-order difference, through cond(S) ~ 1e13, moves the evidence by ~1e-7 nats
    # and L-BFGS's stopping point along flat directions by ~1e-4: equal to the optimiser's resolution
    assert abs(v_got - v_ref) < 1e-6 and np.abs(g_got - g_ref).max() < 1e-5
    assert got.report["kappa_subset"] == pytest.approx(ref.report["kappa_subset"], rel=1e-12)   # subset stage unchanged
    assert got.posterior.kappa == pytest.approx(ref.posterior.kappa, rel=1e-4)   # full-posterior s^2: refit resolution
    assert got.report["logev_full"] == pytest.approx(ref.report["logev_full"], abs=1e-5)
    np.testing.assert_allclose(got.posterior.h, ref.posterior.h, atol=1e-3)
    np.testing.assert_allclose(np.asarray(p_got.F_var), np.asarray(p_ref.F_var), rtol=1e-4)
    Fm = np.asarray(p_ref.F_mean)
    np.testing.assert_allclose(np.asarray(p_got.F_mean), Fm, rtol=0, atol=1e-4 * np.abs(Fm).max())


def test_predict_ard_traces_the_chunked_rows_once(tiny_linear_problem, monkeypatch):
    """predict_ard over a multi-batch Dataset traces linear_rows_chunked ONCE.  Called eagerly,
    its fori_loop is retraced per batch with that batch's arrays baked in as constants: an XLA
    compile per batch (hours at the Cantor basis: ~200 prediction batches)."""
    from ace_jax.fit import rows
    from ace_jax.fit.ard import ARDEvidence, ard_posterior, ard_statistics, body_order_columns, predict_ard
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    assert ds.n_batches >= 2
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}
    with highest_precision():
        ev = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                         body_order_columns(meta, prob.cfg))
        post = ard_posterior(ev, ev.h0(theta), 2.0, meta)
        ref = predict_ard(post, prob, ds)
        n = {"traces": 0}
        orig = rows.linear_rows_chunked

        def counting(*a, **k):
            n["traces"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(rows, "linear_rows_chunked", counting)
        got = predict_ard(post, prob, ds)
    assert n["traces"] == 1
    np.testing.assert_allclose(np.asarray(got.F_var), np.asarray(ref.F_var), rtol=1e-12)


def test_ard_stage_kappa_is_refit_for_the_full_posterior(monkeypatch):
    """kappa calibrates the FULL-refit posterior, not the subset one: the held-out atoms' errors
    come from the subset model (honest), their s^2 from the full posterior (the one served).  The
    misspecification that kappa absorbs does not shrink on the refit while s^2 does, so a subset
    kappa would be ~sqrt(n_train / n_fit) too small (bench365: measured 1.105, predicted 1.118)."""
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    calls = []
    orig = ard.kappa_closed_form

    def spy(e2, s2):
        k = orig(e2, s2)
        calls.append((np.array(e2), np.array(s2), k))
        return k

    monkeypatch.setattr(ard, "kappa_closed_form", spy)
    cfg = _pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
    assert len(calls) == 2
    (e2_sub, s2_sub, k_sub), (e2_full, s2_full, k_full) = calls
    np.testing.assert_array_equal(e2_full, e2_sub)          # the same honest held-out errors
    assert s2_full.mean() < s2_sub.mean()                    # the refit saw the held-out configs
    assert res.posterior.kappa == k_full == res.report["kappa"]
    assert res.report["kappa_subset"] == k_sub and k_full > k_sub


def test_sandwich_scores_sum_to_the_prior_force_at_the_mean(ard_setup):
    """Stationarity: sum_c g~_c = D^-1 Lambda c = lam_prior * x at the posterior mean, so the
    residuals, their whitening and the cluster sums are exactly the posterior's own."""
    from ace_jax.fit.ard import sandwich_scores
    with highest_precision():
        prob, ds, ev, h, post = ard_setup
        G = sandwich_scores(post, prob, ds, ev.sigmas(h))
        _, _, lam, _ = ev._parts(jnp.asarray(h, float))
        x = post.mean / post.dinv                                                    # scaled mean D c
    n_cfg = int(np.asarray(ds.cfg_mask).sum())
    assert G.shape == (prob.cfg.len_basis, n_cfg)                                    # padded configs dropped
    np.testing.assert_allclose(G.sum(1), np.asarray(lam) * x, rtol=1e-6, atol=1e-8 * np.abs(G).max())


def test_sandwich_variance_matches_dense_reference(ard_setup):
    """lam^2 ||Q^T phi~||^2 == lam^2 phi A^-1 M A^-1 phi^T with A and M built densely in the original
    coordinates (M = sum over configs of the outer product of the summed residual-weighted rows)."""
    from ace_jax.fit.ard import sandwich_factor, sandwich_scores
    with highest_precision():
        prob, ds, ev, h, post = ard_setup
        G = sandwich_scores(post, prob, ds, ev.sigmas(h))
        post = post._replace(Q=sandwich_factor(post, G), lam=1.7)
        Ms, _, lam, _ = ev._parts(jnp.asarray(h, float))
        D = 1.0 / post.dinv
        A = np.asarray(Ms + jnp.diag(lam)) * D[:, None] * D[None, :]                 # unscaled A
        Gu = G * D[:, None]                                                          # unscaled scores
        Sig = np.linalg.solve(A, np.linalg.solve(A, Gu @ Gu.T).T)                    # A^-1 M A^-1
        b0 = jax.tree.map(lambda a: a[0], ds)
        from ace_jax.fit.rows import linear_rows
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, b0)[0].F).reshape(-1, prob.cfg.len_basis)
        got = post.force_var_rows(Fr)
    ref = 1.7 ** 2 * np.einsum("nl,lm,nm->n", Fr, Sig, Fr)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-12 * ref.max())


def test_posterior_schema2_roundtrip_and_schema1_loads(ard_setup, tmp_path):
    from ace_jax.fit.ard import ARDPosterior, sandwich_factor, sandwich_scores
    from ace_jax.fit.rows import linear_rows
    with highest_precision():
        prob, ds, ev, h, post = ard_setup
        post = post._replace(Q=sandwich_factor(post, sandwich_scores(post, prob, ds, ev.sigmas(h))), lam=3.0)
        post.save(tmp_path / "p.npz")
        back = ARDPosterior.load(tmp_path / "p.npz")
        Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
        np.testing.assert_allclose(back.forces_std(Fr), post.forces_std(Fr), rtol=1e-3)
        assert back.lam == 3.0 and back.Q.dtype == np.float64
        # a schema-1 file (no Q / lam) still loads and serves kappa variance
        z = dict(np.load(tmp_path / "p.npz"))
        z.pop("Q"); z.pop("lam"); z["schema"] = np.array(1)
        np.savez(tmp_path / "p1.npz", **z)
        old = ARDPosterior.load(tmp_path / "p1.npz")
        assert old.Q is None and old.lam == 1.0
        np.testing.assert_allclose(old.forces_std(Fr), old.kappa * np.sqrt(
            old.var_rows(Fr.reshape(-1, Fr.shape[-1])).reshape(-1, 3).sum(1)), rtol=1e-12)
        zero = np.zeros((2, 3, prob.cfg.len_basis))
        assert np.all(post.forces_std(zero) == 0.0)                                  # no neighbours: 0, not NaN


def test_ard_stage_sandwich_variance_and_lam_rule(monkeypatch):
    """Default variance: Q from the full refit's training residuals, lam by the kappa refit rule
    (held-out subset errors against the served posterior's sandwich variance), F_var = lam^2 sandwich."""
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    calls = []
    orig = ard.kappa_closed_form
    monkeypatch.setattr(ard, "kappa_closed_form", lambda e2, s2: calls.append((np.array(e2), np.array(s2))) or orig(e2, s2))
    cfg = _pipe_cfg(ard_variance="sandwich").validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        pred = ard.predict_ard(res.posterior, b.prob, d.ds_test)
        pred1 = ard.predict_ard(res.posterior._replace(lam=1.0), b.prob, d.ds_test)
    post, rep = res.posterior, res.report
    assert post.Q is not None and post.Q.shape == (b.prob.cfg.len_basis, len(d.train))
    assert isinstance(post.Q, jax.Array)          # stage keeps Q on-device: no per-batch host->device copy
    assert rep["variance"] == "sandwich" and rep["n_clusters"] == len(d.train)
    # kappa_sub, kappa, lam (own cluster left out), lam_incl_own: the same e2
    assert len(calls) == 4 and np.array_equal(calls[2][0], calls[0][0])
    assert rep["lam_incl_own"] == orig(*calls[3])
    assert post.lam == orig(*calls[2]) == rep["lam"] and abs(rep["val_rms_z_sandwich"] - 1.0) < 1e-6
    np.testing.assert_allclose(pred.F_var, post.lam ** 2 * pred1.F_var, rtol=1e-12)


def test_ard_stage_sandwich_in_sequential_mode():
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    theta_ls = lambda t: [float(getattr(t, f"log_sigma_{q}")) for q in "EFV"]
    cfg = _pipe_cfg(ard_variance="sandwich", ard_mode="sequential").validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        post = res.posterior
        # rebuild the stage's full-refit evidence: its sequential sigmas are the fixed linear MAP ones
        ev = ard.ARDEvidence(ard.ard_statistics(theta, b.prob, d.ds_train, "sequential"),
                             np.asarray(b.prob.gamma), ard.body_order_columns(d.meta, b.prob.cfg))
        h = np.asarray(res.report["h"])
        Ms, _, lam_prior, _ = ev._parts(jnp.asarray(h, float))
        G = np.asarray((Ms + jnp.diag(lam_prior)) @ post.Q)          # G~ = S Q: the stage's own scores
        x = post.mean / post.dinv                                     # scaled mean D c
    assert post.Q.shape == (b.prob.cfg.len_basis, len(d.train)) and np.isfinite(post.lam) and post.lam > 0
    # stationarity sum_c g~_c = lam_prior * x: the scores are whitened with ev.sigmas(h) -- the
    # sequential mode's fixed MAP sigmas, not the (absent) fitted ones
    np.testing.assert_allclose(G.sum(1), np.asarray(lam_prior) * x, rtol=1e-6, atol=1e-8 * np.abs(G).max())
    assert np.array_equal(ev.sigmas(h), np.exp(np.asarray(theta_ls(theta))))


def test_sandwich_scores_columns_follow_config_order(ard_setup):
    """Column c of G~ is config c: the batched dataset (3 configs per batch) and one config per batch
    give the same score matrix column by column, so no cid permutation / cross-batch misassignment."""
    from ace_jax.eval import load
    from ace_jax.fit.ard import sandwich_scores
    from ace_jax.fit.data import build_dataset, load_configs
    from conftest import FIXTURE_DIR
    _, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(FIXTURE_DIR / "si_tiny_train.xyz", "dft_energy", "dft_force", "dft_virial")[:6]
    with highest_precision():
        prob, ds, ev, h, post = ard_setup
        sig = ev.sigmas(h)
        G3 = sandwich_scores(post, prob, build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=3), sig)
        G1 = sandwich_scores(post, prob, build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=1), sig)
    assert G3.shape == G1.shape == (prob.cfg.len_basis, len(configs))
    live = np.abs(G1).max(0) > 0                         # config 0 is an isolated atom: zero rows, zero score
    assert live.sum() >= len(configs) - 1
    Gl = G1[:, live]                                     # the live columns are distinct: a permutation shows
    assert all(not np.allclose(Gl[:, i], Gl[:, j]) for i in range(Gl.shape[1]) for j in range(i))
    for c in range(len(configs)):
        np.testing.assert_allclose(G3[:, c], G1[:, c], rtol=1e-10, atol=1e-14 * np.abs(G1).max())


def test_ard_stage_lam_leaves_out_each_held_out_atoms_own_cluster(monkeypatch):
    """lam is fitted against m^2 WITHOUT the held-out atom's own configuration's cluster: a genuinely
    new configuration has no such term, so keeping it biases lam low.  Brute force: per held-out
    config, zero its own column of Q (column idx[j] of the train order), recompute m^2 from that
    config's force rows alone, and lam = kappa_closed_form(e2, m2)."""
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.data import build_dataset
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.rows import linear_rows
    calls = []
    orig = ard.kappa_closed_form
    monkeypatch.setattr(ard, "kappa_closed_form", lambda e2, s2: calls.append(np.array(e2)) or orig(e2, s2))
    cfg = _pipe_cfg(ard_variance="sandwich").validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        post, rep = res.posterior, res.report
        L = b.prob.cfg.len_basis
        idx = np.random.default_rng(cfg.seed).permutation(len(d.train))
        nval = max(1, int(round(cfg.ard_val_frac * len(d.train))))
        Q, dinv = np.asarray(post.Q), np.asarray(post.dinv)
        m2_loo, m2_all = [], []
        for j in range(nval):                               # one config at a time: no batch bookkeeping
            bt = jax.tree.map(lambda a: a[0], build_dataset([d.train[idx[j]]], d.meta, d.E0, 1))
            F = np.asarray(linear_rows(b.prob.model, b.prob.cfg, bt)[0].F)[np.asarray(bt.w_F) > 0]
            F = F[np.abs(F).reshape(len(F), -1).max(1) > 0]                       # the stage's `ok` atoms
            Ft = F.reshape(-1, L) * dinv[None, :]
            Qo = Q.copy()
            Qo[:, idx[j]] = 0.0
            m2_loo.append(((Ft @ Qo) ** 2).reshape(-1, 3 * Q.shape[1]).sum(1))
            m2_all.append(((Ft @ Q) ** 2).reshape(-1, 3 * Q.shape[1]).sum(1))
    m2_loo, m2_all, e2 = np.concatenate(m2_loo), np.concatenate(m2_all), calls[0]
    assert len(m2_loo) == len(e2) == rep["n_val_atoms"]
    np.testing.assert_allclose(post.lam, orig(e2, m2_loo), rtol=1e-8)
    np.testing.assert_allclose(rep["lam_incl_own"], orig(e2, m2_all), rtol=1e-8)
    assert rep["lam"] == post.lam and rep["lam"] > rep["lam_incl_own"]
    assert abs(rep["val_rms_z_sandwich"] - 1.0) < 1e-6                           # z of the served lam


def test_posterior_chol_and_sandwich_factor_stay_on_device(ard_setup):
    """chol (1.8 GB at L = 15k) and Q are float64 device arrays: var_rows / misspec_var_rows run once
    per batch and jnp.asarray of a numpy factor re-uploads it on every call."""
    from ace_jax.fit.ard import sandwich_factor, sandwich_scores
    with highest_precision():
        prob, ds, ev, h, post = ard_setup
        Q = sandwich_factor(post, sandwich_scores(post, prob, ds, ev.sigmas(h)))
    assert isinstance(post.chol, jax.Array) and post.chol.dtype == np.float64
    assert isinstance(Q, jax.Array) and Q.dtype == np.float64


def _table(G=2):
    from ace_jax.fit.conformal import GroupTable
    return GroupTable(alpha=0.1, n_min=20, lam_rms=np.array([2.0, 3.0]), q=np.array([5.0, 9.0]),
                      r=np.array([1.0, 1.2]), n_cfg=np.array([30, 30]), n_cfg_val=np.array([30, 30]),
                      n_cfg_cal=np.array([0, 0]), n_atoms=np.array([100, 100]), merged=[])


def test_schema3_served_quantities(ard_setup, tmp_path):
    from ace_jax.fit.ard import ARDPosterior
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores, shape_factor
    from ace_jax.fit.rows import linear_rows
    prob, ds, ev, h, post = ard_setup
    rc, K = row_clusters(ds, None, float("inf"))
    R = shape_factor(post, press_scores(post, prob, ds, rc, K, ev.sigmas(h))[0])
    tab = _table()
    p = post._replace(R=R, group_consts={"r1": 3.0, "z_star": 4, "edges": []}, group_table=tab.to_dict())
    Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
    g = np.arange(Fr.shape[0]) % 2
    V = p.atom_shape(Fr)
    v = np.trace(V, axis1=1, axis2=2)
    np.testing.assert_allclose(p.forces_std(Fr, g), tab.lam_rms[g] * np.sqrt(v), rtol=1e-12)
    np.testing.assert_allclose(p.forces_cov(Fr, g), tab.lam_rms[g, None, None] ** 2 * V, rtol=1e-12)
    np.testing.assert_allclose(np.trace(p.forces_cov(Fr, g), axis1=1, axis2=2), p.forces_std(Fr, g) ** 2,
                               rtol=1e-12)
    np.testing.assert_allclose(p.forces_q(Fr, g), tab.q[g] * np.sqrt(v / 3), rtol=1e-12)
    pa = p._replace(force_shape="aniso")
    lam_max = np.linalg.eigvalsh(V + p.eps * (v / 3)[:, None, None] * np.eye(3)).max(1)
    np.testing.assert_allclose(pa.forces_q(Fr, g), tab.q[g] * np.sqrt(lam_max), rtol=1e-10)
    np.testing.assert_allclose(pa.forces_q_mahal(g), tab.q[g])
    p.save(tmp_path / "p.npz")
    back = ARDPosterior.load(tmp_path / "p.npz")
    np.testing.assert_allclose(back.forces_std(Fr, g), p.forces_std(Fr, g), rtol=1e-3)   # R stored float32
    assert back.group_table["q"] == tab.q.tolist() and back.force_shape == "iso"


def test_schema2_serves_scalar(ard_setup, tmp_path):
    from ace_jax.fit.ard import ARDPosterior, sandwich_factor, sandwich_scores
    from ace_jax.fit.rows import linear_rows
    prob, ds, ev, h, post = ard_setup
    old = post._replace(Q=sandwich_factor(post, sandwich_scores(post, prob, ds, ev.sigmas(h))), lam=2.0)
    old.save(tmp_path / "p.npz")
    z = dict(np.load(tmp_path / "p.npz"))
    z["schema"] = np.array(2)
    for k in [k for k in z if k.startswith(("R", "group_", "cal_", "support", "force_shape", "eps"))]:
        z.pop(k)
    np.savez(tmp_path / "p2.npz", **z)
    p2 = ARDPosterior.load(tmp_path / "p2.npz")
    Fr = np.asarray(linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0].F)
    np.testing.assert_allclose(p2.forces_std(Fr), old.forces_std(Fr), rtol=1e-3)
    for f in (lambda: p2.forces_q(Fr, None), lambda: p2.forces_cov(Fr, None), lambda: p2.forces_q_mahal(None)):
        with pytest.raises(ValueError, match="refit with --uq ard"):
            f()
