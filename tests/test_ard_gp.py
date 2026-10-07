"""--uq ard-gp: the ARD jackknife-sandwich posterior over the GP arm's joint design [B | k(B, B_M)] at the
fixed theta_MAP (docs/dev/specs/2026-10-05-gp-discrepancy-design.md, Phase 3)."""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ace_jax.eval import highest_precision  # noqa: E402


def _meta(prob):
    from conftest import _orders
    return {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}


@pytest.fixture
def gp_ev(tiny_gp_problem):
    from ace_jax.fit.ard import (ARDEvidence, ard_gamma, ard_statistics, body_order_columns,
                                 gp_body_columns)
    from ace_jax.fit.prior_root import prior_root
    prob, ds, theta = tiny_gp_problem
    M = prob.ind.XM.shape[0]
    with highest_precision():
        st = ard_statistics(theta, prob, ds, "joint", columns="joint")
        ev = ARDEvidence(st, ard_gamma(prob), gp_body_columns(body_order_columns(_meta(prob), prob.cfg), M),
                         root=prior_root(prob, theta))
    return prob, ds, theta, ev


def test_joint_statistics_have_the_inducing_columns(gp_ev):
    prob, _, _, ev = gp_ev
    L, M = prob.cfg.len_basis, prob.ind.XM.shape[0]
    assert ev._data[0][1].shape == (L + M, L + M)          # G_F over the joint design
    assert ev.groups[0] == -1 and len(ev.groups) == 1 + len(set(ev.body_col[:L]) - {0})


def _gp_dense_rows(prob, ds, theta, sig):
    """Whitened joint rows Psi (n, L+M), targets y~ and the live-configuration index of each row, in
    batch / configuration / (E, F x 3 per atom, V x 6) order.  Rows of zero structural weight (a
    configuration without virials, say) are absent, as in the statistics."""
    from ace_jax.fit.rows import batch_rows
    P, Y, cf, g = [], [], [], 0

    def add(phi, yv, w):
        if w > 0:
            P.append(np.asarray(phi) * w); Y.append(float(yv) * w); cf.append(g)

    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        r = batch_rows(theta, prob.spec, prob.model, prob.ind, prob.cfg, b)
        for c in np.flatnonzero(np.asarray(b.cfg_mask)):
            add(r.E[c], b.y_E[c], float(b.w_E[c]) / sig[0])
            for n in np.flatnonzero(np.asarray(b.node_cfg) == c):
                for a in range(3):
                    add(r.F[n, a], b.y_F[n, a], float(b.w_F[n]) / sig[1])
            for v in range(6):
                add(r.V[c, v], b.y_V[c, v], float(b.w_V[c]) / sig[2])
            g += 1
    return np.array(P), np.array(Y), np.array(cf)


def _obs_space_reference(prob, ds, theta):
    """(LML, posterior mean) of the hybrid model at theta from the observation-space covariance
    C~ = Psi Lambda^-1 Psi^T + I (Lambda = blkdiag(Gamma^2/sigma_c^2, K_MM)): independent of the
    weight-space code, and well-conditioned (the identity), unlike G + Lambda on this fixture."""
    from ace_jax.fit.kernels import K_MM
    from ace_jax.fit.stats import sufficient_statistics
    sig = np.exp([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
    Psi, y, _ = _gp_dense_rows(prob, ds, theta, sig)
    ind = prob.ind
    L = prob.cfg.len_basis
    Kinv = np.linalg.inv(np.asarray(K_MM(theta, prob.spec, ind.XM, ind.SM, ind.ZM, ind.embed)))
    lin_var = float(np.exp(2 * theta.log_sigma_c)) / np.asarray(prob.gamma) ** 2
    PL = Psi[:, :L] * lin_var[None, :]
    PG = Psi[:, L:] @ Kinv
    C = PL @ Psi[:, :L].T + PG @ Psi[:, L:].T + np.eye(len(y))
    a = np.linalg.solve(C, y)
    st = sufficient_statistics(theta, prob.spec, prob.model, ind, prob.cfg, ds)
    logtau = sum(float(getattr(st, f"logw_{q}")) - float(getattr(st, f"n_{q}")) * 2 * np.log(sig[i])
                 for i, q in enumerate("EFV"))
    lml = -0.5 * y @ a - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * len(y) * np.log(2 * np.pi) + 0.5 * logtau
    return lml, np.concatenate([PL.T @ a, PG.T @ a]), st


def test_evidence_at_h0_is_the_gp_lml(gp_ev):
    """h0 (a_k = -2 log sigma_c on the body orders, a_GP = 0, the MAP's sigma_q) reproduces the hybrid
    model's log marginal likelihood at theta: the ARD prior at h0 IS blkdiag(Gamma^2/sigma_c^2, K_MM).
    ARDEvidence's constant leaves out the structural log weights and the 2 pi term:
    LML = logev + 1/2 sum_q logw_q - 1/2 N log(2 pi)."""
    prob, ds, theta, ev = gp_ev
    with highest_precision():
        lml, _, st = _obs_space_reference(prob, ds, theta)
        v, _ = ev.value_and_grad(ev.h0(theta))
    N = sum(float(getattr(st, f"n_{q}")) for q in "EFV")
    logw = sum(float(getattr(st, f"logw_{q}")) for q in "EFV")
    assert v + 0.5 * logw - 0.5 * N * np.log(2 * np.pi) == pytest.approx(lml, rel=1e-8)


def test_posterior_mean_at_h0_is_the_gp_map_mean(gp_ev):
    """The linear coefficients to 1e-6; the inducing weights through what they predict (Psi mean): on their
    own they are only as determined as K_MM is conditioned (cond 7e8 here), in both the reference and S."""
    from ace_jax.fit.ard import ard_posterior
    prob, ds, theta, ev = gp_ev
    L = prob.cfg.len_basis
    with highest_precision():
        _, mu, _ = _obs_space_reference(prob, ds, theta)
        post = ard_posterior(ev, ev.h0(theta), 1.0, _meta(prob), theta=theta)
        Psi, _, _ = _gp_dense_rows(prob, ds, theta, np.exp([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"]))
    np.testing.assert_allclose(post.mean[:L], mu[:L], rtol=1e-6, atol=1e-8 * np.abs(mu[:L]).max())
    np.testing.assert_allclose(Psi @ post.mean, Psi @ mu, rtol=1e-6, atol=1e-8 * np.abs(Psi @ mu).max())


def test_gp_evidence_gradient_matches_finite_differences(gp_ev):
    _, _, theta, ev = gp_ev
    h = ev.h0(theta) + 0.1
    _, g = ev.value_and_grad(h)
    for i in range(len(h)):
        e = np.zeros_like(h)
        e[i] = 1e-3          # 1e-5 is roundoff-limited on the noise scales (~1e-7 nats of evidence noise)
        fd = (ev.value_and_grad(h + e)[0] - ev.value_and_grad(h - e)[0]) / 2e-3
        assert g[i] == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_gp_posterior_save_load_roundtrip(gp_ev, tmp_path):
    from ace_jax.fit.ard import ARDPosterior, ard_posterior
    prob, _, theta, ev = gp_ev
    post = ard_posterior(ev, ev.h0(theta), 1.0, _meta(prob), theta=theta)
    post.save(tmp_path / "p.npz", dtype=np.float64)
    back = ARDPosterior.load(tmp_path / "p.npz")
    assert back.prior_root.M == prob.ind.XM.shape[0]
    np.testing.assert_array_equal(np.asarray(back.prior_root.U), np.asarray(post.prior_root.U))
    np.testing.assert_array_equal(back.gp_theta, post.gp_theta)


def _A_gp(post):
    """A = R0^T S R0 from the posterior's factor and prior root (R0 recovered from R0^-1 = rows(I))."""
    root = post.prior_root
    R0 = np.linalg.inv(np.asarray(root.rows(np.eye(root.width))))
    S = np.asarray(post.chol) @ np.asarray(post.chol).T
    return R0.T @ S @ R0


def _row_cluster_labels(ds, rc, sig):
    """The cluster id of each row of _gp_dense_rows (same order, zero-weight rows dropped)."""
    out = []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a, i=i: a[i], ds)
        for c in np.flatnonzero(np.asarray(b.cfg_mask)):
            if float(b.w_E[c]) > 0:
                out.append(rc[i]["E"][c])
            for n in np.flatnonzero(np.asarray(b.node_cfg) == c):
                if float(b.w_F[n]) > 0:
                    out += [rc[i]["F"][n]] * 3
            if float(b.w_V[c]) > 0:
                out += [rc[i]["V"][c]] * 6
    return np.array(out)


@pytest.fixture
def gp_post(gp_ev):
    from ace_jax.fit.ard import ard_posterior
    prob, ds, theta, ev = gp_ev
    h = ev.h0(theta)
    with highest_precision():
        post = ard_posterior(ev, h, 1.0, _meta(prob), theta=theta)
    return prob, ds, theta, ev, h, post


@pytest.mark.parametrize("ell", [float("inf"), 6.0])
def test_gp_press_equals_exact_deletion(gp_post, ell):
    """c - c_(-k) = A^-1 g~_k for every cluster (whole configurations, and spatial blocks), h fixed, over
    the joint design: the spec's PRESS/DFBETA identity, compared through the fitted values Psi (c - c_(-k))
    (the inducing weights alone are only as determined as cond(K_MM) allows; see the mean test)."""
    from ace_jax.fit.ard import rows_fn_for
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, theta, ev, h, post = gp_post
    sig = ev.sigmas(h)
    Psi, y, _ = _gp_dense_rows(prob, ds, theta, sig)
    A = _A_gp(post)
    c = np.asarray(post.mean)
    configs = None
    if np.isfinite(ell):              # spatial blocks need the configurations: the fixture's first 6
        from conftest import FIXTURE_DIR
        from ace_jax.fit.data import load_configs
        configs = load_configs(FIXTURE_DIR / "si_tiny_train.xyz", "dft_energy", "dft_force", "dft_virial")[:6]
    rc, K = row_clusters(ds, configs, ell)
    assert K > 6 if np.isfinite(ell) else K == 6       # spatial blocks split some configuration
    with highest_precision():
        G, _ = press_scores(post, prob, ds, rc, K, sig, rows_fn=rows_fn_for(prob, theta))
    labels = _row_cluster_labels(ds, rc, sig)
    ref, got = [], []
    for k in range(K):
        m = labels == k
        ck = np.linalg.solve(A - Psi[m].T @ Psi[m], Psi[~m].T @ y[~m])
        ref.append(Psi @ (c - ck)); got.append(Psi @ np.linalg.solve(A, G[:, k]))
    ref, got = np.array(ref), np.array(got)
    # one scale for every cluster: a cluster with no rows of weight has c - c_(-k) = 0 to roundoff
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6 * np.abs(ref).max())


def test_gp_push_through_matches_exact(gp_post):
    from ace_jax.fit.ard import rows_fn_for
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores
    prob, ds, theta, ev, h, post = gp_post
    rc, K = row_clusters(ds, None, float("inf"))
    f = rows_fn_for(prob, theta)
    with highest_precision():
        Ge, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), mode="exact", rows_fn=f)
        Gp, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), mode="pushthrough", rows_fn=f)
    np.testing.assert_allclose(Gp, Ge, rtol=1e-7, atol=1e-9 * np.abs(Ge).max())


def test_ard_gp_shape_is_rotation_equivariant(gp_post):
    """V(R x) = R V(x) R^T for the PRESS shape over the joint rows (Review Focus 5)."""
    from scipy.spatial.transform import Rotation
    from ace_jax.fit.ard import rows_fn_for
    from ace_jax.fit.clusters import row_clusters
    from ace_jax.fit.jackknife import press_scores, shape_factor
    prob, ds, theta, ev, h, post = gp_post
    rc, K = row_clusters(ds, None, float("inf"))
    f = rows_fn_for(prob, theta)
    with highest_precision():
        G, _ = press_scores(post, prob, ds, rc, K, ev.sigmas(h), rows_fn=f)
        post = post._replace(R=shape_factor(post, G))
        b = jax.tree.map(lambda a: a[0], ds)
        Rm = Rotation.from_euler("zyx", [0.3, -0.7, 1.1]).as_matrix()
        V0 = post.atom_shape(np.asarray(f(b).F))
        V1 = post.atom_shape(np.asarray(f(b._replace(rij=b.rij @ Rm.T)).F))
    live = np.asarray(b.node_mask)
    np.testing.assert_allclose(V1[live], np.einsum("ab,nbc,dc->nad", Rm, V0[live], Rm), rtol=1e-8,
                               atol=1e-10 * np.abs(V0).max())


def _gp_pipe_cfg(**kw):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=30, ntest=8, batch=4, r0=2.35, arm="gp", m_per_species=8,
                uq="ard-gp", opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False,
                ard_val_frac=0.4, ard_n_min=50, ard_force_shape="aniso")
    return FitConfig(**{**base, **kw})


def _gp_stage(**kw):
    from conftest import FIXTURE_DIR
    from ace_jax.fit import ard
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _gp_pipe_cfg(**kw).validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        res = ard.run_ard_stage(cfg, d, b, theta, log=lambda *a: None)
        pred = ard.predict_ard(res.posterior, b.prob, d.ds_test, rows_fn=ard.rows_fn_for(b.prob, theta))
    return d, b, theta, res, pred


def test_stage_ard_gp_press_shape_and_group_scales():
    d, b, theta, res, pred = _gp_stage()
    post, rep = res.posterior, res.report
    M = b.prob.ind.XM.shape[0]
    assert post.prior_root.M == M and post.R.shape[0] == b.prob.cfg.len_basis + M
    assert "a_GP" in rep["h_names"] and rep["gp_cols"] == M
    assert np.isfinite(post.group_table["q"]).all()
    assert np.isfinite(pred.F_var).all() and np.all(pred.F_var >= 0)


def test_stage_ard_gp_sequential_shared_noise():
    """Review Focus 3: shared noise forces sequential ARD; the GP-arm stage runs end to end."""
    _, _, _, res, pred = _gp_stage(noise="shared", ard_mode="sequential")
    assert res.report["mode"] == "sequential" and np.isfinite(pred.F_var).all()


def test_ard_gp_with_no_inducing_columns_is_the_linear_stage(ard_map):
    """M = 0 reduces to the linear-arm result: the ard-gp stage (joint statistics via
    sufficient_statistics, the PriorRoot path, rows_fn_for) on a problem with no inducing columns gives the
    --uq ard posterior.  The two stream the statistics through different code (sufficient_statistics against
    the bounded linear statistics), so equality is to summation-order roundoff, not bitwise."""
    from ace_jax.fit import ard
    from conftest import ard_pipe_cfg
    d, b, theta = ard_map
    kw = dict(ard_variance="sandwich", ard_val_frac=0.4, ard_n_min=50)
    with highest_precision():
        lin = ard.run_ard_stage(ard_pipe_cfg(**kw).validate(), d, b, theta, log=lambda *a: None)
        gp_cfg = ard_pipe_cfg(**kw)
        gp_cfg.uq = "ard-gp"                    # the stage's ard-gp branch, bypassing validate()'s arm check
        gp = ard.run_ard_stage(gp_cfg, d, b, theta, log=lambda *a: None)
    P, Q = lin.posterior, gp.posterior
    np.testing.assert_allclose(Q.mean, P.mean, rtol=1e-9, atol=1e-12 * np.abs(P.mean).max())
    S = lambda p: np.asarray(p.R) @ np.asarray(p.R).T
    np.testing.assert_allclose(S(Q), S(P), rtol=1e-7, atol=1e-10 * np.abs(S(P)).max())
    for k in ("lam_rms", "q"):
        np.testing.assert_allclose(Q.group_table[k], P.group_table[k], rtol=1e-7)


@pytest.mark.parametrize("kw, msg", [(dict(arm="linear", m_per_species=0), "ard-gp"),
                                     (dict(_shape_variant="legacy", _score_source="mixed"), "legacy"),
                                     (dict(ard_variance="dtc"), "ard_variance"),     # dropped: it failed acceptance
                                     (dict(uq="ard", arm="linear", m_per_species=0, ard_variance="dtc"), "ard_variance")])
def test_ard_gp_config_refusals(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _gp_pipe_cfg(**kw).validate()


def test_ard_gp_fit_writes_ard_mean_gp_model(tmp_path):
    """An ard-gp fit writes gp_model.npz whose one posterior is the ARD posterior (mean, chol(A) with
    A = R0^T S R0), so GPCalculator's mean is the served posterior's, and posterior.npz beside it."""
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import fit, load_fit_data, write_outputs
    from ace_jax.fit.pipeline.export import load_gp_model
    cfg = _gp_pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    res = fit(cfg, d, log=lambda *a: None)
    write_outputs(res, tmp_path, layout=("run", "cli"), log=lambda *a: None)
    fg, _ = load_gp_model(tmp_path / "gp_model.npz")
    mu, L = fg.posteriors[0]
    np.testing.assert_array_equal(np.asarray(mu), res.ard.posterior.mean)
    root = res.ard.posterior.prior_root
    R0 = np.linalg.inv(np.asarray(root.rows(np.eye(root.width))))
    S = np.asarray(res.ard.posterior.chol) @ np.asarray(res.ard.posterior.chol).T
    A = R0.T @ S @ R0
    np.testing.assert_allclose(np.asarray(L) @ np.asarray(L).T, A, rtol=1e-8, atol=1e-10 * np.abs(A).max())
    assert np.allclose(np.triu(np.asarray(L), 1), 0.0)
    assert (tmp_path / "posterior.npz").exists()
    np.testing.assert_array_equal(np.asarray(fg.draws[0]), res.ard.posterior.gp_theta)


@pytest.fixture(scope="module")
def gp_fit_dir(tmp_path_factory):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import fit, load_fit_data, write_outputs
    out = tmp_path_factory.mktemp("ardgp")
    cfg = _gp_pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    res = fit(cfg, d, log=lambda *a: None)
    write_outputs(res, out, layout=("run", "cli"), log=lambda *a: None)
    return out, d, res


def _atoms0():
    from ase import Atoms
    from conftest import FIXTURE_DIR
    from ace_jax.fit.xyz import read_extxyz
    f = next(f for f in read_extxyz(FIXTURE_DIR / "si_tiny_train.xyz") if len(f.numbers) > 1)  # [0] is an isolated atom
    return Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)


def test_gp_calculator_serves_the_ard_posterior(gp_fit_dir):
    from ace_jax.calc.gp import GPCalculator
    out, d, res = gp_fit_dir
    calc = GPCalculator.from_file(out / "gp_model.npz", posterior=out / "posterior.npz")
    at = _atoms0()
    at.calc = calc
    sd = calc.get_property("forces_std", at)
    cov = calc.get_property("forces_cov", at)
    assert sd.shape == (len(at),) and cov.shape == (len(at), 3, 3)
    np.testing.assert_allclose(sd ** 2, np.trace(cov, axis1=1, axis2=2), rtol=1e-6)
    assert np.isfinite(calc.get_property("forces_q", at)).all()
    assert calc.get_property("forces_group", at).dtype.kind == "i"
    assert calc.get_property("forces_q_mahal", at).shape == (len(at),)       # the fixture's posterior is aniso


def test_gp_calculator_refuses_mismatched_posteriors(gp_fit_dir, tmp_path):
    """Review Focus 1: an ard-gp posterior given to ACECalculator, a linear one to GPCalculator, or one
    from a different fit is refused with a message naming the fix."""
    from conftest import FIXTURE_DIR
    from ace_jax import ACECalculator
    from ace_jax.calc.gp import GPCalculator
    from test_ard import _stage           # tests/ is on sys.path under pytest
    out, _, _ = gp_fit_dir
    with pytest.raises(ValueError, match="ard-gp"):
        ACECalculator(str(FIXTURE_DIR / "si_fitted.npz"), posterior=str(out / "posterior.npz"))
    lin = tmp_path / "lin_post.npz"
    _stage(ard_force_shape="aniso", ard_val_frac=0.4, ard_n_min=50)[1].posterior.save(lin)
    with pytest.raises(ValueError, match="linear"):
        GPCalculator.from_file(out / "gp_model.npz", posterior=lin)
    other = tmp_path / "other.npz"
    z = dict(np.load(out / "posterior.npz"))
    z["mean"] = z["mean"] * 1.01
    np.savez(other, **z)
    with pytest.raises(ValueError, match="same fit"):
        GPCalculator.from_file(out / "gp_model.npz", posterior=other)


def test_gp_calculator_served_rows_are_chunked(gp_fit_dir, monkeypatch):
    """Review Focus 4: a tiny ROWS_EDGE_BUDGET forces the node-chunked rows; served values are unchanged."""
    from ace_jax.calc.gp import GPCalculator
    from ace_jax.fit import rows
    out, _, _ = gp_fit_dir
    at = _atoms0()
    ref = GPCalculator.from_file(out / "gp_model.npz", posterior=out / "posterior.npz")
    at.calc = ref
    a = ref.get_property("forces_cov", at)
    monkeypatch.setattr(rows, "ROWS_EDGE_BUDGET", 1000)
    # the served joint rows must actually take the node-chunked path under the small budget: record the
    # chunk sizes rows_node_chunk returns while ard's joint rows are traced
    import inspect
    seen, orig = [], rows.rows_node_chunk

    def spy(*args, **kw):
        nc = orig(*args, **kw)
        if any(f.function == "_joint_rows_body" for f in inspect.stack()):
            seen.append(nc)
        return nc
    monkeypatch.setattr(rows, "rows_node_chunk", spy)
    small = GPCalculator.from_file(out / "gp_model.npz", posterior=out / "posterior.npz")
    at = _atoms0()
    at.calc = small
    b = small.get_property("forces_cov", at)
    assert seen and all(nc is not None for nc in seen), seen      # traced afresh, and chunked
    np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-14 * np.abs(a).max())


def test_gp_calculator_with_a_sandwich_posterior_skips_the_derivative_dtc(gp_fit_dir, monkeypatch):
    """Review fix: with a posterior attached the mixture forces_std is replaced, so the calculator must not
    build the derivative-DTC (n, K, d, 3) arrays it would discard (the big-cell memory hog)."""
    from ace_jax.calc.gp import GPCalculator
    from ace_jax.fit import predict
    out, _, _ = gp_fit_dir
    calls, orig = [], predict._dtc_deriv_residual
    monkeypatch.setattr(predict, "_dtc_deriv_residual", lambda *a, **k: calls.append(1) or orig(*a, **k))
    calc = GPCalculator.from_file(out / "gp_model.npz", posterior=out / "posterior.npz")
    at = _atoms0()
    at.calc = calc
    assert np.isfinite(calc.get_property("forces_std", at)).all()
    assert calls == []


def test_evidence_with_near_duplicate_inducing_points(tiny_gp_problem):
    """Review Focus 2: two inducing points 1e-9 apart (K_MM singular up to its jitter).  The scaled-system
    evidence still matches the observation-space reference, to a tolerance set by cond(K_MM)."""
    from ace_jax.fit.ard import ARDEvidence, ard_gamma, ard_statistics, body_order_columns, gp_body_columns
    from ace_jax.fit.prior_root import prior_root
    prob, ds, theta = tiny_gp_problem
    ind = prob.ind
    prob = prob._replace(ind=ind._replace(XM=ind.XM.at[1].set(ind.XM[0] + 1e-9), SM=ind.SM.at[1].set(ind.SM[0]),
                                          ZM=ind.ZM.at[1].set(ind.ZM[0])))
    with highest_precision():
        st = ard_statistics(theta, prob, ds, "joint", columns="joint")
        ev = ARDEvidence(st, ard_gamma(prob), gp_body_columns(body_order_columns(_meta(prob), prob.cfg), 8),
                         root=prior_root(prob, theta))
        v, g = ev.value_and_grad(ev.h0(theta))
        lml, _, st2 = _obs_space_reference(prob, ds, theta)
    N = sum(float(getattr(st2, f"n_{q}")) for q in "EFV")
    logw = sum(float(getattr(st2, f"logw_{q}")) for q in "EFV")
    assert np.isfinite(g).all()
    assert v + 0.5 * logw - 0.5 * N * np.log(2 * np.pi) == pytest.approx(lml, rel=1e-6)


def test_a_removed_dtc_posterior_is_refused(gp_fit_dir, tmp_path):
    """--ard-variance dtc failed its acceptance and was removed: an old dtc posterior file must not load and
    silently serve the kappa shape."""
    from ace_jax.fit.ard import ARDPosterior
    out, _, _ = gp_fit_dir
    z = dict(np.load(out / "posterior.npz"))
    z["variance"] = np.array("dtc")
    np.savez(tmp_path / "dtc.npz", **z)
    with pytest.raises(ValueError, match="removed"):
        ARDPosterior.load(tmp_path / "dtc.npz")
