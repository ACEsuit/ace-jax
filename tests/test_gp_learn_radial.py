"""fit.radial_learn: VarPro objective, L-BFGS loop, held-out gate."""
import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
MODEL = FIXTURE_DIR / "si_ace_model.npz"
pytestmark = pytest.mark.skipif(not (XYZ.exists() and MODEL.exists()), reason="missing fixtures")

THETA = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
               log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
               log_sigma_F=np.log(0.01), log_sigma_V=np.log(0.01))


def make_problem(ncfg=6, per_batch=3, start=0, force_key="dft_force"):
    """Linear (M = 0) problem on si_tiny with the analytic Si model; gamma = 1."""
    model, meta, z = load(MODEL)
    configs = load_configs(XYZ, "dft_energy", force_key, "dft_virial")[start:start + ncfg]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=per_batch)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=per_batch)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   jnp.ones(cfg.len_basis), default_prior(2.35))
    return prob, ds, meta


def relabel(prob, ds, W, c):
    """ds with targets replaced by the noiseless linear ACE with radials W and readout c."""
    from ace_jax.fit.radial_model import with_radial
    from ace_jax.fit.rows import linear_rows
    m = with_radial(prob.model, W)
    yE, yF, yV = [], [], []
    for i in range(ds.n_batches):
        r, _, _ = linear_rows(m, prob.cfg, jax.tree.map(lambda a: a[i], ds))
        yE.append(r.E @ c); yF.append(r.F @ c); yV.append(r.V @ c)
    return ds._replace(y_E=jnp.stack(yE), y_F=jnp.stack(yF), y_V=jnp.stack(yV))


@pytest.fixture(scope="module")
def small():
    return make_problem()


def _in_memory_residual(W, prob, ds, theta):
    """min_c ||Phi~ c - y~||^2 on the materialised prior-augmented design."""
    from ace_jax.fit.radial_model import with_radial
    from ace_jax.fit.solve import stacked_design
    Phi, y = stacked_design(prob._replace(model=with_radial(prob.model, W)), ds, theta)
    # jnp.linalg.lstsq has no custom VJP: its gradient is plain autodiff through
    # svd, whose singular-VECTOR gradient is NaN on this degenerate ACE design
    # (repeated singular values). Reduced QR is differentiable for a
    # full-column-rank design, and the prior rows make Phi~ full rank.
    Qm, _ = jnp.linalg.qr(Phi)          # reduced (mode="reduced" is the default)
    v = Qm.T @ y
    return y @ y - v @ v


def test_projected_residual_matches_in_memory_lstsq(small):
    from ace_jax.fit.radial_learn import projected_residual
    prob, ds, _ = small
    W = prob.model.rnl_Wnlq
    got = float(projected_residual(W, THETA, prob, ds))
    ref = float(_in_memory_residual(W, prob, ds, THETA))
    assert abs(got - ref) < 1e-7 * abs(ref)


def test_projected_residual_gradient(small):
    from ace_jax.fit.radial_learn import projected_residual
    prob, ds, _ = small
    W = prob.model.rnl_Wnlq
    f = lambda X: projected_residual(X, THETA, prob, ds)
    g = jax.grad(f)(W)
    D = jnp.asarray(np.random.default_rng(0).standard_normal(W.shape))
    h = 1e-5
    fd = (float(f(W + h * D)) - float(f(W - h * D))) / (2 * h)
    assert abs(float(jnp.vdot(g, D)) - fd) < 1e-5 * abs(fd)
    g_mem = jax.grad(lambda X: _in_memory_residual(X, prob, ds, THETA))(W)
    # The streamed path solves the normal equations (error ~ eps * kappa(Phi)^2, with
    # kappa(Phi) ~ 5.7e4 here), the reference a QR of Phi; float64 agreement is
    # therefore ~1e-6 of the gradient scale, and platform-dependent below that
    # (CI Linux vs macOS differ at the 1e-8 level). The FD check above pins the
    # gradient independently.
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_mem), rtol=1e-5,
                               atol=1e-6 * float(jnp.abs(g_mem).max()))


def _temp_bytes(fn, *args):
    ma = jax.jit(fn).lower(*args).compile().memory_analysis()
    if ma is None:
        pytest.skip("backend exposes no memory_analysis")
    return ma.temp_size_in_bytes


def test_gradient_memory_does_not_scale_with_batches():
    from ace_jax.fit.radial_learn import projected_residual
    prob, ds6, _ = make_problem(ncfg=12, per_batch=2)          # 6 equal-shape batches
    ds2 = jax.tree.map(lambda a: a[:2], ds6)
    W = prob.model.rnl_Wnlq
    grad = lambda X, d: jax.grad(lambda Y: projected_residual(Y, THETA, prob, d))(X)
    t2, t6 = _temp_bytes(grad, W, ds2), _temp_bytes(grad, W, ds6)
    val6 = _temp_bytes(lambda X, d: projected_residual(X, THETA, prob, d), W, ds6)
    print(f"temp bytes: grad(2 batches)={t2}  grad(6 batches)={t6}  value(6)={val6}")
    assert t6 <= 1.25 * t2 + 1_000_000


def test_lbfgs_loop_quadratic():
    from ace_jax.fit.radial_learn import lbfgs_loop
    t = jnp.arange(5.0)
    s = jnp.array([1.0, 10.0, 100.0, 0.1, 3.0])
    f = jax.jit(lambda x: jnp.sum(s * (x - t) ** 2))
    x, fx, trace, reason = lbfgs_loop(f, jnp.zeros(5), steps=100)
    np.testing.assert_allclose(np.asarray(x), np.asarray(t), atol=1e-6)
    assert reason in ("converged", "linesearch", "steps") and fx <= trace[0]


def test_lbfgs_loop_zero_steps_returns_start():
    from ace_jax.fit.radial_learn import lbfgs_loop
    x0 = jnp.ones(3)
    x, fx, trace, reason = lbfgs_loop(lambda x: jnp.sum(x ** 2), x0, steps=0)
    assert bool(jnp.all(x == x0)) and trace == [] and reason == "steps" and fx == 3.0


def test_lbfgs_loop_nonfinite_keeps_best():
    from ace_jax.fit.radial_learn import lbfgs_loop
    f = jax.jit(lambda x: jnp.where(x[0] > 0.5, jnp.sum(x ** 2), jnp.nan))
    x0 = jnp.array([2.0, 1.0])
    x, fx, _, _ = lbfgs_loop(f, x0, steps=50)
    assert bool(jnp.all(jnp.isfinite(x))) and np.isfinite(fx)
    assert fx <= float(f(x0)) and float(f(x)) == fx


def _toy_quadratic(x, s, t):
    """Module-level (hence stable-identity, hashable) toy `f` for lbfgs_loop's
    args= interface: f(x, *args, *statics) = f(x, s, t)."""
    return jnp.sum(s * (x - t) ** 2)


def test_lbfgs_loop_no_recompile_across_rounds():
    """Two lbfgs_loop calls with the same f/statics and matching x0/args
    shapes -- exactly how learn_radial drives one round per call -- must
    compile _lbfgs_step at most once between them, not once per call."""
    from ace_jax.fit.radial_learn import _lbfgs_step, lbfgs_loop
    t = jnp.arange(5.0)
    s1 = jnp.array([1.0, 10.0, 100.0, 0.1, 3.0])
    s2 = jnp.array([2.0, 5.0, 50.0, 0.2, 1.0])          # different VALUES, same shape/dtype
    n0 = _lbfgs_step._cache_size()
    lbfgs_loop(_toy_quadratic, jnp.zeros(5), steps=5, args=(s1, t))
    n1 = _lbfgs_step._cache_size()
    lbfgs_loop(_toy_quadratic, jnp.zeros(5), steps=5, args=(s2, t))
    n2 = _lbfgs_step._cache_size()
    print(f"_lbfgs_step cache size: before={n0} after call 1={n1} after call 2={n2}")
    assert n2 - n0 <= 1


def test_learn_radial_requires_x64(small, monkeypatch):
    import types
    from ace_jax.fit import radial_learn
    prob, ds, _ = small
    # require_x64 is learn_radial's first statement, so only its read of
    # jax.config sees the stub; the global flag is never touched
    monkeypatch.setattr(radial_learn, "jax", types.SimpleNamespace(
        config=types.SimpleNamespace(jax_enable_x64=False)))
    with pytest.raises(RuntimeError, match="float64"):
        radial_learn.learn_radial(prob, ds, prob.model.rnl_Wnlq, theta0=THETA, steps=1)


def test_learn_radial_zero_steps_is_normalised_init(small):
    from ace_jax.fit.radial_learn import learn_radial
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    prob, ds, _ = small
    W0 = prob.model.rnl_Wnlq
    W, info = learn_radial(prob, ds, W0, theta0=THETA, profile=False, steps=0)
    ref = normalise(W0, radial_gram(prob.model, ds), row_active(W0))
    np.testing.assert_array_equal(np.asarray(W), np.asarray(ref))
    assert info["steps"] == 0


def _perturbed_truth(prob, seed=0, eps=0.2):
    rng = np.random.default_rng(seed)
    Wt = prob.model.rnl_Wnlq
    W0 = Wt + eps * jnp.abs(Wt).mean() * jnp.asarray(rng.standard_normal(Wt.shape))
    c = jnp.asarray(0.1 * rng.standard_normal(prob.cfg.len_basis))
    return Wt, W0, c


def test_learn_radial_recovers_perturbed_radials():
    from ace_jax.fit.radial_learn import learn_radial, projected_residual
    prob, ds, _ = make_problem(ncfg=12, per_batch=3)
    Wt, W0, c = _perturbed_truth(prob)
    ds = relabel(prob, ds, Wt, c)
    W, info = learn_radial(prob, ds, W0, theta0=THETA, profile=False, steps=30)
    f0 = float(projected_residual(W0, THETA, prob, ds))
    f1 = float(projected_residual(W, THETA, prob, ds))
    print(f"recovery: f0={f0:.4e} f1={f1:.4e} reasons={info['reasons']}")
    assert f1 < 0.3 * f0
    assert all(b <= a * (1 + 1e-12) for a, b in zip(info["trace"], info["trace"][1:]))


def test_learn_radial_profiles_theta(small):
    from ace_jax.fit.radial_learn import learn_radial
    prob, ds, _ = small
    W, info = learn_radial(prob, ds, prob.model.rnl_Wnlq, steps=4, reprofile_every=2, map_steps=50)
    assert len(info["theta"]) >= 2 and all(np.all(np.isfinite(t)) for t in info["theta"])


def test_learn_radial_logs_per_round(small):
    from ace_jax.fit.radial_learn import learn_radial
    prob, ds, _ = small
    lines = []
    learn_radial(prob, ds, prob.model.rnl_Wnlq, theta0=THETA, profile=False, steps=4,
                 reprofile_every=2, log=lines.append)
    round_lines = [l for l in lines if "round" in l]
    assert len(round_lines) >= 2
    assert "eta=" in round_lines[0] and "eta=" not in round_lines[-1]


def test_gate_ties_go_to_first():
    from ace_jax.fit.radial_learn import gate
    label, scores = gate({"init": 1, "learned": 2}, lambda k, w: 0.5)
    assert label == "init" and scores == {"init": 0.5, "learned": 0.5}
    label, _ = gate({"init": 1, "learned": 2}, lambda k, w: 1.0 if w == 1 else 0.1)
    assert label == "learned"
    seen = []
    gate({"init": 1, "learned": 1}, lambda k, w: seen.append(k) or 0.0)   # equal values, distinct labels
    assert seen == ["init", "learned"]


def test_holdout_score_energy_only_split(small):
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.radial_learn import holdout_score
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6, force_key="__no_such_key__")
    from ace_jax.fit.objective import posterior
    from ace_jax.fit.stats import linear_statistics
    a = to_array(THETA)
    W = prob.model.rnl_Wnlq
    s = holdout_score(W, a, a, prob, ds_fit, ds_val)
    val = linear_statistics(prob.model, prob.cfg, ds_val)
    assert float(val.n_F) == 0 and float(val.n_E) > 0
    c, _ = posterior(THETA, linear_statistics(prob.model, prob.cfg, ds_fit), prob)
    e_term = float(val.yy_E - 2.0 * c @ val.b_E + c @ val.G_E @ c) / (float(val.n_E) * 0.01 ** 2)
    assert np.isfinite(s) and s > 0.0
    np.testing.assert_allclose(s, e_term, rtol=1e-10)


def test_fit_radial_gate_prefers_learned_on_recoverable_problem():
    from ace_jax.fit.radial_learn import fit_radial
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, W0, c = _perturbed_truth(prob)
    ds_fit, ds_val = relabel(prob, ds_fit, Wt, c), relabel(prob, ds_val, Wt, c)
    # theta0=None: a0 is the (unconverged) MAP at the init, so this exercises
    # the gate's common re-MAP for every candidate, init included
    W, info = fit_radial(prob, ds_fit, ds_val, W0, lam_grid=(0.0,), theta0=None,
                         profile=False, steps=30, map_steps=50)
    print(info["scores"], info["scores_at_a0"])
    assert info["selected"].startswith("learned")
    assert info["scores"][info["selected"]] < info["scores"]["init"]
    assert info["scores_at_a0"][info["selected"]] < info["scores_at_a0"]["init"]


def test_fit_radial_zero_steps_gate_is_fair(small):
    """steps=0: learned == init, so both are scored by the identical procedure
    and must tie exactly (-> init), whatever the MAP's convergence."""
    from ace_jax.fit.radial_learn import fit_radial
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    W, info = fit_radial(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, lam_grid=(0.0,), theta0=None,
                         steps=0, map_steps=50)
    s = info["scores"]
    print(s, info["scores_at_a0"], info["map_diag"])
    np.testing.assert_allclose(s["learned_lam=0"], s["init"], rtol=1e-6)
    np.testing.assert_allclose(info["scores_at_a0"]["learned_lam=0"], info["scores_at_a0"]["init"],
                               rtol=1e-6)
    assert info["selected"] == "init"
    assert set(info["map_diag"]) == {"init", "learned_lam=0"}
    assert all(np.isfinite(d["grad_norm"]) and np.isfinite(d["dloss_last"])
               for d in info["map_diag"].values())
    assert info["readout"].shape == (prob.cfg.len_basis,)


def test_learn_radial_two_species_roughness():
    """NZ = 2 (SiGe, structurally-zero onehot rows) and lam > 0: zero rows stay
    exactly zero, everything is finite, the roughness penalty lowers the
    learned roughness, and lam_abs is the relative weight times r0/rough0."""
    from ace_jax.fit.radial_learn import learn_radial
    from ace_jax.fit.radial_model import (radial_gram, rnl_degrees, roughness, roughness_matrix,
                                          row_active, to_analytic)
    spline, meta, z = load(FIXTURE_DIR / "sige_nofit.npz")
    model, _ = to_analytic(spline, 12)
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:6]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   jnp.ones(cfg.len_basis), default_prior(2.35))
    W0 = model.rnl_Wnlq
    zero = ~np.asarray(row_active(W0))
    assert cfg.NZ == 2 and zero.any() and (~zero).any()
    wn = 1.0 / (1.0 + rnl_degrees(meta)) ** 2
    D2 = roughness_matrix(model)
    out = {}
    for lam in (0.0, 1e-1):
        W, info = learn_radial(prob, ds, W0, theta0=THETA, lam_rough=lam, rough_weights=wn,
                               steps=5, reprofile_every=5, map_steps=20)
        W = np.asarray(W)
        assert np.all(np.isfinite(W)) and all(np.isfinite(info["trace"]))
        assert np.all(W[zero] == 0.0)
        np.testing.assert_allclose(info["lam_abs"], lam * info["r0"] / info["rough0"], rtol=1e-12)
        out[lam] = float(roughness(jnp.asarray(W), D2, jnp.asarray(wn)))
    print(f"roughness: lam=0 {out[0.0]:.6e}  lam=0.1 {out[1e-1]:.6e}")
    assert out[1e-1] <= out[0.0]


def test_fit_radial_checkpoints_each_lambda(tmp_path, small):
    from ace_jax.fit.radial_learn import fit_radial, save_result
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    calls = []

    def ckpt(label, W, info):
        calls.append(label)
        save_result(tmp_path / f"lam_{label}", W, info)

    W, info = fit_radial(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, lam_grid=(0.0, 1e-2),
                         theta0=THETA, profile=False, steps=1, map_steps=10, checkpoint=ckpt)
    assert calls == ["0", "0.01"]
    import json
    saved = {}
    for lab in calls:
        saved[lab] = np.load(tmp_path / f"lam_{lab}" / "rnl_Wnlq.npy")
        js = json.loads((tmp_path / f"lam_{lab}" / "radial_info.json").read_text())
        assert js["lam_rough"] == float(lab) and js["steps"] == 1
    sel = info["selected"]
    if sel != "init":
        np.testing.assert_array_equal(saved[sel.split("=")[1]], np.asarray(W))


def test_learn_radial_precomputed_pieces_match(small):
    """Q/D2/r0 passed in (as fit_radial does) give the same run as computed."""
    from ace_jax.fit.radial_learn import learn_radial, projected_residual
    from ace_jax.fit.radial_model import normalise, radial_gram, roughness_matrix, row_active
    prob, ds, _ = small
    W0 = prob.model.rnl_Wnlq
    Q, D2 = radial_gram(prob.model, ds), roughness_matrix(prob.model)
    r0 = float(projected_residual(normalise(W0, Q, row_active(W0)), THETA, prob, ds))
    kw = dict(theta0=THETA, profile=False, lam_rough=1e-2, steps=2)
    Wa, ia = learn_radial(prob, ds, W0, **kw)
    Wb, ib = learn_radial(prob, ds, W0, Q=Q, D2=D2, r0=r0, **kw)
    np.testing.assert_allclose(ib["lam_abs"], ia["lam_abs"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(Wb), np.asarray(Wa), rtol=1e-10, atol=1e-12)


def test_learn_radial_rejects_smooth_init_with_lambda(small):
    from ace_jax.fit.radial_learn import learn_radial, relative_lambda
    prob, ds, _ = small
    assert relative_lambda(0.0, 1.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="rough0"):
        relative_lambda(1e-2, 5.0, 1e-13)
    W0 = jnp.zeros_like(prob.model.rnl_Wnlq).at[..., 1].set(1.0)   # degree-1 radials: zero curvature
    with pytest.raises(ValueError, match="rough0"):
        learn_radial(prob, ds, W0, theta0=THETA, profile=False, lam_rough=1e-2, steps=1)


def test_learned_radials_reject_inducing_points(small):
    from ace_jax.fit.radial_learn import fit_radial, learn_radial
    prob, ds, _ = small
    ind = prob.ind._replace(XM=jnp.zeros((3, prob.ind.XM.shape[1])))
    bad = prob._replace(ind=ind)
    with pytest.raises(ValueError, match="M = 0"):
        learn_radial(bad, ds, prob.model.rnl_Wnlq, theta0=THETA, steps=0)
    with pytest.raises(ValueError, match="M = 0"):
        fit_radial(bad, ds, ds, prob.model.rnl_Wnlq, steps=0)


def test_fit_radial_rejects_duplicate_lambdas(small):
    from ace_jax.fit.radial_learn import fit_radial
    prob, ds, _ = small
    with pytest.raises(ValueError, match="duplicate"):
        fit_radial(prob, ds, ds, prob.model.rnl_Wnlq, lam_grid=(0.0, 1e-2, 0.01), steps=0)


def test_save_result_roundtrip(tmp_path, small):
    from ace_jax.fit.radial_learn import save_result
    prob, _, _ = small
    W = 1.5 * prob.model.rnl_Wnlq
    c = np.arange(prob.cfg.len_basis, dtype=float)
    save_result(tmp_path, W, {"selected": "learned", "trace": [1.0, 0.5], "theta": [np.zeros(3)],
                              "readout": c}, src_npz=MODEL, model=prob.model)
    assert np.array_equal(np.load(tmp_path / "rnl_Wnlq.npy"), np.asarray(W))
    assert np.array_equal(np.load(tmp_path / "readout.npy"), c)
    back, meta, _ = load(tmp_path / "model.npz")
    np.testing.assert_array_equal(np.asarray(back.rnl_Wnlq), np.asarray(W))
    nB, nP, NZ = prob.cfg.n_B, prob.cfg.n_pair, prob.cfg.NZ
    np.testing.assert_array_equal(np.asarray(back.WB)[:, 0], c[:nB])
    np.testing.assert_array_equal(np.asarray(back.Wpair)[:, 0], c[NZ * nB:NZ * nB + nP])
    import json
    js = json.loads((tmp_path / "radial_info.json").read_text())
    assert js["selected"] == "learned" and "readout" not in js


def test_save_result_refuses_stale_readout(tmp_path, small):
    from ace_jax.fit.radial_learn import save_result
    prob, _, _ = small
    with pytest.raises(ValueError, match="readout"):
        save_result(tmp_path, prob.model.rnl_Wnlq, {"selected": "init"}, src_npz=MODEL, model=prob.model)
    assert not (tmp_path / "model.npz").exists()
    save_result(tmp_path, prob.model.rnl_Wnlq, {"selected": "init"})      # no model.npz: fine
    assert (tmp_path / "rnl_Wnlq.npy").exists()


def test_readout_to_npz_layout():
    from ace_jax.construct.export import readout_to_npz
    nB, nP, NZ = 3, 2, 2
    c = np.arange((nB + nP) * NZ, dtype=float)
    WB, Wpair = readout_to_npz(c, nB, nP, NZ)
    np.testing.assert_array_equal(WB, [[0, 3], [1, 4], [2, 5]])
    np.testing.assert_array_equal(Wpair, [[6, 8], [7, 9]])
    with pytest.raises(ValueError, match="readout shape"):
        readout_to_npz(c[:-1], nB, nP, NZ)


def test_saved_model_energies_match_fitted_readout(tmp_path, small):
    """model.npz written by save_result evaluates to linear_rows @ readout
    (+ E0): the readout is the one fitted for the selected radials."""
    from ace_jax.fit.radial_learn import fit_radial, save_result
    from ace_jax.fit.radial_model import with_radial
    from ace_jax.fit.rows import linear_rows
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    W, info = fit_radial(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, lam_grid=(0.0,), theta0=THETA,
                         profile=False, steps=2, map_steps=20)
    save_result(tmp_path, W, info, src_npz=MODEL, model=prob.model)
    back, _, z = load(tmp_path / "model.npz")
    E0 = np.asarray(z["E0"])
    m = with_radial(prob.model, W)
    worst = 0.0
    for i in range(ds_val.n_batches):
        b = jax.tree.map(lambda a: a[i], ds_val)
        rows, _, _ = linear_rows(m, prob.cfg, b)
        C = b.y_E.shape[0]
        Ncap, K = b.nbr.shape
        zi = jnp.broadcast_to(b.node_z[:, None], (Ncap, K))
        e = back.site_energies_dense(b.rij, zi, b.node_z[b.nbr], b.nbr_mask, b.node_z)
        E_model = np.asarray(jax.ops.segment_sum(e, b.node_cfg, num_segments=C + 1)[:C])
        live = np.asarray(b.node_mask)
        E0sum = np.asarray(jax.ops.segment_sum(jnp.where(b.node_mask, jnp.asarray(E0)[b.node_z], 0.0),
                                               b.node_cfg, num_segments=C + 1)[:C])
        E_lin = np.asarray(rows.E @ jnp.asarray(info["readout"])) + E0sum
        assert live.any()
        worst = max(worst, float(np.max(np.abs(E_model - E_lin) / np.abs(E_lin))))
    print(f"max relative |E_npz - (rows @ c + E0)| / |E| = {worst:.3e}")
    assert worst < 1e-8


def test_bench_driver_smoke(tmp_path):
    import json, subprocess, sys
    from conftest import ROOT
    r = subprocess.run(
        [sys.executable, str(ROOT / "bench/learn_radial/run.py"), "--model", str(MODEL),
         "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
         "--virial-key", "dft_virial", "--ntrain", "8", "--nval", "8", "--batch", "4",
         "--n-q", "20", "--steps", "3", "--lam-grid", "0", "--map-steps", "20",
         "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["selected"] in s["scores"] and (tmp_path / "model.npz").exists()
    assert (tmp_path / "lam_0" / "rnl_Wnlq.npy").exists()
    assert s["to_analytic_relres_max"] == 0.0            # analytic fixture: widened only
    assert "relres_max=" in r.stdout
