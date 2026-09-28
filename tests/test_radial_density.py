"""Joint radial + density VarPro (fit.density masked columns, fit.radial_density)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from conftest import FIXTURE_DIR
from test_gp_learn_radial import MODEL, THETA, XYZ, make_problem

pytestmark = pytest.mark.skipif(not (XYZ.exists() and MODEL.exists()), reason="missing fixtures")


@pytest.fixture(scope="module")
def small():
    return make_problem()


def test_prior_precision_sizes_from_gamma(small):
    from ace_jax.fit.objective import prior_precision
    prob, _, _ = small
    L = prob.cfg.len_basis
    Lam, logdet = prior_precision(THETA, prob._replace(gamma=jnp.full(L + 3, 2.0)))
    assert Lam.shape == (L + 3, L + 3)
    np.testing.assert_allclose(np.diag(Lam), 4.0 / 0.3 ** 2)
    Lam0, _ = prior_precision(THETA, prob)
    assert Lam0.shape == (L, L)


def test_theta_map_and_holdout_take_precomputed_stats(small):
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.radial_learn import holdout_score, theta_map_linear
    from ace_jax.fit.stats import linear_statistics
    prob, ds, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    W = prob.model.rnl_Wnlq
    lin = linear_statistics(prob.model, prob.cfg, ds)
    a1 = theta_map_linear(prob, ds, W, steps=20)
    a2 = theta_map_linear(prob, ds, None, steps=20, lin=lin)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    a = to_array(THETA)
    lv = linear_statistics(prob.model, prob.cfg, ds_val)
    s1 = holdout_score(W, a, a, prob, ds, ds_val)
    s2 = holdout_score(None, a, a, prob, ds, ds_val, lin_fit=lin, lin_val=lv)
    np.testing.assert_allclose(s1, s2, rtol=1e-12)


def _dens_fd_check(model, cfg, batch, eta, mask):
    """density_rows_masked F/V rows are minus the derivative of its E rows (autodiff through
    compact_basis), as in tests/test_gp_density.py."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.data import VOIGT, flat_edges
    from ace_jax.fit.density import density_rows_masked
    from ace_jax.fit.rows import linear_rows
    Ncap, K = batch.nbr.shape
    ncol = eta.shape[0] * cfg.NZ
    with highest_precision():
        _, X, J = linear_rows(model, cfg, batch)
        dr = density_rows_masked(eta, mask, cfg, batch, X, J)

        def e_dens(rij):
            b = batch._replace(rij=rij)
            r, send, recv, m = flat_edges(rij, batch.nbr, batch.nbr_mask)
            Xr = model.compact_basis(r, batch.node_z[send], batch.node_z[recv], send, Ncap, m)
            return density_rows_masked(eta, mask, cfg, b, Xr, J).E
        T = jax.jacrev(e_dens)(batch.rij)
    node_cfg = np.asarray(batch.node_cfg); nbr = np.asarray(batch.nbr).reshape(-1)
    own = np.minimum(node_cfg, cfg.C - 1)
    Tn = np.where(np.asarray(batch.node_mask)[:, None, None, None],
                  np.asarray(T)[own, :, np.arange(Ncap)], 0.0)               # (Ncap, ncol, K, 3)
    dEdr = np.zeros((Ncap, 3, ncol))
    dEdr -= Tn.sum(2).transpose(0, 2, 1)
    np.add.at(dEdr, nbr, Tn.transpose(0, 2, 3, 1).reshape(Ncap * K, 3, ncol))
    assert np.abs(dEdr).max() > 1e-3
    assert np.abs(np.asarray(dr.F) + dEdr).max() < 1e-9 * max(1.0, np.abs(dEdr).max())
    rij = np.asarray(batch.rij).reshape(Ncap * K, 3)
    Te = Tn.transpose(0, 2, 3, 1).reshape(Ncap * K, 3, ncol)
    Vref = np.zeros((cfg.C + 1, 6, ncol)); ecfg = node_cfg[np.repeat(np.arange(Ncap), K)]
    for v, (a, b_) in enumerate(VOIGT):
        np.add.at(Vref[:, v], ecfg, -0.5 * (Te[:, a] * rij[:, b_, None] + Te[:, b_] * rij[:, a, None]))
    assert np.abs(np.asarray(dr.V) - Vref[:cfg.C]).max() < 1e-9 * max(1.0, np.abs(Vref).max())
    return dr


@pytest.mark.parametrize("span,P", [("full", 2), ("pair", 1), ("pair", 2)])
def test_density_rows_masked_are_derivatives(small, span, P):
    from ace_jax.fit.density import density_mask
    prob, ds, _ = small
    cfg = prob.cfg
    mask = density_mask(cfg, span)
    eta = jnp.asarray(np.random.default_rng(1).standard_normal((P, cfg.NZ, cfg.D))) * mask
    dr = _dens_fd_check(prob.model, cfg, jax.tree.map(lambda a: a[0], ds), eta, mask)
    assert dr.E.shape[-1] == P * cfg.NZ


def test_masked_pair_P1_equals_density_rows(small):
    from ace_jax.fit.density import density_mask, density_rows, density_rows_masked
    from ace_jax.fit.rows import linear_rows
    prob, ds, _ = small
    cfg = prob.cfg
    b = jax.tree.map(lambda a: a[0], ds)
    _, X, J = linear_rows(prob.model, cfg, b)
    eta_p = jnp.asarray(np.random.default_rng(2).standard_normal((cfg.NZ, cfg.n_pair)))
    eta = jnp.zeros((1, cfg.NZ, cfg.D)).at[0, :, cfg.n_B:].set(eta_p)
    a = density_rows(eta_p, cfg, b, X, J)
    m = density_rows_masked(eta, density_mask(cfg, "pair"), cfg, b, X, J)
    for k in "EFV":
        np.testing.assert_allclose(np.asarray(getattr(m, k)), np.asarray(getattr(a, k)), rtol=1e-12, atol=1e-12)


def test_density_statistics_P0_is_linear_statistics(small):
    from ace_jax.fit.density import density_mask, linear_density_statistics
    from ace_jax.fit.stats import linear_statistics
    prob, ds, _ = small
    cfg = prob.cfg
    a = linear_statistics(prob.model, cfg, ds)
    b = linear_density_statistics(prob.model, jnp.zeros((0, cfg.NZ, cfg.D)), density_mask(cfg, "full"), cfg, ds)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _widened_rows(model, cfg, batch, eta, mask):
    from ace_jax.fit.density import density_rows_masked
    from ace_jax.fit.rows import Rows, linear_rows
    r, X, J = linear_rows(model, cfg, batch)
    d = density_rows_masked(eta, mask, cfg, batch, X, J)
    return Rows(*(jnp.concatenate([getattr(r, k), getattr(d, k)], -1) for k in "EFV"))


def test_density_statistics_match_in_memory(small):
    from ace_jax.fit.density import density_mask, linear_density_statistics
    prob, ds, _ = small
    cfg = prob.cfg
    mask = density_mask(cfg, "full")
    eta = jnp.asarray(np.random.default_rng(3).standard_normal((2, cfg.NZ, cfg.D)))
    st = linear_density_statistics(prob.model, eta, mask, cfg, ds)
    G = {k: 0.0 for k in "EFV"}; bb = {k: 0.0 for k in "EFV"}
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        r = _widened_rows(prob.model, cfg, b, eta, mask)
        Dt = r.E.shape[-1]
        for k, R, y, w in (("E", r.E, b.y_E, b.w_E),
                           ("F", r.F.reshape(-1, Dt), b.y_F.reshape(-1), jnp.repeat(b.w_F, 3)),
                           ("V", r.V.reshape(-1, Dt), b.y_V.reshape(-1), jnp.repeat(b.w_V, 6))):
            Pw = R * w[:, None]
            G[k] = G[k] + Pw.T @ Pw; bb[k] = bb[k] + Pw.T @ (y * w)
    for k in "EFV":
        np.testing.assert_allclose(np.asarray(getattr(st, "G_" + k)), np.asarray(G[k]), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(getattr(st, "b_" + k)), np.asarray(bb[k]), rtol=1e-10, atol=1e-10)


def test_widened_projected_residual_matches_qr(small):
    """VarPro objective on the widened statistics == min ||Phi~ c - y~||^2 by QR on the
    materialised prior-augmented [linear | density] design."""
    from ace_jax.fit.density import density_gamma, density_mask, linear_density_statistics
    from ace_jax.fit.radial_learn import projected_residual_from_stats
    prob, ds, _ = small
    cfg = prob.cfg
    mask = density_mask(cfg, "full")
    eta = jnp.asarray(np.random.default_rng(4).standard_normal((2, cfg.NZ, cfg.D))) * 1e-2
    gw = density_gamma(prob.gamma, 2, cfg.NZ)
    got = float(projected_residual_from_stats(THETA, linear_density_statistics(prob.model, eta, mask, cfg, ds), gw))
    se, sf, sv = (float(np.exp(getattr(THETA, "log_sigma_" + k))) for k in "EFV")
    rows, ys = [], []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        r = _widened_rows(prob.model, cfg, b, eta, mask)
        Dt = r.E.shape[-1]
        w = jnp.concatenate([b.w_E / se, jnp.repeat(b.w_F, 3) / sf, jnp.repeat(b.w_V, 6) / sv])
        Phi = jnp.concatenate([r.E, r.F.reshape(-1, Dt), r.V.reshape(-1, Dt)])
        y = jnp.concatenate([b.y_E, b.y_F.reshape(-1), b.y_V.reshape(-1)])
        rows.append(Phi * w[:, None]); ys.append(y * w)
    Phi = jnp.concatenate(rows + [jnp.diag(gw / float(np.exp(THETA.log_sigma_c)))])
    y = jnp.concatenate(ys + [jnp.zeros(gw.shape[0])])
    Qm, _ = jnp.linalg.qr(Phi)
    v = Qm.T @ y
    ref = float(y @ y - v @ v)
    assert abs(got - ref) < 1e-7 * abs(ref)


def test_density_gamma_and_compact_gamma():
    from types import SimpleNamespace
    from ace_jax.fit.density import compact_gamma, density_gamma
    cfg = SimpleNamespace(n_B=2, n_pair=1, NZ=2, D=3)
    g = jnp.asarray([1., 2., 3., 4., 5., 6.])           # B z0 | B z1 | pair z0 | pair z1
    np.testing.assert_array_equal(np.asarray(compact_gamma(g, cfg)), [[1, 2, 5], [3, 4, 6]])
    gw = density_gamma(g, 2, 2)
    assert gw.shape == (10,)
    np.testing.assert_allclose(np.asarray(gw[6:]), np.exp(np.mean(np.log(np.arange(1, 7)))), rtol=1e-12)


def _rd_args(prob, ds, P=2, span="full", seed=0):
    """Full positional args of _objective_rd after (xb, other), plus V, H, statics pieces."""
    from ace_jax.fit.density import compact_gamma, density_gamma, density_mask
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.radial_density import init_density, normalise_rho, rho_gram
    from ace_jax.fit.radial_model import normalise, radial_gram, roughness_matrix, row_active, spectral_weights
    cfg = prob.cfg
    W0 = prob.model.rnl_Wnlq
    active = row_active(W0); Q = radial_gram(prob.model, ds)
    V = normalise(W0, Q, active)
    mask = density_mask(cfg, span)
    S = rho_gram(prob.model, V, mask, cfg, ds)
    H = normalise_rho(init_density(cfg, mask, P, seed), S)
    nq = W0.shape[-1]
    z = lambda: jnp.asarray(0.0)
    common = (to_array(THETA), prob.model, ds, density_gamma(prob.gamma, P, cfg.NZ), Q, active,
              roughness_matrix(prob.model), jnp.ones(W0.shape[2]), z(), V, spectral_weights(nq, 4.0), z(),
              jnp.zeros((cfg.NZ, cfg.NZ, nq, nq)), z(), S, mask, compact_gamma(prob.gamma, cfg),
              jnp.asarray(1e-3))
    shapes = (tuple(V.shape), tuple(H.shape))
    return V, H, common, shapes


@pytest.mark.parametrize("seed", [0, 3])
def test_rd_gradient_matches_fd(small, seed):
    from ace_jax.fit.radial_density import _objective_rd, _pack
    prob, ds, _ = small
    V, H, common, shapes = _rd_args(prob, ds)
    r = jnp.asarray([1.0, 3.0])
    x, other = _pack(V, H, r, "joint")
    f = lambda y: float(_objective_rd(y, other, *common, r, prob.cfg, "joint", shapes))
    g = jax.grad(lambda y: _objective_rd(y, other, *common, r, prob.cfg, "joint", shapes))(x)
    nV = V.size
    # Per block: f ~ 2.5e8 with cond(G) ~ 5e23 leaves ~6.6e-5 absolute roundoff in f, so V needs
    # h = 1e-4 and a gradient-weighted direction (a random one can be near-orthogonal to g_V);
    # the ssqrt kink (sites with rho ~ 0) needs small h in H, hence a 5-point stencil there.
    R = jnp.asarray(np.random.default_rng(seed).standard_normal(x.shape))
    gV = g.at[nV:].set(0.0)
    DV = (gV / jnp.linalg.norm(gV) * np.sqrt(nV) + R.at[nV:].set(0.0)) / np.sqrt(2)
    DH = R.at[:nV].set(0.0)
    h = 1e-4
    fdV = (f(x + h * DV) - f(x - h * DV)) / (2 * h)
    h = 1e-5
    fdH = (-f(x + 2 * h * DH) + 8 * f(x + h * DH) - 8 * f(x - h * DH) + f(x - 2 * h * DH)) / (12 * h)
    eV, eH = abs(float(g @ DV) - fdV) / abs(fdV), abs(float(g @ DH) - fdH) / abs(fdH)
    print(f"seed={seed} V rel={eV:.2e} H rel={eH:.2e}")
    assert eV < 2e-4 and eH < 1e-4


def test_rd_objective_gauge_and_precond_invariant(small):
    from ace_jax.fit.radial_density import _objective_rd, _pack
    prob, ds, _ = small
    V, H, common, shapes = _rd_args(prob, ds)
    one = jnp.asarray([1.0, 1.0])
    f = lambda V_, H_, r: float(_objective_rd(*_pack(V_, H_, r, "joint"), *common, r, prob.cfg, "joint", shapes))
    f0 = f(V, H, one)
    scale = jnp.asarray(np.random.default_rng(1).uniform(0.2, 5.0, H.shape[:2]))[..., None]
    np.testing.assert_allclose(f(V, H * scale, one), f0, rtol=1e-10)               # per-(p, z) scale gauge
    np.testing.assert_allclose(f(V, H.at[1].multiply(-1.0), one), f0, rtol=1e-10)  # sign of density p
    np.testing.assert_allclose(f(V, H, jnp.asarray([1.0, 7.3])), f0, rtol=1e-12)    # preconditioner


def _relabel_rd(prob, ds, W, eta, mask, c):
    from ace_jax.fit.radial_model import with_radial
    m = with_radial(prob.model, W)
    yE, yF, yV = [], [], []
    for i in range(ds.n_batches):
        r = _widened_rows(m, prob.cfg, jax.tree.map(lambda a: a[i], ds), eta, mask)
        yE.append(r.E @ c); yF.append(r.F @ c); yV.append(r.V @ c)
    return ds._replace(y_E=jnp.stack(yE), y_F=jnp.stack(yF), y_V=jnp.stack(yV))


def _density_truth(prob, ds, P=1, span="full", seed=5):
    """Targets from the true radials plus a density whose weights differ from the init."""
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import init_density, normalise_rho, rho_gram
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    cfg = prob.cfg
    rng = np.random.default_rng(seed)
    W0 = prob.model.rnl_Wnlq
    Wt = normalise(W0, radial_gram(prob.model, ds), row_active(W0))
    mask = density_mask(cfg, span)
    S = rho_gram(prob.model, Wt, mask, cfg, ds)
    Ht = init_density(cfg, mask, P, 0) + 0.5 * jnp.asarray(rng.standard_normal((P, cfg.NZ, cfg.D))) * mask
    eta_t = normalise_rho(Ht, S)
    c = jnp.concatenate([0.1 * jnp.asarray(rng.standard_normal(cfg.len_basis)), jnp.full(P * cfg.NZ, 0.5)])
    return Wt, eta_t, mask, c


@pytest.mark.parametrize("mode,factor", [("joint", 0.3), ("alternating", 0.5)])
def test_learn_radial_density_reduces_objective_on_density_data(mode, factor):
    from ace_jax.fit.density import density_gamma, linear_density_statistics
    from ace_jax.fit.radial_density import init_density, learn_radial_density, normalise_rho, rho_gram
    from ace_jax.fit.radial_learn import projected_residual_from_stats
    from ace_jax.fit.radial_model import with_radial
    prob, ds, _ = make_problem(ncfg=12, per_batch=3)
    Wt, eta_t, mask, c = _density_truth(prob, ds)
    ds = _relabel_rd(prob, ds, Wt, eta_t, mask, c)
    gw = density_gamma(prob.gamma, 1, prob.cfg.NZ)
    f = lambda W, eta: float(projected_residual_from_stats(
        THETA, linear_density_statistics(with_radial(prob.model, W), eta, mask, prob.cfg, ds), gw))
    S = rho_gram(prob.model, Wt, mask, prob.cfg, ds)
    f0 = f(Wt, normalise_rho(init_density(prob.cfg, mask, 1), S))
    V, eta, info = learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, P=1, mode=mode,
                                        theta0=THETA, profile=False, steps=30, reprofile_every=10)
    f1 = f(V, eta)
    print(f"{mode}: f0={f0:.4e} f1={f1:.4e} reasons={info['reasons']} precond={info['precond']}")
    assert f1 < factor * f0
    assert info["mode"] == mode and info["P"] == 1 and len(info["precond"]) >= 1


def test_learn_radial_density_zero_steps_is_normalised_init(small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import init_density, learn_radial_density, normalise_rho, rho_gram
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    prob, ds, _ = small
    mask = density_mask(prob.cfg, "full")
    V, eta, info = learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, P=2, theta0=THETA,
                                        profile=False, steps=0)
    W0 = prob.model.rnl_Wnlq
    Vr = normalise(W0, radial_gram(prob.model, ds), row_active(W0))
    np.testing.assert_array_equal(np.asarray(V), np.asarray(Vr))
    S = rho_gram(prob.model, Vr, mask, prob.cfg, ds)
    np.testing.assert_array_equal(np.asarray(eta), np.asarray(normalise_rho(init_density(prob.cfg, mask, 2), S)))
    assert info["steps"] == 0


def test_normalise_rho_unit_mean_square(small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import init_density, normalise_rho, rho_gram
    prob, ds, _ = small
    mask = density_mask(prob.cfg, "full")
    S = rho_gram(prob.model, prob.model.rnl_Wnlq, mask, prob.cfg, ds)
    eta = normalise_rho(init_density(prob.cfg, mask, 2, seed=3), S)
    q = jnp.einsum("pzd,zde,pze->pz", eta, S, eta)
    np.testing.assert_allclose(np.asarray(q), 1.0, rtol=1e-10)


def test_learn_radial_density_single_compile_joint(small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import learn_radial_density
    from ace_jax.fit.radial_learn import _lbfgs_step
    prob, ds, _ = small
    kw = dict(mask=density_mask(prob.cfg, "full"), P=1, theta0=THETA, profile=False, reprofile_every=2)
    learn_radial_density(prob, ds, prob.model.rnl_Wnlq, steps=2, **kw)          # warm the cache
    n0 = _lbfgs_step._cache_size()
    learn_radial_density(prob, ds, prob.model.rnl_Wnlq, steps=6, **kw)          # 3 rounds, new precond each
    assert _lbfgs_step._cache_size() == n0


def test_learn_radial_density_absent_species_is_zero_and_finite():
    """SiGe basis on Si-only data: Ge has no sites, so its density weights are exactly 0."""
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem
    from ace_jax.fit.radial_density import learn_radial_density
    from ace_jax.fit.radial_model import to_analytic
    spline, meta, z = load(FIXTURE_DIR / "sige_nofit.npz")
    model, _ = to_analytic(spline, 12)
    ds = build_dataset(load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:6], meta,
                       np.asarray(z["E0"]), configs_per_batch=3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.ones(cfg.len_basis), default_prior(2.35))
    ge = list(meta["elements"]).index(32)
    V, eta, info = learn_radial_density(prob, ds, model.rnl_Wnlq, mask=density_mask(cfg, "full"), P=1,
                                        theta0=THETA, profile=False, steps=3, reprofile_every=3)
    assert np.all(np.isfinite(np.asarray(eta))) and np.all(np.isfinite(np.asarray(V)))
    assert np.all(np.asarray(eta)[:, ge] == 0.0)
    assert all(np.isfinite(info["trace"]))


def test_learn_radial_density_rejects_bad_options(small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import learn_radial_density
    prob, ds, _ = small
    mask = density_mask(prob.cfg, "full")
    with pytest.raises(ValueError, match="mode"):
        learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, mode="nope", theta0=THETA, steps=1)
    with pytest.raises(ValueError, match="P"):
        learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, P=0, theta0=THETA, steps=1)
    with pytest.raises(ValueError, match="lam_eta"):
        learn_radial_density(prob, ds, prob.model.rnl_Wnlq, mask=mask, lam_eta=-1.0, theta0=THETA, steps=1)


def test_fit_radial_density_gate_prefers_density_on_density_data():
    from ace_jax.fit.radial_density import fit_radial_density
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, eta_t, mask, c = _density_truth(prob, ds_fit)
    ds_fit, ds_val = _relabel_rd(prob, ds_fit, Wt, eta_t, mask, c), _relabel_rd(prob, ds_val, Wt, eta_t, mask, c)
    W, eta, info = fit_radial_density(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, mask=mask, P=1,
                                      theta0=None, profile=False, steps=20, map_steps=50)
    print(info["scores"])
    assert info["selected"].startswith("density") and eta is not None and info["P"] == 1
    assert info["readout"].shape == (prob.cfg.len_basis + prob.cfg.NZ,)
    assert set(info["scores"]) == {"init", "radials_only", "density_lam_eta=0"}


def test_fit_radial_density_gate_rejects_density_at_true_radials():
    """Start the fit at the true radials (no radial error for the density to
    absorb), so the linear-data gate has nothing to gain from the extra
    per-species density flexibility."""
    from test_gp_learn_radial import _perturbed_truth, relabel
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import fit_radial_density
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, _, c = _perturbed_truth(prob)
    W0 = Wt
    ds_fit, ds_val = relabel(prob, ds_fit, Wt, c), relabel(prob, ds_val, Wt, c)
    W, eta, info = fit_radial_density(prob, ds_fit, ds_val, W0, mask=density_mask(prob.cfg, "full"), P=1,
                                      theta0=None, profile=False, steps=20, map_steps=50)
    print(info["scores"])
    assert info["selected"] in ("init", "radials_only") and eta is None and info["P"] == 0
    assert info["readout"].shape == (prob.cfg.len_basis,)


def test_fit_radial_density_gate_selects_min_score():
    """The perturbed-W0 setup (test_fit_radial_density_gate_rejects_density_at_true_radials'
    predecessor): with unconverged radials the density candidate can legitimately win on
    held-out data, because it compensates for the radial error rather than any noise in
    the (noiseless) targets. This test only checks the gate picks the argmin, not which
    label wins."""
    from test_gp_learn_radial import _perturbed_truth, relabel
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import fit_radial_density
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, W0, c = _perturbed_truth(prob)
    ds_fit, ds_val = relabel(prob, ds_fit, Wt, c), relabel(prob, ds_val, Wt, c)
    W, eta, info = fit_radial_density(prob, ds_fit, ds_val, W0, mask=density_mask(prob.cfg, "full"), P=1,
                                      theta0=None, profile=False, steps=20, map_steps=50)
    print(info["scores"])
    assert all(np.isfinite(s) for s in info["scores"].values())
    assert info["selected"] == min(info["scores"], key=info["scores"].get)


def test_saved_density_model_matches_widened_rows(tmp_path):
    """model.npz from save_result(eta=...) evaluates to [linear | density] rows @ readout + E0."""
    from ace_jax.eval import load
    from ace_jax.eval.fs_model import FSModel
    from ace_jax.fit.radial_density import fit_radial_density
    from ace_jax.fit.radial_learn import save_result
    from ace_jax.fit.radial_model import with_radial
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, eta_t, mask, c = _density_truth(prob, ds_fit)
    ds_fit, ds_val = _relabel_rd(prob, ds_fit, Wt, eta_t, mask, c), _relabel_rd(prob, ds_val, Wt, eta_t, mask, c)
    W, eta, info = fit_radial_density(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, mask=mask, P=1,
                                      theta0=None, profile=False, steps=5, map_steps=20)
    assert eta is not None
    save_result(tmp_path, W, info, src_npz=MODEL, model=prob.model, eta=eta, mask=mask)
    back, meta, z = load(tmp_path / "model.npz")
    assert isinstance(back, FSModel) and np.load(tmp_path / "eta.npy").shape == eta.shape
    E0 = np.asarray(z["E0"]); m = with_radial(prob.model, W); cr = jnp.asarray(info["readout"])
    worst = 0.0
    for i in range(ds_val.n_batches):
        b = jax.tree.map(lambda a: a[i], ds_val)
        r = _widened_rows(m, prob.cfg, b, eta, mask)
        C = b.y_E.shape[0]; Ncap, K = b.nbr.shape
        zi = jnp.broadcast_to(b.node_z[:, None], (Ncap, K))
        e = back.site_energies_dense(b.rij, zi, b.node_z[b.nbr], b.nbr_mask, b.node_z)
        E_model = np.asarray(jax.ops.segment_sum(e, b.node_cfg, num_segments=C + 1)[:C])
        E0sum = np.asarray(jax.ops.segment_sum(jnp.where(b.node_mask, jnp.asarray(E0)[b.node_z], 0.0),
                                               b.node_cfg, num_segments=C + 1)[:C])
        E_lin = np.asarray(r.E @ cr) + E0sum
        worst = max(worst, float(np.max(np.abs(E_model - E_lin) / np.abs(E_lin))))
    assert worst < 1e-8


def test_save_result_eta_without_mask_raises(tmp_path):
    from ace_jax.fit.radial_learn import save_result
    with pytest.raises(ValueError, match="mask"):
        save_result(tmp_path, np.zeros(3), {}, eta=np.zeros((1, 2, 3)))


def _driver(tmp_path, *extra, model=MODEL):
    import subprocess, sys
    from conftest import ROOT
    return subprocess.run(
        [sys.executable, str(ROOT / "bench/learn_radial/run.py"), "--model", str(model),
         "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
         "--virial-key", "dft_virial", "--ntrain", "8", "--nval", "8", "--batch", "4",
         "--n-q", "20", "--steps", "3", "--lam-grid", "0", "--map-steps", "20",
         "--out", str(tmp_path), *extra], capture_output=True, text=True)


def test_bench_driver_density_smoke(tmp_path):
    import json, subprocess, sys
    from conftest import ROOT
    r = _driver(tmp_path, "--density", "full", "--P", "1")
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["density"] == "full" and s["selected"] in s["scores"] and "radials_only" in s["scores"]
    assert (tmp_path / "model.npz").exists()
    r2 = subprocess.run([sys.executable, str(ROOT / "bench/learn_radial/rmse_npz.py"), "--model", str(MODEL),
                         "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
                         "--virial-key", "dft_virial", "--ntrain", "8", "--nval", "8",
                         "--model-npz", str(tmp_path / "model.npz"), "--out", str(tmp_path / "rmse.json")],
                        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr[-3000:]
    js = json.loads((tmp_path / "rmse.json").read_text())
    (row,) = js.values()
    assert np.isfinite(row["E_rmse_meV_atom"]) and np.isfinite(row["F_rmse_meV_A"])


def test_bench_driver_density_rejects_grids(tmp_path):
    r = _driver(tmp_path, "--density", "full", "--spec-grid", "0,1e-5")
    assert r.returncode != 0 and "single-valued" in (r.stderr + r.stdout)


def test_bench_driver_unwraps_density_model(tmp_path):
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.eval import load
    base, meta, _ = load(MODEL)
    NZ, D = len(meta["elements"]), meta["n_B"] + meta["n_pair"]
    patch_radial_npz(MODEL, tmp_path / "fs.npz", base,
                     fs=(1e-2 * np.ones((1, NZ, D)), np.ones((1, NZ)), np.ones(D), 1e-6))
    r = _driver(tmp_path / "out", model=tmp_path / "fs.npz")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "density term of the input model is dropped" in r.stdout


def test_fit_radial_density_checkpoints(tmp_path, small):
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import fit_radial_density
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    seen = []
    fit_radial_density(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, mask=density_mask(prob.cfg, "pair"),
                       P=1, lam_eta_grid=(0.0, 1e-2), theta0=THETA, profile=False, steps=2, map_steps=20,
                       checkpoint=lambda k, W, eta, info: seen.append((k, eta is None)))
    assert seen == [("radials_only", True), ("density_lam_eta=0", False), ("density_lam_eta=0.01", False)]
