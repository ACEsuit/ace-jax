"""Joint radial + density VarPro (fit.density masked columns, fit.radial_density)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

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
