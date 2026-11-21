"""`operators.posterior`: the coefficient posterior of a block fit from its statistics."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from ace_jax.fit.linear import stats as ls
from ace_jax.operators import posterior as P
from ace_jax.operators.fit import DesignStats


def _stats(hyper):
    rng = np.random.default_rng(0)
    A, k, nc = rng.normal(size=(40, 5)), 5, 2
    truth = rng.normal(size=(k, 2 * nc))
    Y = A @ truth + 0.1 * rng.normal(size=(40, 2 * nc))
    Q, R = np.linalg.qr(A)
    c = Q.T @ Y
    fac = ls.Factor(R, c, (Y ** 2).sum(0), 40.0, 0.0)
    gamma = np.array([1.0, 2.0, 3.0, 0.0, 0.5])
    if hyper:                                            # per right-hand side: sigma, sigma_c, criterion, sigma_c solved with
        h = np.stack([[0.1, 1.0, 0.0, 1.0], [0.1, 1.0, 0.0, 1.0], [0.2, 0.5, 0.0, 0.5], [0.3, 3.0, 0.0, 3.0]])
        return A, Y, DesignStats(fac, ("a", "b"), h, gamma, 1e-8), nc
    return A, Y, DesignStats(fac, ("a", "b"), None, gamma, 1e-8), nc


def test_draws_have_the_posterior_covariance_at_selected_priors():
    A, Y, st, nc = _stats(True)
    n = 20000
    mean = {"a": np.zeros((nc, 5)), "b": np.ones((nc, 5))}
    out = P.draws(mean, {"d": st}, n, nc, seed=3)
    assert out["a"].shape == (nc * (1 + n), 5) and np.array_equal(out["b"][:nc], mean["b"])
    for q, (key, ch) in enumerate((("a", 0), ("a", 1), ("b", 0), ("b", 1))):
        sigma, sc = st.hyper[q, 0], st.hyper[q, 3]
        lin = np.where(st.gamma > 0, st.gamma ** 2 * sigma ** 2 / sc ** 2, st.ridge)
        cov = sigma ** 2 * np.linalg.inv(A.T @ A + np.diag(lin))
        d = out[key][nc + ch::nc] - mean[key][ch]
        assert np.abs(d.mean(0)).max() < 4 * np.sqrt(cov.diagonal().max() / n)
        assert np.abs(d.T @ d / n - cov).max() < 0.05 * np.abs(cov).max()
    assert np.allclose(np.concatenate([P.noise({"d": st}, nc)[k] for k in ("a", "b")]), st.hyper[:, 0])


def test_fixed_prior_takes_the_residual_as_noise():
    A, Y, st, nc = _stats(False)
    lin = st.ridge + st.gamma
    mu = np.linalg.solve(A.T @ A + np.diag(lin), A.T @ Y)
    sigma = np.sqrt(((A @ mu - Y) ** 2).sum(0) / (40 - 5))
    nz = P.noise({"d": st}, nc)
    assert np.allclose(np.concatenate([nz["a"], nz["b"]]), sigma, rtol=1e-10)
    assert P.channel_names(("H", "S"), 2) == ("H", "S", "H#1", "S#1", "H#2", "S#2")


def test_jackknife_deviations_are_the_refits_without_each_structure():
    from ace_jax.operators.fit import _jackknife
    rng = np.random.default_rng(1)
    G, m, k = 6, 9, 5
    A, Y = rng.normal(size=(G, m, k)), rng.normal(size=(G, m, 3))
    gamma, ridge, rho = np.array([1.0, 2.0, 0.0, 0.5, 3.0]), 1e-8, np.array([-2.0, 0.5, -2.0])
    blocks = []
    for g in range(G):
        Q, R = np.linalg.qr(A[g])
        blocks.append((R, Q.T @ Y[g], np.zeros(3)))
    Af, Yf = A.reshape(-1, k), Y.reshape(-1, 3)
    lin = lambda r: np.where(gamma > 0, gamma ** 2 * 10.0 ** r, ridge)
    solve = lambda X, y, r: np.linalg.solve(X.T @ X + np.diag(lin(r)), X.T @ y)
    mean = np.stack([solve(Af, Yf[:, q], rho[q]) for q in range(3)], axis=1)
    Q, R = np.linalg.qr(Af)
    fac = ls.Factor(R, Q.T @ Yf, (Yf ** 2).sum(0), float(len(Af)), 0.0)
    J = _jackknife(fac, gamma, ridge, blocks, rho, mean)
    assert J.shape == (3, k, G)
    for g in range(G):
        keep = np.arange(G) != g
        for q in range(3):
            ref = solve(A[keep].reshape(-1, k), Y[keep].reshape(-1, 3)[:, q], rho[q]) - mean[:, q]
            assert np.allclose(J[q, :, g], np.sqrt((G - 1) / G) * ref, rtol=1e-8, atol=1e-10)
    st = DesignStats(fac, ("a", "b", "c"), None, gamma, ridge, J)
    out = P.jackknife_draws({key: mean[:, [i]].T for i, key in enumerate("abc")}, {"d": st}, 4, 1, seed=2)
    xi = np.random.default_rng(2).normal(size=(G, 4))
    assert np.allclose(out["b"][1:], mean[:, 1] + (J[1] @ xi).T)
