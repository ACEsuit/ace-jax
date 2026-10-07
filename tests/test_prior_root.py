import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ace_jax.fit.prior_root import PriorRoot, prior_root  # noqa: E402


def _dense(root):
    L, M = len(root.dinv), root.M
    R0 = np.zeros((L + M, L + M))
    R0[np.arange(L), np.arange(L)] = 1.0 / np.asarray(root.dinv)
    if M:
        R0[L:, L:] = np.asarray(root.U)
    return R0


def _random_root(M, seed=0):
    rng = np.random.default_rng(seed)
    dinv = rng.uniform(0.5, 2.0, 7)
    if M == 0:
        return PriorRoot.diag(dinv)
    A = rng.normal(size=(M, M))
    K = A @ A.T + M * np.eye(M)
    return PriorRoot(dinv, jnp.asarray(np.linalg.cholesky(K).T))


@pytest.mark.parametrize("M", [0, 4])
def test_prior_root_ops_match_dense(M):
    root = _random_root(M)
    R0 = _dense(root)
    Ri = np.linalg.inv(R0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2, 3, root.width))
    G = rng.normal(size=(root.width, 5))
    x = rng.normal(size=root.width)
    Mx = rng.normal(size=(root.width, root.width))
    Mx = Mx + Mx.T
    np.testing.assert_allclose(root.rows(X), X @ Ri, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.dual(G), Ri.T @ G, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.dual(x), Ri.T @ x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.primal(x), Ri @ x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.lift(x), R0.T @ x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(root.gram(Mx), Ri.T @ Mx @ Ri, rtol=1e-11, atol=1e-11)


def test_prior_root_diag_is_todays_dinv_bitwise():
    """M = 0 evaluates the linear arm's dinv expressions verbatim (--uq ard stays bit-identical)."""
    root = _random_root(0)
    d = jnp.asarray(root.dinv)
    X = jnp.asarray(np.random.default_rng(2).normal(size=(5, 7)))
    assert np.array_equal(np.asarray(root.rows(X)), np.asarray(X * d[None, :]))
    assert np.array_equal(np.asarray(root.gram(X.T @ X)), np.asarray(d[:, None] * (X.T @ X) * d[None, :]))
    assert np.array_equal(np.asarray(root.lift(X[0])), np.asarray(X[0] / d))


def test_prior_root_from_problem(tiny_gp_problem):
    from ace_jax.fit.ard import ard_gamma
    from ace_jax.fit.kernels import K_MM
    prob, _, theta = tiny_gp_problem
    root = prior_root(prob, theta)
    assert root.M == prob.ind.XM.shape[0] > 0
    np.testing.assert_array_equal(root.dinv, 1.0 / ard_gamma(prob))
    K = np.asarray(K_MM(theta, prob.spec, prob.ind.XM, prob.ind.SM, prob.ind.ZM, prob.ind.embed))
    np.testing.assert_allclose(np.asarray(root.U).T @ np.asarray(root.U), K, rtol=1e-10, atol=1e-14)


def test_prior_root_near_duplicate_inducing_points(tiny_gp_problem):
    """Two inducing points 1e-9 apart: K_MM's jitter keeps chol finite, and rows/dual stay finite."""
    prob, _, theta = tiny_gp_problem
    ind = prob.ind
    XM = ind.XM.at[1].set(ind.XM[0] + 1e-9)
    SM = ind.SM.at[1].set(ind.SM[0])
    ZM = ind.ZM.at[1].set(ind.ZM[0])
    root = prior_root(prob._replace(ind=ind._replace(XM=XM, SM=SM, ZM=ZM)), theta)
    v = jnp.ones(root.width)
    assert np.isfinite(np.asarray(root.U)).all()
    assert np.isfinite(np.asarray(root.dual(v))).all() and np.isfinite(np.asarray(root.rows(v[None]))).all()
