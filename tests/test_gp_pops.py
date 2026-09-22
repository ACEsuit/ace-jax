import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ace_jax.fit.pops import (
    pops_corrections,
    pops_posterior,
    pops_var,
    whiten,
)


def test_whiten_scales_rows_by_w_over_sigma():
    phi = jnp.ones((3, 2)); r = jnp.array([1.0, 2.0, 3.0])
    w = jnp.array([2.0, 2.0, 2.0]); sigma = 4.0
    pt, rt = whiten(phi, r, w, sigma)
    assert jnp.allclose(pt, 0.5 * phi) and jnp.allclose(rt, 0.5 * r)


@pytest.fixture
def pops_synth():
    """A MISSPECIFIED linear synthetic: fit a straight line (features [1, x])
    to strongly quadratic data, so the model cannot fit exactly and the
    residual is systematic. Everything is treated as already-whitened
    (homoscedastic scale 1), so ``Sigma0 = A^{-1}`` with A the ridge
    precision. Returns ``(Sigma0, phi_t, r_t, c, phi_star, epi_var)``.
    """
    N = 60
    x = jnp.linspace(-1.0, 1.0, N)
    # strong quadratic target that a line cannot represent
    y = 2.0 + 0.5 * x + 3.0 * x ** 2
    phi_t = jnp.stack([jnp.ones_like(x), x], axis=1)  # (N, 2)

    lam = 1.0e-3
    A = phi_t.T @ phi_t + lam * jnp.eye(2)
    Sigma0 = jnp.linalg.inv(A)            # epistemic weight covariance A^{-1}
    c = Sigma0 @ (phi_t.T @ y)            # inner-MAP line fit
    r_t = y - phi_t @ c                   # systematic misspecification residual

    x_star = jnp.linspace(-1.0, 1.0, 20)
    phi_star = jnp.stack([jnp.ones_like(x_star), x_star], axis=1)
    epi_var = jnp.sum((phi_star @ Sigma0) * phi_star, axis=1)
    return Sigma0, phi_t, r_t, c, phi_star, epi_var


def test_pops_var_exceeds_epistemic_under_misspecification(pops_synth):
    Sigma0, phi_t, r_t, c, phi_star, epi_var = pops_synth
    d = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=0.0)
    post = pops_posterior(d, c, form="samples")
    v = pops_var(phi_star, post)
    assert jnp.all(v >= 0.5 * epi_var)            # misspecification inflates
    assert v.mean() > epi_var.mean()


def test_pops_corrections_fit_their_own_point(pops_synth):
    # the Newton step for point i must fit point i exactly: phi_i . (c + d_i) = y_i
    Sigma0, phi_t, r_t, c, phi_star, epi_var = pops_synth
    d = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=0.0)
    fitted = jnp.sum(phi_t * d, axis=1)           # phi_i . d_i
    assert jnp.allclose(fitted, r_t, atol=1e-8)


def test_pops_leverage_percentile_keeps_top_fraction(pops_synth):
    Sigma0, phi_t, r_t, c, phi_star, epi_var = pops_synth
    d_all = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=0.0)
    d_top = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=50.0)
    assert d_all.shape[0] == phi_t.shape[0]
    assert d_top.shape[0] <= d_all.shape[0]
    assert d_top.shape[0] >= 1


def test_pops_hypercube_cov_inflates(pops_synth):
    Sigma0, phi_t, r_t, c, phi_star, epi_var = pops_synth
    d = pops_corrections(Sigma0, phi_t, r_t, leverage_pct=0.0)
    post_cov = pops_posterior(d, c, form="hypercube")
    assert post_cov["cov"].shape == (phi_t.shape[1], phi_t.shape[1])
    v_cov = pops_var(phi_star, post_cov)
    assert jnp.all(v_cov >= 0.0)
    # hypercube misspecification variance also exceeds epistemic on average
    assert v_cov.mean() > epi_var.mean()
