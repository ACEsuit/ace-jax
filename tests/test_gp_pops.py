import jax.numpy as jnp

from ace_jax.fit.pops import whiten


def test_whiten_scales_rows_by_w_over_sigma():
    phi = jnp.ones((3, 2)); r = jnp.array([1.0, 2.0, 3.0])
    w = jnp.array([2.0, 2.0, 2.0]); sigma = 4.0
    pt, rt = whiten(phi, r, w, sigma)
    assert jnp.allclose(pt, 0.5 * phi) and jnp.allclose(rt, 0.5 * r)
