import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ace_jax.fit.hypers import Hypers, Prior, default_prior, from_array, log_prior, to_array


def test_roundtrip_and_field_order():
    h = Hypers(*[float(i) for i in range(10)])
    assert from_array(to_array(h)) == h
    assert Hypers._fields[0] == "log_ell" and Hypers._fields[-1] == "log_sigma_V"


def test_log_prior_is_sum_of_normals():
    prior = default_prior(r0=2.35)
    theta = prior.mu
    # at the mean every term is -0.5 log(2 pi sigma^2)
    expect = sum(-0.5 * np.log(2 * np.pi * s * s) for s in prior.sigma)
    assert abs(float(log_prior(theta, prior)) - expect) < 1e-12
    shifted = from_array(to_array(theta) + 1.0)
    assert float(log_prior(shifted, prior)) < float(log_prior(theta, prior))


def test_default_prior_centres_r0():
    prior = default_prior(r0=2.35)
    assert abs(float(jnp.exp(prior.mu.log_r0)) - 2.35) < 1e-12


def test_sigma_type_recovers_injected_ratio(two_type_synthetic):
    # fixture: 2 config-types, type-1 energies noisier by 3x (built in conftest).
    from ace_jax.fit.fit_api import fit_linear_with_sigma_type
    ratios = fit_linear_with_sigma_type(two_type_synthetic)   # exp(log_ratios)[:, 0]
    assert 2.0 < float(ratios[1] / ratios[0]) < 4.5           # ~3x recovered by evidence

