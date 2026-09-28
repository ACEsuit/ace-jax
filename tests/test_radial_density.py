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
