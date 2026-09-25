import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_predict_uses_explicit_stats(tiny_linear_problem):
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.predict import predict_mixture
    from ace_jax.fit.stats import sufficient_statistics
    prob, ds = tiny_linear_problem
    theta = prob.prior.mu
    calls = []
    def stats(th):
        calls.append(1)
        return sufficient_statistics(th, prob.spec, prob.model, prob.ind, prob.cfg, ds)
    d = np.asarray(to_array(theta))[None]
    ref = predict_mixture(d, prob, ds, ds)
    got = predict_mixture(d, prob, ds, ds, stats=stats)
    assert calls == [1]
    for f in ref._fields:
        assert np.allclose(np.asarray(getattr(got, f)), np.asarray(getattr(ref, f)), rtol=1e-12)
