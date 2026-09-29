"""SBessel recurrence: exactness against the pre-change direct evaluation, and
that the transcendental (sin/cos) count is O(1) in K, not O(K)."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from ace_jax.eval import pace_radial as pr


def _direct(r, rc, K):
    """The pre-change definition: one sinc per k."""
    import math
    def f(n):
        pre = ((-1) ** n * math.sqrt(2) * math.pi * (n + 1) * (n + 2)
               / math.sqrt((n + 1) ** 2 + (n + 2) ** 2))
        return pre / rc ** 1.5 * (pr._sinc(r * (n + 1) * math.pi / rc)
                                  + pr._sinc(r * (n + 2) * math.pi / rc))
    g, d_prev = [f(0)], 1.0
    for n in range(1, K):
        en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = 1 - en / d_prev
        g.append((f(n) + math.sqrt(en / d_prev) * g[-1]) / math.sqrt(dn))
        d_prev = dn
    return jnp.stack(g, axis=-1)


def test_sbessel_recurrence_is_exact_and_cheap():
    r = jnp.linspace(1e-6, 4.9, 997)
    for K in (0, 1, 4, 8, 12):
        np.testing.assert_allclose(pr._sbessel(r, 5.0, K), _direct(r, 5.0, K), rtol=1e-12, atol=1e-13)
    hlo = jax.jit(lambda x: pr._sbessel(x, 5.0, 12)).lower(r).as_text()
    # StableHLO op names, not bare "sine"/"cosine": "cosine" contains "sine" as a
    # substring, so the bare form double-counts stablehlo.cosine as a sine too.
    assert hlo.count("stablehlo.sine") <= 1 and hlo.count("stablehlo.cosine") <= 1  # not K+1 of each
