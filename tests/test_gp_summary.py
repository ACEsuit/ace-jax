import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.fit.summary import site_summary, summary_edge_jacobian


def _star(dists):
    """One centre atom (node 0) with neighbours at the given distances along x."""
    rij = jnp.asarray([[d, 0.0, 0.0] for d in dists])
    seg = jnp.zeros(len(dists), jnp.int32)
    return rij, seg


def test_power_mean_recovers_min_at_large_p():
    # rcut far away so f_c ~ 1: the exact limit is r_nn * f_c(r_nn)^(-1/p)
    rij, seg = _star([2.3, 2.5, 3.1])
    s = site_summary(rij, seg, 1, r0=2.3, rcut=50.0, p=200)
    assert abs(float(s[0]) - 2.3) < 1e-3


def test_power_mean_below_min_with_several_close_neighbours():
    rij, seg = _star([2.3, 2.3, 2.3, 2.3])
    s = site_summary(rij, seg, 1, r0=2.3, rcut=5.0, p=10)
    assert float(s[0]) < 2.3


def test_isolated_atom_is_infinite():
    rij = jnp.zeros((1, 3)).at[0, 0].set(5.0)      # a padded edge at the cutoff
    s = site_summary(rij, jnp.zeros(1, jnp.int32), 1, r0=2.3, rcut=5.0,
                     mask=jnp.asarray([False]))
    assert float(s[0]) > 100 * 2.3 and np.isfinite(float(s[0]))


def test_continuous_at_cutoff():
    a = site_summary(*_star([2.3, 4.999]), 1, r0=2.3, rcut=5.0)
    b = site_summary(*_star([2.3, 5.001]), 1, r0=2.3, rcut=5.0)
    assert abs(float(a[0]) - float(b[0])) < 1e-6


def test_edge_jacobian_matches_finite_difference():
    rij, seg = _star([2.3, 2.6, 3.0])
    rij = rij + jnp.asarray([[0.0, 0.1, -0.2], [0.3, 0.0, 0.1], [-0.1, 0.2, 0.0]])
    s, Js = summary_edge_jacobian(rij, seg, 1, r0=2.3, rcut=5.0)
    h = 1e-6
    for e in range(3):
        for c in range(3):
            dp = rij.at[e, c].add(h); dm = rij.at[e, c].add(-h)
            fd = (site_summary(dp, seg, 1, 2.3, 5.0)[0] - site_summary(dm, seg, 1, 2.3, 5.0)[0]) / (2 * h)
            assert abs(float(Js[e, c]) - float(fd)) < 1e-7


def test_masked_zero_edge_gives_zero_jacobian_without_nan():
    rij = jnp.asarray([[2.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    seg = jnp.zeros(2, jnp.int32)
    mask = jnp.asarray([True, False])
    s, Js = summary_edge_jacobian(rij, seg, 1, r0=2.3, rcut=5.0, mask=mask)
    assert np.isfinite(float(s[0])) and bool(jnp.all(jnp.isfinite(Js)))
    assert float(jnp.abs(Js[1]).max()) == 0.0
    s0, _ = summary_edge_jacobian(rij[:1], seg[:1], 1, r0=2.3, rcut=5.0)
    assert abs(float(s[0]) - float(s0[0])) < 1e-12
