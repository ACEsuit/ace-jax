"""Multi-start L-BFGS MAP (fit/multistart.py).  The joint GP LML is multimodal:
on Cantor-1k the same start reached log-posteriors 620 nats apart depending on
round-off (device vs host-cache paths diverged after ~10 evaluations), so the MAP
is chosen as the best of several starts."""
import numpy as np
import pytest

from ace_jax.fit.hypers import Hypers, Prior
from ace_jax.fit.multistart import lbfgs_map, multistart_map, prior_starts


def _double_well(x):
    """log-posterior -(x0^2 - 1)^2 - x1^2 + 0.3 x0: local max near x0 = -1, global near +1."""
    x = np.asarray(x, float)
    v = -(x[0] ** 2 - 1) ** 2 - x[1] ** 2 + 0.3 * x[0]
    g = np.array([-4 * x[0] * (x[0] ** 2 - 1) + 0.3, -2 * x[1]])
    return v, g


LO, HI = np.array([-3.0, -3.0]), np.array([3.0, 3.0])


def test_single_start_stops_in_the_basin_it_starts_in():
    # at the local maximum (grad ~ 0) a single run cannot leave it
    r = lbfgs_map(_double_well, np.array([-0.962, 0.0]), LO, HI, maxiter=100)
    assert r["x"][0] < 0 and r["value"] < _double_well([1.0, 0.0])[0]


def test_multistart_keeps_the_best_run_and_reports_all():
    starts = [np.array([-0.962, 0.0]), np.array([0.8, -0.3]), np.array([-2.0, 1.0])]
    best, runs = multistart_map(_double_well, starts, LO, HI, maxiter=100)
    assert len(runs) == 3 and [r["start"] for r in runs] == [0, 1, 2]
    assert best["x"][0] > 0.9                                           # the global basin
    assert best["value"] == max(r["value"] for r in runs)


def test_one_start_is_exactly_the_single_run():
    x0 = np.array([-0.962, 0.0])
    best, runs = multistart_map(_double_well, [x0], LO, HI, maxiter=100)
    single = lbfgs_map(_double_well, x0, LO, HI, maxiter=100)
    assert np.array_equal(best["x"], single["x"]) and best["value"] == single["value"]


def test_non_finite_points_are_penalised_not_fatal():
    def vg(x):
        return (np.nan, np.full(2, np.nan)) if x[0] > 1.5 else _double_well(x)
    r = lbfgs_map(vg, np.array([2.0, 0.0]), LO, HI, maxiter=50)      # starts in the bad region
    assert np.isfinite(r["value"]) or r["value"] == -np.inf


def _prior():
    mu = Hypers(*np.zeros(10)); sigma = Hypers(*np.full(10, 2.0))
    return Prior(mu, sigma)


def test_prior_starts_first_is_x0_then_seeded_clipped_draws():
    lo, hi = np.full(10, -1.0), np.full(10, 1.0)
    lo[5] = hi[5] = 0.3                                                 # a pinned coordinate
    x0 = np.full(10, 0.3)
    s = prior_starts(_prior(), 4, lo, hi, x0, seed=7)
    assert len(s) == 4 and np.array_equal(s[0], x0)
    for x in s[1:]:
        assert np.all(x >= lo) and np.all(x <= hi) and x[5] == 0.3
    assert not np.allclose(s[1], s[2])
    s2 = prior_starts(_prior(), 4, lo, hi, x0, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(s, s2))              # deterministic


def test_prior_starts_rejects_zero():
    with pytest.raises(ValueError):
        prior_starts(_prior(), 0, np.full(10, -1.0), np.full(10, 1.0), np.zeros(10), seed=0)
