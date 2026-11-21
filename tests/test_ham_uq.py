"""`ham.uq`: split-conformal half-widths of eigenvalues, per energy band and joint."""
import numpy as np
import pytest

from ace_jax.ham import uq


def _pairs(n, seed):
    """Structures whose eigenvalue errors grow with the energy and share a part within a structure."""
    rng, out = np.random.default_rng(seed), []
    for _ in range(n):
        e0 = np.sort(rng.uniform(-10, 4, size=(4, 60)), axis=1)
        sig = 0.002 * (1 + (e0 + 10) / 5)
        out.append((e0, e0 + sig * (0.6 * rng.normal() + 0.8 * rng.normal(size=e0.shape))))
    return out


def test_half_widths_cover_at_their_levels():
    cal = uq.calibrate(_pairs(200, 0), window=(-9.0, 3.0), n_bands=4)
    cov = uq.coverage(cal, _pairs(400, 1))
    for c in cov:                                          # the pointwise and the joint statement, each at its level
        assert abs(c["pointwise"] - c["level"]) < 0.02 and abs(c["joint"] - c["level"]) < 0.06
        assert c["structure_min"] < c["pointwise"] < c["structure_max"]
    p = np.asarray(cal["pointwise"])
    assert np.all(np.diff(p, axis=0) > 0) and np.all(np.diff(p, axis=1) > 0)       # wider at higher energy and at higher level
    assert all(j > o for j, o in zip(cal["joint"], cal["one"])) and cal["n_structures"] == 200
    one = dict(cal, pointwise=[cal["one"]] * 4)            # one width for every energy: the bands' coverages spread
    e0, e1 = (np.concatenate([q[i].ravel() for q in _pairs(400, 1)]) for i in (0, 1))
    spread = lambda c: np.ptp([np.mean((np.abs(e1 - e0) <= uq.half_width(c, e1, 0.9))[(e1 > a) & (e1 < b)]) for a, b in zip(cal["edges"][:-1], cal["edges"][1:])])
    assert spread(cal) < 0.03 < spread(one)


def test_window_levels_and_small_sets():
    cal = uq.calibrate(_pairs(5, 2), window=(-9.0, 3.0), n_bands=2)
    h = uq.half_width(cal, np.array([-9.5, -5.0, 2.0, 3.5]), 0.9)
    assert np.isnan(h[[0, 3]]).all() and np.isfinite(h[[1, 2]]).all()
    assert np.isfinite(cal["joint"][0]) and np.isinf(cal["joint"][1]) and np.isinf(cal["joint"][2])     # 5 structures: 0.68 only
    with pytest.raises(ValueError, match="level"):
        uq.half_width(cal, np.zeros(3), 0.8)
    with pytest.raises(ValueError, match="no eigenvalue"):
        uq.calibrate([(np.full((1, 3), 9.0), np.full((1, 3), 9.0))])
