"""`ham.uq`: split-conformal half-widths of eigenvalues, per energy band and joint, and of matrix elements."""
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


def _block_set(rng, n_struct, scale):
    """Structures of two species with on-site and off-site blocks whose error grows towards short bonds and differs by shell pair."""
    from ace_jax.operators.orbitals import OrbitalBasis
    orb = OrbitalBasis.of({1: (0,), 6: (0, 1)})
    numbers, cell, out = np.array([1, 6, 6, 1]), 6.0 * np.eye(3), []
    for _ in range(n_struct):
        pos = rng.uniform(0, 6, (4, 3)); ref, pred = {}, {}
        for i in range(4):
            for j in range(4):
                for R in ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
                    k = (*R, i, j); on = R == (0, 0, 0) and i == j
                    r = 0.0 if on else np.linalg.norm(pos[j] - pos[i] + np.array(R) @ cell)
                    shape = (orb.n_orb(numbers[i]), orb.n_orb(numbers[j])); s = scale * (5.0 if on else np.exp(-r / 3)) * np.ones(shape)
                    s[0, :] *= 3.0                                                # the first shell of the first atom: larger errors
                    ref[k] = rng.normal(size=shape); pred[k] = ref[k] + s * rng.normal(size=shape)
        out.append((numbers, pos, cell, ref, pred))
    return orb, out


def test_block_half_widths_cover_per_group():
    """Block calibration: the elements of new structures lie inside their half-widths at the levels, in
    every distance band and shell pair (one width for all would not: the errors differ fifteenfold),
    whole-block widths bound the largest element, and a species pair the calibration lacks is inf."""
    rng = np.random.default_rng(0)
    orb, cal_set = _block_set(rng, 60, 1e-3); _, test_set = _block_set(rng, 30, 1e-3)
    be = uq.BlockErrors(orb)
    for s in cal_set:
        be.add("H", *s)
    cal = be.calibrate(n_dist=3)
    assert set(cal["channels"]["H"]) == {"1-1", "1-6", "6-1", "6-6"} and len(cal["channels"]["H"]["6-6"]["bands"]) == 4
    assert cal["channels"]["H"]["1-6"]["bands"][0]["n"] == 0 and cal["channels"]["H"]["1-1"]["bands"][0]["n"] == 120      # on-site blocks: same atom only
    bt = uq.BlockErrors(orb)
    for s in test_set:
        bt.add("H", *s)
    for row in bt.coverage(cal)["H"]:
        assert abs(row["elements"] - row["level"]) < 0.03 and abs(row["whole"] - row["level"]) < 0.06 and row["structure_min"] > row["level"] - 0.2
    numbers, pos, cell, ref, pred = test_set[0]
    h, hw = uq.block_half_widths(cal, orb, numbers, pos, cell, ref, level=0.9), uq.block_half_widths(cal, orb, numbers, pos, cell, ref, level=0.9, whole=True)
    k = (0, 0, 0, 1, 2)                                                           # a C-C bond: s row three times the p rows
    assert h[k].shape == (4, 4) and 2.0 < h[k][0, 1] / h[k][1, 1] < 4.5 and hw[k] > h[k].max() * 0.9
    inside = np.mean([np.mean(np.abs(pred[q] - ref[q]) <= h[q]) for q in ref])
    assert 0.8 < inside < 0.97 and h[(0, 0, 0, 1, 1)][1, 1] > 5 * h[(1, 0, 0, 1, 2)][1, 1]                   # on-site wider than a bond
    with pytest.raises(KeyError):
        uq.block_half_widths(cal, orb, np.array([8, 8, 8, 8]), pos, cell, [(0, 0, 0, 0, 1)])
    assert all(np.isinf(v).all() for v in uq.block_half_widths(cal, orb, numbers, pos, cell, [k], channel="S").values())   # a channel without calibration
    assert "block calibration: 60 structures" in uq.format_block_calibration(cal)[0]
