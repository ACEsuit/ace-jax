"""ACECalculator picks the A-basis form (EdgeSiteModel.edge_a) automatically.

Which form is faster depends on backend, dtype and edge count (on an A4500 the
matmul form is 3x faster for f32 forces and 1.7x slower for f64), so "auto"
calibrates on the real neighbour list -- once per power-of-two edge bucket --
and small systems, where compile time dominates, just use the gather.  The
A-form applies to the sparse layout, so these tests request it explicitly.

The form is an ACEModel property: PACE builds A pool-first and never forms
per-edge A rows, so for PACE the calibration is a no-op (Task 5 ruling; the
form tests here moved from a PACE fixture to an ACEModel one).
"""
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
from ase import Atoms

import ace_jax.calc.point as point
from ace_jax.calc.point import ACECalculator
from conftest import FIXTURE_DIR, pace_fixture

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


def _ref_atoms(z):
    return Atoms(numbers=np.asarray(z["test_Z"]), positions=np.asarray(z["test_pos"]).T,
                 cell=np.asarray(z["test_cell"]).T, pbc=np.asarray(z["test_pbc"]).astype(bool))


@pytest.fixture
def case():
    """A two-species ACEModel (the form applies to ACEModel only) on its test structure."""
    p = FIXTURE_DIR / "sige_nofit.npz"
    if not p.exists():
        pytest.skip(f"{p.name} not generated")
    return str(p), _ref_atoms(np.load(p))


def _efs(calc, at):
    a = at.copy()
    a.calc = calc
    return a.get_potential_energy(), a.get_forces(), a.get_stress()


def test_default_is_auto_and_small_systems_use_gather(case, monkeypatch):
    y, at = case
    calls = []
    monkeypatch.setattr(point, "calibrate_edge_a", lambda *a, **k: calls.append(1))
    calc = ACECalculator(y, layout="sparse")
    assert calc.edge_a_kind == "auto"
    _efs(calc, at)
    assert calls == [] and calc.last_edge_a_kind == "gather"   # below AUTO_MIN_EDGES


def test_auto_calibrates_once_per_edge_bucket(case, monkeypatch):
    y, at = case
    monkeypatch.setattr(point, "AUTO_MIN_EDGES", 0)
    real = point.calibrate_edge_a
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(point, "calibrate_edge_a", spy)
    calc = ACECalculator(y, layout="sparse")
    E_auto, F_auto, S_auto = _efs(calc, at)
    at2 = at.copy()
    at2.rattle(0.01, seed=5)                  # same edge bucket: no re-calibration
    _efs(calc, at2)
    assert len(calls) == 1
    assert calc.last_edge_a_kind in ("gather", "matmul")
    E, F, S = _efs(ACECalculator(y, edge_a_kind="gather", layout="sparse"), at)
    assert E_auto == pytest.approx(E, abs=1e-12)
    np.testing.assert_allclose(F_auto, F, atol=1e-12)
    np.testing.assert_allclose(S_auto, S, atol=1e-12)


@pytest.mark.parametrize("kind", ["gather", "matmul"])
def test_explicit_kind_skips_calibration(case, monkeypatch, kind):
    y, at = case
    monkeypatch.setattr(point, "AUTO_MIN_EDGES", 0)
    monkeypatch.setattr(point, "calibrate_edge_a", lambda *a, **k: pytest.fail("calibrated"))
    calc = ACECalculator(y, edge_a_kind=kind, layout="sparse")
    _efs(calc, at)
    assert calc.last_edge_a_kind == kind


def test_bad_kind_rejected(case):
    y, _ = case
    with pytest.raises(ValueError, match="edge_a_kind"):
        ACECalculator(y, edge_a_kind="scatter")


def test_pace_is_pool_first_and_never_calibrates(monkeypatch):
    """PACE never forms per-edge A rows, so "auto" (and an explicit form) is a
    no-op for it: no calibration, and no form is reported."""
    y = pace_fixture(FIX / "gesi_sbessel.yace")
    ref = np.load(pace_fixture(FIX / "gesi_sbessel_ref.npz"))
    at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=True)
    monkeypatch.setattr(point, "AUTO_MIN_EDGES", 0)
    monkeypatch.setattr(point, "calibrate_edge_a", lambda *a, **k: pytest.fail("calibrated"))
    E = {}
    for kind in ("auto", "matmul"):
        calc = ACECalculator(str(y), edge_a_kind=kind, layout="sparse")
        E[kind] = _efs(calc, at)[0]
        assert calc.last_edge_a_kind is None
    assert E["auto"] == E["matmul"]
