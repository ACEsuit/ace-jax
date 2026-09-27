"""ACECalculator's skin (Verlet) neighbour list: the (n, K_skin) list is built
for cutoff + skin and reused, one compiled step per call, until an atom has
moved skin / 2, or the cell, pbc or species change.  Every result must equal a
fresh rebuild (skin=0) to 1e-12, including the Review Focus cases: atoms with
no neighbours, triclinic cells, small cells with many images per pair, and an
unrelated structure of the same size."""
import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes

import ace_jax.calc.skin as skin_mod
from ace_jax.calc.point import ACECalculator
from conftest import FIXTURE_DIR

M = str(FIXTURE_DIR / "sige_nofit.npz")


def _efs(calc, at):
    calc.calculate(at, ["energy", "forces", "stress"], all_changes)
    r = calc.results
    return r["energy"], r["forces"].copy(), r.get("stress", np.zeros(6)).copy()


def _cell(kind):
    if kind == "tric":
        a = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
        a.set_cell(a.cell.array @ np.array([[1, .08, 0], [0, 1, .05], [0, 0, 1]]), scale_atoms=True)
    elif kind == "small":
        a = bulk("Si", "diamond", a=5.43)                       # many images per pair
    a.numbers[::2] = 32
    return a


def _close(got, ref):
    E, F, S = got
    E0, F0, S0 = ref
    assert abs(E - E0) < 1e-12 * max(1, abs(E0))
    np.testing.assert_allclose(F, F0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(S, S0, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["tric", "small"])
def test_skin_matches_rebuild_along_a_trajectory(kind):
    at = _cell(kind)
    fresh, reuse = ACECalculator(M, layout="dense", skin=0.0), ACECalculator(M, layout="dense", skin=1.0)
    rng = np.random.default_rng(1)
    for step in range(12):
        at.positions += rng.normal(0, 0.03, at.positions.shape)     # ~0.4 A over the run
        E0, F0, S0 = _efs(fresh, at)
        E1, F1, S1 = _efs(reuse, at)
        assert abs(E1 - E0) < 1e-12 * max(1, abs(E0))
        np.testing.assert_allclose(F1, F0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(S1, S0, rtol=1e-12, atol=1e-12)
    assert reuse.last_timing["rebuilds"] < fresh.last_timing["rebuilds"]


def test_rebuild_triggers():
    at = _cell("tric")
    c = ACECalculator(M, layout="dense", skin=1.0)
    _efs(c, at)
    n0 = c.last_timing["rebuilds"]
    _efs(c, at)  # unchanged
    assert c.last_timing["rebuilds"] == n0
    b = at.copy()  # > skin/2
    b.positions[0] += [0.6, 0, 0]
    _efs(c, b)
    assert c.last_timing["rebuilds"] == n0 + 1
    d = at.copy()  # cell change
    d.set_cell(d.cell * 1.01, scale_atoms=True)
    _efs(c, d)
    assert c.last_timing["rebuilds"] == n0 + 2
    e = at.copy()  # species change
    e.numbers[1] = 14 if e.numbers[1] == 32 else 32
    _efs(c, e)
    assert c.last_timing["rebuilds"] == n0 + 3


def test_unrelated_structure_same_size_rebuilds_and_is_correct():
    a, b = _cell("tric"), _cell("tric")
    b.positions = np.random.default_rng(2).permutation(b.positions)    # same set, relabelled
    c, ref = ACECalculator(M, layout="dense", skin=1.0), ACECalculator(M, layout="dense", skin=0.0)
    _efs(c, a)
    E, F, _ = _efs(c, b)
    E0, F0, _ = _efs(ref, b)
    assert abs(E - E0) < 1e-12 * abs(E0)
    np.testing.assert_allclose(F, F0, rtol=1e-12, atol=1e-12)


def test_isolated_atoms():
    at = Atoms(numbers=[14, 32], positions=[[0, 0, 0], [20, 0, 0]], cell=[40] * 3, pbc=False)
    c = ACECalculator(M, layout="dense", skin=1.0)
    E, F, _ = _efs(c, at)
    E0, F0, _ = _efs(ACECalculator(M, layout="dense", skin=0.0), at)
    assert np.isfinite(E) and np.all(np.isfinite(F))
    assert abs(E - E0) < 1e-12 and np.allclose(F, 0.0)


def test_cluster_with_an_isolated_atom():
    """A row with count == 0 next to rows that have neighbours (Review Focus 1)."""
    at = _cell("tric")
    at.pbc = False
    at.center(vacuum=8.0)
    at += Atoms("Si", positions=[at.cell.array.sum(axis=0) - 1.0])        # far corner, alone
    rng = np.random.default_rng(3)
    c, ref = ACECalculator(M, layout="dense", skin=1.0), ACECalculator(M, layout="dense", skin=0.0)
    for _ in range(4):
        at.positions += rng.normal(0, 0.03, at.positions.shape)
        got, want = _efs(c, at), _efs(ref, at)
        assert np.all(np.isfinite(got[1]))
        _close(got, want)
        assert np.allclose(got[1][-1], 0.0)
    assert c._skin_state.rev_s is not None                              # the gather path ran


def test_scatter_fallback_when_reverse_slots_fails(monkeypatch):
    """Ruling 1: reverse_slots raising (a rounding-boundary case) leaves rev_s
    None and the step assembles forces by the scatter, with the same results."""
    def boom(*a, **k):
        raise ValueError("1 edges without a matched reverse")

    monkeypatch.setattr(skin_mod, "reverse_slots", boom)
    at = _cell("small")
    c, ref = ACECalculator(M, layout="dense", skin=1.0), ACECalculator(M, layout="dense", skin=0.0)
    rng = np.random.default_rng(4)
    for _ in range(4):
        at.positions += rng.normal(0, 0.03, at.positions.shape)
        _close(_efs(c, at), _efs(ref, at))
    assert c._skin_state.rev_s is None


def test_overflow_retries_with_a_larger_K():
    """An atom gaining neighbours inside the cutoff past K (without moving skin/2)
    trips the step's overflow flag: the call rebuilds with a larger K and the
    result still equals a fresh rebuild."""
    at = _cell("tric")
    c, ref = ACECalculator(M, layout="dense", skin=1.0), ACECalculator(M, layout="dense", skin=0.0)
    _efs(c, at)
    c._skin_state = dataclasses.replace(c._skin_state, K=1)             # too small now
    n0 = c.last_timing["rebuilds"]
    at.positions[0] += [0.05, 0, 0]
    _close(_efs(c, at), _efs(ref, at))
    assert c.last_timing["rebuilds"] == n0 + 1 and c._skin_state.K > 1


def test_reuse_reports_no_neighbour_list_time():
    at = _cell("tric")
    c = ACECalculator(M, layout="dense", skin=1.0)
    _efs(c, at)
    assert c.last_timing["nlist_s"] > 0                                 # first call built it
    at.positions[0] += [0.01, 0, 0]
    _efs(c, at)
    t = c.last_timing
    assert t["nlist_s"] == 0 and t["model_s"] > 0 and t["nlist_backend"]
    assert c.last_layout == "dense" and c.last_n_edges > 0


def test_skin_zero_rebuilds_every_call():
    at = _cell("tric")
    c = ACECalculator(M, layout="dense", skin=0.0)
    for k in range(1, 4):
        _efs(c, at)
        assert c.last_timing["rebuilds"] == k and c.last_timing["nlist_s"] > 0
    assert c._skin_state is None


def test_float32_skin_matches_rebuild():
    """In float32 the reused list is as accurate as a fresh one: the step adds
    small displacements to the build's edge vectors, rather than differencing
    float32 positions a box length from the origin."""
    import jax.numpy as jnp
    at = _cell("tric")
    at.positions += 100.0                                   # far from the origin
    at.rattle(0.05, seed=5)
    c = ACECalculator(M, layout="dense", skin=1.0, dtype=jnp.float32)
    ref = ACECalculator(M, layout="dense", skin=0.0, dtype=jnp.float32)
    exact = ACECalculator(M, layout="dense", skin=0.0)
    for _ in range(3):
        at.positions += np.random.default_rng(6).normal(0, 0.02, at.positions.shape)
        got, want, E64 = _efs(c, at), _efs(ref, at), _efs(exact, at)
        err = lambda r: np.max(np.abs(r[1] - E64[1]))                          # noqa: E731
        assert abs(got[0] - E64[0]) < 1e-5 * abs(E64[0])
        assert err(got) <= max(2 * err(want), 1e-4)
    assert c.last_timing["rebuilds"] == 1


def test_unknown_element_named():
    at = _cell("tric")
    at.numbers[3] = 6
    with pytest.raises(ValueError, match=r"\b6\b"):
        _efs(ACECalculator(M, layout="dense", skin=1.0), at)
