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
    for _ in range(12):
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
        err = lambda r: np.max(np.abs(r[1] - E64[1]))  # noqa: E731, B023 (used this iteration)
        assert abs(got[0] - E64[0]) < 1e-5 * abs(E64[0])
        assert err(got) <= max(2 * err(want), 1e-4)
    assert c.last_timing["rebuilds"] == 1


def test_unknown_element_named():
    at = _cell("tric")
    at.numbers[3] = 6
    with pytest.raises(ValueError, match=r"\b6\b"):
        _efs(ACECalculator(M, layout="dense", skin=1.0), at)


def _counting(monkeypatch):
    """Count traces of the dense model body (the skin step's jit traces it once)."""
    from ace_jax.eval.edge_model import EdgeSiteModel
    calls = []
    orig = EdgeSiteModel.energy_forces_virial_dense

    def counted(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(EdgeSiteModel, "energy_forces_virial_dense", counted)
    return calls


def _perturbed(model):
    """The same structure with other weights: E0 shifted, the pair readout scaled."""
    import equinox as eqx
    return eqx.tree_at(lambda m: (m.E0, m.Wpair), model, (model.E0 + 1.0, model.Wpair * 1.1))


@pytest.mark.parametrize("layout", ["dense", "sparse"])
def test_model_swap_between_calls_uses_the_new_model(monkeypatch, layout):
    """Setting calc.model takes effect on the next call (as it did before the
    skin list), with no retrace when only the weights changed."""
    calls = _counting(monkeypatch)
    at = _cell("tric")
    c = ACECalculator(M, layout=layout, skin=1.0)
    old = _efs(c, at)
    c.model = _perturbed(c.model)
    at.positions[0] += [0.01, 0, 0]                        # inside the skin: no rebuild needed
    got = _efs(c, at)
    want = _efs(ACECalculator(c.model, c.meta, layout=layout, skin=0.0), at)
    _close(got, want)
    assert abs(got[0] - old[0] - len(at)) > 1e-6         # the pair weights took effect too
    if layout == "dense":
        assert c.last_layout == "dense"
        assert len(calls) == 2                             # c's one trace + the fresh calc's
    c.get_potential_energy(at)
    c.model = _perturbed(c.model)                          # ASE's result cache is dropped too
    assert c.get_potential_energy(at) != got[0]
    # and back again, still without a retrace
    n_calls = len(calls)
    c.model = ACECalculator(M).model
    at.positions[0] -= [0.01, 0, 0]
    _close(_efs(c, at), old)
    if layout == "dense":
        assert len(calls) == n_calls


def test_skin_toggled_after_construction():
    """skin 0 -> 1 -> 0 on a live calculator: the skin list is built on first
    use and dropped when skin is set back to 0."""
    at = _cell("tric")
    ref = ACECalculator(M, layout="dense", skin=0.0)
    c = ACECalculator(M, layout="dense", skin=0.0)
    _close(_efs(c, at), _efs(ref, at))
    c.skin = 1.0
    n0 = c.last_timing["rebuilds"]
    for _ in range(2):
        at.positions[0] += [0.01, 0, 0]
        _close(_efs(c, at), _efs(ref, at))
    assert c._skin_state is not None and c.last_timing["rebuilds"] == n0 + 1
    c.skin = 0.0
    assert c._skin_state is None
    _close(_efs(c, at), _efs(ref, at))
    assert c.last_timing["rebuilds"] == n0 + 2 and c._skin_state is None


def _pbc(a):
    a.pbc = (True, True, False)
    return a, 1


def _drop_atom(a):
    del a[5]
    return a, 1


def _lattice_wrap(a):
    # the same structure, some atoms one or two lattice vectors over: displacement
    # is a box length, so a rebuild, and nothing may depend on the image an atom is in
    a.positions[:7] += a.cell[0] - a.cell[2]
    a.positions[7:9] -= a.cell[1]
    return a, 1


def _cell_change(a):
    a.set_cell(a.cell * 1.01, scale_atoms=True)
    return a, 1


def _species(a):
    a.numbers[1] = 14 if a.numbers[1] == 32 else 32
    return a, 1


def _into_cutoff(a):
    """Two atoms 0.45 A (0.9 x skin / 2) each toward each other: a pair in the
    skin shell (rc < d) closes to inside rc with no rebuild.  Not atom 0, which
    the follow-up step moves again, and well under skin / 2: at 0.499 A the
    pair's pick followed the neighbour-list sort order, which differs between
    numpy builds (x86 SIMD sort), and on Linux it picked atom 0, whose extra
    0.0245 A step then took it past skin / 2 -- a legitimate rebuild."""
    from ase.neighborlist import neighbor_list
    rc = float(ACECalculator(M).cutoff)
    i, j, d, D = neighbor_list("ijdD", a, rc + 1.0)
    # the closest shell pair (away from atom 0) that 2 * 0.45 A brings inside rc
    ok = (d > rc + 0.05) & (d < rc + 0.85) & (i < j) & (i != 0)
    k = np.flatnonzero(ok)[np.argmin(d[ok])]
    step = 0.45 * D[k] / d[k]
    a.positions[i[k]] += step
    a.positions[j[k]] -= step
    assert np.linalg.norm(D[k] - 2 * step) < rc - 0.05         # now inside the cutoff
    return a, 0


MUTATIONS = {"pbc": _pbc, "atom_count": _drop_atom, "lattice_wrap": _lattice_wrap,
             "cell": _cell_change, "species": _species, "shell_into_cutoff": _into_cutoff}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_skin_equals_rebuild_after_each_change(name):
    """Each change a skin list must notice (or, below the threshold, must not
    need to): the result equals a fresh rebuild (skin=0) to 1e-12 on the call
    itself and on the next small step, with the rebuild counter as expected."""
    at = _cell("tric")
    at.rattle(0.02, seed=7)
    c, ref = ACECalculator(M, layout="dense", skin=1.0), ACECalculator(M, layout="dense", skin=0.0)
    _close(_efs(c, at), _efs(ref, at))
    n0 = c.last_timing["rebuilds"]
    at, rebuilt = MUTATIONS[name](at.copy())
    _close(_efs(c, at), _efs(ref, at))
    assert c.last_timing["rebuilds"] == n0 + rebuilt
    at.positions[0] += [0.02, -0.01, 0.01]               # just after (or without) the rebuild
    _close(_efs(c, at), _efs(ref, at))
    assert c.last_timing["rebuilds"] == n0 + rebuilt


def test_step_counts_are_integers_in_float32():
    """The step's flags and counts come back as integers, not in the model dtype:
    float32 holds integers exactly only to 2**24, a 16.8M-edge system."""
    import jax.numpy as jnp
    at = _cell("tric")
    c = ACECalculator(M, layout="dense", skin=1.0, dtype=jnp.float32)
    _efs(c, at)
    st = c._skin_state
    out = c._step()(jax.device_put(st.displacements(at.positions, np.float32)), st.arrays, K=st.K)
    values, counts = out
    assert values.dtype == np.float32 and jnp.issubdtype(counts.dtype, jnp.integer)
    _, _, _, drift, overflow, k_max, n_edges = skin_mod.unpack(out, len(at))
    assert (drift, overflow, k_max, n_edges) == (False, False, st.k_max, st.n_edges)
    n, big = 2, 2**24 + 1                                 # not representable in float32
    got = skin_mod.unpack((np.zeros(10 + 3 * n, np.float32), np.array([0, 1, 7, big], np.int32)), n)
    assert got[3:] == (False, True, 7, big)
