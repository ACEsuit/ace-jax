"""ACECalculator evaluates the model through a compiled function.

Run eagerly, `energy_forces_virial[_dense]` dispatches thousands of ops one by
one: a flat ~0.6 s per call on an A100 whatever the system size, which is what
the scaling benchmark measured before this was jitted.  The model's Python
body must run once per (layout, shape), not once per call.
"""
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
from ase import Atoms
from ase.calculators.calculator import all_changes

from ace_jax.calc.point import ACECalculator
from ace_jax.eval.edge_model import EdgeSiteModel
from conftest import pace_fixture

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


def _bulk():
    ref = np.load(pace_fixture(FIX / "gesi_sbessel_ref.npz"))
    return Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"],
                 pbc=ref["pbc_bulk"])


@pytest.mark.parametrize("layout,method", [("dense", "energy_forces_virial_dense"),
                                           ("sparse", "energy_forces_virial")])
def test_model_body_runs_once_per_shape(monkeypatch, layout, method):
    calls = []
    orig = getattr(EdgeSiteModel, method)

    def counted(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(EdgeSiteModel, method, counted)
    calc = ACECalculator(str(pace_fixture(FIX / "gesi_sbessel.yace")), layout=layout)
    at = _bulk()
    E = []
    for _ in range(3):                           # as the benchmark: no ASE result cache
        calc.calculate(at, ["energy", "forces", "stress"], all_changes)
        E.append(calc.results["energy"])
    assert calc.last_layout == layout
    assert len(calls) == 1                       # traced once, then the compiled call
    assert E[0] == E[1] == E[2]


def test_last_timing_splits_neighbour_list_and_model():
    """The benchmark reports model time on its own; `call - nlist` also counted
    host regrouping and transfers as 'model'."""
    calc = ACECalculator(str(pace_fixture(FIX / "gesi_sbessel.yace")))
    at = _bulk()
    for _ in range(2):
        calc.calculate(at, ["energy", "forces", "stress"], all_changes)
    t = calc.last_timing
    assert t["nlist_s"] > 0 and t["model_s"] > 0 and t["nlist_backend"]


def test_dense_path_uses_native_neighbour_matrix(monkeypatch):
    """With matscipy_neighbours, the dense (n, K) graph comes straight from
    neighbour_matrix (no sparse build + regroup), K cached across calls and
    re-learnt when an atom outgrows it; results match the sparse layout."""
    import ace_jax.eval.nlist as nl
    if not nl.have_matscipy_neighbours():
        pytest.skip("matscipy_neighbours not installed")
    import matscipy_neighbours
    calls = []
    orig = matscipy_neighbours.neighbour_matrix

    def counted(*a, **k):
        calls.append(k.get("max_neighbours"))
        return orig(*a, **k)

    monkeypatch.setattr(matscipy_neighbours, "neighbour_matrix", counted)
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    at = _bulk()
    dense = ACECalculator(y, layout="dense")
    for _ in range(3):
        dense.calculate(at, ["energy", "forces", "stress"], all_changes)
    assert len(calls) >= 2 and dense.last_timing["nlist_backend"] == "neighbour_matrix"
    dense._k_hint = 2                                  # too small: must recover
    dense.calculate(at, ["energy", "forces", "stress"], all_changes)
    E, F = dense.results["energy"], dense.results["forces"].copy()
    sparse = ACECalculator(y, layout="sparse")
    sparse.calculate(at, ["energy", "forces", "stress"], all_changes)
    assert E == pytest.approx(sparse.results["energy"], abs=1e-10)
    np.testing.assert_allclose(F, sparse.results["forces"], atol=1e-9)


def test_sparse_layout_does_not_recompile_as_the_edge_count_changes(monkeypatch):
    """MD moves atoms, so the edge count changes from step to step; unpadded,
    every new count was a new shape and a full XLA compile.  The edge list is
    padded to a power-of-two bucket, so traces stay bounded."""
    calls = []
    orig = EdgeSiteModel.energy_forces_virial

    def counted(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(EdgeSiteModel, "energy_forces_virial", counted)
    calc = ACECalculator(str(pace_fixture(FIX / "gesi_sbessel.yace")), layout="sparse")
    at = _bulk()
    rng = np.random.default_rng(0)
    counts = set()
    ref = ACECalculator(str(pace_fixture(FIX / "gesi_sbessel.yace")), layout="dense")
    for _ in range(6):
        a = at.copy()
        a.positions += rng.normal(0, 0.15, a.positions.shape)
        calc.calculate(a, ["energy", "forces"], all_changes)
        ref.calculate(a, ["energy", "forces"], all_changes)
        np.testing.assert_allclose(calc.results["forces"], ref.results["forces"], atol=1e-10)
        counts.add(calc.last_n_edges)
    assert len(counts) > 1                         # the edge count really changed
    assert len(calls) <= 2                         # ... but not the compiled shape
