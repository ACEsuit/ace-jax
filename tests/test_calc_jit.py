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
