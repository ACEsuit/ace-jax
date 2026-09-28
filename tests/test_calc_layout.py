"""ACECalculator chooses the neighbour layout (dense or sparse) by memory.

Dense (batched outer product per node) was 3-13x faster for forces on
A4500/A100/H100, and lighter for single-element models, but its memory grows
with neighbour-species channels and with (n, K) padding; sparse grows only with
the edge count.  "auto" uses dense when its estimate fits the budget and the
padding is efficient, else sparse.
"""
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
from ase import Atoms

import ace_jax.calc.point as point
from ace_jax.calc.point import ACECalculator
from ace_jax.eval import load
from ace_jax.eval.edge_model import estimate_a_bytes
from conftest import pace_fixture

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


def _atoms(name, s):
    ref = np.load(pace_fixture(FIX / f"{name}_ref.npz"))
    return Atoms(numbers=ref[f"Z_{s}"], positions=ref[f"pos_{s}"], cell=ref[f"cell_{s}"],
                 pbc=ref[f"pbc_{s}"])


def _efs(calc, at):
    a = at.copy()
    a.calc = calc
    S = a.get_stress() if a.pbc.all() else np.zeros(6)
    return a.get_potential_energy(), a.get_forces(), S


@pytest.fixture
def yace():
    return str(pace_fixture(FIX / "gesi_sbessel.yace"))


def test_estimate_formula(yace):
    m, _, _ = load(yace)
    nc, ny = m.edge_a_widths()
    C, nA = m.a_channels, int(m.aspec_r.shape[0])
    n, E, K = 64, 1500, 40
    assert estimate_a_bytes(m, "dense", n, E, K, 8) == 8 * (2 * n * K * (C * nc + ny) + 3 * n * C * nc * ny)
    assert estimate_a_bytes(m, "sparse", n, E, K, 8) == 8 * 2 * E * (nc + ny + nA)
    with pytest.raises(ValueError):
        estimate_a_bytes(m, "banded", n, E, K, 8)


def test_auto_picks_dense_for_uniform_bulk(yace):
    calc = ACECalculator(yace)
    assert calc.layout == "auto"
    _efs(calc, _atoms("gesi_sbessel", "bulk"))
    assert calc.last_layout == "dense"


def test_auto_falls_back_to_sparse_over_budget(yace, monkeypatch):
    monkeypatch.setattr(point, "dense_budget_bytes", lambda: 1)
    calc = ACECalculator(yace)
    _efs(calc, _atoms("gesi_sbessel", "bulk"))
    assert calc.last_layout == "sparse"


def test_auto_falls_back_to_sparse_when_padding_is_wasteful(yace, monkeypatch):
    monkeypatch.setattr(point, "MIN_DENSE_FILL", 1.01)       # no layout can meet it
    calc = ACECalculator(yace)
    _efs(calc, _atoms("gesi_sbessel", "bulk"))
    assert calc.last_layout == "sparse"


@pytest.mark.parametrize("s", ["bulk", "close", "isolated", "dimer_2.3"])
def test_layouts_agree(yace, s):
    at = _atoms("gesi_sbessel", s)
    Ed, Fd, Sd = _efs(ACECalculator(yace, layout="dense"), at)
    Es, Fs, Ss = _efs(ACECalculator(yace, layout="sparse"), at)
    assert Ed == pytest.approx(Es, abs=1e-10)
    np.testing.assert_allclose(Fd, Fs, atol=1e-10)
    np.testing.assert_allclose(Sd, Ss, atol=1e-10)


def test_bad_layout_rejected(yace):
    with pytest.raises(ValueError, match="layout"):
        ACECalculator(yace, layout="banded")
