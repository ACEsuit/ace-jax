"""ACECalculator(energy_reference="E0") reports the energy relative to the isolated
atoms, sum_i (E_i - E0[z_i]): the per-atom constant never enters the summed values,
so the total keeps resolution for line searches on large cells
(docs/dev/energy-sum-results.md).  Forces and stress are unchanged; the absolute
energy is `energy + results["e0_offset"]`."""
import math

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
from ase.build import bulk

from ace_jax.calc.point import ACECalculator
from conftest import FIXTURE_DIR

ACE = str(FIXTURE_DIR / "si_fitted.npz")
PACE = str(FIXTURE_DIR / "pace" / "gesi_sbessel.yace")


def _atoms(symbols="Si", seed=0):
    at = bulk("Si", "diamond", a=5.45, cubic=True) * (2, 2, 2)
    if symbols == "SiGe":
        at.numbers[::3] = 32
    at.rattle(0.05, seed=seed)
    return at


def _efs(calc, at):
    at = at.copy()
    at.calc = calc
    return at.get_potential_energy(), at.get_forces(), at.get_stress(), calc


@pytest.mark.parametrize("path,symbols", [(ACE, "Si"), (PACE, "SiGe")])
def test_e0_relative_energy_is_absolute_minus_isolated_atoms(path, symbols):
    at = _atoms(symbols)
    E_abs, F_abs, S_abs, ca = _efs(ACECalculator(path), at)
    E_rel, F_rel, S_rel, cr = _efs(ACECalculator(path, energy_reference="E0"), at)
    e0 = np.asarray(ca.model.E0, float)
    zi = [ca._z2i[int(z)] for z in at.numbers]
    offset = math.fsum(e0[i] for i in zi)
    assert cr.results["e0_offset"] == offset
    assert abs(E_rel - (E_abs - offset)) <= 1e-9 * abs(E_abs)
    assert abs(E_rel) < abs(E_abs) or offset == 0.0
    np.testing.assert_allclose(F_rel, F_abs, rtol=0, atol=1e-12)
    np.testing.assert_allclose(S_rel, S_abs, rtol=0, atol=1e-12)


def test_absolute_is_the_default_and_reports_a_zero_offset():
    at = _atoms()
    _, _, _, c = _efs(ACECalculator(ACE), at)
    assert c.energy_reference == "absolute"
    assert c.results["e0_offset"] == 0.0


def test_e0_reference_survives_a_model_swap():
    at = _atoms()
    c = ACECalculator(ACE, energy_reference="E0")
    E1, *_ = _efs(c, at)
    c.model = c.model                         # the setter rebuilds the evaluation model
    E2, *_ = _efs(c, at)
    assert E2 == pytest.approx(E1, abs=1e-9)
    assert float(np.max(np.abs(np.asarray(c.eval_model.E0)))) == 0.0
    assert float(np.max(np.abs(np.asarray(c.model.E0)))) > 0.0   # the model as given is untouched


def test_unknown_energy_reference_is_refused():
    with pytest.raises(ValueError, match="energy_reference"):
        ACECalculator(ACE, energy_reference="vacuum")
