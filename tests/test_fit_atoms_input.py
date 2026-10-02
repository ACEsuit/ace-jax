"""load_configs / load_fit_data take ase.Atoms lists as well as files; stress labels
convert to virials (virial = -stress * volume)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress

from conftest import FIXTURE_DIR

from ace_jax.fit.data import load_configs

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
KEYS = dict(energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")


def _atoms_from_file(n=6):
    from ace_jax.fit.xyz import read_extxyz
    from ase import Atoms
    out = []
    for f in read_extxyz(XYZ)[:n]:
        a = Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
        a.info.update(f.info); [a.set_array(k, v) for k, v in f.arrays.items()]
        out.append(a)
    return out


def test_atoms_list_gives_the_same_configs_as_the_file():
    a, b = load_configs(_atoms_from_file(), **KEYS), load_configs(XYZ, **KEYS)[:6]
    for x, y in zip(a, b):
        assert x.energy == y.energy and x.config_type == y.config_type and (x.w_E, x.w_F) == (y.w_E, y.w_F)
        np.testing.assert_array_equal(x.forces, y.forces)
        np.testing.assert_array_equal(x.positions, y.positions)
        assert (x.virial is None) == (y.virial is None)


def test_singlepoint_calculator_results_are_labels():
    at = bulk("Si", "diamond", a=5.43, cubic=True)
    F = np.random.default_rng(0).normal(size=(len(at), 3))
    at.calc = SinglePointCalculator(at, energy=-40.0, forces=F)
    (c,) = load_configs([at])
    assert c.energy == -40.0
    np.testing.assert_array_equal(c.forces, F)


@pytest.mark.parametrize("form", ["voigt", "full", "flat"])
def test_stress_converts_to_virial_on_a_triclinic_cell(form):
    at = bulk("Si", "diamond", a=5.43)                       # fcc primitive: not orthogonal
    at.set_cell(at.cell.array @ np.array([[1.0, 0.03, 0.0], [0.0, 1.0, 0.02], [0.0, 0.0, 1.0]]),
                scale_atoms=True)
    s = np.array([[0.01, 0.002, -0.003], [0.002, -0.02, 0.004], [-0.003, 0.004, 0.015]])
    at.info["stress"] = {"voigt": full_3x3_to_voigt_6_stress(s), "full": s, "flat": s.ravel()}[form]
    at.info["energy"] = -1.0
    (c,) = load_configs([at], stress_key="stress")
    np.testing.assert_allclose(c.virial, -s * at.get_volume(), rtol=0, atol=1e-12)


def test_no_virial_and_no_stress_stays_unlabelled():
    at = bulk("Si", "diamond", a=5.43, cubic=True); at.info["energy"] = -40.0
    (c,) = load_configs([at], stress_key="stress")
    assert c.virial is None


def test_load_fit_data_accepts_atoms_lists():
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), arm="linear", m_per_species=0, r0=2.35, **KEYS)
    atoms = _atoms_from_file(12)
    d = load_fit_data(cfg, train=atoms[:8], test=atoms[8:], log=lambda *a: None)
    assert (len(d.train), len(d.test)) == (8, 4)


def test_a_calculator_shared_across_structures_is_not_read_as_labels():
    # `for a in s: a.calc = calc; a.get_forces()` leaves calc.results for the LAST structure
    # on every one of them; reading those would label each structure with another's results
    from ase.calculators.emt import EMT
    calc = EMT()
    s = [bulk("Cu", "fcc", a=3.6 * x, cubic=True) for x in (0.98, 1.02)]
    for a in s:
        a.calc = calc
        a.get_forces()
    with pytest.raises(ValueError, match="shared across structures"):
        load_configs(s, energy_key="energy", force_key="forces", virial_key=None)
    one = s[1]                                             # the structure calc last saw: still valid
    assert load_configs([one], energy_key="energy", force_key="forces", virial_key=None)[0].energy \
        == pytest.approx(one.get_potential_energy())


def _read_labelled(tmp_path):
    from ase.calculators.emt import EMT
    from ase.io import read, write
    s = []
    for x in (0.98, 1.02):
        a = bulk("Cu", "fcc", a=3.6 * x, cubic=True); a.rattle(0.02, seed=3); a.positions[0] += 0.3 * a.cell[0]
        a.calc = EMT(); a.get_forces(); a.get_stress(); s.append(a)
    write(tmp_path / "s.xyz", s)
    return read(tmp_path / "s.xyz", ":")


def test_harmless_edits_to_read_frames_keep_their_labels(tmp_path):
    keys = dict(energy_key="energy", force_key="forces", virial_key=None, stress_key="stress")
    ref = load_configs(_read_labelled(tmp_path), **keys)
    wrapped = _read_labelled(tmp_path)
    for a in wrapped:
        a.wrap()                                           # positions move by lattice vectors only
    mag = _read_labelled(tmp_path)
    for a in mag:
        a.set_initial_magnetic_moments([1.0] * len(a))     # not a label input
    for frames in (wrapped, mag):
        got = load_configs(frames, **keys)
        for g, r in zip(got, ref):
            assert g.energy == r.energy and np.allclose(g.forces, r.forces) and np.allclose(g.virial, r.virial)


def test_info_labels_with_a_shared_live_calculator_are_read(tmp_path):
    # labels in info/arrays: the calculator's (stale) results are never used, so nothing to reject
    from ase.calculators.emt import EMT
    calc, s = EMT(), []
    for x in (0.98, 1.02):
        a = bulk("Cu", "fcc", a=3.6 * x, cubic=True); a.info["dft_energy"] = -x
        a.arrays["dft_forces"] = np.zeros((len(a), 3)); a.calc = calc; a.get_forces(); s.append(a)
    got = load_configs(s, energy_key="dft_energy", force_key="dft_forces", virial_key=None)
    assert [g.energy for g in got] == [-0.98, -1.02]


def test_moved_atoms_name_what_changed(tmp_path):
    a = _read_labelled(tmp_path)[0]; a.positions[1] += 0.1
    with pytest.raises(ValueError, match="positions"):
        load_configs([a], energy_key="energy", force_key="forces", virial_key=None)
