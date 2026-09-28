"""Training/eval data is read with libAtoms `extxyz` (ace_jax.fit.xyz), not
ase.io: labels must come back exactly as written, whatever they are called.
ASE >= 3.23 moves `energy` / `forces` / `stress` into a SinglePointCalculator,
which made the CLI's default label names read as absent.  Files here are
written by the installed ASE (3.29 when this was added), so its current
encodings (special 3x3 keys in Fortran order, `_JSON` 2-D info arrays) are the
ones exercised."""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from conftest import FIXTURE_DIR

from ace_jax.fit.data import load_configs

RNG = np.random.default_rng(7)


def _si(n_rattle=0):
    at = bulk("Si", cubic=True)
    at.rattle(0.05, seed=n_rattle)
    return at


def test_default_label_names_round_trip(tmp_path):
    """energy / forces / virial -- the CLI defaults -- come back as written."""
    at = _si()
    F = RNG.normal(size=(len(at), 3)); V = RNG.normal(size=(3, 3))
    at.calc = SinglePointCalculator(at, energy=-12.25, forces=F)
    at.info["virial"] = V
    p = tmp_path / "d.xyz"; write(p, at)
    F = read(p).calc.results["forces"]           # per-atom columns are written to 8 decimals
    c, = load_configs(p)
    assert c.energy == -12.25
    assert c.forces is not None and np.array_equal(c.forces, F)
    assert c.virial is not None and np.array_equal(c.virial, V)


def test_custom_label_names(tmp_path):
    at = _si(1)
    F = RNG.normal(size=(len(at), 3)); V = RNG.normal(size=(3, 3))   # 2-D, non-special: ASE writes _JSON
    at.info.update(dft_energy=-3.5, dft_virial=V, config_type="bulk")
    at.arrays["dft_force"] = F
    p = tmp_path / "d.xyz"; write(p, at)
    assert "_JSON" in p.read_text()
    F = read(p).arrays["dft_force"]              # per-atom columns are written to 8 decimals
    c, = load_configs(p, energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
    assert c.energy == -3.5 and np.array_equal(c.forces, F) and np.array_equal(c.virial, V)


def test_flat_nine_vector_keeps_ase_order(tmp_path):
    """A non-special 9-vector (what the committed fixtures hold) reshapes in the
    order it is written, exactly as the ASE reader + reshape(3, 3) did."""
    at = _si(2)
    V = RNG.normal(size=(3, 3))
    at.info.update(energy_x=1.0, v9=V.ravel())
    p = tmp_path / "d.xyz"; write(p, at)
    c, = load_configs(p, energy_key="energy_x", virial_key="v9")
    assert np.array_equal(c.virial, V)


def test_stress_written_by_ase_calculator(tmp_path):
    """A calculator stress (Voigt 6 in ASE) is written as a full 3x3; reading it
    under its own name gives that 3x3 back, not None."""
    at = _si(3)
    s = RNG.normal(size=6)
    at.calc = SinglePointCalculator(at, energy=0.5, stress=s)
    p = tmp_path / "d.xyz"; write(p, at)
    ref = read(p).calc.results["stress"]                        # ASE: the same numbers, Voigt
    c, = load_configs(p, virial_key="stress")
    full = np.array([[ref[0], ref[5], ref[4]], [ref[5], ref[1], ref[3]], [ref[4], ref[3], ref[2]]])
    assert np.array_equal(c.virial, full)


def test_isolated_atom_frame(tmp_path):
    at = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 20.0, pbc=True)
    at.info.update(config_type="isolated_atom", dft_energy=-158.5)
    p = tmp_path / "d.xyz"; write(p, [at, _si()])
    c = load_configs(p, energy_key="dft_energy")
    assert len(c) == 2 and c[0].energy == -158.5 and c[1].energy is None
    assert c[0].numbers.tolist() == [14] and c[0].positions.shape == (1, 3)
    assert np.array_equal(c[0].cell, np.eye(3) * 20.0) and c[0].pbc.all()


def test_no_lattice_is_not_periodic(tmp_path):
    p = tmp_path / "d.xyz"
    p.write_text("2\nProperties=species:S:1:pos:R:3 energy=-1.0\nSi 0 0 0\nSi 0 0 2.3\n")
    c, = load_configs(p)
    assert not c.pbc.any() and not c.cell.any() and c.energy == -1.0


def test_unreadable_label_fails_loudly(tmp_path):
    p = tmp_path / "d.xyz"
    p.write_text('1\nLattice="9 0 0 0 9 0 0 0 9" Properties=species:S:1:pos:R:3 energy=abc pbc="T T T"\n'
                 "Si 0 0 0\n")
    with pytest.raises(ValueError, match="energy"):
        load_configs(p)


def test_fixture_matches_the_ase_reader():
    """On the committed Si fixture (no calculator-named labels), every field is
    what ase.io.read + the old lookup produced, bit for bit."""
    path = FIXTURE_DIR / "si_tiny_train.xyz"
    ref = read(path, ":")
    got = load_configs(path, energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
    assert len(got) == len(ref)
    for a, c in zip(ref, got, strict=True):
        assert np.array_equal(a.numbers, c.numbers) and np.array_equal(a.positions, c.positions)
        assert np.array_equal(a.cell.array, c.cell) and np.array_equal(a.pbc, c.pbc)
        assert c.energy == float(a.info["dft_energy"])
        assert np.array_equal(c.forces, a.arrays["dft_force"])
        V = a.info.get("dft_virial")
        assert (c.virial is None if V is None
                else np.array_equal(c.virial, np.asarray(V, float).reshape(3, 3)))


def test_triclinic_cell_and_z_column_match_ase(tmp_path):
    """Cell orientation (extxyz reads Lattice column-major) and a Z column."""
    at = bulk("SiGe", "zincblende", a=5.6).repeat((2, 1, 1))
    at.set_cell(at.cell.array @ (np.eye(3) + 0.05 * RNG.normal(size=(3, 3))), scale_atoms=True)
    at.arrays["Z"] = at.numbers.copy()
    at.info["dft_energy"] = 2.0
    p = tmp_path / "d.xyz"; write(p, at)
    ref = read(p)
    c, = load_configs(p, energy_key="dft_energy")
    assert np.array_equal(c.cell, ref.cell.array) and np.array_equal(c.positions, ref.positions)
    assert c.numbers.tolist() == ref.numbers.tolist() == [14, 32, 14, 32]
