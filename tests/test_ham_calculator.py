"""`HamiltonianCalculator`: band energy, forces and stress through ASE, with eigenvalues, the Fermi
level and the H blocks, on a random two-element model."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from ase import Atoms
from conftest import require_optional
from test_ham_operator import _env_predictor, models, structure  # noqa: F401 (models: a fixture)

from ace_jax.ham import deeph
from ace_jax.ham.calculator import HamiltonianCalculator


@pytest.fixture(scope="module")
def setup(models):  # noqa: F811
    pos, cell, numbers = structure()
    calc = HamiltonianCalculator(_env_predictor(models), kpts=(2, 1, 2), nelectrons=40, width=0.2, overlap=None)
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True), calc


def test_energy_forces_and_stress(setup):
    """Forces and stress are the derivatives of the free energy (central differences through new
    calculations), the occupations hold the electrons, and the Fermi level is among the bands."""
    atoms, calc = setup
    a = atoms.copy(); a.calc = calc
    F, s, e = a.get_forces(), a.get_stress(voigt=False), a.get_potential_energy()
    fe = a.get_potential_energy(force_consistent=True)
    assert F.shape == (4, 3) and np.abs(F.sum(axis=0)).max() < 1e-8 * np.abs(F).max() and fe < e
    ev, mu = calc.results["eigenvalues"], calc.get_fermi_level()
    assert ev.shape == (4, calc.get_number_of_bands()) and ev.min() < mu < ev.max()
    occ = 2.0 / (1.0 + np.exp((ev - mu) / 0.2))
    assert abs((occ * calc.get_k_point_weights()[:, None]).sum() - 40) < 1e-8
    assert abs((occ * ev * calc.get_k_point_weights()[:, None]).sum() - e) < 1e-8
    assert np.allclose(calc.get_eigenvalues(kpt=2), ev[2]) and calc.get_ibz_k_points().shape == (4, 3)

    def free(x, c):
        b = Atoms(numbers=atoms.numbers, positions=x, cell=c, pbc=True); b.calc = calc
        return b.get_potential_energy(force_consistent=True)
    h = 1e-4
    for i, d in ((0, 0), (2, 1), (3, 2)):
        dx = np.zeros_like(atoms.positions); dx[i, d] = h
        fd = -(free(atoms.positions + dx, atoms.cell[:]) - free(atoms.positions - dx, atoms.cell[:])) / (2 * h)
        assert abs(fd - F[i, d]) < 1e-5 * max(1.0, np.abs(F).max()), (i, d, fd, F[i, d])
    V = atoms.get_volume()
    for (p, q) in ((0, 0), (1, 2)):
        eps = np.zeros((3, 3)); eps[p, q] += 0.5; eps[q, p] += 0.5
        up, dn = np.eye(3) + h * eps, np.eye(3) - h * eps
        fd = (free(atoms.positions @ up.T, atoms.cell[:] @ up.T) - free(atoms.positions @ dn.T, atoms.cell[:] @ dn.T)) / (2 * h) / V
        assert abs(fd - s[p, q]) < 1e-5 * max(1.0, np.abs(s).max()), (p, q, fd, s[p, q])


def test_hamiltonian_blocks_and_files(setup, tmp_path):
    """`get_hamiltonian`: every bond's reverse is its transpose, on-site blocks are symmetric; and
    `write_results` writes a DeepH directory that reads back to the same blocks."""
    require_optional("h5py")
    atoms, calc = setup
    full = HamiltonianCalculator(calc.predictor, kpts=(1, 1, 1), nelectrons=40, overlap=None)
    full.overlap = "S"                                   # the blocks of both channels (S is not solved with here)
    a = atoms.copy(); a.calc = full
    H = full.get_hamiltonian(a)
    assert all(np.allclose(H[(-k[0], -k[1], -k[2], k[4], k[3])], v.T, atol=1e-10) for k, v in H.items())
    assert (0, 0, 0, 0, 0) in H and len(H) > 20
    full.write_results(tmp_path / "pred", a)
    back = deeph.read(tmp_path / "pred", operators=(("H", "hamiltonians_pred.h5"), ("S", "overlaps_pred.h5")))
    assert set(back.blocks["H"]) == set(H) and all(np.allclose(back.blocks["H"][k], v, atol=1e-12) for k, v in H.items())
    assert np.allclose(back.positions, a.positions) and list(back.numbers) == list(a.numbers)


def test_electron_counts():
    c = HamiltonianCalculator.__new__(HamiltonianCalculator)
    for spec, want in ((None, 14 + 6 + 14), (7.0, 7.0), ({14: 4, 6: 4}, 12.0)):
        c.nelectrons = spec
        assert c.electrons(np.array([14, 6, 14])) == want
