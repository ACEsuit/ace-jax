"""DeepH-layout IO and the orbital convention, on real data (fixtures/ham/sic_deeph:
a subset of Quoll.jl's SiC regression data, shells s, p, d on C and s, p, d, f on Si)."""
import jax

jax.config.update("jax_enable_x64", True)

import json
import pathlib

import numpy as np
import pytest
from conftest import require_optional

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.basis import cg
from ace_jax.ham import deeph
from ace_jax.ham.orbitals import OrbitalBasis

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ham" / "sic_deeph"


@pytest.fixture(scope="module")
def data():
    require_optional("h5py")
    return deeph.read(FIX)


def test_orbital_basis():
    b = OrbitalBasis.of({14: [0, 0, 1, 2, 3], 6: (0, 1)})
    assert b.species == (6, 14) and b.ls(14) == (0, 0, 1, 2, 3) and b.lmax == 3
    assert b.n_orb(14) == 17 and b.n_orb(6) == 4
    assert list(b.offsets(14)) == [0, 1, 2, 5, 10, 17] and b.shell_slice(14, 3) == slice(5, 10)
    with pytest.raises(KeyError):
        b.ls(1)


def test_read(data):
    assert data.basis == OrbitalBasis.of({6: (0, 0, 0, 0, 1, 1, 1, 2, 2), 14: (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)})
    assert list(data.numbers) == [14, 6, 14, 6] and data.positions.shape == (4, 3)
    assert np.allclose(data.cell, [[0, 4.3596, 4.3596], [2.1798, 0, 2.1798], [2.1798, 2.1798, 0]])
    assert set(data.blocks) == {"H", "S"} and len(data.blocks["H"]) == len(data.blocks["S"]) == 68
    assert data.blocks["S"][(0, 0, 0, 0, 1)].shape == (30, 23)
    dists = sorted({round(float(np.linalg.norm(data.bond(k))), 3) for k in data.blocks["S"]})
    assert dists[0] == 0 and 1.8 < dists[1] < 1.95 and dists[-1] < 3.2
    # eV: the Si 1s on-site energy
    assert -1800 < data.blocks["H"][(0, 0, 0, 0, 0)][0, 0] < -1785


def test_reverse_bond_is_the_transpose(data):
    """H_ji(-R) = H_ij(R)^T, and the overlap likewise."""
    for blocks in data.blocks.values():
        n = 0
        for (Rx, Ry, Rz, i, j), B in blocks.items():
            rev = (-Rx, -Ry, -Rz, j, i)
            assert rev in blocks
            assert np.array_equal(blocks[rev], B.T)
            n += 1
        assert n == 68


def violation(data, order=None):
    """Largest departure of an overlap shell block from the two-centre form: its
    L parts must be multiples of Y_L(bond) for l1+l2+L even and zero for odd.
    `order` overrides DEEPH_ORDER (to show a wrong convention is detected)."""
    worst = 0.0
    for key, B in data.blocks["S"].items():
        r = data.bond(key)
        if np.linalg.norm(r) < 1e-6:
            continue
        zi, zj = int(data.numbers[key[3]]), int(data.numbers[key[4]])
        Y = np.asarray(real_spherical_harmonics(r[None], 6))[0]
        for a, l1 in enumerate(data.basis.ls(zi)):
            for b, l2 in enumerate(data.basis.ls(zj)):
                blk = B[data.basis.shell_slice(zi, a), data.basis.shell_slice(zj, b)]
                if order is not None:                 # undo the reader's order, apply another
                    p1 = np.argsort(order[l1])[list(deeph.DEEPH_ORDER[l1])]
                    p2 = np.argsort(order[l2])[list(deeph.DEEPH_ORDER[l2])]
                    blk = blk[p1][:, p2]
                bad = 0.0
                for L, part in cg.project(blk, l1, l2).items():
                    y = Y[L * L:(L + 1) ** 2]
                    bad += (part ** 2).sum() if (l1 + l2 + L) % 2 else ((part - (part @ y) / (y @ y) * y) ** 2).sum()
                worst = max(worst, float(np.sqrt(bad)))
    return worst


def test_overlap_blocks_are_harmonics_of_the_bond(data):
    """The convention test.  An overlap is a two-centre integral, so this holds
    exactly in the true convention, with no model.  The data's own integration
    error is ~1e-3 (the on-site overlap departs from the identity by 2e-3)."""
    assert violation(data) < 1.5e-3
    assert abs(data.blocks["S"][(0, 0, 0, 0, 0)][3, 3] - 1) > 1e-3        # the noise floor quoted above


@pytest.mark.parametrize("l", [1, 2, 3])
def test_a_wrong_order_is_detected(data, l):
    """The same measure with m = -l..l taken as the DeepH order of one l: far above the floor."""
    wrong = {**deeph.DEEPH_ORDER, l: tuple(range(2 * l + 1))}
    assert violation(data, wrong) > 0.05


def test_write_reproduces_quolls_files(data, tmp_path):
    """Writing what was read gives Quoll's datasets and structure files back."""
    import h5py
    deeph.write(tmp_path, data)
    for name in ("hamiltonians.h5", "overlaps.h5"):
        with h5py.File(FIX / name, "r") as a, h5py.File(tmp_path / name, "r") as b:
            assert sorted(a) == sorted(b)
            for k in a:
                assert np.array_equal(a[k][()], b[k][()]), (name, k)
    for name in ("lat.dat", "site_positions.dat", "element.dat", "R_list.dat"):
        assert np.allclose(np.loadtxt(FIX / name), np.loadtxt(tmp_path / name), atol=0, rtol=1e-15), name
    assert (tmp_path / "orbital_types.dat").read_text().split() == (FIX / "orbital_types.dat").read_text().split()
    assert json.loads((tmp_path / "info.json").read_text()) == {"isspinful": False}
    again = deeph.read(tmp_path)
    assert all(np.array_equal(again.blocks["H"][k], v) for k, v in data.blocks["H"].items())


def test_write_under_another_name(data, tmp_path):
    deeph.write(tmp_path, data._replace(blocks={"H": data.blocks["H"]}), files={"H": "hamiltonians_pred.h5"})
    assert (tmp_path / "hamiltonians_pred.h5").exists() and not (tmp_path / "overlaps.h5").exists()
    got = deeph.read(tmp_path, operators=[("H", "hamiltonians_pred.h5")])
    assert np.array_equal(got.blocks["H"][(0, 0, 0, 0, 1)], data.blocks["H"][(0, 0, 0, 0, 1)])


def test_inconsistent_files_are_refused(data, tmp_path):
    deeph.write(tmp_path, data)
    (tmp_path / "orbital_types.dat").write_text("0\t1\n0\n0\t1\n0\n")
    with pytest.raises(ValueError, match="block shape"):
        deeph.read(tmp_path)
    (tmp_path / "orbital_types.dat").write_text("0\t1\n0\n0\n0\n")
    with pytest.raises(ValueError, match="different orbital bases"):
        deeph.read(tmp_path)


def test_no_order_for_g_shells():
    with pytest.raises(ValueError, match="l = 4"):
        deeph._atom_perm((0, 4))
