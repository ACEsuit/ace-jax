"""On-site model: equivariance and symmetry of the blocks, and the block fit on real data."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import numpy as np
import pytest
from conftest import require_optional

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.ham import cg, deeph
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.fit import fit_onsite
from ace_jax.ham.onsite import OnsiteModel, parts_of, shell_pairs
from ace_jax.ham.orbitals import OrbitalBasis

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE, FIX = ROOT / "fixtures" / "ham" / "coupling_cache", ROOT / "fixtures" / "ham" / "sic_deeph"
ORB = OrbitalBasis.of({14: (0, 0, 1, 1, 2), 6: (0, 1, 2)})


@pytest.fixture(scope="module")
def basis():
    return build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))


def random_model(basis, seed=0):
    rng = np.random.default_rng(seed)
    coef = {}
    for z in ORB.species:
        for s1, s2, l1, l2 in shell_pairs(ORB, z):
            for L, p in parts_of(l1, l2, s1 == s2):
                n = basis.n(L, p) + (L == 0)
                if n:
                    coef[(z, s1, s2, L)] = rng.normal(size=(2, n))
    return OnsiteModel(basis=basis, coef=coef, orbitals=ORB, channels=("H", "S"))


def structure(seed=0):
    rng = np.random.default_rng(seed)
    cell = 4.4 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    numbers = np.array([14, 6, 14, 6, 6])
    return rng.uniform(size=(5, 3)) @ cell, cell, numbers


def wigner(L, Q):
    pts = np.random.default_rng(1).normal(size=(80, 3))
    sl = slice(L * L, (L + 1) ** 2)
    Y0 = np.asarray(real_spherical_harmonics(pts, max(L, 1)))[:, sl]
    Y1 = np.asarray(real_spherical_harmonics(pts @ Q.T, max(L, 1)))[:, sl]
    return np.linalg.lstsq(Y0, Y1, rcond=None)[0].T


def atom_rotation(z, Q):
    """Block-diagonal D over the shells of species z: orbitals rotate as their harmonics."""
    import scipy.linalg
    return scipy.linalg.block_diag(*[wigner(l, Q) if l else np.ones((1, 1)) for l in ORB.ls(z)])


def test_parts():
    assert parts_of(1, 1, True) == [(0, 1), (2, 1)]                       # same shell: symmetric, even L
    assert parts_of(1, 1, False) == [(0, 1), (1, 1), (2, 1)]              # (1, +1): the axial, antisymmetric part
    assert parts_of(0, 1, False) == [(1, -1)] and parts_of(1, 2, False) == [(1, -1), (2, -1), (3, -1)]
    assert [(a, b) for a, b, _, _ in shell_pairs(ORB, 6)] == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def test_blocks_are_symmetric_and_shaped(basis):
    pos, cell, numbers = structure()
    out = random_model(basis).predict(pos, cell, numbers)
    assert set(out) == {"H", "S"} and set(out["H"]) == {(0, 0, 0, i, i) for i in range(5)}
    for name in out:
        for (_, _, _, i, _), B in out[name].items():
            n = ORB.n_orb(numbers[i])
            assert B.shape == (n, n) and np.allclose(B, B.T, atol=1e-13) and np.abs(B).max() > 1e-3
    assert not np.allclose(out["H"][(0, 0, 0, 0, 0)], out["S"][(0, 0, 0, 0, 0)])


@pytest.mark.parametrize("improper", [False, True])
def test_blocks_rotate_with_the_orbitals(basis, improper):
    """H(Q R) = D(Q) H(R) D(Q)^T for a rotation, and with the orbital parities for an inversion;
    random coefficients, so every part (pseudo-tensor parts included) is exercised."""
    pos, cell, numbers = structure()
    m = random_model(basis)
    q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    Q = q * np.sign(np.linalg.det(q))
    a = m.predict(pos, cell, numbers)
    b = m.predict(pos @ Q.T * (-1 if improper else 1), cell @ Q.T * (-1 if improper else 1), numbers)
    for name in a:
        for key, B in a[name].items():
            z = int(numbers[key[3]])
            D = atom_rotation(z, Q)
            if improper:
                D = D * np.concatenate([[(-1) ** l] * (2 * l + 1) for l in ORB.ls(z)])[:, None]
            assert np.abs(b[name][key] - D @ B @ D.T).max() < 1e-9 * np.abs(B).max(), (name, key)


def test_isolated_atom_is_the_constant_part(basis):
    """No neighbour within the cutoff: only the L = 0 constants remain, so each
    same-l block is a multiple of the identity and the rest vanish."""
    m = random_model(basis)
    B = m.predict(np.zeros((1, 3)), 50 * np.eye(3), np.array([6]))["H"][(0, 0, 0, 0, 0)]
    for s1, s2, l1, l2 in shell_pairs(ORB, 6):
        blk = B[ORB.shell_slice(6, s1), ORB.shell_slice(6, s2)]
        if l1 == l2:
            T0 = cg.real_cg(l1, l2, 0)[:, :, 0]                      # +-identity / sqrt(2l+1)
            assert np.allclose(blk, m.coef[(6, s1, s2, 0)][0, -1] * T0, atol=1e-13)
            assert np.allclose(np.abs(T0), np.eye(2 * l1 + 1) / np.sqrt(2 * l1 + 1), atol=1e-14)
        else:
            assert not blk.any()


def test_species_without_orbitals(basis):
    pos, cell, numbers = structure()
    with pytest.raises(KeyError, match="14"):
        OnsiteModel(basis=basis, coef={}, orbitals=OrbitalBasis.of({6: (0,)}), channels=("H",)).predict(
            pos, cell, numbers)


@pytest.fixture(scope="module")
def basis_spdf():
    """l up to 6 as single-particle functions (wL = 1), so every part of an spdf block has features."""
    return build_site_basis([14, 6], order=2, totaldegree=7, lmax_feat=6, rcut=4.0, wL=1.0,
                            coupling_cache_dir=str(CACHE))


def test_fit_reproduces_real_onsite_blocks(basis_spdf):
    """SiC fixture (4 atoms, s, p, d and f shells): the fitted model returns the on-site H and S
    blocks.  With two equivalent atoms per species this is interpolation, so it checks the
    whole path on real data (conventions, design against readout, channels, transposed shell
    pairs, the pseudo-tensor parts), not the model's accuracy."""
    require_optional("h5py")
    d = deeph.read(FIX)
    model, report = fit_onsite(basis_spdf, [d])
    assert model.channels == ("H", "S") and model.orbitals == d.basis
    assert all(cols > 0 for _, cols, _ in report.values())
    pred = model.predict(d.positions, d.cell, d.numbers)
    for name, tol in (("H", 1e-6), ("S", 1e-9)):
        err = max(np.abs(pred[name][k] - d.blocks[name][k]).max() for k in pred[name])
        assert err < tol, (name, err)
    assert pred["H"][(0, 0, 0, 0, 0)][0, 0] < -1700 and abs(pred["S"][(0, 0, 0, 0, 0)][0, 0] - 1) < 1e-3


def test_parts_without_features_are_reported(basis):
    """The small basis has no L >= 4 feature: those parts get no coefficients, are predicted
    as zero, and the report gives the size of what is missed.  (Some parts it does have
    features for are still missed here: its only (3, -1) feature is a product of l = 1 and
    l = 2 functions, which vanish at a tetrahedral site.)"""
    require_optional("h5py")
    d = deeph.read(FIX)
    model, report = fit_onsite(basis, [d])
    empty = [k for k, (_, cols, _) in report.items() if cols == 0]
    assert empty and all(k not in model.coef for k in empty)
    assert {k for k in report if k[3] >= 4} <= set(empty)                      # and some pseudo-tensor parts
    assert max(float(report[k][2].max()) for k in empty) > 1.0                   # eV: a real omission
    assert np.isfinite(model.predict(d.positions, d.cell, d.numbers)["H"][(0, 0, 0, 0, 0)]).all()


def test_compiled_readout_is_the_reference_readout(basis):
    """`OnsitePredictor` (one jitted program, dense tensors) against `OnsiteModel.predict`
    (a loop over shell pairs), for a model with every part, and for an isolated atom."""
    from ace_jax.ham.onsite import OnsitePredictor
    m = random_model(basis)
    fast = OnsitePredictor(m)
    for pos, cell, numbers in (structure(), structure(3), (np.zeros((1, 3)), 50 * np.eye(3), np.array([6]))):
        a, b = m.predict(pos, cell, numbers), fast(pos, cell, numbers)
        assert set(a["H"]) == set(b["H"])
        for name in a:
            for k, v in a[name].items():
                assert np.abs(b[name][k] - v).max() < 1e-11 * max(1.0, np.abs(v).max()), (name, k)
    atoms, blocks = fast.dense(*structure())
    assert [list(x) for x in atoms] == [[1, 3, 4], [0, 2]] and blocks[0].shape == (3, 2, ORB.n_orb(6), ORB.n_orb(6))
