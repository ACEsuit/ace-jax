"""Off-site model from shared features: equivariance, exact bond reversal, the streamed fit."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import numpy as np
import pytest
import scipy.linalg
from conftest import require_optional

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.basis import cg
from ace_jax.ham import deeph
from ace_jax.operators import bond
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import fit_offsite
from ace_jax.operators.offsite import OffsiteModel, block_lps, canonical_pairs
from ace_jax.operators.orbitals import OrbitalBasis

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE, FIX = ROOT / "fixtures" / "ham" / "coupling_cache", ROOT / "fixtures" / "ham" / "sic_deeph"
ORB = OrbitalBasis.of({14: (0, 1, 1, 2), 6: (0, 1)})


@pytest.fixture(scope="module")
def basis():
    site = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))
    return bond.BondBasis(site=site, rcut_bond=5.0, n_bond=4, n_cpl=2, l_cpl=2)


def random_model(basis, seed=0, channels=("H", "S")):
    rng = np.random.default_rng(seed)
    coef = {}
    for zi in ORB.species:
        for zj in ORB.species:
            for s1, s2, l1, l2, same in canonical_pairs(ORB, zi, zj):
                p = 1 if (l1 + l2) % 2 == 0 else -1
                for L in cg.Ls(l1, l2):
                    k0, k1 = bond.n_features(basis, L, p)
                    n = k0 + (k1 if same else 2 * k1) + bond.n_env_features(basis, L, p, same=same)
                    if n:
                        coef[(zi, zj, s1, s2, L)] = rng.normal(size=(len(channels), n))
    return OffsiteModel(bond=basis, coef=coef, orbitals=ORB, channels=channels)


def structure(seed=0):
    rng = np.random.default_rng(seed)
    cell = 4.6 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    return rng.uniform(size=(4, 3)) @ cell, cell, np.array([14, 6, 14, 6])


def wigner(L, Q):
    pts = np.random.default_rng(1).normal(size=(80, 3))
    sl = slice(L * L, (L + 1) ** 2)
    Y0 = np.asarray(real_spherical_harmonics(pts, max(L, 1)))[:, sl]
    Y1 = np.asarray(real_spherical_harmonics(pts @ Q.T, max(L, 1)))[:, sl]
    return np.linalg.lstsq(Y0, Y1, rcond=None)[0].T


def atom_rotation(z, Q, improper=False):
    D = scipy.linalg.block_diag(*[wigner(l, Q) if l else np.ones((1, 1)) for l in ORB.ls(z)])
    return D * np.concatenate([[(-1) ** l] * (2 * l + 1) for l in ORB.ls(z)])[:, None] if improper else D


def test_canonical_pairs_and_symmetries():
    assert canonical_pairs(ORB, 14, 6) == []
    assert len(canonical_pairs(ORB, 6, 14)) == 2 * 4
    same = canonical_pairs(ORB, 14, 14)
    assert len(same) == 10 and [s for *_, s in same].count(True) == 4
    assert (1, 1) in block_lps(ORB) and (4, 1) in block_lps(ORB) and (3, -1) in block_lps(ORB)


def test_every_bond_and_its_reverse(basis):
    """H_ji(-R) = H_ij(R)^T for every pair, self-image pairs included, in each channel."""
    pos, cell, numbers = structure()
    out = random_model(basis).predict(pos, cell, numbers)
    assert set(out) == {"H", "S"} and len(out["H"]) > 50
    n_self = 0
    for name, blocks in out.items():
        for (Rx, Ry, Rz, i, j), blk in blocks.items():
            assert blk.shape == (ORB.n_orb(numbers[i]), ORB.n_orb(numbers[j])) and np.abs(blk).max() > 1e-6
            assert np.allclose(blocks[(-Rx, -Ry, -Rz, j, i)], blk.T, atol=1e-12), (name, Rx, Ry, Rz, i, j)
            n_self += i == j
    assert n_self > 0
    assert not np.allclose(out["H"][next(iter(out["H"]))], out["S"][next(iter(out["S"]))])


@pytest.mark.parametrize("improper", [False, True])
def test_blocks_rotate_with_the_orbitals(basis, improper):
    pos, cell, numbers = structure()
    m = random_model(basis)
    q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    Q = q * np.sign(np.linalg.det(q))
    s = -1 if improper else 1
    a, b = m.predict(pos, cell, numbers), m.predict(s * pos @ Q.T, s * cell @ Q.T, numbers)
    for name in a:
        assert set(a[name]) == set(b[name])
        for key, blk in a[name].items():
            Di = atom_rotation(int(numbers[key[3]]), Q, improper)
            Dj = atom_rotation(int(numbers[key[4]]), Q, improper)
            assert np.abs(b[name][key] - Di @ blk @ Dj.T).max() < 1e-9 * np.abs(blk).max(), (name, key)


def test_two_centre_only_model_is_a_function_of_the_bond(basis):
    """With no coupled features the block depends on the bond vector alone (a Slater-Koster table)."""
    two = basis._replace(n_cpl=0)
    m = random_model(two)
    pos, cell, numbers = structure()
    a = m.predict(pos, cell, numbers)["H"]
    pos2 = pos.copy(); pos2[2:] += 0.2                                   # move atoms 2 and 3 together
    b = m.predict(pos2, cell, numbers)["H"]
    k = next(k for k in a if k[3] == 2 and k[4] == 3 and k in b)         # their bond is unchanged
    assert np.allclose(a[k], b[k], atol=1e-12)
    k = next(k for k in a if k[3] == 0 and k[4] == 2 and k in b)
    assert not np.allclose(a[k], b[k], atol=1e-6)


def test_fit_on_real_data(basis):
    """SiC fixture: blocks of pairs closer than 3.2 angstrom, s to f shells.  A two-centre
    model with enough radial functions reproduces the overlap (it IS two-centre) to the data's
    own integration error, and the streamed fit equals the fit in one chunk."""
    require_optional("h5py")
    d = deeph.read(FIX)
    two = bond.BondBasis(site=basis.site, rcut_bond=3.3, n_bond=10)
    model, report = fit_offsite(two, [d], channels=("S", "H"))
    assert model.channels == ("S", "H") and len(report) > 50
    pred = model.predict(d.positions, d.cell, d.numbers)
    off = [k for k in d.blocks["S"] if np.linalg.norm(d.bond(k)) > 1e-6]
    assert set(off) <= set(pred["S"])
    err = max(np.abs(pred["S"][k] - d.blocks["S"][k]).max() for k in off)
    assert err < 3e-3, err                                              # the data's integration error is ~1e-3
    again, _ = fit_offsite(two, [d], channels=("S", "H"), chunk=7)
    pred2 = again.predict(d.positions, d.cell, d.numbers)                # few bonds: compare predictions, not coefficients
    assert max(np.abs(pred2["S"][k] - pred["S"][k]).max() for k in off) < 1e-6
    # H is not two-centre: the same model leaves a real residual, far above the overlap's
    errH = max(np.abs(pred["H"][k] - d.blocks["H"][k]).max() for k in off)
    assert errH > 10 * err


def test_compiled_prediction_is_the_reference(basis):
    """`OffsitePredictor` (coefficients contracted with the site features once per atom, a
    matrix product per atom over its bonds) against `OffsiteModel.predict`, every bond and
    channel, both species orders, with and without coupled features."""
    from ace_jax.operators.offsite_fast import OffsitePredictor
    for b in (basis, basis._replace(n_cpl=0)):
        m = random_model(b)
        fast = OffsitePredictor(m)
        for seed in (0, 3):
            pos, cell, numbers = structure(seed)
            ref, got = m.predict(pos, cell, numbers), fast(pos, cell, numbers)
            for name in ref:
                assert set(ref[name]) == set(got[name])
                worst = max(np.abs(got[name][k] - v).max() / max(1.0, np.abs(v).max()) for k, v in ref[name].items())
                assert worst < 1e-10, (name, worst)


@pytest.fixture(scope="module")
def basis_env(basis):
    return basis._replace(r_env=4.5, n_env=3, l_env=2, n_ebond=2, l_ebond=2)


@pytest.mark.parametrize("improper", [False, True])
def test_bond_environment_model_symmetries(basis_env, improper):
    """With the bond-environment group: every bond's reverse is still its transpose, and the
    blocks rotate and invert with the orbitals."""
    pos, cell, numbers = structure()
    m = random_model(basis_env)
    a = m.predict(pos, cell, numbers)
    for name, blocks in a.items():
        for (Rx, Ry, Rz, i, j), blk in blocks.items():
            assert np.allclose(blocks[(-Rx, -Ry, -Rz, j, i)], blk.T, atol=1e-12), (name, Rx, Ry, Rz, i, j)
    q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    Q = q * np.sign(np.linalg.det(q))
    s = -1 if improper else 1
    b = m.predict(s * pos @ Q.T, s * cell @ Q.T, numbers)
    for name in a:
        for key, blk in a[name].items():
            Di, Dj = atom_rotation(int(numbers[key[3]]), Q, improper), atom_rotation(int(numbers[key[4]]), Q, improper)
            assert np.abs(b[name][key] - Di @ blk @ Dj.T).max() < 1e-9 * np.abs(blk).max(), (name, key)
    # the environment group matters: the same coefficients without it give other blocks
    k = next(iter(a["H"]))
    assert not np.allclose(a["H"][k], random_model(basis_env._replace(n_env=0)).predict(pos, cell, numbers)["H"][k])


def test_predict_selected_bonds(basis_env):
    pos, cell, numbers = structure()
    m = random_model(basis_env)
    full = m.predict(pos, cell, numbers)
    some = m.predict(pos, cell, numbers, only=lambda g: np.flatnonzero(g.i == 0))
    assert set(some["H"]) == {k for k in full["H"] if k[3] == 0} and len(some["H"]) > 5
    assert all(np.allclose(some["H"][k], full["H"][k], atol=1e-13) for k in some["H"])


def test_compiled_predictor_with_the_environment_group(basis_env):
    """The compiled predictor with the bond-environment group equals the reference predictor."""
    from ace_jax.operators.offsite_fast import OffsitePredictor
    m = random_model(basis_env)
    fast = OffsitePredictor(m)
    for seed in (0, 3):
        pos, cell, numbers = structure(seed)
        ref, got = m.predict(pos, cell, numbers), fast(pos, cell, numbers)
        for name in ref:
            assert set(ref[name]) == set(got[name])
            worst = max(np.abs(got[name][k] - v).max() / max(1.0, np.abs(v).max()) for k, v in ref[name].items())
            assert worst < 1e-10, (name, worst)


@pytest.mark.parametrize("threads", [2, 5])
def test_threaded_prediction_is_the_single_call(basis_env, threads):
    """Rows in chunks on a thread pool (`threads` > 1): the same blocks as the one compiled call,
    with the coupled and bond-environment groups on, for chunk counts that do and do not divide
    the number of atoms of a species."""
    from ace_jax.operators.offsite_fast import OffsitePredictor
    m = random_model(basis_env)
    one, many = OffsitePredictor(m), OffsitePredictor(m, threads=threads)
    for seed in (0, 3):
        pos, cell, numbers = structure(seed)
        ref, got = one(pos, cell, numbers), many(pos, cell, numbers)
        for name in ref:
            assert set(ref[name]) == set(got[name])
            assert max(np.abs(got[name][k] - v).max() for k, v in ref[name].items()) < 1e-12 * max(
                np.abs(v).max() for v in ref[name].values())


def test_species_aware_environment_on_real_data(basis):
    """SiC fixture, H: the bond environment fits the off-site blocks better than the two-centre
    model alone, and with one channel per element no worse than blind to the species of its atoms
    (in this perfect crystal every bond of a kind has the same environment, so the blind group
    already fits what the group can: the species dependence itself is tested on a random
    structure in test_operators_bond.py).  The compiled predictor (one call, and chunks on
    threads) returns the reference predictor's blocks for the two-element structure."""
    from ace_jax.operators.offsite_fast import OffsitePredictor
    require_optional("h5py")
    d = deeph.read(FIX)
    off = [k for k in d.blocks["H"] if np.linalg.norm(d.bond(k)) > 1e-6]
    two = bond.BondBasis(site=basis.site, rcut_bond=3.3, n_bond=6)
    aware = two._replace(r_env=3.5, n_env=2, l_env=2, n_ebond=2, l_ebond=2)
    blind = aware._replace(env_emb=((1.0,), (1.0,)))
    rms = {}
    for name, b in (("two", two), ("blind", blind), ("aware", aware)):
        model, _ = fit_offsite(b, [d], channels=("H",), ridge=1e-8)
        pred = model.predict(d.positions, d.cell, d.numbers)["H"]
        rms[name] = np.sqrt(np.mean(np.concatenate([(pred[k] - d.blocks["H"][k]).ravel() for k in off]) ** 2))
        if name == "aware":
            for fast in (OffsitePredictor(model), OffsitePredictor(model, threads=3)):
                got = fast(d.positions, d.cell, d.numbers)["H"]
                assert set(got) >= set(off)
                assert max(np.abs(got[k] - pred[k]).max() for k in off) < 1e-9 * max(np.abs(pred[k]).max() for k in off)
    assert rms["aware"] < rms["blind"] * (1 + 1e-6) and rms["blind"] < 0.8 * rms["two"], rms
