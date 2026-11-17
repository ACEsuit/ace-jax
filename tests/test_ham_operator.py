"""H(k), S(k) and eigenvalues on the device against a host assembly of the predicted blocks."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import numpy as np
import pytest
import scipy.linalg

from ace_jax.ham import bond, cg
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.offsite import OffsiteModel, canonical_pairs
from ace_jax.ham.onsite import OnsiteModel, parts_of, shell_pairs
from ace_jax.ham.operator import HamiltonianPredictor
from ace_jax.ham.orbitals import OrbitalBasis

CACHE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ham" / "coupling_cache"
ORB = OrbitalBasis.of({14: (0, 1, 1, 2), 6: (0, 1)})
CH = ("H", "S")


@pytest.fixture(scope="module")
def models():
    site = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))
    basis = bond.BondBasis(site=site, rcut_bond=5.0, n_bond=4, n_cpl=2, l_cpl=2)
    rng = np.random.default_rng(0)
    on = {(z, s1, s2, L): 0.05 * rng.normal(size=(2, site.n(L, p) + (L == 0)))
          for z in ORB.species for s1, s2, l1, l2 in shell_pairs(ORB, z) for L, p in parts_of(l1, l2, s1 == s2)
          if site.n(L, p) + (L == 0)}
    for (z, s1, s2, L), c in on.items():          # make S well conditioned: identity plus small terms
        if L == 0 and s1 == s2:
            c[1, -1] = np.sqrt(2 * ORB.ls(z)[s1] + 1) * (1 if cg.real_cg(ORB.ls(z)[s1], ORB.ls(z)[s1], 0)[0, 0, 0] > 0 else -1)
    off = {}
    for zi in ORB.species:
        for zj in ORB.species:
            for s1, s2, l1, l2, same in canonical_pairs(ORB, zi, zj):
                p = 1 if (l1 + l2) % 2 == 0 else -1
                for L in cg.Ls(l1, l2):
                    k0, k1 = bond.n_features(basis, L, p)
                    if k0 + k1:
                        off[(zi, zj, s1, s2, L)] = 0.01 * rng.normal(size=(2, k0 + (k1 if same else 2 * k1)))
    return (OnsiteModel(basis=site, coef=on, orbitals=ORB, channels=CH),
            OffsiteModel(bond=basis, coef=off, orbitals=ORB, channels=CH))


def structure(seed=0):
    rng = np.random.default_rng(seed)
    cell = 4.6 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    return rng.uniform(size=(4, 3)) @ cell, cell, np.array([14, 6, 14, 6])


def host_matrix(onsite, offsite, pos, cell, numbers, k, channel):
    """The same Bloch sum from the reference predictions, in numpy."""
    off = np.concatenate([[0], np.cumsum([ORB.n_orb(z) for z in numbers])])
    M = np.zeros((off[-1], off[-1]), complex)
    blocks = {**onsite.predict(pos, cell, numbers)[channel], **offsite.predict(pos, cell, numbers)[channel]}
    for (Rx, Ry, Rz, i, j), b in blocks.items():
        M[off[i]:off[i + 1], off[j]:off[j + 1]] += b * np.exp(2j * np.pi * np.dot(k, (Rx, Ry, Rz)))
    return M


def test_bloch_matrices_match_the_host_assembly(models):
    onsite, offsite = models
    pos, cell, numbers = structure()
    hp = HamiltonianPredictor(onsite, offsite)
    prep = hp.prepare(pos, cell, numbers)
    assert prep.n_orb == 2 * ORB.n_orb(14) + 2 * ORB.n_orb(6)
    for k in ((0.0, 0.0, 0.0), (0.31, -0.2, 0.45)):
        for c in CH:
            got, ref = np.asarray(hp.hk(prep, k, c)), host_matrix(onsite, offsite, pos, cell, numbers, k, c)
            assert np.abs(got - ref).max() < 1e-11 * max(1.0, np.abs(ref).max()), (k, c)
            assert np.abs(got - got.conj().T).max() < 1e-11                     # Hermitian: bond reversal is exact
    assert np.abs(np.asarray(hp.hk(prep, (0, 0, 0), "H")).imag).max() < 1e-12     # real at Gamma


def test_bands_match_scipy(models):
    onsite, offsite = models
    pos, cell, numbers = structure()
    hp = HamiltonianPredictor(onsite, offsite)
    prep = hp.prepare(pos, cell, numbers)
    kpts = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.13, 0.27, -0.4)]
    e = np.asarray(hp.bands(prep, kpts))
    assert e.shape == (3, prep.n_orb)
    for k, ek in zip(kpts, e):
        H = host_matrix(onsite, offsite, pos, cell, numbers, k, "H")
        S = host_matrix(onsite, offsite, pos, cell, numbers, k, "S")
        assert np.linalg.eigvalsh(S).min() > 0.2
        assert np.abs(ek - scipy.linalg.eigh(H, S, eigvals_only=True)).max() < 1e-9
    e0 = np.asarray(hp.bands(prep, [kpts[1]], overlap=None))[0]
    assert np.abs(e0 - np.linalg.eigvalsh(host_matrix(onsite, offsite, pos, cell, numbers, kpts[1], "H"))).max() < 1e-9
    # k and k + G give the same spectrum
    assert np.abs(np.asarray(hp.bands(prep, [(1.13, 0.27, -1.4)]))[0] - e[2]).max() < 1e-9


def test_models_must_share_the_site_basis(models):
    onsite, offsite = models
    other = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))
    with pytest.raises(ValueError, match="share one site basis"):
        HamiltonianPredictor(OnsiteModel(basis=other, coef=onsite.coef, orbitals=ORB, channels=CH), offsite)
    with pytest.raises(ValueError, match="channels"):
        HamiltonianPredictor(OnsiteModel(basis=onsite.basis, coef={}, orbitals=ORB, channels=("H",)), offsite)
