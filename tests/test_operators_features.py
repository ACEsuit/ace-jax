"""Site features for all L from one AA: rotation, inversion, the empty environment."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import numpy as np
import pytest

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.operators.features import build_site_basis

CACHE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ham" / "coupling_cache"


@pytest.fixture(scope="module")
def basis():
    return build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))


def wigner(L, Q):
    pts = np.random.default_rng(1).normal(size=(80, 3))
    sl = slice(L * L, (L + 1) ** 2)
    Y0 = np.asarray(real_spherical_harmonics(pts, max(L, 1)))[:, sl]
    Y1 = np.asarray(real_spherical_harmonics(pts @ Q.T, max(L, 1)))[:, sl]
    return np.linalg.lstsq(Y0, Y1, rcond=None)[0].T


def cluster(seed=0, n=6):
    """A centre (node 0) with n neighbours of two species: edges i = 0 -> j."""
    rng = np.random.default_rng(seed)
    rij = rng.normal(size=(n, 3)) * 1.2
    zj = rng.integers(0, 2, size=n).astype(np.int32)
    return rij, np.zeros(n, np.int32), zj, np.zeros(n, np.int32)


def feats(basis, rij, zi, zj, seg, n_nodes=1):
    return {k: np.asarray(v) for k, v in basis.features(rij, zi, zj, seg, n_nodes).items()}


def test_every_symmetry_is_present(basis):
    """Order 2 with l <= 3: the true tensors up to L = 3 and the axial vector; a pseudoscalar
    needs three vectors (order 3), so (0, -1) is absent and reads as zero features."""
    assert set(basis.lps) == {(0, 1), (1, -1), (1, 1), (2, 1), (3, -1)}
    assert basis.n(0, 1) > 0 and basis.n(0, -1) == 0 and basis.n(9, 1) == 0
    assert basis.elements == (14, 6)


def test_rotation_and_inversion(basis):
    rij, zi, zj, seg = cluster()
    B = feats(basis, rij, zi, zj, seg)
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    Q = q * np.sign(np.linalg.det(q))
    Br, Bi = feats(basis, rij @ Q.T, zi, zj, seg), feats(basis, -rij, zi, zj, seg)
    for (L, p), b in B.items():
        assert b.shape == (1, basis.n(L, p), 2 * L + 1) and np.abs(b).max() > 1e-6, (L, p)
        D = wigner(L, Q) if L else np.ones((1, 1))
        scale = np.abs(b).max()
        assert np.abs(Br[(L, p)] - b @ D.T).max() < 1e-10 * scale, (L, p)
        assert np.abs(Bi[(L, p)] - p * b).max() < 1e-12 * scale, (L, p)


def test_species_matter(basis):
    rij, zi, zj, seg = cluster()
    a, b = feats(basis, rij, zi, zj, seg), feats(basis, rij, zi, 1 - zj, seg)
    assert np.abs(a[(2, 1)] - b[(2, 1)]).max() > 1e-6


def test_no_neighbours_and_masked_edges(basis):
    """An atom with no neighbour has zero features (its block is then the constant part alone)."""
    rij, zi, zj, seg = cluster()
    B = feats(basis, rij, zi, zj, seg, n_nodes=2)                    # node 1 has no edge
    for b in B.values():
        assert np.isfinite(b).all() and not b[1].any()
    mask = np.arange(len(rij)) < 3
    half = {k: np.asarray(v) for k, v in basis.features(rij, zi, zj, seg, 1, mask).items()}
    ref = feats(basis, rij[:3], zi[:3], zj[:3], seg[:3])
    assert all(np.allclose(half[k], ref[k], atol=1e-13) for k in ref)


def test_unknown_element(basis):
    assert list(basis.species_index([6, 14, 14])) == [1, 0, 0]
    with pytest.raises(KeyError, match="Z = 8"):
        basis.species_index([14, 8])
