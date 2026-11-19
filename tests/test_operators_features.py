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


def test_provider_protocol_and_radial_table():
    """The site basis is a `SiteFeatures` provider that holds no model, and its radial tabulated
    in r (`with_table`, the CPU form) gives the same features at physical distances."""
    from ace_jax.eval.features import SiteFeatures
    from ace_jax.eval.radial import TensorRadial
    from ace_jax.operators.features import RadialTable
    b = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))
    assert isinstance(b, SiteFeatures) and isinstance(b.radial, TensorRadial)
    t = b.with_table()
    assert isinstance(t.radial, RadialTable) and t.with_table() is t and isinstance(t, SiteFeatures)
    rng = np.random.default_rng(0)
    E, n = 300, 20
    u = rng.normal(size=(E, 3))
    rij = u / np.linalg.norm(u, axis=1)[:, None] * rng.uniform(1.2, 4.3, (E, 1))        # some past the cutoff
    zi, zj = rng.integers(0, 2, E).astype(np.int32), rng.integers(0, 2, E).astype(np.int32)
    seg = np.sort(rng.integers(0, n, E)).astype(np.int32)
    fa, fb = b.features(rij, zi, zj, seg, n), t.features(rij, zi, zj, seg, n)
    scale = max(float(np.abs(v).max()) for v in fa.values())
    assert max(float(np.abs(fa[k] - fb[k]).max()) for k in fa) < 1e-11 * scale


# ---- element embedding (the species-compressed provider)
ELS = (1, 6, 7, 8, 9)


def test_embedding_provider_is_equivariant_and_compressed():
    """Five elements through two embedding channels at order 2: every feature rotates with the
    Wigner matrix of its L and has its parity, and there are as many features as a two-element
    basis of the same radials has channels for, not five elements' worth."""
    from ace_jax.eval.features import SiteFeatures
    from ace_jax.operators.features import build_embedding_site_basis
    kw = dict(order=2, totaldegree=5, lmax_feat=3, rcut=4.0, coupling_cache_dir=str(CACHE))
    b = build_embedding_site_basis(ELS, d_max=2, **kw)                       # the default: a random embedding
    emb = np.asarray(b.radial.rnl_embedding)
    assert emb.shape == (5, 2) and np.allclose(np.linalg.norm(emb, axis=1), 1.0)
    assert np.array_equal(emb, np.asarray(build_embedding_site_basis(ELS, d_max=2, **kw).radial.rnl_embedding))   # seeded
    assert not np.allclose(emb, np.asarray(build_embedding_site_basis(ELS, d_max=2, seed=1, **kw).radial.rnl_embedding))
    assert isinstance(b, SiteFeatures) and b.with_table() is b and b.elements == ELS
    two = build_embedding_site_basis(ELS[:2], embedding="identity", d_max=2, **kw)     # order-2 width 2, not C(3, 2)
    assert b.lps == two.lps and b.n_B == two.n_B
    rng = np.random.default_rng(0)
    rij = rng.normal(size=(7, 3)) * 1.2
    zi, zj, seg = np.zeros(7, np.int32), rng.integers(0, 5, 7).astype(np.int32), np.zeros(7, np.int32)
    B = feats(b, rij, zi, zj, seg)
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    Q = q * np.sign(np.linalg.det(q))
    Br, Bi = feats(b, rij @ Q.T, zi, zj, seg), feats(b, -rij, zi, zj, seg)
    for (L, p), v in B.items():
        scale = np.abs(v).max()
        assert v.shape == (1, b.n(L, p), 2 * L + 1) and scale > 1e-6
        assert np.abs(Br[(L, p)] - v @ (wigner(L, Q) if L else np.ones((1, 1))).T).max() < 1e-10 * scale, (L, p)
        assert np.abs(Bi[(L, p)] - p * v).max() < 1e-12 * scale, (L, p)
    assert np.abs(feats(b, rij, zi, (zj + 1) % 5, seg)[(2, 1)] - B[(2, 1)]).max() > 1e-6       # species matter


def test_identity_embedding_is_the_categorical_basis():
    """Order 1 with the identity embedding: channel k is "neighbours of element k".  The features
    of a neighbourhood are the sum over elements of the features of that element's neighbours
    alone, each of which lives in its own channel, and relabelling the elements of a
    one-element neighbourhood moves the same numbers to another channel."""
    from ace_jax.operators.features import build_embedding_site_basis
    b = build_embedding_site_basis(ELS[:3], 1, 6, 3, "identity", rcut=4.0, wL=0.5, coupling_cache_dir=str(CACHE))
    rng = np.random.default_rng(1)
    rij = rng.normal(size=(9, 3)) * 1.3
    zi, zj, seg = np.zeros(9, np.int32), rng.integers(0, 3, 9).astype(np.int32), np.zeros(9, np.int32)
    full = feats(b, rij, zi, zj, seg)
    parts = [feats(b, rij[zj == k], zi[zj == k], zj[zj == k], seg[zj == k]) for k in range(3)]
    as_0 = [feats(b, rij[zj == k], zi[zj == k], np.zeros((zj == k).sum(), np.int32), seg[zj == k]) for k in range(3)]
    for lp, v in full.items():
        assert np.allclose(v, sum(p[lp] for p in parts), atol=1e-12 * np.abs(v).max())
        live = [np.flatnonzero(np.abs(p[lp][0]).max(axis=1) > 1e-12) for p in parts]
        assert all(len(a) == b.n(*lp) // 3 for a in live) and len(np.unique(np.concatenate(live))) == b.n(*lp)
        for k in range(3):                                # the same numbers in channel 0 when the element is relabelled
            lv0 = np.flatnonzero(np.abs(as_0[k][lp][0]).max(axis=1) > 1e-12)
            assert np.allclose(parts[k][lp][0][live[k]], as_0[k][lp][0][lv0], atol=1e-12 * np.abs(v).max())


def test_embedding_widths_per_order():
    """`widths` sets the channels of each order: order 1 at full width (it is linear in each
    element's neighbour density, so fewer channels lose information), order 2 compressed."""
    from ace_jax.operators.features import build_embedding_site_basis
    kw = dict(rcut=4.0, wL=0.5, coupling_cache_dir=str(CACHE))
    n = lambda b: sum(b.n(*lp) for lp in b.lps)
    full, one = build_embedding_site_basis(ELS[:3], 2, 6, 2, d_max=3, **kw), build_embedding_site_basis(ELS[:3], 2, 6, 2, d_max=1, **kw)
    mixed = build_embedding_site_basis(ELS[:3], 2, 6, 2, widths=(3, 1), **kw)
    first = lambda b: n(build_embedding_site_basis(ELS[:3], 1, 6, 2, d_max=b, **kw))
    assert n(one) < n(mixed) < n(full)
    assert n(mixed) - n(one) == first(3) - first(1)                      # the two more order-1 channels, nothing else
    assert np.asarray(mixed.radial.rnl_embedding).shape[-1] == 3
