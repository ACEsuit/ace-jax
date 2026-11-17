"""Bond features from shared site features: graph, rotation, inversion, bond reversal."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import numpy as np
import pytest

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.operators import bond
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.onsite import _features, site_graph

CACHE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ham" / "coupling_cache"
LPS = [(0, 1), (1, -1), (1, 1), (2, 1), (2, -1), (3, -1), (4, 1)]


@pytest.fixture(scope="module")
def basis():
    site = build_site_basis([14, 6], order=2, totaldegree=5, lmax_feat=4, rcut=4.0, coupling_cache_dir=str(CACHE))
    return bond.BondBasis(site=site, rcut_bond=5.5, n_bond=4, n_cpl=2, l_cpl=2)


def structure(seed=0):
    rng = np.random.default_rng(seed)
    cell = 4.6 * np.eye(3) + 0.3 * rng.normal(size=(3, 3))
    return rng.uniform(size=(4, 3)) @ cell, cell, np.array([14, 6, 14, 6])


def all_features(basis, pos, cell, numbers):
    rij, zi, zj, seg, n = site_graph(basis.site, pos, cell, numbers)
    F = {k: np.asarray(v) for k, v in _features(basis.site, rij, zi, zj, seg, n).items()}
    g = bond.bond_graph(basis, pos, cell, numbers)
    return g, bond.features(basis, F, g, np.arange(len(g.i)), LPS)


def wigner(L, Q):
    pts = np.random.default_rng(1).normal(size=(80, 3))
    sl = slice(L * L, (L + 1) ** 2)
    Y0 = np.asarray(real_spherical_harmonics(pts, max(L, 1)))[:, sl]
    Y1 = np.asarray(real_spherical_harmonics(pts @ Q.T, max(L, 1)))[:, sl]
    return np.linalg.lstsq(Y0, Y1, rcond=None)[0].T


def test_radial():
    r = np.linspace(0, 6, 50)
    R = bond.radial(r, 5.5, 6)
    assert R.shape == (50, 6) and not R[r >= 5.5].any() and np.abs(R[1]).max() > 0.5
    assert np.linalg.matrix_rank(R[r < 5.5]) == 6
    h = 1e-5
    assert np.abs(bond.radial(5.5 - h, 5.5, 6)).max() < 1e-8                 # value and slope vanish at the cutoff
    assert bond.radial(r, 5.5, 0).shape == (50, 0)


def test_graph_and_reverse(basis):
    pos, cell, numbers = structure()
    g = bond.bond_graph(basis, pos, cell, numbers)
    n = len(g.i)
    assert n > 100 and np.array_equal(g.reverse[g.reverse], np.arange(n))
    assert np.array_equal(g.i[g.reverse], g.j) and np.array_equal(g.R[g.reverse], -g.R)
    assert np.allclose(g.rij[g.reverse], -g.rij, atol=1e-12)
    assert np.allclose(g.rij, pos[g.j] + g.R @ cell - pos[g.i], atol=1e-10)
    assert np.linalg.norm(g.rij, axis=1).max() < basis.rcut_bond
    self_image = (g.i == g.j)
    assert self_image.any() and (np.abs(g.R[self_image]).sum(1) > 0).all()    # an atom and its own image


def test_layout_and_counts(basis):
    two, cpl = bond.layout(basis, 2, 1)
    assert two and all((-1) ** l * pi == 1 and abs(l - lam) <= 2 <= l + lam for l, lam, pi, _ in cpl)
    assert bond.layout(basis, 1, 1)[0] is False                                # an axial vector has no two-centre part
    k0, k1 = bond.n_features(basis, 2, 1)
    assert k0 == 4 and k1 == 2 * sum(nB for *_, nB in cpl) and k1 > 0
    assert bond.n_features(basis._replace(n_cpl=0), 2, 1) == (4, 0)


def test_rotation_and_inversion(basis):
    pos, cell, numbers = structure()
    g, f = all_features(basis, pos, cell, numbers)
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    Q = q * np.sign(np.linalg.det(q))
    gr, fr = all_features(basis, pos @ Q.T, cell @ Q.T, numbers)
    gi, fi = all_features(basis, -pos, -cell, numbers)
    order = lambda gg, s=1: {(int(a), int(b), *map(int, s * c)): k for k, (a, b, c) in enumerate(zip(gg.i, gg.j, gg.R))}
    o0 = order(g)
    pr = np.asarray([order(gr)[k] for k in o0])                                 # same (i, j, R): the cell rotates too
    pi_ = np.asarray([order(gi)[k] for k in o0])
    for (L, p), d in f.items():
        D = wigner(L, Q) if L else np.ones((1, 1))
        for grp in ("bond", "i", "j"):
            a = d[grp]
            if a.shape[1] == 0:
                continue
            scale = np.abs(a).max()
            assert scale > 1e-8, (L, p, grp)
            assert np.abs(fr[(L, p)][grp][pr] - a @ D.T).max() < 1e-9 * scale, (L, p, grp)
            assert np.abs(fi[(L, p)][grp][pi_] - p * a).max() < 1e-11 * scale, (L, p, grp)


def test_reverse_bond_swaps_the_groups(basis):
    """Features of (j, i, -R): the two-centre group times (-1)^L, the i and j groups exchanged
    with the sign (-1)^l of each feature.  The model's reversal symmetry rests on this."""
    pos, cell, numbers = structure()
    g, f = all_features(basis, pos, cell, numbers)
    for (L, _), d in f.items():
        assert np.allclose(d["bond"][g.reverse], (-1) ** L * d["bond"], atol=1e-12)
        s = d["sign"][None, :, None]
        assert np.allclose(d["i"][g.reverse], s * d["j"], atol=1e-12)
        assert np.allclose(d["j"][g.reverse], s * d["i"], atol=1e-12)
        # the symmetrised design of a same-shell block: reverse bond gives (-1)^L times it
        A = bond.design(d, L, same=True)
        assert np.allclose(A[g.reverse], (-1) ** L * A, atol=1e-12)
        assert bond.design(d, L, same=False).shape[1] == d["bond"].shape[1] + 2 * d["i"].shape[1]


@pytest.fixture(scope="module")
def basis_env(basis):
    return basis._replace(r_env=4.5, n_env=3, l_env=2, n_ebond=2, l_ebond=2)


def env_features(basis, pos, cell, numbers):
    rij, zi, zj, seg, n = site_graph(basis.site, pos, cell, numbers)
    F = {k: np.asarray(v) for k, v in _features(basis.site, rij, zi, zj, seg, n).items()}
    g = bond.bond_graph(basis, pos, cell, numbers)
    return g, bond.features(basis, F, g, np.arange(len(g.i)), LPS, env=bond.env_lists(basis, pos, cell))


def test_bond_environment_group(basis_env):
    """The environment about the midpoint: rotates with the Wigner matrices, parity (-1)^(l+lam)
    under inversion, and for the reverse bond the same up to the sign (-1)^l of its bond harmonic."""
    pos, cell, numbers = structure()
    g, f = env_features(basis_env, pos, cell, numbers)
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    Q = q * np.sign(np.linalg.det(q))
    gr, fr = env_features(basis_env, pos @ Q.T, cell @ Q.T, numbers)
    gi, fi = env_features(basis_env, -pos, -cell, numbers)
    order = lambda gg: {(int(a), int(b), *map(int, c)): k for k, (a, b, c) in enumerate(zip(gg.i, gg.j, gg.R))}
    pr = np.asarray([order(gr)[k] for k in order(g)])
    pi_ = np.asarray([order(gi)[k] for k in order(g)])
    seen = 0
    for (L, p), d in f.items():
        a = d["env"]
        assert a.shape[1] == bond.n_env_features(basis_env, L, p) == len(d["env_sign"])
        if a.shape[1] == 0:
            continue
        seen += 1
        scale = np.abs(a).max()
        assert scale > 1e-8
        D = wigner(L, Q) if L else np.ones((1, 1))
        assert np.abs(fr[(L, p)]["env"][pr] - a @ D.T).max() < 1e-9 * scale, (L, p)
        assert np.abs(fi[(L, p)]["env"][pi_] - p * a).max() < 1e-11 * scale, (L, p)
        assert np.allclose(a[g.reverse], d["env_sign"][None, :, None] * a, atol=1e-12 * scale)
        A = bond.design(d, L, same=True)
        assert np.allclose(A[g.reverse], (-1) ** L * A, atol=1e-12 * max(1.0, np.abs(A).max()))
        assert A.shape[1] == d["bond"].shape[1] + d["i"].shape[1] + bond.n_env_features(basis_env, L, p, same=True)
    assert seen >= 5
    # the pseudo-tensor symmetries get environment features too: (1, +1) has no two-centre part
    assert bond.n_env_features(basis_env, 1, 1) > 0 and f[(1, 1)]["bond"].shape[1] == 0


def test_bond_environment_sees_atoms_the_ends_do_not_share(basis_env):
    """Moving an atom near a bond changes the bond's environment features; an atom far from both
    ends does not; and a basis with an environment needs its lists."""
    pos, cell, numbers = structure()
    g, f = env_features(basis_env, pos, cell, numbers)
    with pytest.raises(ValueError, match="env_lists"):
        bond.features(basis_env, {}, g, np.arange(2), LPS)
    assert bond.env_lists(basis_env._replace(n_env=0), pos, cell) is None
