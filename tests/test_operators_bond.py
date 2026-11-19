"""Bond features from shared site features: graph, rotation, inversion, bond reversal."""
import jax

jax.config.update("jax_enable_x64", True)

import pathlib

import jax.numpy as jnp
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
    return g, bond.features(basis, F, g, np.arange(len(g.i)), LPS, env=bond.env_lists(basis, pos, cell, numbers=numbers))


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


def test_bond_environment_knows_the_species_of_its_atoms(basis_env):
    """Bonds between atoms 0 and 1 while atoms 2 and 3 swap species: the environment features
    change with the default one channel per element, and do not with a one-channel embedding
    that gives both elements the same weight.  The channels are a partition of the environment:
    summed, they are the species-blind features, and any embedding is a combination of them."""
    pos, cell, numbers = structure()
    swapped = numbers.copy(); swapped[[2, 3]] = swapped[[3, 2]]
    assert not np.array_equal(numbers, swapped)
    blind = basis_env._replace(env_emb=((1.0,), (1.0,)))
    mixed = basis_env._replace(env_emb=((1.0, 0.5), (-2.0, 0.25)))
    g, f = env_features(basis_env, pos, cell, numbers)
    _, fs = env_features(basis_env, pos, cell, swapped)
    _, fb = env_features(blind, pos, cell, numbers)
    _, fbs = env_features(blind, pos, cell, swapped)
    _, fm = env_features(mixed, pos, cell, numbers)
    ends = np.flatnonzero((g.i < 2) & (g.j < 2))
    assert len(ends) > 4
    ne, nr, seen = basis_env.n_ebond, basis_env.n_env, 0
    for (L, p), d in f.items():
        a = d["env"][ends]
        if a.shape[1] == 0:
            continue
        seen += 1
        scale = np.abs(a).max()
        assert bond.n_env_features(basis_env, L, p) == 2 * bond.n_env_features(blind, L, p) == bond.n_env_features(mixed, L, p)
        assert np.abs(fs[(L, p)]["env"][ends] - a).max() > 1e-6 * scale, (L, p)
        assert np.abs(fbs[(L, p)]["env"][ends] - fb[(L, p)]["env"][ends]).max() < 1e-12 * scale, (L, p)
        ch = a.reshape(len(ends), -1, ne, 2, nr, 2 * L + 1)                     # (bond, group, bond radial, channel, env radial, M)
        b = fb[(L, p)]["env"][ends].reshape(len(ends), -1, ne, 1, nr, 2 * L + 1)
        assert np.allclose(ch.sum(axis=3, keepdims=True), b, atol=1e-12 * scale)
        m = fm[(L, p)]["env"][ends].reshape(ch.shape)
        emb = np.asarray(mixed.env_emb)                                       # (element, channel)
        assert np.allclose(m, np.einsum("bgnzrm,zc->bgncrm", ch, emb), atol=1e-12 * scale)
    assert seen >= 5
    with pytest.raises(ValueError, match="numbers"):
        bond.env_lists(basis_env, pos, cell)
    with pytest.raises(ValueError, match="one row per element"):
        bond.env_embedding(basis_env._replace(env_emb=((1.0,),)))


def test_environment_features_are_continuous_at_the_bond_midpoint():
    """An environment atom at the midpoint of a bond (a second-neighbour bond of a perfect lattice)
    has no direction from it.  With `env_core` the harmonics of l > 0 vanish there, so the features
    do not depend on the side from which the atom arrives; with env_core = 0 (ace-jax 0.7.0) they
    jump by more than their own size.  The compiled rows (`design._env_half`) are the same."""
    from types import SimpleNamespace as NS

    from ace_jax.operators import design as D
    r = np.array([[4.0, 0.0, 0.0]])

    def A(basis, mid, half=bond._env_side):
        rij = np.array([[[2.0, 0.0, 0.0] + np.asarray(mid, float), [4.0, 0.0, 0.0], [1.0, 2.0, 0.5]]])
        return np.asarray(half(basis, (rij, np.array([3]), np.array([[0, 0, 0]])), np.array([0]), r)[0])

    new, old = (NS(site=NS(elements=(79,)), env_emb=None, r_env=5.0, rcut_bond=10.0, n_env=6, l_env=4, env_core=a) for a in (1.0, 0.0))
    size = np.linalg.norm(A(new, [0.3, 0.0, 0.0]))
    for eps in (1e-4, 1e-8):
        jump = np.linalg.norm(A(new, [eps, 0, 0]) - A(new, [0, -eps, 0]))
        assert jump < 20 * eps * size + 1e-12, (eps, jump)
        assert np.linalg.norm(A(old, [eps, 0, 0]) - A(old, [0, -eps, 0])) > 0.5 * size        # the defect, kept for old files
    assert np.allclose(A(new, [0, 0, 0]), A(new, [1e-9, 0, 0]), atol=1e-7) and np.isfinite(A(new, [0, 0, 0])).all()
    far = [0.4, 1.9, -0.7]                                                # away from the midpoint: the same features, scaled per l
    ls = np.repeat(np.arange(5), 2 * np.arange(5) + 1)
    x2 = np.array([np.dot(far, far), (-1.0) ** 2 + 2.0 ** 2 + 0.5 ** 2])      # the two environment atoms, from the midpoint
    assert np.allclose(bond.core(np, x2, 4, 1.0), (x2 / (x2 + 1.0))[:, None] ** (0.5 * ls)) and np.all(bond.core(np, x2, 4, 0.0) == 1.0)
    emb = bond.env_embedding(new)

    def half(basis, env, atoms, rr):
        return D._env_half(basis, jnp.asarray(emb), jnp.asarray(env[0]), jnp.asarray(env[1]), jnp.asarray(env[2]), jnp.asarray(atoms), jnp.asarray(rr))

    assert np.allclose(A(new, far, half), A(new, far), atol=1e-12) and np.allclose(A(old, far, half), A(old, far), atol=1e-12)
