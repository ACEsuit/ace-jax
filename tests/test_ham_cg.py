"""Real Clebsch-Gordan tensors against Wigner matrices of ace-jax's own harmonics."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.ham import cg

LL = [(l1, l2) for l1 in range(4) for l2 in range(4)]


def wigner(L, Q):
    """D with Y_L(Q r) = D Y_L(r)."""
    pts = np.random.default_rng(1).normal(size=(80, 3))
    sl = slice(L * L, (L + 1) ** 2)
    Y0 = np.asarray(real_spherical_harmonics(pts, max(L, 1)))[:, sl]
    Y1 = np.asarray(real_spherical_harmonics(pts @ Q.T, max(L, 1)))[:, sl]
    return np.linalg.lstsq(Y0, Y1, rcond=None)[0].T


def rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


@pytest.mark.parametrize("l1, l2", LL)
def test_stack_is_orthogonal(l1, l2):
    S = np.concatenate([cg.real_cg(l1, l2, L).reshape(-1, 2 * L + 1) for L in cg.Ls(l1, l2)], axis=1)
    assert S.shape[0] == S.shape[1]
    assert np.allclose(S.T @ S, np.eye(S.shape[0]), atol=1e-13)


@pytest.mark.parametrize("l1, l2", LL)
def test_intertwines_the_harmonics_rotations(l1, l2):
    Q = rotation(3)
    D1, D2 = wigner(l1, Q), wigner(l2, Q)
    for L in cg.Ls(l1, l2):
        T = cg.real_cg(l1, l2, L)
        assert np.allclose(np.einsum("ai,bj,abM->ijM", D1, D2, T), np.einsum("ijN,MN->ijM", T, wigner(L, Q)),
                           atol=1e-11)


@pytest.mark.parametrize("l1, l2", LL)
def test_product_of_harmonics_is_a_harmonic(l1, l2):
    """Two-centre form: sum T Y_l1(r) Y_l2(r) is c Y_L(r) for l1+l2+L even (c != 0) and 0 for odd."""
    pts = np.random.default_rng(5).normal(size=(9, 3))
    Y = np.asarray(real_spherical_harmonics(pts, 6))
    y = lambda l: Y[:, l * l:(l + 1) ** 2]
    for L in cg.Ls(l1, l2):
        F = np.einsum("pi,pj,ijM->pM", y(l1), y(l2), cg.real_cg(l1, l2, L))
        if (l1 + l2 + L) % 2:
            assert np.abs(F).max() < 1e-13
        else:
            c = (F * y(L)).sum() / (y(L) ** 2).sum()
            assert abs(c) > 1e-3 and np.allclose(F, c * y(L), atol=1e-12)


def test_transpose_sign():
    """T(l2, l1, L)[j, i, M] = (-1)^(l1+l2-L) T(l1, l2, L)[i, j, M]: the sign bond reversal uses."""
    for l1, l2 in LL:
        for L in cg.Ls(l1, l2):
            assert np.allclose(cg.real_cg(l2, l1, L).transpose(1, 0, 2), (-1) ** (l1 + l2 - L) * cg.real_cg(l1, l2, L),
                               atol=1e-14)


def test_project_assemble_roundtrip():
    b = np.random.default_rng(0).normal(size=(4, 5, 3))
    parts = cg.project(b, 2, 1)
    assert set(parts) == {1, 2, 3} and parts[2].shape == (4, 5)
    assert np.allclose(cg.assemble(parts, 2, 1), b, atol=1e-13)


def test_bad_L():
    with pytest.raises(ValueError):
        cg.real_cg(1, 1, 3)
