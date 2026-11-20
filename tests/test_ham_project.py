"""Core-orbital projection (`ham.project.core_projection`) on a small synthetic periodic system."""
import jax

jax.config.update("jax_enable_x64", True)

import itertools

import numpy as np
import scipy.linalg

from ace_jax.ham.operator import BlockSum
from ace_jax.ham.project import core_projection
from ace_jax.operators.data import OperatorData
from ace_jax.operators.orbitals import OrbitalBasis


def system(coupling=0.05, seed=0):
    """Two atoms, shells s s p (the first s is the core, an eigenfunction of H at -50 up to
    `coupling`), blocks for every pair in the cells R in {-1, 0, 1}^3: with a 3 x 3 x 3 grid the
    back transform of any function of k is exact."""
    rng = np.random.default_rng(seed)
    basis = OrbitalBasis.of({1: (0, 0, 1)})
    n = basis.n_orb(1)
    H, S = {}, {}
    for R in itertools.product((-1, 0, 1), repeat=3):
        for i in range(2):
            for j in range(2):
                key, rev = (*R, i, j), (-R[0], -R[1], -R[2], j, i)
                if rev in H:
                    H[key], S[key] = H[rev].T.copy(), S[rev].T.copy()
                    continue
                on = R == (0, 0, 0) and i == j
                h, s = 0.05 * rng.normal(size=(n, n)), 0.02 * rng.normal(size=(n, n))
                if on:
                    h, s = 0.5 * (h + h.T) + np.diag([0.0, -1.0, 0.5, 0.5, 0.5]), 0.5 * (s + s.T) + np.eye(n)
                # the core function is an eigenfunction at -50 up to `coupling`: H c = -50 S c + noise
                noise = coupling * rng.normal(size=(2, n))
                h[0, :] = -50.0 * s[0, :] + noise[0]
                h[:, 0] = -50.0 * s[:, 0] + noise[1]
                if on:
                    h[0, 1:] = h[1:, 0] = 0.5 * (h[0, 1:] + h[1:, 0])
                H[key], S[key] = h, s
    cell = 4.0 * np.eye(3)
    return OperatorData(cell, np.array([[0.0, 0, 0], [1.9, 2.1, 2.0]]), np.array([1, 1]), basis, {"H": H, "S": S})


def test_projection_is_the_orthogonalisation_to_the_core():
    """On the grid, the Bloch matrices of the projected blocks are the valence blocks of
    B^+ H B and B^+ S B with B = [[1, -S_cc^-1 S_cv], [0, 1]]; the blocks are real, each bond's
    reverse is its transpose, and the valence basis is the shells that are left."""
    d = system()
    p, info = core_projection(d, {1: (0,)})
    assert info["nk"] == (3, 3, 3) and info["n_core"] == 2 and info["n_valence"] == 8 and p.basis.ls(1) == (0, 1)
    assert set(p.blocks["H"]) == set(d.blocks["H"])
    for name in ("H", "S"):
        for (Rx, Ry, Rz, i, j), b in p.blocks[name].items():
            assert b.shape == (4, 4) and np.allclose(p.blocks[name][(-Rx, -Ry, -Rz, j, i)], b.T, atol=1e-12)
    ks = np.array([[a / 3, b / 3, c / 3] for a in range(3) for b in range(3) for c in range(3)])
    full, val = BlockSum(list(d.blocks["H"]), d.numbers, d.basis), BlockSum(list(p.blocks["H"]), p.numbers, p.basis)
    c, v = np.array([0, 5]), np.array([1, 2, 3, 4, 6, 7, 8, 9])
    for name in ("H", "S"):
        M, Sk, got = np.asarray(full.matrices(d.blocks[name], ks)), np.asarray(full.matrices(d.blocks["S"], ks)), np.asarray(val.matrices(p.blocks[name], ks))
        for q in range(len(ks)):
            X = np.linalg.solve(Sk[q][np.ix_(c, c)], Sk[q][np.ix_(c, v)])
            Bm = np.zeros((10, 8), complex)
            Bm[c] = -X
            Bm[v] = np.eye(8)
            assert np.abs(got[q] - Bm.conj().T @ M[q] @ Bm).max() < 1e-10, (name, q)


def test_valence_spectrum_survives_when_the_core_is_decoupled():
    """With a small core-valence block of H the valence eigenvalues of the projected problem are
    those of the full one (to second order in the coupling), and the error grows with it."""
    ks = np.array([[0.0, 0.0, 0.0], [1 / 3, 0.0, 2 / 3], [1 / 3, 1 / 3, 1 / 3]])
    err = []
    for coupling in (1e-4, 1e-2):
        d = system(coupling)
        p, info = core_projection(d, {1: (0,)})
        full, val = BlockSum(list(d.blocks["H"]), d.numbers, d.basis), BlockSum(list(p.blocks["H"]), p.numbers, p.basis)
        e_full = np.array([scipy.linalg.eigh(h, s, eigvals_only=True) for h, s in zip(np.asarray(full.matrices(d.blocks["H"], ks)), np.asarray(full.matrices(d.blocks["S"], ks)))])
        e_val = np.array([scipy.linalg.eigh(h, s, eigvals_only=True) for h, s in zip(np.asarray(val.matrices(p.blocks["H"], ks)), np.asarray(val.matrices(p.blocks["S"], ks)))])
        assert (e_full[:, :2] < -40).all()                               # the two core levels
        err.append(np.abs(e_val - e_full[:, 2:]).max())
    assert err[0] < 1e-6 and err[0] < 0.01 * err[1]
