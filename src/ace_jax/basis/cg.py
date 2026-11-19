"""Real Clebsch-Gordan tensors in ace-jax's harmonics convention.

`real_cg(l1, l2, L)` is T[m1, m2, M] (indices m + l) with: for f, g that rotate
as the real Y_l1, Y_l2 of `eval/harmonics.py` (no Condon-Shortley phase, order
m = -l..l), F_M = sum T[m1, m2, M] f_m1 g_m2 rotates as the real Y_L.  The stack
over L = |l1-l2|..l1+l2 is an orthogonal (2l1+1)(2l2+1) square matrix, so a
block of an operator between shells l1 and l2 splits exactly into its L parts
(`project`) and is rebuilt from them (`assemble`).

T comes from the complex coefficients (Racah's formula) and the unitary change
to the real harmonics; in the real basis it is purely real or purely imaginary,
and the nonzero part is kept, which fixes its sign.  tests/test_basis_cg.py checks
it against Wigner matrices taken from the harmonics themselves."""
import functools
import math

import numpy as np


def _cg(j1, m1, j2, m2, J, M):
    """<j1 m1 j2 m2 | J M>, integer arguments (Racah)."""
    if m1 + m2 != M or not abs(j1 - j2) <= J <= j1 + j2 or abs(M) > J:
        return 0.0
    f = math.factorial
    pre = math.sqrt((2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J) / f(j1 + j2 + J + 1))
    pre *= math.sqrt(f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2))
    s = 0.0
    for k in range(max(0, j2 - J - m1, j1 + m2 - J), min(j1 + j2 - J, j1 - m1, j2 + m2) + 1):
        s += (-1) ** k / (f(k) * f(j1 + j2 - J - k) * f(j1 - m1 - k) * f(j2 + m2 - k)
                          * f(J - j2 + m1 + k) * f(J - j1 - m2 + k))
    return pre * s


def _real_from_complex(l):
    """U with Y^real_m = sum_mu U[m, mu] Y^complex_mu (complex with the
    Condon-Shortley phase, real as eval/harmonics.py: without it)."""
    U = np.zeros((2 * l + 1, 2 * l + 1), complex)
    U[l, l] = 1.0
    r = 1 / math.sqrt(2)
    for k in range(1, l + 1):
        U[l + k, l + k], U[l + k, l - k] = (-1) ** k * r, r                # sqrt2 P cos(k phi)
        U[l - k, l + k], U[l - k, l - k] = (-1) ** k * r / 1j, -r / 1j     # sqrt2 P sin(k phi)
    return U


@functools.cache
def real_cg(l1, l2, L):
    """T[m1, m2, M], shape (2l1+1, 2l2+1, 2L+1), float64; sum over (m1, m2) of T T = identity in M."""
    if not abs(l1 - l2) <= L <= l1 + l2:
        raise ValueError(f"L = {L} is not in |l1-l2|..l1+l2 for l1, l2 = {l1}, {l2}")
    C = np.array([[[_cg(l1, a, l2, b, L, c) for c in range(-L, L + 1)] for b in range(-l2, l2 + 1)]
                  for a in range(-l1, l1 + 1)])
    T = np.einsum("Mc,abc,ia,jb->ijM", _real_from_complex(L), C,
                  _real_from_complex(l1).conj(), _real_from_complex(l2).conj())
    re, im = np.abs(T.real).max(), np.abs(T.imag).max()
    if min(re, im) > 1e-12:
        raise AssertionError(f"real CG ({l1}, {l2}, {L}) is neither real nor imaginary")
    T = np.ascontiguousarray(T.real if re > im else T.imag)
    T.setflags(write=False)
    return T


def Ls(l1, l2):
    return range(abs(l1 - l2), l1 + l2 + 1)


def project(block, l1, l2):
    """{L: (..., 2L+1)} parts of a (..., 2l1+1, 2l2+1) block."""
    return {L: np.einsum("...ij,ijM->...M", block, real_cg(l1, l2, L)) for L in Ls(l1, l2)}


def assemble(parts, l1, l2):
    """The block of `project`'s parts (missing L count as zero)."""
    return sum(np.einsum("...M,ijM->...ij", parts[L], real_cg(l1, l2, L)) for L in Ls(l1, l2) if L in parts)
