"""ace1-compatible radial pieces for Python model authoring (pure numpy).

What `ace1_model` / `ace_embedding_model(ace1_compat=true)` build in Julia:

    tensor radial   Agnesi (p, q) = (2, 4), PolyEnvelope2sX(-1, 1, 2, 2), and
                    Jacobi(4, 4) polynomials -- ACE1 folds the envelope into the
                    orthogonality, int Q_p Q_q env^2 dx = delta_pq -- then SPLINED
                    (`splinify`: cubic B-spline over 100 nodes of x in [-1, 1])
    pair radial     Agnesi (1, 3), ACE1_PolyEnvelope1sR(rcut, r0, 2), Legendre,
                    one-hot species weights, splined the same way

Parity: tests/test_construct_radial_ace1.py against Julia-exported models.
Polynomials4ML normalises the recurrence by quadgk (atol 1e-10); here the
normalising integrals are exact (Gauss-Legendre on a polynomial integrand), so
the tables agree to ~1e-10 rather than bit-for-bit.
"""
import math

import numpy as np

from .radial_init import BOND_LEN, agnesi_transform_params, envelope2sx_params, poly_eval

SPLINE_NODES = 100                   # Rnl_basis.jl splinify(nnodes = 100)


def jacobi_3term(N, alpha, beta):
    """(A, B, C) of the normalised Jacobi(alpha, beta) basis of length N, in the
    OrthPolyBasis1D3T convention (Polynomials4ML `orthpolybasis`):
    Q_0 = A[0];  Q_1 = A[1] x + B[1];  Q_k = (A[k] x + B[k]) Q_{k-1} + C[k] Q_{k-2},
    orthonormal under (1-x)^alpha (1+x)^beta on [-1, 1].  alpha = beta = 0 is
    `legendre_basis`."""
    a, b = float(alpha), float(beta)
    A, B, C = np.zeros(N), np.zeros(N), np.zeros(N)
    A[0] = 1.0
    if N > 1:
        A[1] = (a + b + 2) / 2
        B[1] = -(a + b + 2) / 2 + (a + 1)
    for n in range(2, N):                    # Julia n = 2:N-1 sets index n+1
        c1 = 2 * n * (n + a + b) * (2 * n + a + b - 2)
        c2 = 2 * n + a + b - 1
        A[n] = (2 * n + a + b) * (2 * n + a + b - 2) * c2 / c1
        B[n] = (a * a - b * b) * c2 / c1
        C[n] = -2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b) / c1
    g = _norms(A, B, C, a, b)
    A[0] /= g[0]
    if N > 1:
        A[1] /= g[1]; B[1] /= g[1]
    for n in range(2, N):
        A[n] *= g[n - 1] / g[n]; B[n] *= g[n - 1] / g[n]; C[n] *= g[n - 2] / g[n]
    return A, B, C


def _norms(A, B, C, a, b):
    """sqrt(int Q_k^2 (1-x)^a (1+x)^b dx) of the unnormalised recurrence, exactly:
    for integer a, b the integrand is a polynomial of degree 2(N-1) + a + b, which
    Gauss-Legendre integrates exactly with enough nodes."""
    if a != int(a) or b != int(b) or a < 0 or b < 0:
        raise NotImplementedError("exact Jacobi normalisation implemented for integer alpha, beta >= 0")
    npts = len(A) + int(a + b) // 2 + 2
    x, w = np.polynomial.legendre.leggauss(npts)
    Q = poly_eval(x, A, B, C)
    return np.sqrt(((Q * Q) * (w * (1 - x) ** a * (1 + x) ** b)[:, None]).sum(0))


def cubic_bspline_coefs(y):
    """Interpolations.jl `cubic_spline_interpolation` coefficients on a uniform
    grid: BSpline(Cubic(Line(OnGrid()))).  y (n, m) nodal values -> (n+2, m)
    coefficients (one pad each side): interior rows (c[i-1] + 4 c[i] + c[i+1])/6
    = y[i], boundary rows c0 - 2 c1 + c2 = 0 (y'' = 0 at the end knots)."""
    y = np.asarray(y, dtype=np.float64)
    squeeze = y.ndim == 1
    y = y[:, None] if squeeze else y
    n = y.shape[0]
    M = np.zeros((n + 2, n + 2))
    M[0, :3] = (1.0, -2.0, 1.0)
    M[-1, -3:] = (1.0, -2.0, 1.0)
    i = np.arange(1, n + 1)
    M[i, i - 1] = 1.0 / 6.0; M[i, i] = 2.0 / 3.0; M[i, i + 1] = 1.0 / 6.0
    rhs = np.zeros((n + 2, y.shape[1])); rhs[1:-1] = y
    c = np.linalg.solve(M, rhs)
    return c[:, 0] if squeeze else c


def spline_nodes(nnodes=SPLINE_NODES):
    return np.linspace(-1.0, 1.0, nnodes)


def jacobi_spline_table(n1, alpha=4.0, beta=4.0, nnodes=SPLINE_NODES):
    """(nnodes+2, n1): spline coefficients of the normalised Jacobi Q_0..Q_{n1-1}
    -- the factorised ace1 tensor radial (export's rnl_spline_coefs_single)."""
    A, B, C = jacobi_3term(n1, alpha, beta)
    return cubic_bspline_coefs(poly_eval(spline_nodes(nnodes), A, B, C))


def pair_spline_coefs(NZ, maxq, nnodes=SPLINE_NODES):
    """(NZ, NZ, nnodes+2, maxq*NZ): the splined one-hot Legendre pair basis,
    R_n(r, Z1, Z2) = Q_{n'}(x) delta_{z', Z2} with z' = mod1(n, NZ),
    n' = div(n-1, NZ) + 1 (`set_onehot_weights!`)."""
    spl = jacobi_spline_table(maxq, 0.0, 0.0, nnodes)                 # (ncoef, maxq)
    out = np.zeros((NZ, NZ, spl.shape[0], maxq * NZ))
    for n in range(maxq * NZ):
        zp, npr = n % NZ, n // NZ
        out[:, zp, :, n] = spl[:, npr]
    return out


def uniform_cutoffs(elements, rcut=None):
    """(rin, r0, rcut) shared by every species pair, as `ace_embedding_model`
    (uniform_cutoffs=true): r0 = mean tabulated bond length over the elements
    that have one, rcut = 2.5 r0 unless given, rin = 0."""
    known = [z for z in elements if z in BOND_LEN]
    if not known:
        raise ValueError(f"no element in {list(elements)} has a tabulated bond length; pass rcut")
    r0 = sum(BOND_LEN[z] for z in known) / len(known)
    return (0.0, float(r0), float(2.5 * r0 if rcut is None else rcut))


def transform_table(NZ, cut, p, q):
    """(NZ, NZ, 7) Agnesi 7-tuples, all pairs sharing the uniform cutoffs."""
    rin, r0, rcut = cut
    t = agnesi_transform_params(rin, r0, rcut, p, q)
    return np.tile(np.asarray(t), (NZ, NZ, 1))


def tensor_envelope_table(NZ):
    """(NZ, NZ, 5) PolyEnvelope2sX(-1, 1, 2, 2) params."""
    return np.tile(np.asarray(envelope2sx_params(-1.0, 1.0, 2, 2)), (NZ, NZ, 1))


def pair_envelope_table(NZ, cut, p=2):
    """(NZ, NZ, 3) ACE1_PolyEnvelope1sR(rcut, r0, p) params."""
    _, r0, rcut = cut
    return np.tile(np.asarray([rcut, r0, float(p)]), (NZ, NZ, 1))
