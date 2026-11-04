"""The host QR merge (stats._host_merge): folding rows into a triangular factor by LAPACK tpqrt, which keeps
R triangular (2 m L^2 flops for m rows) where geqrf of the stacked matrix treats R as dense (~3.3 L^3 per L
rows).  The merged factor is a QR factor of the stacked system: same Gram R^T R and R^T c as numpy's QR."""
import numpy as np
import pytest

from ace_jax.fit.stats import _host_merge


def _tri(rng, n, scale=1.0):
    R = np.linalg.qr(rng.normal(size=(2 * n, n)) * scale, mode="r")
    return R, rng.normal(size=n)


def _reference(R, c, P, y):
    A = np.block([[R, c[:, None]], [P, y[:, None]]])
    Ra = np.linalg.qr(A, mode="r")
    n = R.shape[1]
    return Ra[:n, :n], Ra[:n, n]


def _check(got, ref):
    (R, c), (Rr, cr) = got, ref
    assert np.array_equal(R, np.triu(R))
    G, Gr = R.T @ R, Rr.T @ Rr
    np.testing.assert_allclose(G, Gr, rtol=0, atol=1e-13 * np.abs(Gr).max())
    np.testing.assert_allclose(R.T @ c, Rr.T @ cr, rtol=0, atol=1e-13 * np.abs(Rr.T @ cr).max())


@pytest.mark.parametrize("n,m", [(1, 1), (5, 3), (40, 40), (40, 97), (300, 150)])
def test_host_merge_dense_rows(n, m):
    rng = np.random.default_rng(n * 1000 + m)
    R, c = _tri(rng, n)
    P, y = rng.normal(size=(m, n)), rng.normal(size=m)
    _check(_host_merge(R, c, P, y), _reference(R, c, P, y))


@pytest.mark.parametrize("n", [1, 7, 200])
def test_host_merge_two_triangles(n):
    """upper=True: the rows are themselves an upper-triangular factor (merging MPI ranks' factors)."""
    rng = np.random.default_rng(n)
    (R, c), (R2, c2) = _tri(rng, n), _tri(rng, n, scale=1e-6)        # very different scales
    _check(_host_merge(R, c, R2, c2, upper=True), _reference(R, c, R2, c2))
    _check(_host_merge(R, c, R2, c2), _reference(R, c, R2, c2))       # the dense form agrees too


def test_host_merge_zero_start_and_inputs_untouched():
    rng = np.random.default_rng(3)
    n, m = 30, 50
    R0, c0 = np.zeros((n, n)), np.zeros(n)
    P, y = rng.normal(size=(m, n)), rng.normal(size=m)
    keep = P.copy(), y.copy()
    _check(_host_merge(R0, c0, P, y), _reference(R0, c0, P, y))
    assert np.array_equal(P, keep[0]) and np.array_equal(y, keep[1]) and not R0.any() and not c0.any()
