"""ace1-compatible radial pieces for Python embedded-model authoring: normalised
Jacobi / Legendre recurrences (Polynomials4ML), Interpolations' cubic B-spline
coefficients (Line(OnGrid)) on the 100-node splinify grid, and the uniform ace1
transforms / envelopes -- against Julia-exported embedded models
(julia/export_model.jl ... embedding, fixtures/emb_ref_*.npz)."""
import pathlib

import numpy as np
import pytest

from ace_jax.construct import radial_ace1 as ra
from ace_jax.construct.radial_init import legendre_3term, poly_eval

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
CASES = [("emb_ref_SiGe_o2d6", [14, 32]), ("emb_ref_CrMnFe_o3d5_dmax4", [24, 25, 26])]


def _ref(name):
    p = FIX / f"{name}.npz"
    if not p.exists():
        pytest.skip(f"missing {p.name}; see julia/export_model.jl (embedding kind)")
    return np.load(p)


def test_jacobi_00_is_the_normalised_legendre_recurrence():
    A, B, C = ra.jacobi_3term(12, 0.0, 0.0)
    Al, Bl, Cl = legendre_3term(12)
    assert np.allclose(A, Al, rtol=1e-13) and np.allclose(B, Bl, atol=1e-14) and np.allclose(C, Cl, rtol=1e-13)


def test_jacobi_44_is_orthonormal_under_its_weight():
    A, B, C = ra.jacobi_3term(20, 4.0, 4.0)
    x, w = np.polynomial.legendre.leggauss(60)
    Q = poly_eval(x, A, B, C)
    G = (Q * (w * (1 - x) ** 4 * (1 + x) ** 4)[:, None]).T @ Q
    assert np.allclose(G, np.eye(20), atol=1e-12)


def test_cubic_bspline_coefs_interpolate_and_are_linear_at_the_ends():
    x = np.linspace(-1, 1, 100)
    y = np.stack([np.sin(3 * x), x ** 3], axis=1)
    c = ra.cubic_bspline_coefs(y)                                    # (102, 2)
    assert c.shape == (102, 2)
    inner = (c[:-2] + 4 * c[1:-1] + c[2:]) / 6
    assert np.allclose(inner, y, atol=1e-13)                          # interpolation at the knots
    assert np.allclose(c[0] - 2 * c[1] + c[2], 0, atol=1e-13)         # Line(OnGrid): y'' = 0 at x = -1
    assert np.allclose(c[-1] - 2 * c[-2] + c[-3], 0, atol=1e-13)


@pytest.mark.parametrize("name,els", CASES)
def test_tensor_radial_spline_table_matches_julia(name, els):
    """The factorised table has one column per (n', l) block of the radial spec
    (the ace1 radial is l-independent, so Q_{n'-1} repeats across l); every column
    is the spline of its normalised Jacobi(4,4) Q_{n'-1}, and within each l block
    n' counts up from 1.  (The block order itself is the spec's -- stage 3.)"""
    z = _ref(name)
    ref = z["rnl_spline_coefs_single"]                                # (102, n_blocks)
    T = ra.jacobi_spline_table(ref.shape[1], 4.0, 4.0)
    npr = []
    for k in range(ref.shape[1]):
        err = np.abs(T - ref[:, [k]]).max(0) / np.abs(ref[:, k]).max()
        m = int(np.argmin(err))
        assert err[m] < 1e-12, (k, err[m])
        npr.append(m)
    starts = [k for k in range(len(npr)) if npr[k] == 0]
    for a_, b_ in zip(starts, starts[1:] + [len(npr)]):
        assert npr[a_:b_] == list(range(b_ - a_))


@pytest.mark.parametrize("name,els", CASES)
def test_pair_spline_table_matches_julia(name, els):
    z = _ref(name)
    ref = z["pair_spline_coefs"]                                      # (NZ, NZ, 102, maxq*NZ)
    NZ = len(els)
    got = ra.pair_spline_coefs(NZ, ref.shape[-1] // NZ)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-10 * np.abs(ref).max())


@pytest.mark.parametrize("name,els", CASES)
def test_uniform_ace1_transforms_and_envelopes_match_julia(name, els):
    z = _ref(name)
    cut = ra.uniform_cutoffs(els)                                      # (rin, r0, rcut)
    NZ = len(els)
    tt = ra.transform_table(NZ, cut, 2, 4)
    tp = ra.transform_table(NZ, cut, 1, 3)
    assert np.allclose(tt, z["rnl_transform"], rtol=1e-12)
    assert np.allclose(tp, z["pair_transform"], rtol=1e-12)
    assert np.allclose(ra.tensor_envelope_table(NZ), z["rnl_envelope"], rtol=1e-14)
    assert np.allclose(ra.pair_envelope_table(NZ, cut), z["pair_envelope"], rtol=1e-12)
    assert np.allclose(z["rcuts"], cut[2]) and np.allclose(z["pair_rcuts"], cut[2])
