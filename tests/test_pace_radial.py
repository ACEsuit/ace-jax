import pathlib
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import pace_radial as pr

REF = pathlib.Path(__file__).parent.parent / "fixtures" / "pace" / "unit_ref.npz"


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.skip("unit_ref.npz not generated (Task 1)")
    return np.load(REF)


@pytest.mark.parametrize("name", ["ChebExpCos", "ChebPow", "ChebLinear", "SBessel"])
@pytest.mark.parametrize("inner", ["density", "distance", "zbl"])
def test_radbase(ref, name, inner):
    g = pr.radbase(jnp.asarray(ref["r"]), name, inner, 2.5, 5.0, 0.3, 1.2, 0.4, 8)
    np.testing.assert_allclose(g, ref[f"g_{name}_{inner}"], rtol=1e-12, atol=1e-13)


def test_zbl_cut_in_quirk(ref):
    g = pr.radbase(jnp.asarray(ref["r"]), "ChebExpCos", "zbl", 2.5, 5.0, 0.3, 1.2, 0.0, 8)
    np.testing.assert_allclose(g, ref["g_ChebExpCos_zbl_dcutin0"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("inner", ["density", "distance"])
def test_radcore(ref, inner):
    c = pr.radcore(jnp.asarray(ref["r"]), 3.0, 0.7, 5.0, 1.2, 0.4, inner)
    np.testing.assert_allclose(c, ref[f"cr_{inner}"], rtol=1e-12, atol=1e-13)


def test_zbl(ref):
    c = pr.pace_zbl(jnp.asarray(ref["r"]), 14, 32, 5.0, 0.3, 1.0)
    np.testing.assert_allclose(c, ref["cr_zbl"], rtol=1e-11, atol=1e-12)


def test_cutoff_func_poly(ref):
    np.testing.assert_allclose(pr.cutoff_func_poly(jnp.asarray(ref["r"]), 3.0, 1.0),
                               ref["fcpoly"], rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("m", [0.5, 1.0, 2.0])
def test_embedding(ref, m):
    x = jnp.asarray(ref["fx"])
    tag = f"{m:g}"
    np.testing.assert_allclose(pr.fexp(x, m), ref[f"fexp_m{tag}"], rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(pr.fexp_shifted_scaled(x, m), ref[f"fexpss_m{tag}"],
                               rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("fn", [pr.fexp, pr.fexp_shifted_scaled])
@pytest.mark.parametrize("m", [0.5, 1.0, 2.0])
def test_embedding_grad_finite_at_zero(fn, m):
    g = jax.grad(lambda x: fn(x, m))(0.0)
    assert np.isfinite(g)


@pytest.mark.parametrize("name", ["ChebExpCos", "ChebPow", "ChebLinear", "SBessel"])
def test_radbase_grad_finite_at_and_beyond_cutoff(name):
    for r in (5.0, 5.5):
        J = jax.jacfwd(lambda rr: pr.radbase(rr, name, "density", 2.0, 5.0, 0.3, 0.0, 1e-5, 6))(r)
        assert np.all(np.isfinite(J)) and np.all(J == 0)
