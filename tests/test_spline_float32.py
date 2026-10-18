"""float32 spline tables (`radial.spline_local`, applied by `with_dtype`).

A uniform cubic B-spline's derivative is sum_k w'_k(t) c_k / h: a difference of
neighbouring coefficients over the interval.  Rounded to float32 each c_k carries
~6e-8 |c_k| independently per column, so S' carries ~6e-8 |c| / h of noise that
grows with the interval count, and a readout whose coefficients cancel amplifies
it (o4d12 at 2048 intervals: 74 meV/A against 0.73 analytic; docs/dev/fp32-results.md).
The local form keeps c_ix and the differences c_j - c_ix, formed in float64, so S'
reads only the differences, at their own relative precision.
"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load, with_dtype
from ace_jax.eval.model import lean
from ace_jax.eval.radial import spline_eval, spline_local
from test_lean import _efv, _structure
from test_to_spline import CANTOR, _perturbed


def _table(n=4096, offset=100.0, F=3):
    """B-spline coefficients (n+2, F) of offset + smooth columns on x in [-1, 1]."""
    x = np.linspace(-1.0 - 2.0 / n, 1.0 + 2.0 / n, n + 2)
    return offset + np.stack([np.sin(3 * x + k) for k in range(F)], -1), (-1.0, 2.0 / n, n)


def _deriv(c, grid, xs):
    x0, h, n = grid
    return np.asarray(jax.vmap(jax.jacfwd(lambda x: spline_eval(x, c, x0, h, n)))(xs), np.float64)


def test_local_form_is_exact_in_float64():
    c, (x0, h, n) = _table()
    xs = jnp.linspace(-0.999, 0.999, 3001)
    ref = spline_eval(xs, jnp.asarray(c), x0, h, n)
    got = spline_eval(xs, spline_local(jnp.asarray(c)), x0, h, n)
    assert np.abs(np.asarray(got) - np.asarray(ref)).max() <= 1e-12 * np.abs(c).max()
    dref, dgot = _deriv(jnp.asarray(c), (x0, h, n), xs), _deriv(spline_local(jnp.asarray(c)), (x0, h, n), xs)
    assert np.abs(dgot - dref).max() <= 1e-9 * np.abs(dref).max()


def test_float32_derivative_does_not_grow_with_the_grid():
    """Plain float32 coefficients: S' error ~ |c| eps / h; local form: ~ |S'| eps."""
    c, (x0, h, n) = _table()
    xs = jnp.linspace(-0.999, 0.999, 3001)
    ref = _deriv(jnp.asarray(c), (x0, h, n), xs)
    xs32 = xs.astype(jnp.float32)
    plain = _deriv(jnp.asarray(c, jnp.float32), (x0, h, n), xs32)
    local = _deriv(spline_local(jnp.asarray(c)).astype(jnp.float32), (x0, h, n), xs32)
    e_plain, e_local = np.abs(plain - ref).max(), np.abs(local - ref).max()
    print(f"\n  S' error, max|S'| {np.abs(ref).max():.2f}: plain {e_plain:.2e}  local {e_local:.2e}")
    assert e_local <= 1e-5 * np.abs(ref).max()
    assert e_local <= e_plain / 30


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_with_dtype_float32_splined_as_accurate_as_analytic(layout):
    """A fine-grid splined model cast by with_dtype is as accurate in float32 as
    the analytic model it approximates."""
    m, meta, _ = load(str(CANTOR))
    a = _perturbed(m)
    ms = lean(a, spline_tol=1e-10, spline_intervals=4096)
    ma = lean(a, spline_tol=None)
    s32 = with_dtype(ms, jnp.float32)
    assert s32.rnl_coefs.ndim == ms.rnl_coefs.ndim + 1          # local form
    assert with_dtype(ms, jnp.float64).rnl_coefs.shape == ms.rnl_coefs.shape
    at = _structure(meta)
    err = {}
    for name, m64 in (("spline", ms), ("analytic", ma)):
        F64 = _efv(m64, meta, at, layout)[1]
        F32 = _efv(with_dtype(m64, jnp.float32), meta, at, layout)[1]
        err[name] = np.abs(F32 - F64).max() / np.abs(F64).max()
    print(f"\n  float32 |dF|/max|F|: {err}")
    assert err["spline"] <= 3 * err["analytic"] + 1e-6
