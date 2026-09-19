"""The two A-basis forms must be interchangeable.

`edge_a_kind` picks how `A = Rnl[aspec_r] * Ylm[aspec_y]` is formed: a gather, or
an algebraically identical one-hot matmul.  They exist because their *adjoints*
differ -- the gather's is an axis-1 scatter whose cost per slot grows with buffer
length, the matmul's is a matmul and is flat -- and neither wins on every
backend.  Whatever the cost, the numbers must not move, so these tests demand
bit-identity rather than a tolerance.
"""
import jax

jax.config.update("jax_enable_x64", True)   # test-local, never library-level
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import species_index

from ace_jax.eval import calibrate_edge_a, load, with_edge_a_kind


def _case(npz, dtype, kind):
    model, meta, z = load(npz, dtype=dtype, edge_a_kind=kind)
    n = int(z["test_pos"].shape[1])
    send = jnp.asarray(z["test_edge_i"], jnp.int32)
    recv = jnp.asarray(z["test_edge_j"], jnp.int32)
    rij = jnp.asarray(np.asarray(z["test_edge_rij"]).T, dtype)
    nz = species_index(z)
    return model, (rij, nz[send], nz[recv], send, n), z


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32], ids=["f64", "f32"])
def test_forms_agree_bitwise(npz, dtype):
    """Values and gradients agree; how tightly depends on the dtype.

    Values are bit-identical in both dtypes.  GRADIENTS are bit-identical in f64
    but only close in f32: the two forms' adjoints are a scatter and a matmul,
    and XLA is free to accumulate them in different orders.  Measured 0.0 on
    Apple Silicon and ~4e-6 on x86 -- so an exact assertion here passes on one
    architecture and fails on the other, which is how this was found.
    """
    mg, args, _ = _case(npz, dtype, "gather")
    mm, _, _ = _case(npz, dtype, "matmul")
    rij, zi, zj, send, n = args
    nzero = jnp.zeros(n, jnp.int32)
    ev = lambda m: m.site_energies(rij, zi, zj, send, n, nzero)
    dv = np.max(np.abs(np.asarray(ev(mg) - ev(mm))))
    gr = lambda m: jax.grad(lambda r: jnp.sum(m.site_energies(r, zi, zj, send, n, nzero)))(rij)
    dg = np.max(np.abs(np.asarray(gr(mg) - gr(mm))))
    print(f"\n  {dtype.__name__}: values {dv:.1e}  grads {dg:.1e}")
    assert dv == 0.0, f"values differ by {dv}"
    if dtype is jnp.float64:
        assert dg == 0.0, f"f64 gradients differ by {dg}"
    else:
        scale = float(np.max(np.abs(np.asarray(gr(mg)))))
        assert dg <= 1e-5 * scale, f"f32 gradients differ by {dg} (scale {scale})"


def test_switching_preserves_results(npz):
    """`with_edge_a_kind` round-trips without touching the numbers."""
    mg, (rij, zi, zj, send, n), _ = _case(npz, jnp.float64, "gather")
    nzero = jnp.zeros(n, jnp.int32)
    ref = np.asarray(mg.site_energies(rij, zi, zj, send, n, nzero))
    for kind in ("matmul", "gather"):
        m = with_edge_a_kind(mg, kind)
        assert m.edge_a_kind == kind
        got = np.asarray(m.site_energies(rij, zi, zj, send, n, nzero))
        assert np.array_equal(got, ref)
    assert with_edge_a_kind(mg, "gather") is mg          # no-op returns the same object


def test_calibration_picks_one_and_is_correct(npz):
    """Calibration must return a working model, whichever form it chooses."""
    mg, (rij, zi, zj, send, n), _ = _case(npz, jnp.float64, "gather")
    nzero = jnp.zeros(n, jnp.int32)
    best, timings = calibrate_edge_a(mg, rij, zi, zj, send, n, nzero, reps=2)
    print(f"\n  timings/ms {timings}  -> {best.edge_a_kind}")
    assert set(timings) == {"gather", "matmul"}
    assert best.edge_a_kind == min(timings, key=timings.get)
    ref = np.asarray(mg.site_energies(rij, zi, zj, send, n, nzero))
    assert np.array_equal(np.asarray(best.site_energies(rij, zi, zj, send, n, nzero)), ref)


def test_bad_kind_rejected(npz):
    with pytest.raises(ValueError, match="edge_a_kind"):
        load(npz, edge_a_kind="scatter")
