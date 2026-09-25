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

from conftest import FIXTURE_DIR, MODELS, pace_fixture, species_index

from ace_jax.eval import calibrate_edge_a, load, sparse_graph, with_edge_a_kind

# Both model families share the A-basis forms (eval/edge_model.py), so every
# test here runs on the ACE npz models and on PACE .yace fixtures alike.
PACE = ("gesi_sbessel", "sige_zbl")


@pytest.fixture(params=[f"ace:{k}" for k in MODELS] + [f"pace:{k}" for k in PACE])
def model_path(request):
    fam, name = request.param.split(":")
    if fam == "ace":
        p = FIXTURE_DIR / MODELS[name]
        if not p.exists():
            pytest.skip(f"{p.name} not generated")
        return p
    return pace_fixture(FIXTURE_DIR / "pace" / f"{name}.yace")


def _case(model_path, dtype, kind):
    """(model, (rij, zi, zj, senders, n_nodes, node_z)) for either family."""
    model, meta, z = load(str(model_path), dtype=dtype, edge_a_kind=kind)
    if str(model_path).endswith(".yace"):
        from ase import Atoms
        ref = np.load(str(model_path).replace(".yace", "_ref.npz"))
        at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=True)
        g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
        z2i = {zz: i for i, zz in enumerate(meta["elements"])}
        nz = jnp.asarray([z2i[int(x)] for x in at.numbers], jnp.int32)
        send, recv = jnp.asarray(g.senders, jnp.int32), jnp.asarray(g.receivers, jnp.int32)
        return model, (jnp.asarray(g.rij, dtype), nz[send], nz[recv], send, len(at), nz)
    n = int(z["test_pos"].shape[1])
    send = jnp.asarray(z["test_edge_i"], jnp.int32)
    recv = jnp.asarray(z["test_edge_j"], jnp.int32)
    rij = jnp.asarray(np.asarray(z["test_edge_rij"]).T, dtype)
    nz = species_index(z)
    return model, (rij, nz[send], nz[recv], send, n, jnp.zeros(n, jnp.int32))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32], ids=["f64", "f32"])
def test_forms_agree_bitwise(model_path, dtype):
    """Values and gradients agree; how tightly depends on the dtype.

    Values are bit-identical in both dtypes.  GRADIENTS are bit-identical in f64
    but only close in f32: the two forms' adjoints are a scatter and a matmul,
    and XLA is free to accumulate them in different orders.  Measured 0.0 on
    Apple Silicon and ~4e-6 on x86 -- so an exact assertion here passes on one
    architecture and fails on the other, which is how this was found.
    """
    mg, args = _case(model_path, dtype, "gather")
    mm, _ = _case(model_path, dtype, "matmul")
    assert (mg.edge_a_kind, mm.edge_a_kind) == ("gather", "matmul")   # else this compares a form with itself
    rij, zi, zj, send, n, nzero = args
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


def test_switching_preserves_results(model_path):
    """`with_edge_a_kind` round-trips without touching the numbers."""
    mg, (rij, zi, zj, send, n, nzero) = _case(model_path, jnp.float64, "gather")
    ref = np.asarray(mg.site_energies(rij, zi, zj, send, n, nzero))
    for kind in ("matmul", "gather"):
        m = with_edge_a_kind(mg, kind)
        assert m.edge_a_kind == kind
        got = np.asarray(m.site_energies(rij, zi, zj, send, n, nzero))
        assert np.array_equal(got, ref)
    assert with_edge_a_kind(mg, "gather") is mg          # no-op returns the same object


def test_calibration_picks_one_and_is_correct(model_path):
    """Calibration must return a working model, whichever form it chooses."""
    mg, (rij, zi, zj, send, n, nzero) = _case(model_path, jnp.float64, "gather")
    best, timings = calibrate_edge_a(mg, rij, zi, zj, send, n, nzero, reps=2)
    print(f"\n  timings/ms {timings}  -> {best.edge_a_kind}")
    assert set(timings) == {"gather", "matmul"}
    assert best.edge_a_kind == min(timings, key=timings.get)
    ref = np.asarray(mg.site_energies(rij, zi, zj, send, n, nzero))
    assert np.array_equal(np.asarray(best.site_energies(rij, zi, zj, send, n, nzero)), ref)


def test_bad_kind_rejected(model_path):
    with pytest.raises(ValueError, match="edge_a_kind"):
        load(str(model_path), edge_a_kind="scatter")



def _dot_generals(jaxpr):
    """All dot_general equations, recursing into sub-jaxprs (pjit, custom_vjp...)."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            yield eqn
        for v in eqn.params.values():
            for sub in (v if isinstance(v, (list, tuple)) else [v]):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _dot_generals(inner)


def test_matmul_form_selects_at_full_precision(model_path):
    """The one-hot matmuls only *select* values, so they must not round them:
    at default precision an f32 matmul on Ampere+ GPUs is TF32 (10-bit mantissa),
    which silently truncates every A entry (measured 3e-3 on an A4500).  CPUs
    have no TF32, so this checks the traced graph, not the numbers."""
    mm, (rij, zi, zj, send, n, nz) = _case(model_path, jnp.float32, "matmul")
    closed = jax.make_jaxpr(lambda r: mm.site_energies(r, zi, zj, send, n, nz))(rij)
    n_a = mm.a_sel_r.shape[1]
    sel = [e for e in _dot_generals(closed.jaxpr)
           if e.outvars[0].aval.shape == (rij.shape[0], n_a)]
    assert len(sel) == 2, f"expected the two selection matmuls, found {len(sel)}"
    for e in sel:
        prec = e.params["precision"]
        assert prec is not None and all(p == jax.lax.Precision.HIGHEST for p in prec), prec
