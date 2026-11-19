"""The coupled form of the product basis (`eval/coupled.py`) against the monomials it replaces, and the C++
kernel's coupled passes against its DAG: one, two and five species, spherical and solid harmonics, a species
embedding, correlation orders 3 and 4."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import coupled, ffi, load
from ace_jax.eval.model import fold_readout, lean
from conftest import FIXTURE_DIR as FIXTURES
from ace_jax.eval.api import with_dtype
from test_ffi import backend_kernel, cluster, device, kernel, matrix_inputs  # noqa: F401  (the compiled kernels)

NAMES = ["si_fitted", "si_ace_model", "sige_nofit", "ace_cantor5_small", "si_m710"]


def _lean(name):
    m, meta, _ = load(str(FIXTURES / f"{name}.npz"))
    return lean(m, spline_tol=None, radial_table=True), meta


@pytest.mark.parametrize("name", NAMES)
def test_plan_equals_the_monomials(name):
    """E to roundoff of the terms' sum at random A, per centre species, and dE/dA against autodiff."""
    m, _ = _lean(name)
    specs, ct = [np.asarray(s) for s in m.aa_specs], np.asarray(m.ctilde)
    P = ffi.coupled_of(m)
    w = P.weights(ct)
    assert coupled.check(P, specs, ct, w) < 1e-12
    st = P.stats()
    assert st["weights"] + st["linear"] <= sum(len(s) for s in specs)   # no more weights than monomials
    A = np.random.default_rng(1).normal(size=(2, P.n_a))

    def mono(a, c):
        e, off = 0.0, 0
        for s in specs:
            if s.size:
                e = e + jnp.prod(a[s], axis=1) @ c[off:off + len(s)]
            off += len(s)
        return e
    for zc in range(ct.shape[1]):
        E, dA = coupled.evaluate(P, w[:, zc], A)
        ref = np.stack([jax.grad(mono)(jnp.asarray(a), jnp.asarray(ct[:, zc])) for a in A])
        np.testing.assert_allclose(dA, ref, atol=1e-11 * np.abs(ref).max())


def test_weights_are_linear_in_the_readout_and_the_pairing_is_the_specs():
    """`Plan.weights` is one sparse product, so a new readout needs no new plan; the integer program only
    lowers the cost of the X blocks."""
    m, _ = _lean("si_fitted")
    P = ffi.coupled_of(m)
    ct = np.asarray(m.ctilde)
    np.testing.assert_allclose(P.weights(3.0 * ct), 3.0 * P.weights(ct), rtol=1e-13)
    assert ffi.coupled_of(m) is P                          # kept per spec
    specs = [np.asarray(s) for s in m.aa_specs]
    P2 = coupled.build(m.aspec_r, m.aspec_y, specs, ilp_seconds=2.0)
    assert coupled.check(P2, specs, ct) < 1e-12
    assert P2.n_x <= P.n_x + 2


def test_no_plan_above_order_four_or_for_a_broken_block():
    m, _ = _lean("si_fitted")
    specs = [np.asarray(s) for s in m.aa_specs]
    with pytest.raises(coupled.NoPlan, match="order"):
        coupled.build(m.aspec_r, m.aspec_y, specs + [np.zeros((1, 5), np.int64)])
    ay = np.asarray(m.aspec_y)
    keep = np.flatnonzero(ay != ay.max())                  # an A block less one of its m
    with pytest.raises(coupled.NoPlan, match="whole"):
        coupled.build(np.asarray(m.aspec_r)[keep], ay[keep], [np.zeros((1, 1), np.int64)])


@pytest.mark.parametrize("precision", [None, "mixed", "float32"])
@pytest.mark.parametrize("name", NAMES)
def test_kernel_coupled_form_matches_its_dag(device, monkeypatch, name, precision):  # noqa: F811
    """A kernel in coupled form against the same kernel on the DAG, the C++ one and the CUDA one: float64 to
    roundoff, mixed precision and (the GPU's) float32 as close as a float32 product basis allows."""
    gpu = device == "gpu"
    if precision == "float32" and not gpu:
        pytest.skip("the C++ kernel is float64 or mixed")
    m, meta = _lean(name)
    if precision == "float32":
        m, precision = with_dtype(m, "float32"), None
    rcut = float(meta["rcut"])
    at = cluster(meta)
    els = [int(z) for z in meta["elements"]]
    out = []
    with jax.default_device(jax.devices(device)[0]):
        pos, z, graph = matrix_inputs(at, rcut + 1.0, [els.index(int(n)) for n in at.numbers])
        for form in ("dag", "coupled"):
            monkeypatch.setattr(ffi, "FORM", form)
            attrs, _ = ffi.kernel_inputs(m, gpu, precision)
            assert attrs["form"] == (form == "coupled")
            efn, ffn = ffi.lammps_fns(m, len(els), list(range(len(els))), rcut, gpu, precision)
            out.append([np.asarray(a) for a in (efn(pos.astype(m.E0.dtype), z, graph), ffn(pos.astype(m.E0.dtype), z, graph))])
    (e0, f0), (e1, f1) = out
    te, tf = (1e-12, 1e-10) if precision is None and m.E0.dtype == jnp.float64 else (4e-5, 4e-4)
    print(f"\n  max|de| {np.abs(e1 - e0).max():.1e}  max|dF| {np.abs(f1 - f0).max():.1e}")
    assert np.abs(e1 - e0).max() <= te * max(np.abs(e0).max(), 1.0)
    assert np.abs(f1 - f0).max() <= tf * max(np.abs(f0).max(), 1.0)


def test_auto_takes_the_form_with_fewer_operations(monkeypatch):
    m, _ = _lean("si_fitted")
    n_nodes = len(ffi.plan(m)[2])
    ops = ffi.coupled_plan(m)[2]
    monkeypatch.setattr(ffi, "FORM", None)
    assert ffi.choose_form(m, "auto", n_nodes)[0] == ("coupled" if ops <= ffi.COUPLED_GAIN * 3 * n_nodes else "dag")
    monkeypatch.setattr(ffi, "COUPLED_GAIN", 1e9)
    assert ffi.choose_form(m, None, n_nodes)[0] == "coupled"
    monkeypatch.setattr(ffi, "COUPLED_GAIN", 0.0)
    assert ffi.choose_form(m, None, n_nodes)[0] == "dag"
    with pytest.raises(ValueError, match="ffi_form"):
        ffi.choose_form(m, "tree", n_nodes)


def emulate_gpu(ints, reals, A):
    """The GPU kernel's coupled pass (`ace_coupled`) on one site's A, from `ffi.coupled_gpu_plan`'s arrays for
    its centre species: (E, dE/dA).  Thread by thread, as the kernel: any sum two threads would share shows as
    a difference from the reference only if the plan is wrong, so the targets written are also counted."""
    ints = np.asarray(ints).view(np.uint32).astype(np.int64)
    T, n_a, n_x, pieces, ns = ints[:5]
    n_tot = n_a + n_x
    st = ints[5:5 + 5 * ns].reshape(ns, 5)
    fx = ints[5 + 5 * ns:5 + 5 * ns + 4].reshape(2, 2)
    Z = np.zeros(n_tot); Z[:n_a] = A
    G = np.zeros(n_tot + pieces)
    writes = np.zeros(n_tot + pieces, int)
    lin = reals[:n_a]
    E = float(lin @ A)

    def run(s, kind, nc=1):
        nonlocal E
        steps, ntask, io, ho, ro = st[s]
        items = ints[io:io + steps * T].reshape(steps, T)
        hdr = ints[ho:ho + ntask * T].reshape(ntask, T)
        co = reals[ro:ro + steps * T].reshape(steps, T)
        for t in range(T):
            acc, k = np.zeros(nc), 0
            for j in range(steps):
                it, c = items[j, t], co[j, t]
                lo, hi = it & 0x7fff, (it >> 15) & 0xffff
                if kind == 1:
                    acc[0] += c * Z[lo] * Z[hi]
                elif kind == 2:
                    acc += c * Z[lo:lo + nc]
                else:
                    acc[0] += c * G[lo] * Z[hi]
                if it >> 31:
                    dst, u = hdr[k, t] & 0xffff, hdr[k, t] >> 16
                    k += 1
                    if kind == 1:
                        Z[dst] = acc[0]
                    else:
                        G[dst:dst + nc] += acc
                        writes[dst:dst + nc] += 1
                    if kind == 2:
                        E += 0.5 * float(acc @ Z[u:u + nc])
                    acc[:] = 0

    def fix(f):
        n, o = fx[f]
        for w0, w1 in ints[o:o + 2 * n].reshape(n, 2):
            dst, cnt, slot, stride = w0 & 0xffff, w0 >> 16, w1 & 0xffff, w1 >> 16
            G[dst] += G[slot:slot + cnt * stride:stride].sum()
            G[slot:slot + cnt * stride:stride] = 0
    run(0, 1)
    for nc in range(1, ffi.GPU_NC + 1):
        run(nc, 2, nc)
    assert writes[:n_a].max(initial=0) <= 1 and writes[n_a:].max(initial=0) <= 1   # one owner per target
    fix(0)
    writes[:] = 0
    run(ns - 1, 3)
    fix(1)
    return E, G[:n_a] + lin


@pytest.mark.parametrize("T", [32, 256])
@pytest.mark.parametrize("name", NAMES)
def test_gpu_plan_emulated_equals_the_reference(name, T):
    """`ffi.coupled_gpu_plan`'s streams, run as the GPU kernel runs them, against `coupled.evaluate`."""
    m, _ = _lean(name)
    P = ffi.coupled_of(m)
    ct = np.asarray(m.ctilde)
    w = P.weights(ct)
    ints, reals, threads = ffi.coupled_gpu_plan(m, T)
    assert threads == T and ints.dtype == np.int32
    order, new = ffi._a_order(m)
    A = np.random.default_rng(2).normal(size=P.n_a)
    for zc in range(ct.shape[1]):
        E0, d0 = coupled.evaluate(P, w[:, zc], A[None])
        E1, d1 = emulate_gpu(ints, reals[zc], A[order])
        assert E1 == pytest.approx(E0[0], rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(d1[new], d0[0], atol=1e-11 * np.abs(d0).max())


@pytest.mark.parametrize("elements,order,degree", [(("Si",), 3, 8), (("Si", "Ge"), 3, 6), (("Si",), 4, 12)])
def test_gpu_analytic_radial_matches_the_model(backend_kernel, monkeypatch, elements, order, degree):  # noqa: F811
    """The CUDA kernel with the tensor and pair radials evaluated analytically (`ffi.RADIAL`, a lane per
    neighbour) against the JAX model without a radial table, for bases `build_basis` makes (polynomial radials)
    with a random readout: energy, forces and stress to roundoff, where the table kernel is good to ~1e-8.
    One and two elements; few A (the analytic A pass too) and more (the table's)."""
    import dataclasses

    from ase.build import bulk

    from ace_jax.basis.model import BasisSpec, build_basis
    from ace_jax.calc.point import ACECalculator
    from conftest import require_coupling_lib
    if backend_kernel != "gpu":
        pytest.skip("the analytic radial is the CUDA kernel's")
    require_coupling_lib()
    model, meta = build_basis(BasisSpec(order=order, max_degree=degree, elements=elements, rcut=5.0)).eval_pair()
    WB = jnp.asarray(np.random.default_rng(0).normal(scale=0.05, size=model.WB.shape))
    model = fold_readout(dataclasses.replace(model, WB=WB, folded=False))
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    at.numbers = np.asarray([int(z) for z in meta["elements"]])[np.arange(len(at)) % len(elements)]
    at.rattle(0.08, seed=1)
    monkeypatch.setattr(ffi, "RADIAL", "analytic")
    monkeypatch.setattr(ffi, "FORM", "coupled")
    k, j = ACECalculator(model, meta, ffi=True), ACECalculator(model, meta)
    assert dict(k.eval_model.attrs)["nq"] > 0 and j.ffi is None
    a1, a2 = at.copy(), at.copy()
    a1.calc, a2.calc = k, j
    F2 = a2.get_forces()
    print(f"\n  n_a {int(k.eval_model.consts[5].shape[1])}  max|dF| {np.abs(a1.get_forces() - F2).max():.1e} of {np.abs(F2).max():.2f}")
    assert a1.get_potential_energy() == pytest.approx(a2.get_potential_energy(), rel=1e-11, abs=1e-10)
    np.testing.assert_allclose(a1.get_forces(), F2, atol=2e-10 * max(1.0, np.abs(F2).max()))
    np.testing.assert_allclose(a1.get_stress(), a2.get_stress(), atol=1e-11)
