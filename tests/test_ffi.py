"""The FFI kernels on lammps-jax's neighbour matrix, C++ on the CPU and CUDA on a GPU, against the
matrix-layout program of the lean model with the same radial table: atom energies and forces to roundoff,
since both read one table and differ only in summation order.  Fixtures cover spherical and solid
harmonics, 1, 2 and 5 species, and correlation order 4; periodic cells with ghost atoms (newton on)
against ACECalculator; the GPU kernel in float32 too.  The GPU cases need nvcc and a CUDA jax."""
import dataclasses
import os
import pathlib
import shutil
from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk

from ace_jax.calc.point import ACECalculator
from ace_jax.eval import ffi, load
from ace_jax.eval.api import with_dtype
from ace_jax.eval.model import lean
from ace_jax.export.lammps import make_energy_fn
from conftest import FIXTURE_DIR as FIXTURES
from conftest import require_optional

NAMES = ["si_fitted", "si_ace_model", "sige_nofit", "ace_cantor5_small", "si_m710"]
Graph = namedtuple("Graph", "neighbors num_neighbors")


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    if shutil.which("c++") is None:                    # ACEJAX_REQUIRE_OPTIONAL: a missing compiler fails
        (pytest.fail if os.environ.get("ACEJAX_REQUIRE_OPTIONAL") else pytest.skip)("no C++ compiler for the FFI kernel")
    ffi.register(ffi.library(flags=("-O3",), cache=tmp_path_factory.mktemp("ffi")))


@pytest.fixture(scope="module", params=["cpu", "gpu"])
def device(request, tmp_path_factory):
    """The platform: the C++ kernel, or the GPU kernel, built and registered."""
    if request.param == "cpu":
        request.getfixturevalue("kernel")
    elif ffi.cuda_compiler() is None or jax.default_backend() != "gpu":
        pytest.skip("no nvcc and CUDA GPU for the GPU kernel")
    else:
        ffi.register_gpu(ffi.library(gpu=True, cache=tmp_path_factory.getbasetemp() / "ffi-gpu"))
    return request.param


def cluster(meta, seed=0):
    """A rattled cluster with every species of the model present."""
    els = [int(z) for z in meta["elements"]]
    at = (bulk("Ni", "fcc", a=3.6, cubic=True).repeat(2) if len(els) > 2
          else bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1)))
    at.numbers = np.asarray(els)[np.arange(len(at)) % len(els)]
    np.random.default_rng(seed).shuffle(at.numbers)
    at.rattle(0.08, seed=seed)
    at.pbc = False
    return at


def matrix_inputs(at, cut, z):
    """Positions, species and the graph: a (K, n) neighbour matrix of the pairs within `cut`, padded with
    n, and the counts."""
    pos = at.positions
    d = np.linalg.norm(pos[None] - pos[:, None], axis=-1)
    lists = [np.nonzero((d[i] < cut) & (np.arange(len(at)) != i))[0] for i in range(len(at))]
    nb = np.full((max(map(len, lists)), len(at)), len(at), np.int32)
    for i, js in enumerate(lists):
        nb[:len(js), i] = js
    num = jnp.asarray([len(js) for js in lists], jnp.int32)
    return jnp.asarray(pos), jnp.asarray(z, jnp.int32), Graph(jnp.asarray(nb), num)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("name", NAMES)
def test_ffi_matches_matrix_program(device, name, dtype):
    """float32: the GPU kernel on `with_dtype(model, "float32")`, to float32 roundoff."""
    if device == "cpu" and dtype == "float32":
        pytest.skip("the C++ kernel is float64 only")
    m, meta, _ = load(str(FIXTURES / f"{name}.npz"))
    m1, rcut = lean(m, spline_tol=None, radial_table=True), float(meta["rcut"])
    at = cluster(meta)
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    ns = len(meta["elements"])
    with jax.default_device(jax.devices(device)[0]):
        pos, z, graph = matrix_inputs(at, rcut + 1.0, [z2i[int(x)] for x in at.numbers])   # with skin pairs
        efn = make_energy_fn(m1, ns, "matrix", rcut=rcut)
        e0, f0 = jax.jit(lambda p: (efn(p, z, graph), -jax.grad(lambda q: efn(q, z, graph).sum())(p)))(pos)
        efn1, ffn1 = ffi.lammps_fns(with_dtype(m1, dtype), ns, list(range(ns)), rcut, device == "gpu")
        e1, f1 = jax.jit(lambda p: (efn1(p, z, graph), ffn1(p, z, graph)))(pos.astype(dtype))
    print(f"\n  max|de| {np.abs(e1 - e0).max():.1e}"
          f"  max|dF|/max|F| {np.abs(f1 - f0).max() / np.abs(f0).max():.1e}")
    tol_e, tol_f = (1e-12, 1e-10) if dtype == "float64" else (1e-5, 3e-5)   # float32 XLA: 1.5e-6, 3.6e-6
    assert e1.dtype == f1.dtype == dtype
    assert np.abs(e1 - e0).max() <= tol_e * np.abs(e0).max()
    assert np.abs(f1 - f0).max() <= tol_f * np.abs(f0).max()


@pytest.mark.parametrize("device", ["gpu"], indirect=True)
def test_gpu_ffi_matches_cpu_kernel_on_a_radial_column_of_two_degrees(device, kernel):
    """A radial column read at l = 1 and 2, which `prune_columns` allows: the GPU kernel's runs must not cross
    a degree. Energies and forces equal the C++ kernel's."""
    m, meta, _ = load(str(FIXTURES / "si_fitted.npz"))
    m1, rcut = lean(m, spline_tol=None, radial_table=True), float(meta["rcut"])
    ar, deg = np.asarray(m1.aspec_r).copy(), np.sqrt(np.asarray(m1.aspec_y)).astype(int)
    ar[deg == 2] = ar[deg == 1][0]
    m2 = dataclasses.replace(m1, aspec_r=jnp.asarray(ar, m1.aspec_r.dtype))
    at = cluster(meta)
    pos, z, graph = matrix_inputs(at, rcut + 1.0, [0] * len(at))
    out = []
    for dev in ("cpu", "gpu"):
        with jax.default_device(jax.devices(dev)[0]):
            efn, ffn = ffi.lammps_fns(m2, 1, [0], rcut, dev == "gpu")
            run = jax.jit(lambda p, efn=efn, ffn=ffn: (efn(p, z, graph), ffn(p, z, graph)))
            out.append([np.asarray(a) for a in run(pos)])
    (e0, f0), (e1, f1) = out
    assert np.abs(e1 - e0).max() <= 1e-12 * np.abs(e0).max()
    assert np.abs(f1 - f0).max() <= 1e-10 * np.abs(f0).max()


def test_ffi_needs_a_table():
    m, _, _ = load(str(FIXTURES / "si_fitted.npz"))
    with pytest.raises(ValueError, match="radial table"):
        ffi.kernel_inputs(lean(m, spline_tol=None))


def periodic_ghosts(at, cutoff):
    """LAMMPS's periodic picture: owned atoms, then one ghost per (atom, image) some owned atom lists
    within `cutoff`.  Returns positions, the owned rows' neighbour lists and every ghost's owner."""
    from ase.neighborlist import neighbor_list
    i, j, S = neighbor_list("ijS", at, cutoff)
    n, ghosts = len(at), {}
    recv = [jj if s == (0, 0, 0) else n + ghosts.setdefault((jj, s), len(ghosts)) for jj, s in zip(j, map(tuple, S))]
    owner = np.array([jj for jj, _ in ghosts], np.int64)
    shift = np.array([s for _, s in ghosts], float).reshape(-1, 3) @ at.cell.array
    lists = [np.asarray(recv)[i == k] for k in range(n)]
    return np.concatenate([at.positions, at.positions[owner] + shift]), lists, owner


@pytest.mark.parametrize("name", ["si_fitted", "sige_nofit", "ace_cantor5_small"])
def test_ffi_periodic_ghosts_match_calculator(device, name):
    """Owned rows only (rows < atoms), three padding rows past nlocal, padding slots, skin pairs and an
    unused position tail: owned energies, and forces with the ghost rows folded back onto their owners
    (newton on), equal ACECalculator with the same radial table; padding rows and the tail get none."""
    path = str(FIXTURES / f"{name}.npz")
    m, meta, _ = load(path)
    els, rcut = [int(z) for z in meta["elements"]], float(meta["rcut"])
    at = cluster(meta, seed=1)
    at.pbc = True
    pos, lists, owner = periodic_ghosts(at, rcut + 1.0)
    n, N, rows, tail = len(at), len(pos), len(at) + 3, 8
    nb = np.full((max(map(len, lists)) + 4, rows), N + tail, np.int32)   # padding indexes past max_atoms
    for i, js in enumerate(lists):
        nb[:len(js), i] = js
    num = np.r_[[len(js) for js in lists], [0] * (rows - n)].astype(np.int32)
    z2i = {z: i for i, z in enumerate(els)}
    m1 = lean(m, spline_tol=None, radial_table=True)
    with jax.default_device(jax.devices(device)[0]):
        z = jnp.asarray([z2i[int(x)] for x in np.r_[at.numbers, at.numbers[owner]]] + [0] * tail, jnp.int32)
        graph = Graph(jnp.asarray(nb), jnp.asarray(num))
        efn, ffn = ffi.lammps_fns(m1, len(els), list(range(len(els))), rcut, device == "gpu")
        x = jnp.asarray(np.r_[pos, np.zeros((tail, 3))])
        e, f = jax.jit(lambda p: (efn(p, z, graph), ffn(p, z, graph)))(x)
    e, f = np.asarray(e), np.asarray(f)
    F = f[:n].copy()
    np.add.at(F, owner, f[n:N])
    at.calc = ACECalculator(path, radial_table=True, spline_tol=None, skin=0.0)
    assert e[:n].sum() == pytest.approx(at.get_potential_energy(), abs=1e-10)
    np.testing.assert_allclose(F, at.get_forces(), atol=1e-10)
    assert np.all(e[rows:] == 0.0) and np.all(f[N:] == 0.0)


def test_library_is_cached_and_keyed(kernel, tmp_path, monkeypatch):
    """library(): one build per key (source, jaxlib, compiler, flags, machine), atomically placed; the
    second call reuses it; other flags are another entry; no compiler is FFIUnavailable, not a crash."""
    calls = []
    real = ffi.build
    monkeypatch.setattr(ffi, "build", lambda *a: calls.append(a) or real(*a))
    a = ffi.library(flags=("-O1",), cache=tmp_path)
    assert ffi.library(flags=("-O1",), cache=tmp_path) == a and len(calls) == 1
    b = ffi.library(flags=("-O0",), cache=tmp_path)
    assert b != a and len(calls) == 2 and sorted(p.name for p in tmp_path.iterdir()) == sorted([a.name, b.name])
    assert ffi.handlers(a) == f"ace_jax_direct={a.resolve()}:AceDirect"
    monkeypatch.setenv("ACEJAX_FFI_FLAGS", "-O1")
    assert ffi.library(cache=tmp_path) == a and len(calls) == 2
    monkeypatch.setattr(ffi, "compiler", lambda: None)
    with pytest.raises(ffi.FFIUnavailable, match="C\\+\\+17 compiler"):
        ffi.library(flags=("-O2",), cache=tmp_path)


def test_export_ffi_bundle(kernel, tmp_path, monkeypatch):
    """export_lammps(ffi=True): the kernel's custom call in a CPU matrix bundle, newton on, the radial
    table turned on, the metadata the kernel applies and the library it built; the other layouts and
    float32 are refused."""
    require_optional("lammps_jax")
    monkeypatch.setenv("ACEJAX_FFI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ACEJAX_FFI_FLAGS", "-O1")
    from ace_jax.export.lammps import export_lammps
    m, meta, _ = load(str(FIXTURES / "si_fitted.npz"))
    kw = dict(max_atoms=256, max_owned=64, k_dense=40, max_neighbors=48, platform="cpu", ffi=True)
    b = export_lammps(m, meta, tmp_path / "b.json", **kw)
    aj = b["ace_jax"]
    lib = aj["ffi"]["library"]
    assert aj["ffi"] == {"target": ffi.TARGET, "library": lib, "handlers": ffi.handlers(lib)}
    assert pathlib.Path(lib).parent == tmp_path / "cache" and aj["layout"] == "matrix" and aj["radial_table"]
    assert aj["aa_dag"] == "greedy" and aj["k_dense"] is None and aj["block_rows"] is None
    assert ffi.TARGET in str(b)                       # the contract names the custom call
    for bad in (dict(layout="dense"), dict(dtype="float32")):
        with pytest.raises(ValueError, match="ffi=True"):
            export_lammps(m, meta, tmp_path / "x.json", **{**kw, **bad})


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_export_ffi_gpu_bundle(tmp_path, dtype):
    """export_lammps(ffi=<library>) on a CUDA platform, float64 or float32: the GPU kernel's custom call, its
    target and the LAMMPS_JAX_FFI_HANDLERS value; the export lowers without a GPU, the library is only
    named."""
    require_optional("lammps_jax")
    from ace_jax.export.lammps import export_lammps
    m, meta, _ = load(str(FIXTURES / "si_fitted.npz"))
    lib = str(tmp_path / "libace_jax_ffi_gpu.so")
    b = export_lammps(m, meta, tmp_path / "b.json", max_atoms=256, max_owned=64, k_dense=40, max_neighbors=48,
                      platform="cuda", ffi=lib, dtype=dtype)
    aj = b["ace_jax"]
    assert aj["ffi"] == {"target": ffi.TARGET_GPU, "library": lib, "handlers": ffi.handlers(lib, gpu=True)}
    assert aj["ffi"]["handlers"] == f"ace_jax_direct_gpu={lib}:AceDirectGpu"
    assert ffi.TARGET_GPU in str(b) and aj["layout"] == "matrix" and aj["radial_table"]
    assert b["contract"]["precision"] == dtype


def test_gpu_library_needs_nvcc_and_a_gpu(tmp_path, monkeypatch):
    """library(gpu=True) without nvcc, or without a GPU to take the architecture from, is FFIUnavailable."""
    monkeypatch.delenv("ACEJAX_FFI_GPU_FLAGS", raising=False)
    monkeypatch.setattr(ffi, "cuda_compiler", lambda: None)
    with pytest.raises(ffi.FFIUnavailable, match="nvcc"):
        ffi.library(gpu=True, flags=("-O3", "-arch=sm_80"), cache=tmp_path)
    monkeypatch.setattr(jax, "devices", lambda *a: (_ for _ in ()).throw(RuntimeError("no gpu")))
    with pytest.raises(ffi.FFIUnavailable, match="no CUDA GPU"):
        ffi.library(gpu=True, cache=tmp_path)


@pytest.mark.parametrize("name", ["si_fitted", "si_ace_model", "ace_cantor5_small"])
def test_ffi_grad_and_vjp_match_matrix_program(device, name):
    """jax.grad through the FFI energy (its custom VJP: a second kernel call with the cotangent as site
    weights): grad of the summed energy is minus the kernel's forces and the XLA program's gradient, and
    the VJP of a random per-atom cotangent is the XLA program's, under jit and value_and_grad."""
    m, meta, _ = load(str(FIXTURES / f"{name}.npz"))
    m1, rcut = lean(m, spline_tol=None, radial_table=True), float(meta["rcut"])
    at = cluster(meta, seed=2)
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    ns = len(meta["elements"])
    with jax.default_device(jax.devices(device)[0]):
        pos, z, graph = matrix_inputs(at, rcut + 1.0, [z2i[int(x)] for x in at.numbers])
        efx = make_energy_fn(m1, ns, "matrix", rcut=rcut)
        efn, ffn = ffi.lammps_fns(m1, ns, list(range(ns)), rcut, device == "gpu")
        E1, G1 = jax.jit(jax.value_and_grad(lambda p: efn(p, z, graph).sum()))(pos)
        E0, G0 = jax.jit(jax.value_and_grad(lambda p: efx(p, z, graph).sum()))(pos)
        F1 = ffn(pos, z, graph)
        ct = jnp.asarray(np.random.default_rng(0).normal(size=pos.shape[0]))
        v1 = jax.jit(lambda p: jax.vjp(lambda q: efn(q, z, graph), p)[1](ct)[0])(pos)
        v0 = jax.jit(lambda p: jax.vjp(lambda q: efx(q, z, graph), p)[1](ct)[0])(pos)
    scale = np.abs(G0).max()
    assert float(E1) == pytest.approx(float(E0), rel=1e-12)
    np.testing.assert_allclose(G1, -F1, atol=1e-13 * scale)
    np.testing.assert_allclose(G1, G0, atol=1e-10 * scale)
    np.testing.assert_allclose(v1, v0, atol=1e-10 * np.abs(v0).max())


def cpu_capacity(meta):
    from ace_jax.export.lammps import neighbour_capacity
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    return neighbour_capacity(at, float(meta["rcut"]))


def test_auto_takes_the_kernel_for_a_cpu_bundle(kernel, tmp_path, monkeypatch):
    """export_lammps's default ffi="auto": a CPU bundle of a linear ACE model is the C++ kernel (no reason
    recorded), and its program compiles on XLA's CPU backend once the handler is registered."""
    require_optional("lammps_jax")
    import base64
    from jax._src import xla_bridge
    from jaxlib import xla_client
    from ace_jax.export.lammps import export_lammps
    monkeypatch.setenv("ACEJAX_FFI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ACEJAX_FFI_FLAGS", "-O1")
    m, meta, _ = load(str(FIXTURES / "si_ace_model.npz"))
    cap = cpu_capacity(meta)
    b = export_lammps(m, meta, tmp_path / "b.json", max_atoms=cap["max_atoms"], max_owned=cap["max_owned"],
                      k_dense=cap["k_dense"], max_neighbors=cap["max_neighbors"], platform="cpu", device_memory=2**30)
    aj = b["ace_jax"]
    assert aj["ffi"]["target"] == ffi.TARGET and aj["ffi_reason"] is None and aj["layout"] == "matrix"
    assert pathlib.Path(aj["ffi"]["library"]).parent == tmp_path / "cache" and aj["radial_table"]
    be = xla_bridge.get_backend("cpu")
    opts = xla_client.CompileOptions.ParseFromString(base64.b64decode(b["compile_options_b64"]))
    be.compile_and_load(base64.b64decode(b["programs"]["energy_and_forces_mlir_b64"]),
                        xla_client.DeviceList(tuple(be.local_devices()[:1])), opts)


AUTO_KW = dict(platform="cpu", dtype="float64", layout="auto", lean=True, radial_table=None, max_neighbors=64,
               aa_dag="auto")


@pytest.mark.parametrize("change,why", [
    ({"platform": "cuda"}, None),
    ({"platform": None}, None),
    ({"dtype": "float32"}, "dtype float32"),
    ({"lean": False}, "lean=False"),
    ({"layout": "dense"}, "layout 'dense'"),
    ({"max_neighbors": None}, "no max_neighbors"),
    ({"radial_table": False}, "radial_table switched off"),
    ({"aa_dag": "cueq"}, "aa_dag='cueq'"),
])
def test_auto_falls_back_with_its_reason(change, why):
    """choose_ffi("auto") takes the XLA program, and says why for a CPU bundle; a GPU bundle has no reason."""
    from ace_jax.export.lammps import choose_ffi
    m, _, _ = load(str(FIXTURES / "si_ace_model.npz"))
    got, reason = choose_ffi("auto", m, **(AUTO_KW | change))
    assert got is False and (reason is None if why is None else why in reason)


def test_auto_falls_back_without_a_kernel_model_lammps_jax_or_compiler(monkeypatch):
    """choose_ffi("auto"): a PACE model, a lammps-jax older than 0.2.1 and no compiler each take the XLA
    program with their reason; an explicit ffi is returned as given (the export then checks it)."""
    from ace_jax.export import lammps
    from conftest import pace_fixture
    m, _, _ = load(str(FIXTURES / "si_ace_model.npz"))
    pace, _, _ = load(str(pace_fixture(FIXTURES / "pace" / "si_chebpow_fs.yace")))
    assert "not a linear ACEModel" in lammps.choose_ffi("auto", pace, **AUTO_KW)[1]
    monkeypatch.setattr(lammps, "lammps_jax_version", lambda: "0.2.0")
    assert "lammps-jax 0.2.0" in lammps.choose_ffi("auto", m, **AUTO_KW)[1]
    monkeypatch.setattr(lammps, "lammps_jax_version", lambda: "0.2.1+abc1234")
    monkeypatch.setattr(ffi, "compiler", lambda env=None: None)
    monkeypatch.setenv("ACEJAX_FFI_FLAGS", "-O0 -DACEJAX_UNCACHED_KEY")   # a key not yet built: library() compiles
    assert "C++17 compiler" in lammps.choose_ffi("auto", m, **AUTO_KW)[1]
    assert lammps.choose_ffi(True, m, **AUTO_KW) == (True, None)
    assert lammps.choose_ffi(False, m, **AUTO_KW) == (False, None)


def test_explicit_ffi_still_refuses_an_ineligible_export(tmp_path):
    """ffi=True requires the kernel: a float32 CPU bundle raises rather than falling back."""
    require_optional("lammps_jax")
    from ace_jax.export.lammps import export_lammps
    m, meta, _ = load(str(FIXTURES / "si_ace_model.npz"))
    cap = cpu_capacity(meta)
    kw = dict(max_atoms=cap["max_atoms"], max_owned=cap["max_owned"], k_dense=cap["k_dense"],
              max_neighbors=cap["max_neighbors"], ffi=True)
    with pytest.raises(ValueError, match="ffi=True"):
        export_lammps(m, meta, tmp_path / "x.json", **kw, platform="cpu", dtype="float32")


def test_ffi_matches_matrix_program_on_a_species_embedding_model(device):
    """The kernel on a species-embedding model (`build_embedding_model`, the factorised radial), with random
    coefficients: atom energies and forces equal the matrix-layout program's to roundoff, as for the
    categorical fixtures (the radial table holds each species pair's embedded radial)."""
    import dataclasses
    from conftest import require_coupling_lib
    require_coupling_lib()
    from ace_jax.basis.model import build_embedding_model
    from ace_jax.eval.model import fold_readout
    els = [14, 32]
    m, meta = build_embedding_model(els, 3, 6, embedding=(els, np.array([[1.0, 0.3], [0.6, -0.8]]))).eval_pair()
    rng = np.random.default_rng(0)
    m = fold_readout(dataclasses.replace(m, WB=jnp.asarray(rng.normal(size=m.WB.shape)),
                                         Wpair=jnp.asarray(0.1 * rng.normal(size=m.Wpair.shape))))
    assert m.radial_kind == "spline_factorised"
    m1, rcut = lean(m, spline_tol=None, radial_table=True), float(meta["rcut"])
    at = cluster({"elements": els}, seed=3)
    with jax.default_device(jax.devices(device)[0]):
        pos, z, graph = matrix_inputs(at, rcut + 1.0, [els.index(int(x)) for x in at.numbers])
        efx = make_energy_fn(m1, len(els), "matrix", rcut=rcut)
        e0, f0 = jax.jit(lambda p: (efx(p, z, graph), -jax.grad(lambda q: efx(q, z, graph).sum())(p)))(pos)
        efn, ffn = ffi.lammps_fns(m1, len(els), [0, 1], rcut, device == "gpu")
        e1, f1 = jax.jit(lambda p: (efn(p, z, graph), ffn(p, z, graph)))(pos)
    assert np.abs(e1 - e0).max() <= 1e-12 * np.abs(e0).max()
    assert np.abs(f1 - f0).max() <= 1e-10 * np.abs(f0).max()


def _md_cell(meta, r=3):
    els = [int(z) for z in meta["elements"]]
    at = (bulk("Ni", "fcc", a=3.6, cubic=True) if len(els) > 2 else bulk("Si", "diamond", a=5.43, cubic=True)).repeat(r)
    at.numbers = np.asarray(els)[np.arange(len(at)) % len(els)]
    return at


@pytest.mark.parametrize("skin", [1.0, 0.0])
@pytest.mark.parametrize("name", ["si_fitted", "sige_nofit", "ace_cantor5_small"])
def test_calculator_kernel_matches_the_jax_model(kernel, name, skin):
    """ACECalculator(ffi=True) against the JAX model with the same radial table, over MD-like steps (the
    skin list reused, and rebuilt by one larger move): energy, forces and stress to roundoff, the dense
    layout.  (Atoms pushed below the table's r_min, 0.5 A, extrapolate its end cubic in both: there the
    two summation orders differ at ~1e-9 relative, outside physical geometries.)"""
    path = str(FIXTURES / f"{name}.npz")
    _, meta, _ = load(path)
    k, j = ACECalculator(path, ffi=True, skin=skin), ACECalculator(path, radial_table=True, skin=skin)
    assert k.ffi and k.ffi_reason is None and j.ffi is None
    at, rng = _md_cell(meta), np.random.default_rng(0)
    for step in range(5):
        at.positions += rng.normal(scale=0.2 if step == 2 else 0.02, size=at.positions.shape)
        a1, a2 = at.copy(), at.copy()
        a1.calc, a2.calc = k, j
        F2 = a2.get_forces()
        assert a1.get_potential_energy() == pytest.approx(a2.get_potential_energy(), rel=1e-12, abs=1e-10)
        np.testing.assert_allclose(a1.get_forces(), F2, atol=1e-11 * max(1.0, np.abs(F2).max()))
        np.testing.assert_allclose(a1.get_stress(), a2.get_stress(), atol=1e-12)
    assert k.last_layout == "dense" and k.last_timing["ffi"] and k.last_timing["rebuilds"] >= 2


def test_calculator_kernel_threads_model_swap_and_e0(kernel):
    """One thread and XLA's pool give identical results (rows write only their own outputs); a calc.model
    swap keeps the kernel with the new coefficients; energy_reference="E0" drops E0 as the JAX model does."""
    import dataclasses
    path = str(FIXTURES / "sige_nofit.npz")
    m, meta, _ = load(path)
    at = _md_cell(meta)
    at.rattle(0.05, seed=1)
    pool, one = ACECalculator(path, ffi=True), ACECalculator(path, ffi=True, ffi_threads=1)
    assert one.ffi["threads"] == 1
    a1, a2 = at.copy(), at.copy()
    a1.calc, a2.calc = pool, one
    assert a1.get_potential_energy() == a2.get_potential_energy()
    np.testing.assert_array_equal(a1.get_forces(), a2.get_forces())
    k, j = ACECalculator(m, meta, ffi=True), ACECalculator(m, meta, radial_table=True)
    m2 = dataclasses.replace(m, WB=1.1 * m.WB)
    k.model, j.model = m2, m2
    a1, a2 = at.copy(), at.copy()
    a1.calc, a2.calc = k, j
    assert k.ffi is not None
    np.testing.assert_allclose(a1.get_forces(), a2.get_forces(), atol=1e-11)
    k, j = (ACECalculator(path, energy_reference="E0", **kw) for kw in (dict(ffi=True), dict(radial_table=True)))
    a1, a2 = at.copy(), at.copy()
    a1.calc, a2.calc = k, j
    assert a1.get_potential_energy() == pytest.approx(a2.get_potential_energy(), abs=1e-10)


def test_calculator_ffi_auto_and_refusals(kernel):
    """ffi="auto" keeps the JAX model where the kernel cannot run, with the reason; ffi=True raises there;
    a PACE model is not a linear ACEModel."""
    from conftest import pace_fixture
    path = str(FIXTURES / "si_fitted.npz")
    for kw, why in ((dict(layout="sparse"), "layout='sparse'"), (dict(lean=False), "lean=False"),
                    (dict(radial_table=False), "radial_table switched off")):
        c = ACECalculator(path, ffi="auto", **kw)
        assert c.ffi is None and why in c.ffi_reason
        with pytest.raises(ValueError, match="cannot run here"):
            ACECalculator(path, ffi=True, **kw)
    assert ACECalculator(path, ffi="auto").ffi is not None
    c = ACECalculator(str(pace_fixture(FIXTURES / "pace" / "si_chebpow_fs.yace")), ffi="auto")
    assert c.ffi is None and "not a linear ACEModel" in c.ffi_reason
    with pytest.raises(ValueError, match="ffi must be"):
        ACECalculator(path, ffi="yes")


def test_aj_native_reports_and_builds(kernel, tmp_path, monkeypatch, capsys):
    """`aj native`: the compiler, flags, CPU, kernel cache state and neighbour list; --build compiles the kernel
    into the cache (then reported as built); without a compiler the status says so and --build exits 1."""
    from ace_jax.cli import main
    monkeypatch.setenv("ACEJAX_FFI_CACHE", str(tmp_path / "cache"))
    main(["native", "--flags=-O1"])
    out = capsys.readouterr().out
    assert "C++ compiler:" in out and "not built yet" in out and "neighbours:" in out and "-O1" in out
    main(["native", "--build", "--flags=-O1"])
    out = capsys.readouterr().out
    assert "built in" in out and len(list((tmp_path / "cache").glob("libace_jax_ffi-*.so"))) == 1
    main(["native", "--flags=-O1"])
    assert "C++ kernel:    built:" in capsys.readouterr().out
    monkeypatch.setattr(ffi, "compiler", lambda env=None: None)
    main(["native"])
    assert "C++ compiler:  none" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["native", "--build"])
    assert e.value.code == 1


def test_edge_kernel_vmaps_by_flattening_rows(kernel):
    """AceEdges under jax.vmap: the batch flattened into rows, one kernel call, results (and dE/drij
    through the custom VJP) identical to calling it per batch element; unbatched row inputs broadcast;
    batched model constants fall back to one call per element."""
    from ace_jax.eval.ffi import KernelModel, edge_site_energies
    m, meta, _ = load(str(FIXTURES / "ace_cantor5_small.npz"))
    km = KernelModel.of(lean(m, spline_tol=None, radial_table=True))
    rng = np.random.default_rng(0)
    B, n, K = 3, 7, 12
    rij = jnp.asarray(rng.normal(size=(B, n, K, 3)) * 1.5 + 0.5)
    z = jnp.asarray(rng.integers(0, 5, (B, n)), jnp.int32)
    zj = jnp.asarray(rng.integers(0, 5, (B, n, K)), jnp.int32)
    mask = jnp.asarray(rng.random((B, n, K)) < 0.8)
    f = lambda r, z_, zj_, m_: km.site_energies_dense(r, None, zj_, m_, z_)
    one = lambda b: f(rij[b], z[b], zj[b], mask[b])
    assert np.array_equal(jax.jit(jax.vmap(f))(rij, z, zj, mask), jnp.stack([one(b) for b in range(B)]))
    g = jax.jit(jax.vmap(jax.grad(lambda *a: f(*a).sum())))(rij, z, zj, mask)
    assert np.array_equal(g, jnp.stack([jax.grad(lambda r, b=b: f(r, z[b], zj[b], mask[b]).sum())(rij[b])
                                        for b in range(B)]))
    shared = jax.vmap(lambda r: f(r, z[0], zj[0], mask[0]))(rij)              # species not batched
    assert np.array_equal(shared, jnp.stack([f(rij[b], z[0], zj[0], mask[0]) for b in range(B)]))
    hlo = jax.jit(jax.vmap(f)).lower(rij, z, zj, mask).as_text()
    assert hlo.count("ace_jax_edges") == 1
    consts = tuple(jnp.stack([c] * B) for c in km.consts)                     # a model per element
    e = jax.vmap(lambda r, c: edge_site_energies(km.attrs, r, z[0], zj[0], mask[0], *c))(rij, consts)
    assert np.array_equal(e, shared)
