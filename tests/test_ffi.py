"""The C++ FFI kernel on lammps-jax's neighbour matrix against the matrix-layout program of the lean model
with the same radial table: atom energies and forces to roundoff, since both read one table and differ
only in summation order.  Fixtures cover spherical and solid harmonics, 1, 2 and 5 species, and
correlation order 4; periodic cells with ghost atoms (newton on) against ACECalculator."""
import os
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
    ffi.register(ffi.build(tmp_path_factory.mktemp("ffi") / "libace_jax_ffi.so"))


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


@pytest.mark.parametrize("name", NAMES)
def test_ffi_matches_matrix_program(kernel, name):
    m, meta, _ = load(str(FIXTURES / f"{name}.npz"))
    m1, rcut = lean(m, spline_tol=None, radial_table=True), float(meta["rcut"])
    at = cluster(meta)
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    pos, z, graph = matrix_inputs(at, rcut + 1.0, [z2i[int(x)] for x in at.numbers])   # with skin pairs
    ns = len(meta["elements"])
    efn = make_energy_fn(m1, ns, "matrix", rcut=rcut)
    e0, f0 = jax.jit(lambda p: (efn(p, z, graph), -jax.grad(lambda q: efn(q, z, graph).sum())(p)))(pos)
    efn1, ffn1 = ffi.lammps_fns(m1, ns, list(range(ns)), rcut)
    e1, f1 = jax.jit(lambda p: (efn1(p, z, graph), ffn1(p, z, graph)))(pos)
    print(f"\n  max|de| {np.abs(e1 - e0).max():.1e}"
          f"  max|dF|/max|F| {np.abs(f1 - f0).max() / np.abs(f0).max():.1e}")
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
def test_ffi_periodic_ghosts_match_calculator(kernel, name):
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
    z = jnp.asarray([z2i[int(x)] for x in np.r_[at.numbers, at.numbers[owner]]] + [0] * tail, jnp.int32)
    graph = Graph(jnp.asarray(nb), jnp.asarray(num))
    efn, ffn = ffi.lammps_fns(lean(m, spline_tol=None, radial_table=True), len(els), list(range(len(els))), rcut)
    e, f = jax.jit(lambda p: (efn(p, z, graph), ffn(p, z, graph)))(jnp.asarray(np.r_[pos, np.zeros((tail, 3))]))
    e, f = np.asarray(e), np.asarray(f)
    F = f[:n].copy()
    np.add.at(F, owner, f[n:N])
    at.calc = ACECalculator(path, radial_table=True, spline_tol=None, skin=0.0)
    assert e[:n].sum() == pytest.approx(at.get_potential_energy(), abs=1e-10)
    np.testing.assert_allclose(F, at.get_forces(), atol=1e-10)
    assert np.all(e[rows:] == 0.0) and np.all(f[N:] == 0.0)


def test_export_ffi_bundle(tmp_path):
    """export_lammps(ffi=True): the kernel's custom call in a CPU matrix bundle, newton on, the radial
    table turned on, the metadata the kernel applies; the other layouts and float32 are refused."""
    require_optional("lammps_jax")
    from ace_jax.export.lammps import export_lammps
    m, meta, _ = load(str(FIXTURES / "si_fitted.npz"))
    kw = dict(max_atoms=256, max_owned=64, k_dense=40, max_neighbors=48, platform="cpu", ffi=True)
    b = export_lammps(m, meta, tmp_path / "b.json", **kw)
    aj = b["ace_jax"]
    assert aj["ffi"] == ffi.TARGET and aj["layout"] == "matrix" and aj["radial_table"]
    assert aj["aa_dag"] == "greedy" and aj["k_dense"] is None and aj["block_rows"] is None
    assert ffi.TARGET in str(b)                       # the contract names the custom call
    for bad in (dict(layout="dense"), dict(dtype="float32"), dict(platform="cuda")):
        with pytest.raises(ValueError, match="ffi=True"):
            export_lammps(m, meta, tmp_path / "x.json", **{**kw, **bad})
