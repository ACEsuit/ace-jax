"""The C++ FFI kernel on lammps-jax's neighbour matrix against the matrix-layout program of the lean model
with the same radial table: atom energies and forces to roundoff, since both read one table and differ
only in summation order.  Fixtures cover spherical and solid harmonics, 1, 2 and 5 species, and
correlation order 4."""
import pathlib
import shutil
from collections import namedtuple

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk

from ace_jax.eval import ffi, load
from ace_jax.eval.model import lean
from ace_jax.export.lammps import make_energy_fn

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
NAMES = ["si_fitted", "si_ace_model", "sige_nofit", "ace_cantor5_small", "si_m710"]
Graph = namedtuple("Graph", "neighbors num_neighbors")


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    if shutil.which("c++") is None:
        pytest.skip("no C++ compiler for the FFI kernel")
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
