"""ace-jax models as lammps-jax energy functions.

The exported function sees LAMMPS's packed edge buffer: masked padding, edges
in whatever order LAMMPS built them, receivers that may be ghost atoms.  On an
open cluster (no ghosts needed) it must equal ACECalculator exactly, in both
layouts; periodic parity is checked end-to-end in LAMMPS (bench parity gate).
"""
import json
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms
from ase.build import bulk

from ace_jax.calc.point import ACECalculator
from ace_jax.eval import load, sparse_graph
from ace_jax.export.lammps import make_energy_fn
from conftest import pace_fixture, require_optional

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


class Graph:                                    # the lammps-jax graph fields
    def __init__(self, s, r, m):
        self.senders, self.receivers, self.edge_mask = s, r, m


def _cluster(name="gesi_sbessel"):
    ref = np.load(pace_fixture(FIX / f"{name}_ref.npz"))
    at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=False)
    at.center(vacuum=8.0)
    return at


def _lammps_graph(at, rcut, n_pad=37, seed=0):
    g = sparse_graph(at.positions, at.cell.array, at.pbc, rcut)
    order = np.random.default_rng(seed).permutation(len(g.senders))   # LAMMPS order is not ours
    s, r = g.senders[order], g.receivers[order]
    m = np.ones(len(s), bool)
    s = np.concatenate([s, np.zeros(n_pad, int)]); r = np.concatenate([r, np.zeros(n_pad, int)])
    m = np.concatenate([m, np.zeros(n_pad, bool)])
    return Graph(jnp.asarray(s, jnp.int32), jnp.asarray(r, jnp.int32), jnp.asarray(m)), g


MODELS = {"pace": lambda: str(pace_fixture(FIX / "gesi_sbessel.yace")),
          "ace": lambda: str(FIX.parent / "sige_nofit.npz")}      # ACEModel: pad_cutoff is a float()


@pytest.mark.parametrize("kind", sorted(MODELS))
@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_energy_fn_matches_calculator(layout, kind):
    y = MODELS[kind]()
    model, meta, _ = load(y)
    at = _cluster()
    if kind == "ace":                                   # the cluster's species, as Si/Ge
        at.numbers = np.where(at.numbers == 32, 32, 14)
    graph, g = _lammps_graph(at, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    species = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    K = int(np.bincount(g.senders, minlength=len(at)).max())
    f = make_energy_fn(model, len(meta["elements"]), layout, k_dense=K + 3)
    pos = jnp.asarray(at.positions)
    # jit: the bundle export traces energy_fn, so nothing in it may concretize
    E, G = jax.jit(jax.value_and_grad(lambda p: jnp.sum(f(p, species, graph))))(pos)
    at.calc = ACECalculator(y, layout="sparse")
    assert float(E) == pytest.approx(at.get_potential_energy(), abs=1e-10)
    np.testing.assert_allclose(-np.asarray(G), at.get_forces(), atol=1e-9)


def test_dense_overflow_is_loud():
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    at = _cluster()
    graph, _ = _lammps_graph(at, meta["rcut"])
    species = jnp.zeros(len(at), jnp.int32)
    f = make_energy_fn(model, len(meta["elements"]), "dense", k_dense=2)
    assert np.isnan(float(jnp.sum(f(jnp.asarray(at.positions), species, graph))))


def test_bundle_written(tmp_path):
    require_optional("lammps_jax")
    from ace_jax.export.lammps import export_lammps
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    b = export_lammps(model, meta, tmp_path / "m.json", max_atoms=256, max_edges=256 * 64,
                      k_dense=64, dtype="float64", layout="dense")
    on_disk = json.loads((tmp_path / "m.json").read_text())
    assert on_disk["contract"]["n_species"] == 2
    assert on_disk["ace_jax"] == {"layout": "dense", "elements": [32, 14],
                                  "type_elements": [32, 14], "k_dense": 64, "owned_rows": None}
    assert b["ace_jax"]["layout"] == "dense"


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_lammps_type_order_differs_from_model_order(layout):
    """LAMMPS hands species = type - 1 in the *input's* element order; a .yace
    lists its elements in its own order ([Ge, Si] here).  type_map bridges them."""
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    assert list(meta["elements"]) == [32, 14]
    at = _cluster()
    graph, g = _lammps_graph(at, meta["rcut"])
    types = [14, 32]                                      # LAMMPS: type 1 = Si, 2 = Ge
    species = jnp.asarray([types.index(int(z)) for z in at.numbers], jnp.int32)
    K = int(np.bincount(g.senders, minlength=len(at)).max())
    f = make_energy_fn(model, 2, layout, k_dense=K + 3,
                       type_map=[list(meta["elements"]).index(z) for z in types])
    E = float(jnp.sum(f(jnp.asarray(at.positions), species, graph)))
    at.calc = ACECalculator(y, layout="sparse")
    assert E == pytest.approx(at.get_potential_energy(), abs=1e-10)


def test_bundle_records_owned_rows_even_for_sparse(tmp_path):
    """The sparse layout has no row concept, but max_owned is still recorded
    (the value the caller sized the bundle's neighbour slots for)."""
    require_optional("lammps_jax")
    from ace_jax.export.lammps import export_lammps
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    b = export_lammps(model, meta, tmp_path / "m.json", max_atoms=256, max_edges=256 * 64,
                      layout="sparse", max_owned=200)
    assert b["ace_jax"]["owned_rows"] == 200 and b["ace_jax"]["layout"] == "sparse"


# JSON keys lammps-jax reads from a bundle (cpp/lammps_jax_model.cpp).  Its
# reader matches a key name anywhere in the file, nested objects included, so
# ace-jax's own metadata must not reuse one: a nested "max_owned" was taken as
# the contract's and LAMMPS refused the bundle.
LAMMPS_JAX_KEYS = {"comm_sites", "comm_widths", "custom_call_targets", "cutoff", "edge_pairing",
                   "energy_and_forces_mlir", "energy_mlir", "force_mlir", "force_output",
                   "format", "input_layout", "max_atoms", "max_edges", "max_neighbors",
                   "max_owned", "n_hops", "n_species", "newton", "pair_sum", "precision",
                   "unit_style", "uses_box"}


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_bundle_metadata_keys_do_not_shadow_the_contract(tmp_path, layout):
    require_optional("lammps_jax")
    from ace_jax.export.lammps import export_lammps
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    b = export_lammps(model, meta, tmp_path / "m.json", max_atoms=256, max_edges=256 * 64,
                      k_dense=64, layout=layout, max_owned=200)
    assert not set(b["ace_jax"]) & LAMMPS_JAX_KEYS


def test_bundle_records_type_order(tmp_path):
    require_optional("lammps_jax")
    from ace_jax.export.lammps import export_lammps
    model, meta, _ = load(str(pace_fixture(FIX / "gesi_sbessel.yace")))
    b = export_lammps(model, meta, tmp_path / "m.json", max_atoms=64, max_edges=64 * 64,
                      k_dense=64, layout="sparse", type_elements=[14, 32])
    assert b["ace_jax"]["type_elements"] == [14, 32]
    with pytest.raises(ValueError, match="not in the model"):
        export_lammps(model, meta, tmp_path / "x.json", max_atoms=64, max_edges=64,
                      layout="sparse", type_elements=[14, 6])


def test_dense_overflow_makes_forces_nan_too():
    """jnp.where(overflow, nan, e) sends no cotangent to e: energy NaN but
    forces exactly zero, so MD between thermo steps would run silently wrong."""
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    at = _cluster()
    graph, _ = _lammps_graph(at, meta["rcut"])
    species = jnp.zeros(len(at), jnp.int32)
    f = make_energy_fn(model, len(meta["elements"]), "dense", k_dense=2)
    g = jax.grad(lambda p: jnp.sum(f(p, species, graph)))(jnp.asarray(at.positions))
    assert np.isnan(np.asarray(g)).any()


@pytest.mark.parametrize("layout", ["dense"])
def test_owned_rows_bundle_matches_calculator(layout):
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    at = _cluster()
    graph, g = _lammps_graph(at, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    species = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    K = int(np.bincount(g.senders, minlength=len(at)).max())
    f = make_energy_fn(model, len(meta["elements"]), layout, k_dense=K + 3,
                       n_rows=int(np.ceil(1.1 * len(at))))
    E, G = jax.value_and_grad(lambda p: jnp.sum(f(p, species, graph)))(jnp.asarray(at.positions))
    at.calc = ACECalculator(y, layout="sparse", skin=0.0)
    assert float(E) == pytest.approx(at.get_potential_energy(), abs=1e-10)
    np.testing.assert_allclose(-np.asarray(G), at.get_forces(), atol=1e-9)


def test_owned_rows_no_overflow_matches_unrestricted():
    """n_rows < n with no overflow: this is owned-rows evaluation itself, not
    just its overflow guard.  Build a periodic supercell graph and keep only
    edges whose sender is < nr (receivers may be any atom -- the ghost-like
    case LAMMPS actually presents), so restricting to nr rows drops no real
    edge.  Rows < nr must then match the unrestricted computation exactly,
    and rows >= nr must be exactly zero (not the isolated-atom energy the
    unrestricted computation would give them)."""
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    a = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    a.numbers[::2] = 32                                       # Ge/Si mix
    graph, _ = _lammps_graph(a, meta["rcut"])
    n = len(a)
    nr = n // 2
    senders, receivers, mask = (np.asarray(x) for x in
                                (graph.senders, graph.receivers, graph.edge_mask))
    keep = mask & (senders < nr)                              # owned senders only
    s = np.where(keep, senders, 0)
    r = np.where(keep, receivers, 0)
    filtered = Graph(jnp.asarray(s, jnp.int32), jnp.asarray(r, jnp.int32), jnp.asarray(keep))
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    species = jnp.asarray([z2i[int(z)] for z in a.numbers], jnp.int32)
    K = int(np.bincount(senders[keep], minlength=n).max())
    pos = jnp.asarray(a.positions)

    f_all = make_energy_fn(model, len(meta["elements"]), "dense", k_dense=K + 3)
    f_owned = make_energy_fn(model, len(meta["elements"]), "dense", k_dense=K + 3, n_rows=nr)

    _, g_all = jax.value_and_grad(lambda p: jnp.sum(f_all(p, species, filtered)))(pos)
    _, g_owned = jax.value_and_grad(lambda p: jnp.sum(f_owned(p, species, filtered)))(pos)
    e_all_arr = f_all(pos, species, filtered)
    e_owned_arr = f_owned(pos, species, filtered)

    np.testing.assert_allclose(np.asarray(e_owned_arr[:nr]), np.asarray(e_all_arr[:nr]),
                               rtol=1e-12, atol=1e-12)
    assert np.all(np.asarray(e_owned_arr[nr:]) == 0.0)
    np.testing.assert_allclose(np.asarray(g_owned), np.asarray(g_all), rtol=1e-12, atol=1e-12)


def test_owned_rows_overflow_is_nan():
    """A sender >= n_rows must make the energy NaN, and (multiplicative
    overflow, not jnp.where(overflow, nan, e)) the forces too."""
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    at = _cluster()
    graph, _ = _lammps_graph(at, meta["rcut"])
    n_rows = len(at) // 2
    species = jnp.zeros(len(at), jnp.int32)
    f = make_energy_fn(model, len(meta["elements"]), "dense", k_dense=64, n_rows=n_rows)
    E, G = jax.value_and_grad(lambda p: jnp.sum(f(p, species, graph)))(jnp.asarray(at.positions))
    assert np.isnan(float(E))
    assert np.isnan(np.asarray(G)).any()


# ------------------------------------------------------------------ row blocks
# The dense bundle evaluates its rows in blocks of lammps.BUNDLE_BLOCK_ROWS
# (lax.map + jax.checkpoint) once they exceed one block; at or below one block
# the program is the unblocked one.  docs/perf-lammps-large-n.md.

def _big_cluster(kind):
    """213 atoms (3x3x3 diamond, 3 removed): with a 64-row block that is four
    blocks of 54 plus 3 padding rows -- not a multiple of the block."""
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3))
    del at[[0, 50, 100]]
    at.numbers[::3] = 32
    at.pbc = False
    at.center(vacuum=8.0)
    return at


def _blocked_inputs(kind, type_order, n_ghost=0):
    model, meta, _ = load(MODELS[kind]())
    at = _big_cluster(kind)
    graph, g = _lammps_graph(at, meta["rcut"])
    elements = [int(z) for z in meta["elements"]]
    types = elements if type_order == "model" else elements[::-1]
    type_map = None if type_order == "model" else [elements.index(z) for z in types]
    species = jnp.asarray([types.index(int(z)) for z in at.numbers], jnp.int32)
    pos = jnp.asarray(at.positions)
    if n_ghost:                              # rows >= n_rows no edge points at
        pos = jnp.concatenate([pos, pos[:n_ghost] + 100.0])
        species = jnp.concatenate([species, species[:n_ghost]])
    K = int(np.bincount(g.senders, minlength=len(at)).max())
    return model, len(elements), K + 3, type_map, species, pos, graph, len(at)


@pytest.mark.parametrize("n_ghost", [0, 11])
@pytest.mark.parametrize("type_order", ["model", "reversed"])
@pytest.mark.parametrize("kind", sorted(MODELS))
def test_dense_blocks_match_single_block(monkeypatch, kind, type_order, n_ghost):
    from ace_jax.export import lammps
    model, nsp, k_dense, type_map, species, pos, graph, n_real = _blocked_inputs(
        kind, type_order, n_ghost)
    n_rows = n_real if n_ghost else None

    def run(block):
        monkeypatch.setattr(lammps, "BUNDLE_BLOCK_ROWS", block)
        f = make_energy_fn(model, nsp, "dense", k_dense=k_dense, type_map=type_map,
                           n_rows=n_rows)
        e = jax.jit(lambda p: f(p, species, graph))(pos)
        G = jax.jit(jax.grad(lambda p: jnp.sum(f(p, species, graph))))(pos)
        return np.asarray(e), np.asarray(G)

    e1, G1 = run(10 ** 9)
    e64, G64 = run(64)
    assert np.isfinite(e1).all() and np.isfinite(G1).all()
    np.testing.assert_allclose(e64, e1, rtol=1e-12, atol=0)
    np.testing.assert_allclose(G64, G1, rtol=1e-12, atol=1e-12 * np.abs(G1).max())


def _loop_primitives(jaxpr):
    """Names of loop / remat primitives anywhere in a (closed) jaxpr."""
    found = set()

    def walk(jx):
        for eqn in jx.eqns:
            if eqn.primitive.name in ("scan", "while", "checkpoint", "remat", "remat2"):
                found.add(eqn.primitive.name)
            for p in eqn.params.values():
                for sub in (p if isinstance(p, (list, tuple)) else [p]):
                    inner = getattr(sub, "jaxpr", sub)
                    if hasattr(inner, "eqns"):
                        walk(inner)
    walk(jaxpr.jaxpr)
    return found


@pytest.mark.parametrize("block, blocked", [(64, True), (213, False), (10 ** 9, False)])
def test_dense_blocks_only_above_one_block(monkeypatch, block, blocked):
    from ace_jax.export import lammps
    model, nsp, k_dense, type_map, species, pos, graph, _ = _blocked_inputs("pace", "model")
    monkeypatch.setattr(lammps, "BUNDLE_BLOCK_ROWS", block)
    f = make_energy_fn(model, nsp, "dense", k_dense=k_dense)
    jx = jax.make_jaxpr(jax.value_and_grad(lambda p: jnp.sum(f(p, species, graph))))(pos)
    prims = _loop_primitives(jx)
    if blocked:
        assert prims & {"scan", "while"} and prims & {"checkpoint", "remat", "remat2"}, prims
    else:
        assert not prims, prims


def test_auto_layout_judges_one_dense_block(tmp_path, monkeypatch):
    """The dense bundle runs in BUNDLE_BLOCK_ROWS blocks, so only one block's
    temporaries are live: auto must stay dense when a block fits the budget
    even though all rows at once would not (the benchmark exports with auto)."""
    require_optional("lammps_jax")
    from ace_jax.calc import point
    from ace_jax.eval.edge_model import estimate_a_bytes
    from ace_jax.export import lammps as lx
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    k, block, n = 64, lx.BUNDLE_BLOCK_ROWS, 4 * lx.BUNDLE_BLOCK_ROWS
    one = estimate_a_bytes(model, "dense", block, block * k, k, 8)
    monkeypatch.setattr(point, "dense_budget_bytes", lambda: 2 * one)   # a block fits, 4 do not
    assert estimate_a_bytes(model, "dense", n, n * k, k, 8) > 2 * one
    b = lx.export_lammps(model, meta, tmp_path / "m.json", max_atoms=n + 100, max_edges=n * k,
                         k_dense=k, max_owned=n, layout="auto")
    assert b["ace_jax"]["layout"] == "dense"
