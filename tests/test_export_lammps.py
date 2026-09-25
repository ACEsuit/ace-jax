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

from ace_jax.calc.point import ACECalculator
from ace_jax.eval import load, sparse_graph
from ace_jax.export.lammps import make_energy_fn
from conftest import pace_fixture

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


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_energy_fn_matches_calculator(layout):
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    at = _cluster()
    graph, g = _lammps_graph(at, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    species = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    K = int(np.bincount(g.senders, minlength=len(at)).max())
    f = make_energy_fn(model, len(meta["elements"]), layout, k_dense=K + 3)
    pos = jnp.asarray(at.positions)
    E, G = jax.value_and_grad(lambda p: jnp.sum(f(p, species, graph)))(pos)
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
    pytest.importorskip("lammps_jax")
    from ace_jax.export.lammps import export_lammps
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    b = export_lammps(model, meta, tmp_path / "m.json", max_atoms=256, max_edges=256 * 64,
                      k_dense=64, dtype="float64", layout="dense")
    on_disk = json.loads((tmp_path / "m.json").read_text())
    assert on_disk["contract"]["n_species"] == 2
    assert on_disk["ace_jax"] == {"layout": "dense", "elements": [32, 14], "k_dense": 64}
    assert b["ace_jax"]["layout"] == "dense"
