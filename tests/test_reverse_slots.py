"""`reverse_slots` matches each live edge to its reverse, per periodic image;
`energy_forces_virial_dense(..., rev=...)` must then give the same forces as
the default scatter-add assembly (Review Focus 2: triclinic and small cells
where one pair appears as several images)."""
import pathlib

import numpy as np
import jax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk

from ace_jax.eval import load
from ace_jax.eval.nlist import dense_graph, reverse_slots

FIX = pathlib.Path(__file__).parent.parent / "fixtures"


def _cells():
    tric = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    tric.set_cell(tric.cell.array @ np.array([[1, .08, 0], [0, 1, .05], [0, 0, 1]]), scale_atoms=True)
    return {"tric": tric, "small": bulk("Si", "diamond", a=5.43)}      # small: images


@pytest.mark.parametrize("name", ["tric", "small"])
def test_reverse_slots_point_back(name):
    at = _cells()[name]
    dg = dense_graph(at.positions, at.cell.array, at.pbc, 5.0, 128)
    rev = reverse_slots(dg.idx, dg.rij, dg.count)
    live = np.arange(128)[None, :] < dg.count[:, None]
    i, k = np.nonzero(live)
    j = dg.idx[i, k]
    assert (dg.idx[j, rev[i, k]] == i).all()
    np.testing.assert_allclose(dg.rij[j, rev[i, k]], -dg.rij[i, k], atol=1e-10)


def test_rev_gather_equals_scatter():
    model, meta, _ = load(str(FIX / "sige_nofit.npz"))
    at = _cells()["small"].repeat((3, 3, 3))
    at.numbers[::2] = 32
    dg = dense_graph(at.positions, at.cell.array, at.pbc, meta["rcut"], 96)
    node_z = jnp.asarray((at.numbers == 32).astype(int))
    idx = jnp.asarray(dg.idx, jnp.int32)
    mask = jnp.arange(96)[None, :] < jnp.asarray(dg.count)[:, None]
    args = (jnp.asarray(dg.rij), jnp.broadcast_to(node_z[:, None], idx.shape), node_z[idx],
            idx, mask, node_z)
    E0, F0, V0 = model.energy_forces_virial_dense(*args)
    rev = jnp.asarray(reverse_slots(dg.idx, dg.rij, dg.count))
    E1, F1, V1 = model.energy_forces_virial_dense(*args, rev=rev)
    np.testing.assert_allclose(F1, F0, rtol=1e-12, atol=1e-13)
