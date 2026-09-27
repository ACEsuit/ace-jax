"""Chunked dense evaluation must match the single-block (unchunked) result.

`EdgeSiteModel.energy_forces_virial_dense` blocks rows in groups of `chunk`
via `lax.map`; a site energy depends only on its own row (`rij`, `zj`, `mask`,
`node_z`), so blocking is exact -- padding rows are fully masked and must
contribute nothing (Review Focus 4)."""
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.eval.nlist import dense_graph
from conftest import pace_fixture

FIX = pathlib.Path(__file__).parent.parent / "fixtures"


def _dense_inputs(at, model, meta, z2i):
    counts_K = 64
    dg = dense_graph(at.positions, at.cell.array, at.pbc, meta["rcut"], counts_K)
    idx = jnp.asarray(dg.idx, jnp.int32)
    mask = jnp.arange(counts_K)[None, :] < jnp.asarray(dg.count)[:, None]
    node_z = jnp.asarray([z2i[int(z)] for z in at.numbers])
    zi = jnp.broadcast_to(node_z[:, None], idx.shape)
    return jnp.asarray(dg.rij), zi, node_z[idx], idx, mask, node_z


@pytest.mark.parametrize("path", [pace_fixture(FIX / "pace" / "gesi_sbessel.yace"),
                                  FIX / "sige_nofit.npz"])
@pytest.mark.parametrize("chunk", [7, 64, 16384])          # 7: 5+ blocks, n % 7 != 0
def test_chunked_equals_unchunked(path, chunk):
    from ase.build import bulk
    model, meta, _ = load(str(path))
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))        # 64 atoms
    at.numbers[::3] = meta["elements"][-1]
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    args = _dense_inputs(at, model, meta, z2i)
    E0, F0, V0 = model.energy_forces_virial_dense(*args, chunk=10 ** 9)   # one block
    E1, F1, V1 = model.energy_forces_virial_dense(*args, chunk=chunk)
    np.testing.assert_allclose(E1, E0, rtol=1e-13)
    np.testing.assert_allclose(F1, F0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(V1, V0, rtol=1e-12, atol=1e-12)
