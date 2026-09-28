"""The dense (n, K) layout must give the sparse layout's energies and forces.

Dense builds the A basis per node as a batched outer product of the two
per-edge factors (EdgeSiteModel.pool_a_dense); sparse sums per-edge A rows
(pool_a_sparse).  Same mathematics, different summation order: equal to
roundoff.  Both model families, multi-element PACE with its neighbour-species
channel, ZBL close contact, and an atom with no neighbours.
"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms

from conftest import FIXTURE_DIR, MODELS, pace_fixture
from ace_jax.eval import load, sparse_graph
from ace_jax.eval.nlist import dense_from_sparse

PACE = ("gesi_sbessel", "sige_zbl", "si_chebpow_fs")


@pytest.fixture(params=[f"ace:{k}" for k in MODELS] + [f"pace:{k}" for k in PACE])
def model_path(request):
    fam, name = request.param.split(":")
    if fam == "ace":
        p = FIXTURE_DIR / MODELS[name]
        if not p.exists():
            pytest.skip(f"{p.name} not generated")
        return p
    return pace_fixture(FIXTURE_DIR / "pace" / f"{name}.yace")


def _structures(path):
    """(label, Atoms) to test on, for either family."""
    if str(path).endswith(".yace"):
        ref = np.load(str(path).replace(".yace", "_ref.npz"))
        out = []
        for s in ("bulk", "close", "isolated"):
            out.append((s, Atoms(numbers=ref[f"Z_{s}"], positions=ref[f"pos_{s}"],
                                 cell=ref[f"cell_{s}"], pbc=ref[f"pbc_{s}"])))
        return out
    z = np.load(path)
    return [("test", Atoms(numbers=np.asarray(z["test_Z"]), positions=np.asarray(z["test_pos"]).T,
                           cell=np.asarray(z["test_cell"]).T, pbc=np.asarray(z["test_pbc"]).astype(bool)))]


def _both_layouts(model, meta, at):
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(x)] for x in at.numbers], jnp.int32)
    d = dense_from_sparse(g, meta["rcut"])
    sparse = (jnp.asarray(g.rij), nz[jnp.asarray(g.senders)], nz[jnp.asarray(g.receivers)],
              jnp.asarray(g.senders), jnp.asarray(g.receivers), len(at), nz)
    idx = jnp.asarray(d.idx)
    dense = (jnp.asarray(d.rij), jnp.broadcast_to(nz[:, None], idx.shape), nz[idx],
             idx, jnp.asarray(d.mask), nz)
    return sparse, dense


def test_site_energies_dense_equal_sparse(model_path):
    model, meta, _ = load(str(model_path))
    for label, at in _structures(model_path):
        (rij, zi, zj, s, r, n, nz), (rd, zid, zjd, idx, mask, nz) = _both_layouts(model, meta, at)
        e_s = np.asarray(model.site_energies(rij, zi, zj, s, n, nz))
        e_d = np.asarray(model.site_energies_dense(rd, zid, zjd, mask, nz))
        np.testing.assert_allclose(e_d, e_s, rtol=1e-12, atol=1e-12, err_msg=label)


def test_energy_forces_virial_dense_equal_sparse(model_path):
    model, meta, _ = load(str(model_path))
    for label, at in _structures(model_path):
        (rij, zi, zj, s, r, n, nz), (rd, zid, zjd, idx, mask, nz) = _both_layouts(model, meta, at)
        E, F, V = model.energy_forces_virial(rij, zi, zj, s, r, n, nz)
        Ed, Fd, Vd = model.energy_forces_virial_dense(rd, zid, zjd, idx, mask, nz)
        assert float(Ed) == pytest.approx(float(E), rel=1e-12, abs=1e-12), label
        np.testing.assert_allclose(np.asarray(Fd), np.asarray(F), atol=1e-10, err_msg=label)
        np.testing.assert_allclose(np.asarray(Vd), np.asarray(V), atol=1e-10, err_msg=label)
        assert np.all(np.isfinite(np.asarray(Fd)))
