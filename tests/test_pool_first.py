"""Pool-first A (EdgeSiteModel.pool_first_dense / pool_first_sparse) for PACE.

The per-edge radial R_nl = crad[z_i, z_j] . g is never formed: g_k (x) Y_lm is
pooled per (node, neighbour species) and the coefficients are applied per node
afterwards.  The frozen references (test_perf_parity.py) are the equality oracle
for E, F and V; these tests pin the pieces: A itself against the per-edge form,
both layouts, and -- Review Focus 5 -- that crad stays trainable (W is built from
crad inside the trace, not frozen into a host table).
"""
import dataclasses
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms
from ase.build import bulk

from ace_jax.eval import load
from ace_jax.eval.nlist import dense_from_sparse, sparse_graph
from conftest import pace_fixture

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"
ALL = ["gesi_sbessel", "sige_zbl", "sige_distance", "sige_density_core", "si_chebexpcos",
       "si_chebpow_fs", "si_cheblinear"]


@pytest.mark.parametrize("name", ["gesi_sbessel", "sige_zbl", "sige_distance", "si_chebexpcos"])
def test_energy_gradient_wrt_crad_matches_reference(name):
    """Pool-first builds W from crad inside the trace: dE/dcrad must match the
    old path's (frozen as finite differences of the reference energy)."""
    model, meta, _ = load(str(pace_fixture(FIX / f"{name}.yace")))
    at = bulk("Si", "diamond", a=5.43, cubic=True)
    at.numbers[::2] = meta["elements"][-1]
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers])
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)

    def E(crad):
        m = dataclasses.replace(model, crad=crad)
        return jnp.sum(m.site_energies(jnp.asarray(g.rij), nz[s], nz[r], s, len(at), nz))

    grad = jax.grad(E)(model.crad)
    d = np.zeros(model.crad.shape); d.flat[np.argmax(np.abs(np.asarray(grad)))] = 1e-6
    fd = (E(model.crad + d) - E(model.crad - d)) / 2e-6
    assert abs(float(jnp.vdot(grad, d / 1e-6)) - float(fd)) < 1e-6 * max(1.0, abs(float(fd)))


def _per_edge_A(model, rij, zi, zj, seg, n):
    """The pre-pool-first A (n, C * n_a): per-edge [g | R_nl][aspec_r] * Y[aspec_y],
    summed per (node, neighbour species).  Written out here as the oracle."""
    g, Y = model.edge_basis_factors(rij, zi, zj)
    R = jnp.einsum("ek,enlk->enl", g, model.crad[zi, zj]).reshape(g.shape[0], -1)
    cols = jnp.concatenate([g, R], axis=-1)
    rows = cols[:, model.aspec_r] * Y[:, model.aspec_y]
    C = model.a_channels
    A = jax.ops.segment_sum(rows, seg * C + zj, num_segments=n * C)
    return A.reshape(n, -1)


def _case(name):
    model, meta, _ = load(str(pace_fixture(FIX / f"{name}.yace")))
    ref = np.load(pace_fixture(FIX / f"{name}_ref.npz"))
    at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=True)
    at.rattle(0.05, seed=1)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    return model, meta, at, g, nz


@pytest.mark.parametrize("name", ALL)
def test_pool_first_A_equals_per_edge_A(name):
    """Both layouts' feature-major A_t (C * n_a, n) equal the per-edge form, transposed."""
    model, meta, at, g, nz = _case(name)
    n = len(at)
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    rij, zi, zj = jnp.asarray(g.rij), nz[s], nz[r]
    ref = np.asarray(_per_edge_A(model, rij, zi, zj, s, n)).T

    b, Y = model.edge_basis_factors(rij, zi, zj)
    At = model.pool_first_sparse(b, Y, s, zj, nz, n)
    assert At.shape == ref.shape
    np.testing.assert_allclose(np.asarray(At), ref, rtol=1e-12, atol=1e-12)

    d = dense_from_sparse(g, meta["rcut"])
    idx, mask = jnp.asarray(d.idx), jnp.asarray(d.mask)
    K = idx.shape[1]
    zid, zjd = jnp.broadcast_to(nz[:, None], idx.shape), nz[idx]
    flat = lambda a: a.reshape(n * K, *a.shape[2:])
    bd, Yd = model.edge_basis_factors(flat(jnp.asarray(d.rij)), flat(zid), flat(zjd), flat(mask))
    Atd = model.pool_first_dense(bd.reshape(n, K, -1), Yd.reshape(n, K, -1), zjd, nz)
    np.testing.assert_allclose(np.asarray(Atd), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ["gesi_sbessel", "sige_zbl"])
def test_pool_first_weights_follow_crad(name):
    """W is a function of crad at call time: changing crad changes W (a host table
    built at load would not), and its g-columns are delta rows."""
    model, *_ = _case(name)
    W = model.pool_first_weights()
    nz, n_a, kr = model.nz, int(model.aspec_r.shape[0]), model.nradbase
    assert W.shape == (nz, nz, n_a, kr)
    W2 = dataclasses.replace(model, crad=2.0 * model.crad).pool_first_weights()
    is_g = np.asarray(model.aspec_r) < kr
    np.testing.assert_array_equal(np.asarray(W2)[:, :, is_g], np.asarray(W)[:, :, is_g])
    np.testing.assert_allclose(np.asarray(W2)[:, :, ~is_g], 2.0 * np.asarray(W)[:, :, ~is_g])


def test_zbl_close_contact_dense_equals_sparse():
    """The shared energy tail keeps the ZBL core switch in both layouts (the
    prototype's feature-major path had dropped it)."""
    model, meta, _ = load(str(pace_fixture(FIX / "sige_zbl.yace")))
    assert model.inner_cutoff_type == "zbl"
    ref = np.load(pace_fixture(FIX / "sige_zbl_ref.npz"))
    at = Atoms(numbers=ref["Z_close"], positions=ref["pos_close"], cell=ref["cell_close"],
               pbc=ref["pbc_close"])
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    s = jnp.asarray(g.senders)
    e_s = model.site_energies(jnp.asarray(g.rij), nz[s], nz[jnp.asarray(g.receivers)], s,
                              len(at), nz)
    d = dense_from_sparse(g, meta["rcut"])
    idx = jnp.asarray(d.idx)
    e_d = model.site_energies_dense(jnp.asarray(d.rij), jnp.broadcast_to(nz[:, None], idx.shape),
                                    nz[idx], jnp.asarray(d.mask), nz)
    assert np.all(np.isfinite(np.asarray(e_s)))
    np.testing.assert_allclose(np.asarray(e_d), np.asarray(e_s), rtol=1e-12, atol=1e-12)
