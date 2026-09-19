"""Folding the linear readout through A2B (PACE's C-tilde) is exact.

`e_i = WB[:,z_i] . (A2B . AA_i)` == `(A2B^T WB[:,z_i]) . AA_i`.  The fold
removes the A2B contraction and its adjoint from the evaluation path; nothing
about the numbers may change, and descriptors (which need B) must be untouched.
"""
import jax
import numpy as np
import pytest

from conftest import species_index

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import fold_readout, highest_precision, load

TOL_SAME_CODE = 1e-12
TOL_JULIA = 1e-10


def _edges(z):
    send = jnp.asarray(np.asarray(z["test_edge_i"], np.int32))
    recv = jnp.asarray(np.asarray(z["test_edge_j"], np.int32))
    rij = jnp.asarray(np.asarray(z["test_edge_rij"]).T)
    n_nodes = int(z["test_pos"].shape[1])
    return rij, send, recv, n_nodes, species_index(z)


def _efv(model, z):
    rij, send, recv, n_nodes, node_z = _edges(z)
    with highest_precision():
        E, F, V = model.energy_forces_virial(rij, node_z[send], node_z[recv],
                                            send, recv, n_nodes, node_z)
    return float(E), np.asarray(F), np.asarray(V)


@pytest.mark.parametrize("kind", ["gather", "matmul"])
@pytest.mark.parametrize("sparse", [False, True])
def test_fold_matches_unfolded(npz, kind, sparse):
    m0, meta, z = load(npz, a2b_sparse=sparse, edge_a_kind=kind, fold=False)
    m1 = fold_readout(m0)
    assert not m0.folded and m1.folded
    assert m1.ctilde.shape == (m0.A2B.shape[1], m0.WB.shape[1])
    E0, F0, V0 = _efv(m0, z)
    E1, F1, V1 = _efv(m1, z)
    print(f"\n  |dE| {abs(E0-E1):.2e}  |dF| {np.abs(F0-F1).max():.2e}  |dV| {np.abs(V0-V1).max():.2e}")
    # Energy is a total over 64 atoms (~1e4 eV): an absolute 1e-12 is below one
    # ulp there, so the fold's re-association is judged relative; forces and
    # the virial are O(1) and stay absolute.
    assert abs(E0 - E1) <= TOL_SAME_CODE * max(1.0, abs(E0))
    assert np.abs(F0 - F1).max() < TOL_SAME_CODE
    assert np.abs(V0 - V1).max() < TOL_SAME_CODE


def test_folded_matches_julia(npz):
    """The default load is folded; it must still hit the Julia reference."""
    m, meta, z = load(npz)
    assert m.folded
    E, F, V = _efv(m, z)
    assert abs(E - float(z["test_E"][0])) < TOL_JULIA
    assert np.abs(F - np.asarray(z["test_F"]).T).max() < TOL_JULIA
    assert np.abs(V - np.asarray(z["test_V"])).max() < TOL_JULIA


def test_fold_is_idempotent(npz):
    m, _, _ = load(npz, fold=False)
    m1 = fold_readout(m)
    assert fold_readout(m1) is m1


def test_site_basis_unchanged_by_fold(npz):
    m0, meta, z = load(npz, fold=False)
    m1 = fold_readout(m0)
    rij, send, recv, n_nodes, node_z = _edges(z)
    with highest_precision():
        B0, P0 = m0.site_basis(rij, node_z[send], node_z[recv], send, n_nodes)
        B1, P1 = m1.site_basis(rij, node_z[send], node_z[recv], send, n_nodes)
    assert np.array_equal(np.asarray(B0), np.asarray(B1))
    assert np.array_equal(np.asarray(P0), np.asarray(P1))


def test_dense_pooling_folded(npz):
    from ace_jax.eval import dense_graph
    m0, meta, z = load(npz, fold=False)
    m1 = fold_readout(m0)
    from ase import Atoms
    atoms = Atoms(numbers=np.asarray(z["test_Z"]), positions=np.asarray(z["test_pos"]).T,
                  cell=np.asarray(z["test_cell"]).T, pbc=np.asarray(z["test_pbc"]).astype(bool))
    g = dense_graph(atoms.get_positions(), atoms.get_cell().array, atoms.get_pbc(),
                    meta["rcut"], max_neighbours=96)      # Si at rcut 6 has ~46 neighbours
    node_z = species_index(z)
    mask = jnp.asarray(g.mask)                             # DenseGraph.mask is a property
    zi = jnp.broadcast_to(node_z[:, None], mask.shape)
    zj = node_z[jnp.where(mask, jnp.asarray(g.idx), 0)]
    with highest_precision():
        e0 = m0.site_energies_dense(jnp.asarray(g.rij), zi, zj, mask, node_z)
        e1 = m1.site_energies_dense(jnp.asarray(g.rij), zi, zj, mask, node_z)
    assert np.abs(np.asarray(e0) - np.asarray(e1)).max() < TOL_SAME_CODE
