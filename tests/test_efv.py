"""Phases 3-4 gate.

From an ASE Atoms object -- own neighbour list, periodic images and all --
reproduce Julia's AtomsCalculators.energy_forces_virial on the same fitted
model to 1e-10 in f64: energy, forces AND virial.

The core is validated elsewhere against a Julia-supplied edge list; this closes the
loop: matscipy-neighbours builds the edges here, so a wrong cutoff, a missing
periodic image or a shift-convention error would show up.
"""
import json
import pathlib

import jax
import numpy as np
import pytest

from conftest import species_index

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import (ACECalculator, dense_graph, dense_to_sparse,
                    highest_precision, load, sparse_graph)

TOL = 1e-10




@pytest.fixture
def case(npz):
    model, meta, z = load(npz)
    from ase import Atoms
    atoms = Atoms(numbers=np.asarray(z["test_Z"]),
                  positions=np.asarray(z["test_pos"]).T,
                  cell=np.asarray(z["test_cell"]).T,
                  pbc=np.asarray(z["test_pbc"]).astype(bool))
    return model, meta, z, atoms


def test_nlist_matches_julia(case):
    """The neighbour list itself, before any model evaluation."""
    model, meta, z, atoms = case
    g = sparse_graph(atoms.get_positions(), atoms.get_cell().array,
                     atoms.get_pbc(), meta["rcut"])
    assert len(g.senders) == len(z["test_edge_i"]), "edge count differs from Julia"
    # same (i, rij) multiset -- ordering within a centre need not match
    def key(i, r):
        return sorted(zip(i.tolist(), [tuple(np.round(v, 9)) for v in r]))
    assert key(g.senders, g.rij) == key(np.asarray(z["test_edge_i"]),
                                        np.asarray(z["test_edge_rij"]).T)
    print(f"\n  edges {len(g.senders)}, sorted by i: {bool(np.all(np.diff(g.senders) >= 0))}")


def test_energy_forces_virial(case):
    model, meta, z, atoms = case
    g = sparse_graph(atoms.get_positions(), atoms.get_cell().array,
                     atoms.get_pbc(), meta["rcut"])
    node_z = species_index(z)
    send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    with highest_precision():
        E, F, V = model.energy_forces_virial(jnp.asarray(g.rij), node_z[send],
                                            node_z[recv], send, recv, g.n_nodes, node_z)
    eE = abs(float(E) - float(z["test_E"][0]))
    eF = np.max(np.abs(np.asarray(F) - np.asarray(z["test_F"]).T))
    eV = np.max(np.abs(np.asarray(V) - np.asarray(z["test_V"])))
    print(f"\n  energy |d| = {eE:.3e} eV")
    print(f"  forces |d| = {eF:.3e} eV/A   (scale {np.max(np.abs(z['test_F'])):.4f})")
    print(f"  virial |d| = {eV:.3e} eV     (scale {np.max(np.abs(z['test_V'])):.4f})")
    assert eE < TOL and eF < TOL and eV < TOL


def test_virial_sign_is_julia_convention(case):
    """Guard the sign explicitly: a flipped virial would still be 'close' in
    magnitude, so assert against the reference rather than |V|."""
    model, meta, z, atoms = case
    g = sparse_graph(atoms.get_positions(), atoms.get_cell().array,
                     atoms.get_pbc(), meta["rcut"])
    node_z = species_index(z)
    send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    with highest_precision():
        _, _, V = model.energy_forces_virial(jnp.asarray(g.rij), node_z[send],
                                            node_z[recv], send, recv, g.n_nodes, node_z)
    Vref = np.asarray(z["test_V"])
    assert np.max(np.abs(np.asarray(V) - Vref)) < TOL
    assert np.max(np.abs(np.asarray(V) + Vref)) > 1.0, "sign test is degenerate"


def test_dense_and_sparse_pooling_agree(case):
    """The two neighbour-list layouts must give the same answer; neither is
    hard-wired, and lammps-jax needs sparse while dense avoids a scatter."""
    model, meta, z, atoms = case
    # capacity from the real neighbour count, not a hardcoded 64: a denser
    # system truncates silently and the comparison then fails for the wrong
    # reason (TiAl at these cutoffs has ~86 neighbours per atom)
    s0 = sparse_graph(atoms.get_positions(), atoms.get_cell().array,
                      atoms.get_pbc(), meta["rcut"])
    kmax = int(np.bincount(np.asarray(s0.senders)).max())
    d = dense_graph(atoms.get_positions(), atoms.get_cell().array,
                    atoms.get_pbc(), meta["rcut"], kmax)
    n, K = d.idx.shape
    node_z = species_index(z)
    zi = jnp.broadcast_to(node_z[:, None], (n, K))        # centre species per row
    zj = jnp.asarray(node_z)[jnp.asarray(d.idx)]          # neighbour species per slot
    mask = jnp.asarray(d.mask)
    # padded slots must not be allowed to blow up: park them at the cutoff
    rij = np.where(d.mask[..., None], d.rij, np.array([meta["rcut"], 0.0, 0.0]))
    with highest_precision():
        e_dense = model.site_energies_dense(jnp.asarray(rij), zi, zj, mask, node_z)
    s = dense_to_sparse(d)
    send = jnp.asarray(s.senders)
    with highest_precision():
        e_sparse = model.site_energies(jnp.asarray(s.rij), node_z[send],
                                       node_z[jnp.asarray(s.receivers)], send,
                                       s.n_nodes, node_z)
    err = np.max(np.abs(np.asarray(e_dense) - np.asarray(e_sparse)))
    print(f"\n  dense vs sparse pooling: max|d| = {err:.3e}")
    assert err < 1e-11


def test_ase_calculator(case):
    model, meta, z, atoms = case
    atoms = atoms.copy()
    atoms.calc = ACECalculator(model, meta)
    eE = abs(atoms.get_potential_energy() - float(z["test_E"][0]))
    eF = np.max(np.abs(atoms.get_forces() - np.asarray(z["test_F"]).T))
    s = atoms.get_stress(voigt=False)
    eV = np.max(np.abs(-s * atoms.get_volume() - np.asarray(z["test_V"])))
    print(f"\n  ASE: |dE| = {eE:.3e}  |dF| = {eF:.3e}  |dV| = {eV:.3e}")
    assert eE < TOL and eF < TOL and eV < TOL


def test_all_neighbour_backends_agree(case):
    """The three backends must produce *identical* edge sets, not similar ones.

    matscipy is the hard dependency; matscipy-neighbours is an optional faster
    one; the numpy fallback exists so the package still works without either.
    A user's results must not depend on which is installed.
    """
    from ace_jax.eval.nlist import (_neighbour_list, backend, have_matscipy,
                              have_matscipy_neighbours)
    model, meta, z, atoms = case
    args = (atoms.get_positions(), atoms.get_cell().array, atoms.get_pbc(),
            float(meta["rcut"]))
    avail = ["ase"]
    if have_matscipy():
        avail.append("matscipy")
    if have_matscipy_neighbours():
        avail.append("matscipy-neighbours")

    key = lambda i, D: sorted(zip(i.tolist(), [tuple(np.round(v, 9)) for v in D]))
    ref_name = avail[0]
    i0, j0, D0, _ = _neighbour_list(*args, force_backend=ref_name)
    ref = key(i0, D0)
    for name in avail[1:]:
        i1, j1, D1, _ = _neighbour_list(*args, force_backend=name)
        assert len(i1) == len(i0), f"{name}: {len(i1)} edges vs {ref_name}'s {len(i0)}"
        assert key(i1, D1) == ref, f"{name} disagrees with {ref_name}"
    print(f"\n  backends agree ({len(i0)} edges): {', '.join(avail)}"
          f"   [active: {backend()}]")


def test_dense_from_sparse_matches_neighbour_matrix(case):
    """The dense layout built by grouping the sparse list must match the native
    `neighbour_matrix` where that exists -- otherwise the two code paths could
    diverge silently on machines that have it."""
    from ace_jax.eval.nlist import _dense_from_sparse, _neighbour_list, have_matscipy_neighbours
    if not have_matscipy_neighbours():
        pytest.skip("no neighbour_matrix to compare against")
    from matscipy_neighbours import neighbour_matrix
    model, meta, z, atoms = case
    pos, cell, pbc, rcut = (atoms.get_positions(), atoms.get_cell().array,
                            atoms.get_pbc(), float(meta["rcut"]))
    i0, _, _, _ = _neighbour_list(pos, cell, pbc, rcut)
    K = int(np.bincount(i0).max())        # not a hardcoded 64: see above
    idx_n, dist_n, cnt_n = neighbour_matrix(positions=pos, cell=cell, pbc=tuple(pbc),
                                            cutoff=rcut, max_neighbours=K)
    i, j, D, _ = _neighbour_list(pos, cell, pbc, rcut)
    idx_s, dist_s, cnt_s = _dense_from_sparse(i, j, D, len(pos), K)
    assert np.array_equal(cnt_n, cnt_s)
    # rows may be ordered differently within a centre; compare as sets
    for a in range(len(pos)):
        n = cnt_n[a]
        set_n = sorted(zip(idx_n[a, :n].tolist(),
                           [tuple(np.round(v, 9)) for v in dist_n[a, :n]]))
        set_s = sorted(zip(idx_s[a, :n].tolist(),
                           [tuple(np.round(v, 9)) for v in dist_s[a, :n]]))
        assert set_n == set_s, f"atom {a} differs"


def test_sparse_a2b_matches_dense(case, npz):
    """A2B has ~one nonzero per column, so the dense contraction does n_B * n_AA
    multiply-adds where nnz would do.  The sparse path must give the same answer
    -- it is an optimisation, not an approximation."""
    from ace_jax.eval import load
    model_d, meta, z = load(npz, a2b_sparse=False)
    model_s, _, _ = load(npz, a2b_sparse=True)
    n = int(z["test_pos"].shape[1])
    send = jnp.asarray(z["test_edge_i"], jnp.int32)
    recv = jnp.asarray(z["test_edge_j"], jnp.int32)
    rij = jnp.asarray(z["test_edge_rij"].T)
    nz = jnp.zeros(n, jnp.int32)
    with highest_precision():
        ed = model_d.site_energies(rij, nz[send], nz[recv], send, n, nz)
        es = model_s.site_energies(rij, nz[send], nz[recv], send, n, nz)
    err = float(np.max(np.abs(np.asarray(ed) - np.asarray(es))))
    nnz = int(model_d.a2b_vals.shape[0])
    dense = int(model_d.A2B.shape[0]) * int(model_d.A2B.shape[1])
    print(f"\n  {npz.stem}: A2B {model_d.A2B.shape} nnz={nnz} "
          f"({100*nnz/dense:.3f}% occupied)  sparse-vs-dense max|d| = {err:.3e}")
    assert err < 1e-10
