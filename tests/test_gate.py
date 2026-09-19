"""Core gate: the descriptor against Julia.

A FITTED ace1_model (acefit! on Si_tiny, BLR) exported and reproducing Julia's
site energies and forces to 1e-10 in f64.

Forces come from the edge-vector core by autodiff on the edge vectors and a
scatter, since rij_e = pos[recv_e] - pos[send_e] + shift_e gives
    F_a = sum_e dE/drij_e (delta_{a,send_e} - delta_{a,recv_e}).
This exercises the core exactly as the LAMMPS path will, with no cell involved.
"""
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)   # test-local, never library-level
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from conftest import species_index

TOL = 1e-10


@pytest.fixture
def case(npz):
    model, meta, z = load(npz)
    n_nodes = int(z["test_pos"].shape[1])
    send = jnp.asarray(z["test_edge_i"], jnp.int32)      # centre atom (0-based)
    recv = jnp.asarray(z["test_edge_j"], jnp.int32)
    rij = jnp.asarray(z["test_edge_rij"].T)              # (E,3)
    node_z = species_index(z)
    zi, zj = node_z[send], node_z[recv]
    return model, meta, z, n_nodes, send, recv, rij, node_z, zi, zj


def test_site_energies(case):
    model, meta, z, n_nodes, send, recv, rij, node_z, zi, zj = case
    with highest_precision():
        e = model.site_energies(rij, zi, zj, send, n_nodes, node_z)
    err = np.max(np.abs(np.asarray(e) - z["test_site_E"]))
    scale = np.max(np.abs(z["test_site_E"]))
    print(f"\n  site energies: max|abs| = {err:.3e}  (scale {scale:.4f}, rel {err/scale:.2e})")
    assert err < TOL, f"site energy error {err:.3e}"


def test_total_energy(case):
    model, meta, z, n_nodes, send, recv, rij, node_z, zi, zj = case
    with highest_precision():
        E = float(jnp.sum(model.site_energies(rij, zi, zj, send, n_nodes, node_z)))
    ref = float(z["test_E"][0])
    print(f"\n  total energy: jax {E:.10f}  julia {ref:.10f}  |d| = {abs(E-ref):.3e}")
    assert abs(E - ref) < TOL, f"energy error {abs(E-ref):.3e}"


def test_forces(case):
    model, meta, z, n_nodes, send, recv, rij, node_z, zi, zj = case

    def total(r):
        return jnp.sum(model.site_energies(r, zi, zj, send, n_nodes, node_z))

    with highest_precision():
        g = jax.grad(total)(rij)                       # dE/drij_e, shape (E,3)
    F = (jnp.zeros((n_nodes, 3), rij.dtype)
         .at[send].add(g).at[recv].add(-g))            # F_a = sum_e dE/dr (d_send - d_recv)
    err = np.max(np.abs(np.asarray(F) - z["test_F"].T))
    scale = np.max(np.abs(z["test_F"]))
    print(f"\n  forces: max|abs| = {err:.3e} eV/A  (scale {scale:.4f}, rel {err/scale:.2e})")
    assert err < TOL, f"force error {err:.3e}"
