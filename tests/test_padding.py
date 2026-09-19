"""Padded edges must not perturb results, and must not NaN the gradient.

The plan flags this: r = |rij| is non-differentiable at 0 and the Agnesi
transform divides, so padding with a zero vector NaNs the gradient silently,
only under grad.  lammps-jax's nequip template pads at the cutoff instead,
where the envelope vanishes and the derivative stays defined; that is the
convention adopted here.
"""
import pathlib

import jax
import numpy as np
import pytest

from conftest import species_index

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load




@pytest.fixture
def case(npz):
    return load(npz)


def _padded(model, z, n_pad, pad_vec):
    """Append n_pad masked edges carrying pad_vec, all pointing at node 0."""
    n_nodes = int(z["test_pos"].shape[1])
    send = np.asarray(z["test_edge_i"], np.int32)
    recv = np.asarray(z["test_edge_j"], np.int32)
    rij = np.asarray(z["test_edge_rij"].T)
    E = len(send)
    send = jnp.asarray(np.concatenate([send, np.zeros(n_pad, np.int32)]))
    recv = jnp.asarray(np.concatenate([recv, np.zeros(n_pad, np.int32)]))
    rij = jnp.asarray(np.concatenate([rij, np.tile(pad_vec, (n_pad, 1))]))
    mask = jnp.concatenate([jnp.ones(E, bool), jnp.zeros(n_pad, bool)])
    node_z = species_index(z)
    return rij, send, recv, mask, node_z, n_nodes


def test_padding_does_not_change_energy(case):
    model, meta, z = case
    rcut = float(meta["rcut"])
    rij, send, recv, mask, node_z, n_nodes = _padded(model, z, 500, [rcut, 0.0, 0.0])
    zi, zj = node_z[send], node_z[recv]
    with highest_precision():
        E = float(jnp.sum(model.site_energies(rij, zi, zj, send, n_nodes, node_z, mask)))
    ref = float(z["test_E"][0])
    print(f"\n  padded energy: {E:.10f} vs {ref:.10f}  |d| = {abs(E-ref):.3e}")
    assert abs(E - ref) < 1e-10


def test_padded_gradient_is_finite(case):
    """The real hazard: masking the energy is not enough, the gradient must be
    finite too.  A zero pad vector would NaN here even though the energy is right."""
    model, meta, z = case
    rcut = float(meta["rcut"])
    rij, send, recv, mask, node_z, n_nodes = _padded(model, z, 500, [rcut, 0.0, 0.0])
    zi, zj = node_z[send], node_z[recv]

    def total(r):
        return jnp.sum(model.site_energies(r, zi, zj, send, n_nodes, node_z, mask))

    with highest_precision():
        g = np.asarray(jax.grad(total)(rij))
    n_bad = int((~np.isfinite(g)).sum())
    print(f"  non-finite gradient entries with pad at rcut: {n_bad}")
    assert n_bad == 0, f"{n_bad} non-finite gradient entries"
    assert np.max(np.abs(g[len(z['test_edge_i']):])) == 0.0, "padding carries gradient"


def test_zero_pad_would_nan(case):
    """Documents WHY the cutoff pad is used: a zero pad vector does NaN."""
    model, meta, z = case
    rij, send, recv, mask, node_z, n_nodes = _padded(model, z, 500, [0.0, 0.0, 0.0])
    zi, zj = node_z[send], node_z[recv]

    def total(r):
        return jnp.sum(model.site_energies(r, zi, zj, send, n_nodes, node_z, mask))

    with highest_precision():
        g = np.asarray(jax.grad(total)(rij))
    n_bad = int((~np.isfinite(g)).sum())
    print(f"  non-finite gradient entries with pad at ZERO: {n_bad}")
    # recorded, not asserted either way: this is the trap, not a requirement
    if n_bad == 0:
        pytest.skip("zero pad happens not to NaN for this model; cutoff pad still used")


def test_dense_padded_slots_are_parked_at_cutoff(case):
    """`neighbour_matrix` leaves unused slots as zero vectors, which would NaN
    the gradient exactly as the sparse zero pad does.  dense_graph must park them
    at the cutoff so a caller cannot get this wrong by forgetting."""
    model, meta, z = case
    from ace_jax.eval import dense_graph
    rcut = float(meta["rcut"])
    # This test is ABOUT the padded slots, so the capacity needs headroom above
    # the real neighbour count -- enough that padding exists by construction.
    # A hardcoded 64 gave neither guarantee: too small for a dense system (it
    # truncates silently) and, where every atom has the same neighbour count,
    # exactly-max leaves no padding at all and the vacuity guard below fires.
    kmax = int(np.bincount(np.asarray(z["test_edge_i"])).max()) + 8
    g = dense_graph(np.asarray(z["test_pos"]).T, np.asarray(z["test_cell"]).T,
                    np.asarray(z["test_pbc"]).astype(bool), rcut, kmax)
    dead = ~g.mask
    assert dead.any(), "no padded slots in this fixture; test is vacuous"
    lengths = np.linalg.norm(g.rij[dead], axis=-1)
    assert np.allclose(lengths, rcut), "padded slots not parked at the cutoff"

    n, K = g.idx.shape
    node_z = jnp.zeros(g.n_nodes, jnp.int32)
    zi = jnp.zeros((n, K), jnp.int32)
    mask = jnp.asarray(g.mask)

    def total(r):
        return jnp.sum(model.site_energies_dense(r, zi, zi, mask, node_z))

    with highest_precision():
        grad = np.asarray(jax.grad(total)(jnp.asarray(g.rij)))
    n_bad = int((~np.isfinite(grad)).sum())
    print(f"  non-finite dense gradient entries: {n_bad}")
    assert n_bad == 0
