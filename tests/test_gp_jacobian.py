"""edge_jacobian (hybrid analytic push) against
a naive jacrev of the compact site basis, and force rows built from it against
-d/dr of the summed basis.  f64 only."""
import jax
import pytest
pytestmark = pytest.mark.heavy
import numpy as np

from conftest import species_index

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load


def _case(npz):
    model, meta, z = load(npz, fold=False)
    n = int(z["test_pos"].shape[1])
    send = jnp.asarray(z["test_edge_i"], jnp.int32)
    recv = jnp.asarray(z["test_edge_j"], jnp.int32)
    nz = species_index(z)
    rij = jnp.asarray(z["test_edge_rij"].T)
    return model, z, n, send, recv, nz, rij


def test_edge_jacobian_matches_naive(npz):
    model, z, n, send, recv, nz, rij = _case(npz)
    zi, zj = nz[send], nz[recv]
    with highest_precision():
        X, J = model.edge_jacobian(rij, zi, zj, send, n)
        X0 = model.compact_basis(rij, zi, zj, send, n)
        Jfull = jax.jacrev(lambda r: model.compact_basis(r, zi, zj, send, n))(rij)
    Jn = Jfull[send, :, jnp.arange(len(send)), :]          # (E, D, 3)
    assert X.shape == X0.shape and J.shape == (len(send), X.shape[1], 3)
    assert float(jnp.abs(X - X0).max()) < 1e-11
    scale = float(jnp.abs(Jn).max())
    assert float(jnp.abs(J - Jn).max()) < 1e-11 * max(scale, 1.0)


def test_force_rows_from_edge_jacobian(npz):
    """F_k(alpha) = -d/dr_alpha sum_i X_ik  ==  -(scatter_recv(J) - scatter_send(J))."""
    model, z, n, send, recv, nz, rij = _case(npz)
    zi, zj = nz[send], nz[recv]
    pos = jnp.asarray(z["test_pos"].T)
    shifts = rij - (pos[recv] - pos[send])

    def esum(p):
        r = p[recv] - p[send] + shifts
        return model.compact_basis(r, zi, zj, send, n).sum(0)   # (D,)

    with highest_precision():
        G = jax.jacrev(esum)(pos)                                 # (D, n, 3)
        _, J = model.edge_jacobian(rij, zi, zj, send, n)
    Frow = -(jax.ops.segment_sum(J, recv, n) - jax.ops.segment_sum(J, send, n))  # (n, D, 3)
    assert float(jnp.abs(-jnp.transpose(G, (1, 0, 2)) - Frow).max()) < 1e-10
