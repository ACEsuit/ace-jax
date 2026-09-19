"""One-point environment summary s(B): the power-mean nearest-neighbour
distance.  It is the coordinate of the non-stationary amplitude delta(s) and
must be smooth in the positions -- a hard min has a kink whenever two
neighbours swap rank, which delta'(s) ds/dr would put into the forces.

    s_i = r0 * [ sum_j f_c(r_ij) (r0 / r_ij)^p ]^(-1/p)

p -> inf recovers min_j r_ij. The cutoff factor biases s by
f_c(r_nn)^(-1/p) (about 5% at p = 10 for r_nn/rcut ~ 0.46); it is a smooth
monotone distortion that the location hyperparameter r0 of delta(s) absorbs.
An atom crossing rcut enters continuously via f_c; an isolated atom
(sum -> 0) gives s -> inf, so delta(s) -> 0."""
import jax
import jax.numpy as jnp

from ..eval import pool_sparse


# A node with no live edges (padding, or a genuinely isolated atom) has S = 0.
# The floor keeps s finite (~1e3 * r0) so delta(s) underflows to exactly 0 with
# a zero gradient, instead of inf/NaN leaking into the sufficient statistics.
_FLOOR = 1e-30


def cutoff_fn(r, rcut):
    x = jnp.clip(r / rcut, 0.0, 1.0)
    return (1.0 - x * x) ** 2


def _edge_terms(rij, r0, rcut, p, mask):
    # Guard the *input* to norm, not just its value: jnp.linalg.norm has a
    # 0/0 (NaN) gradient at rij=0, and multiplying that NaN local Jacobian by
    # a zero upstream cotangent from a value-level jnp.where(mask, r, ...)
    # still yields NaN (0 * NaN = NaN). Substituting a safe rij before norm
    # keeps the whole computation -- value and gradient -- finite for masked
    # edges; the term is zeroed afterwards regardless by jnp.where(mask, t, 0.0).
    if mask is not None:
        rij = jnp.where(mask[:, None], rij, r0)
    r = jnp.linalg.norm(rij, axis=-1)
    t = cutoff_fn(r, rcut) * (r0 / r) ** p
    return t if mask is None else jnp.where(mask, t, 0.0)


def site_summary(rij, segment_ids, n_nodes, r0, rcut, p=10, mask=None):
    t = _edge_terms(rij, r0, rcut, p, mask)                     # (E,)
    S = pool_sparse(t[:, None], segment_ids, n_nodes)[:, 0] + _FLOOR
    return r0 * S ** (-1.0 / p)


def summary_edge_jacobian(rij, segment_ids, n_nodes, r0, rcut, p=10, mask=None):
    """s and ds[segment_ids[e]]/d rij[e].  Edge-local: s_i depends on rij[e]
    only through t_e, so the Jacobian is (ds_i/dS_i) * (dt_e/d rij[e])."""
    t = _edge_terms(rij, r0, rcut, p, mask)
    S = pool_sparse(t[:, None], segment_ids, n_nodes)[:, 0] + _FLOOR
    s = r0 * S ** (-1.0 / p)
    dsdS = -(r0 / p) * S ** (-1.0 / p - 1.0)                    # (n_nodes,)
    one = lambda r1, m1: _edge_terms(r1[None, :], r0, rcut, p,
                                     None if mask is None else m1[None])[0]
    dt = jax.vmap(jax.grad(one))(rij, mask if mask is not None else jnp.ones(len(rij), bool))
    return s, dsdS[segment_ids][:, None] * dt
