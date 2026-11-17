"""Equivariant site features: the protocol an operator model is written against, and the one-particle
basis of a linear ACE without a model.

A site feature provider (`SiteFeatures`) gives, per atom, features of every symmetry (L, parity) it has,
{(L, p): (n_nodes, n(L, p), 2L+1)}, parity p = (-1)^(sum of l).  Where they come from is the provider's
business: a linear ACE with categorical species or an element embedding (`pooled_a` here, then the
product basis and a coupling), or another architecture's chain.
docs/dev/specs/2026-11-17-package-structure-addendum-feature-providers.md."""
from typing import Protocol, runtime_checkable

from .harmonics import real_solid_harmonics, real_spherical_harmonics
from .product import edge_a, pool_sparse


@runtime_checkable
class SiteFeatures(Protocol):
    lps: tuple                    # the (L, parity) present
    rcut: float

    def n(self, L, p):
        """Number of features of symmetry (L, p); 0 if the provider has none."""

    def species_index(self, numbers):
        """Species indices (the zi, zj of `features`) of atomic numbers."""

    def features(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """{(L, p): (n_nodes, n(L, p), 2L+1)} from edge vectors rij (E, 3), the species indices zi, zj (E,)
        of each edge's centre and neighbour, and segment_ids (E,) its centre atom."""


def harmonics(rij, lmax, solid=True):
    """Real solid (r^l Y_lm) or spherical harmonics up to lmax of edge vectors rij (E, 3): (E, (lmax+1)^2)."""
    return real_solid_harmonics(rij, lmax) if solid else real_spherical_harmonics(rij, lmax)


def edge_a_rows(radial, rij, zi, zj, aspec_r, aspec_y, lmax, solid=True):
    """Per-edge one-particle basis rows A_e = R_nl(r) Y_lm(rij) for the (n, l, m) of aspec_r / aspec_y
    (columns of the radial and of the harmonics): (E, n_A).  radial: a `radial.TensorRadial`, or any
    callable (r, zi, zj) -> (E, n_rnl)."""
    import jax.numpy as jnp
    return edge_a(radial(jnp.linalg.norm(rij, axis=-1), zi, zj), harmonics(rij, lmax, solid), aspec_r, aspec_y)


def pooled_a(radial, rij, zi, zj, segment_ids, n_nodes, aspec_r, aspec_y, lmax, solid=True, mask=None):
    """The pooled one-particle basis A (n_nodes, n_A) = sum over each atom's edges of `edge_a_rows`: what
    `product.product_basis` multiplies.  Equal to an ACEModel's own A for its radial and specs."""
    return pool_sparse(edge_a_rows(radial, rij, zi, zj, aspec_r, aspec_y, lmax, solid), segment_ids, n_nodes, mask)
