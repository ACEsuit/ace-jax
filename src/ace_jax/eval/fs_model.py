"""A linear ACE model plus a frozen Finnis-Sinclair sqrt-density term.

    E_i = c_z . X_i + sum_p d[p, z] ssqrt(eta[p, z] . (mask * X_i)),
    ssqrt(rho) = rho (rho^2 + eps)^(-1/4),

X_i the compact site basis [B | Apair] of `base`.  eta (the density weights,
learned by fit.radial_density and then frozen) and d (their readout, fitted with
c) are data; forces and virial come from EdgeSiteModel's one value_and_grad over
the edge vectors, so there is no bespoke derivative code.  The linear part is
evaluated through `base._readout` on the materialised B (not the folded
readout), since the density needs B anyway.  Written by
construct.export.patch_radial_npz(fs=...), loaded by eval.io.load.
See docs/specs/2026-09-28-radial-density-varpro-design.md.
"""
import dataclasses

import equinox as eqx
import jax.numpy as jnp

from .edge_model import EdgeSiteModel


def ssqrt(rho, eps):
    return rho * (rho * rho + eps) ** -0.25


class FSModel(EdgeSiteModel):
    base: EdgeSiteModel
    eta: jnp.ndarray            # (P, NZ, D), zero off the mask
    d: jnp.ndarray              # (P, NZ)
    mask: jnp.ndarray           # (D,) 0/1
    eps: float = eqx.field(static=True, default=1e-6)

    def _energies(self, B, Apair, node_z):
        X = jnp.concatenate([B, Apair], axis=1) * self.mask
        rho = jnp.einsum("nd,pnd->np", X, self.eta[:, node_z])
        fs = jnp.sum(ssqrt(rho, self.eps) * self.d[:, node_z].T, axis=1)
        return self.base._readout(B, Apair, node_z) + fs

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        return self._energies(*self.base.site_basis(rij, zi, zj, segment_ids, n_nodes, mask), node_z)

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        return self._energies(*self.base.site_basis_dense(rij, zi, zj, mask), node_z)

    def with_base(self, base):
        return dataclasses.replace(self, base=base)

    # the base's geometry, A-basis form and descriptors, for the calculators
    def pad_cutoff(self):
        return self.base.pad_cutoff()

    def edge_a_widths(self):
        return self.base.edge_a_widths()

    def edge_a_factors(self, *args, **kw):
        return self.base.edge_a_factors(*args, **kw)

    def site_basis(self, *args, **kw):
        return self.base.site_basis(*args, **kw)

    def site_basis_dense(self, *args, **kw):
        return self.base.site_basis_dense(*args, **kw)

    def compact_basis(self, *args, **kw):
        return self.base.compact_basis(*args, **kw)

    def site_descriptors(self, *args, **kw):
        return self.base.site_descriptors(*args, **kw)

    a_channels = property(lambda self: self.base.a_channels)
    aspec_r = property(lambda self: self.base.aspec_r)
    aspec_y = property(lambda self: self.base.aspec_y)
    edge_a_kind = property(lambda self: self.base.edge_a_kind)
    a_sel_r = property(lambda self: self.base.a_sel_r)
    a_sel_y = property(lambda self: self.base.a_sel_y)
    E0 = property(lambda self: self.base.E0)
    elements = property(lambda self: self.base.elements)
