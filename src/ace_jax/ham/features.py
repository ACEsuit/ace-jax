"""Equivariant site features B^(L, p) for all L from ONE A / AA evaluation.

A carrier `ACEModel` (from `basis.model.build_model`) supplies the radials, the
harmonics and the A / AA product; its `aspec` and `aa_specs` are replaced by the
union of an `EquivariantSet`, so one AA vector per atom serves every (L, p):
B^(L, p)[b, m] = sum_k A2B[b, m, k] AA_k, a sparse contraction each.  Every
shell pair and every operator channel then reads the same features, which is
what keeps the cost of a structure independent of the number of orbitals."""
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..basis.coupling import couple_equivariant_set
from ..basis.model import build_model
from ..basis.spec import build_spec
from ..eval.model import ACEModel, pool_sparse


class SiteBasis(eqx.Module):
    """carrier: radials + A / AA product (its own readout is unused); lps: the
    (L, parity) present; per (L, parity) the COO coupling as flat-index arrays."""

    carrier: ACEModel
    flat: tuple                 # per lp: row * (2L+1) + m, int32 (nnz,)
    cols: tuple                 # per lp: AA column, int32 (nnz,)
    vals: tuple                 # per lp: float64 (nnz,)
    lps: tuple = eqx.field(static=True)
    n_B: tuple = eqx.field(static=True)
    elements: tuple = eqx.field(static=True)
    rcut: float = eqx.field(static=True)

    def n(self, L, p):
        """Number of features of symmetry (L, p); 0 if the basis has none."""
        return self.n_B[self.lps.index((L, p))] if (L, p) in self.lps else 0

    def species_index(self, numbers):
        idx = {z: k for k, z in enumerate(self.elements)}
        try:
            return np.asarray([idx[int(z)] for z in numbers], np.int32)
        except KeyError as e:
            raise KeyError(f"the site basis has no element Z = {e.args[0]} (it has {self.elements})") from None

    def aa(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """(n_nodes, n_AA): the pooled A basis and its products, once."""
        edge_A, _ = self.carrier.edge_features(rij, zi, zj)
        return self.carrier._aa(pool_sparse(edge_A, segment_ids, n_nodes, mask))

    def from_aa(self, AA):
        """{(L, p): (n, n_B, 2L+1)} from an AA matrix (n, n_AA)."""
        out = {}
        for (L, p), nB, flat, cols, vals in zip(self.lps, self.n_B, self.flat, self.cols, self.vals):
            k = 2 * L + 1
            B = jax.ops.segment_sum((AA[:, cols] * vals).T, flat, num_segments=nB * k)
            out[(L, p)] = B.T.reshape(AA.shape[0], nB, k)
        return out

    def features(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """{(L, p): (n_nodes, n_B, 2L+1)}; zi, zj are species indices (`species_index`), one per edge."""
        return self.from_aa(self.aa(rij, zi, zj, segment_ids, n_nodes, mask))


def build_site_basis(elements, order, totaldegree, lmax_feat, *, rcut, wL=1.5, coupling_cache_dir=None,
                     **radial):
    """A `SiteBasis` on the TotalDegree basis of `build_model` (same radials and
    one-particle basis), with every (L, +-1), L <= lmax_feat, that has a
    function.  lmax_feat = 2 * (largest orbital l).  `radial`: `build_model`'s
    r0, rin, radial_mode, seed, n_q_factor, rnl_basis."""
    base = build_model(elements, order, totaldegree, wL=wL, rcut=rcut, with_gamma=False,
                       coupling_cache_dir=coupling_cache_dir, **radial)
    NZ = len(base.model.elements)
    mb = build_spec(NZ, order, totaldegree, wL, admit=lambda bb: True)[0]
    lps = [(L, p) for L in range(lmax_feat + 1) for p in (1, -1)]
    es = couple_equivariant_set(lps, mb, list(base.Rnl_spec), list(base.Ylm_spec), cache_dir=coupling_cache_dir)
    lps = [lp for lp in lps if es.A2B[lp].shape[0]]
    carrier = dataclasses.replace(
        base.model,
        aspec_r=jnp.asarray([a[0] for a in es.aspec], jnp.int32),
        aspec_y=jnp.asarray([a[1] for a in es.aspec], jnp.int32),
        aa_specs=tuple(jnp.asarray(g, jnp.int32) for g in es.aa_specs))
    k = lambda lp: 2 * lp[0] + 1
    return SiteBasis(
        carrier=carrier,
        flat=tuple(jnp.asarray(es.A2B[lp].rows * k(lp) + es.A2B[lp].ms, jnp.int32) for lp in lps),
        cols=tuple(jnp.asarray(es.A2B[lp].cols, jnp.int32) for lp in lps),
        vals=tuple(jnp.asarray(es.A2B[lp].vals, jnp.float64) for lp in lps),
        lps=tuple(lps), n_B=tuple(int(es.A2B[lp].shape[0]) for lp in lps),
        elements=tuple(int(z) for z in base.model.elements), rcut=float(rcut))
