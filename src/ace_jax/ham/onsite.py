"""On-site blocks: block(s1, s2) of an atom = sum_L CG_L . (C[s1 s2 L] . B^(L, p)).

One coefficient vector per (species, shell pair s1 <= s2, L, channel) on the
features of parity p = (-1)^(l1+l2), the parity of the block under inversion
(an orbital of angular momentum l has parity (-1)^l).  When (-1)^L != p the
part is a pseudo-tensor (e.g. the antisymmetric L = 1 part between two p
shells).  An on-site operator is symmetric, so block(s2, s1) is the transpose
and a same-shell block has only even L.  The L = 0 part has a constant column:
the block of the isolated atom."""
import equinox as eqx
import jax
import numpy as np

from ..eval.nlist import sparse_graph
from . import cg
from .features import SiteBasis
from .orbitals import OrbitalBasis


def parts_of(l1, l2, same_shell):
    """[(L, p)] the block between shells l1, l2 has; p = (-1)^(l1+l2)."""
    p = 1 if (l1 + l2) % 2 == 0 else -1
    return [(L, p) for L in cg.Ls(l1, l2) if not (same_shell and L % 2)]


def shell_pairs(orbitals, z):
    """[(s1, s2, l1, l2)], s1 <= s2, of species z."""
    ls = orbitals.ls(z)
    return [(a, b, ls[a], ls[b]) for a in range(len(ls)) for b in range(a, len(ls))]


def site_graph(basis, positions, cell, numbers, pbc=True):
    """(rij, zi, zj, centre index per edge, n_atoms) for `SiteBasis.features`: every
    neighbour j of atom i within the basis cutoff, as the vector from i to j."""
    g = sparse_graph(np.asarray(positions, float), np.asarray(cell, float), pbc, basis.rcut)
    sp = basis.species_index(numbers)
    return g.rij, sp[g.senders], sp[g.receivers], g.senders, len(sp)


class OnsiteModel(eqx.Module):
    """coef[(z, s1, s2, L)]: (n_channel, n_B(L, p) [+ 1 for L = 0, the constant])."""

    basis: SiteBasis
    coef: dict
    orbitals: OrbitalBasis = eqx.field(static=True)
    channels: tuple = eqx.field(static=True)

    def site_features(self, positions, cell, numbers, pbc=True):
        rij, zi, zj, seg, n = site_graph(self.basis, positions, cell, numbers, pbc)
        if len(rij) == 0:                                    # no neighbours anywhere: all features vanish
            return {(L, p): np.zeros((n, self.basis.n(L, p), 2 * L + 1)) for L, p in self.basis.lps}
        return {k: np.asarray(v) for k, v in _features(self.basis, rij, zi, zj, seg, n).items()}

    def predict(self, positions, cell, numbers, pbc=True):
        """{channel: {(0, 0, 0, i, i): (n_orb, n_orb)}} for every atom."""
        numbers = np.asarray(numbers)
        for z in set(int(z) for z in numbers):
            self.orbitals.ls(z)                              # KeyError for a species without orbitals
        F = self.site_features(positions, cell, numbers, pbc)
        out = {c: {} for c in self.channels}
        for z in sorted(set(int(z) for z in numbers)):
            atoms = np.flatnonzero(numbers == z)
            full = np.zeros((len(atoms), len(self.channels), self.orbitals.n_orb(z), self.orbitals.n_orb(z)))
            for s1, s2, l1, l2 in shell_pairs(self.orbitals, z):
                blk = 0.0
                for L, p in parts_of(l1, l2, s1 == s2):
                    part = part_values(self.coef.get((z, s1, s2, L)), F.get((L, p)), atoms, L, len(self.channels))
                    blk = blk + np.einsum("acM,ijM->acij", part, cg.real_cg(l1, l2, L))
                a, b = self.orbitals.shell_slice(z, s1), self.orbitals.shell_slice(z, s2)
                full[:, :, a, b] = blk
                if s1 != s2:
                    full[:, :, b, a] = np.swapaxes(blk, -1, -2)
            for k, i in enumerate(atoms):
                for c, name in enumerate(self.channels):
                    out[name][(0, 0, 0, int(i), int(i))] = full[k, c]
        return out


def _features_impl(basis, rij, zi, zj, seg, n):
    return basis.features(rij, zi, zj, seg, n)


_features = jax.jit(_features_impl, static_argnums=5)      # the atom count sizes the pooling


def design(F, atoms, L):
    """(n_atoms, 2L+1, n_B [+ 1]) regression rows of an L part: the features, and for
    L = 0 a constant column.  `F` is the (n, n_B, 2L+1) feature array or None (no feature)."""
    cols = np.zeros((len(atoms), 2 * L + 1, 0)) if F is None else np.swapaxes(F[atoms], 1, 2)
    if L == 0:
        cols = np.concatenate([cols, np.ones((len(atoms), 1, 1))], axis=2)
    return cols


def part_values(coef, F, atoms, L, n_channel):
    """(n_atoms, n_channel, 2L+1): the L part predicted for each atom and channel
    (zero when the model has no coefficients for it)."""
    if coef is None:
        return np.zeros((len(atoms), n_channel, 2 * L + 1))
    return np.einsum("amk,ck->acm", design(F, atoms, L), np.asarray(coef))
