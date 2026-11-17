"""Off-site blocks from the bond features of `bond`: block(s1, s2) of the bond i -> j is
sum_L CG_L . (C[zi zj s1 s2 L] . Phi^(L, p)), p = (-1)^(l1+l2).

Only canonical blocks carry coefficients: (z_i, s1) <= (z_j, s2).  The others are the
transpose of the reverse bond's canonical block, so H_ji(-R) = H_ij(R)^T holds exactly;
between equal shells of equal species both directions are canonical and the symmetrised
design (`bond.design(same=True)`) makes them consistent."""
import equinox as eqx
import numpy as np

from . import bond as B
from . import cg
from .onsite import _features, site_graph
from .orbitals import OrbitalBasis


def canonical_pairs(orbitals, zi, zj):
    """[(s1, s2, l1, l2, same)] of the canonical blocks of a bond between species zi and zj
    (none when zi > zj: that bond is the transpose of its reverse)."""
    if zi > zj:
        return []
    li, lj = orbitals.ls(zi), orbitals.ls(zj)
    return [(a, b, li[a], lj[b], zi == zj and a == b) for a in range(len(li)) for b in range(len(lj))
            if zi < zj or a <= b]


def block_lps(orbitals):
    """Every (L, p) some off-site block has."""
    out = set()
    for zi in orbitals.species:
        for zj in orbitals.species:
            for _, _, l1, l2, _ in canonical_pairs(orbitals, zi, zj):
                out |= {(L, 1 if (l1 + l2) % 2 == 0 else -1) for L in cg.Ls(l1, l2)}
    return sorted(out)


def site_features(basis, positions, cell, numbers, pbc=True):
    rij, zi, zj, seg, n = site_graph(basis.site, positions, cell, numbers, pbc)
    return {k: np.asarray(v) for k, v in _features(basis.site, rij, zi, zj, seg, n).items()}


class OffsiteModel(eqx.Module):
    """coef[(zi, zj, s1, s2, L)]: (n_channel, n_feat) on `bond.design` of that block."""

    bond: B.BondBasis
    coef: dict
    orbitals: OrbitalBasis = eqx.field(static=True)
    channels: tuple = eqx.field(static=True)

    def predict(self, positions, cell, numbers, pbc=True, chunk=4096, only=None):
        """{channel: {(Rx, Ry, Rz, i, j): (n_orb_i, n_orb_j)}} for every pair within the bond cutoff.
        `only`: a function of the bond graph giving the bond indices wanted (default all); their
        reverse bonds are evaluated too, since a block takes its non-canonical part from them."""
        numbers = np.asarray(numbers)
        orb, nc = self.orbitals, len(self.channels)
        F = site_features(self.bond, positions, cell, numbers, pbc)
        g = B.bond_graph(self.bond, positions, cell, numbers, pbc)
        env = B.env_lists(self.bond, positions, cell, pbc)
        lps = block_lps(orb)
        zi, zj = numbers[g.i], numbers[g.j]
        want = np.arange(len(g.i)) if only is None else np.asarray(only(g), np.int64)
        need = np.zeros(len(g.i), bool)
        need[want] = need[g.reverse[want]] = True
        blocks = [None] * len(g.i)                       # per bond: (n_channel, n_i, n_j), canonical parts filled
        for za in orb.species:
            for zb in orb.species:
                pairs = canonical_pairs(orb, za, zb)
                sel_all = np.flatnonzero((zi == za) & (zj == zb) & need)
                na, nb = orb.n_orb(za), orb.n_orb(zb)
                for lo in range(0, len(sel_all), chunk):
                    sel = sel_all[lo:lo + chunk]
                    full = np.zeros((len(sel), nc, na, nb))
                    if pairs:
                        f = B.features(self.bond, F, g, sel, lps, env=env)
                        for s1, s2, l1, l2, same in pairs:
                            p = 1 if (l1 + l2) % 2 == 0 else -1
                            blk = 0.0
                            for L in cg.Ls(l1, l2):
                                c = self.coef.get((za, zb, s1, s2, L))
                                if c is None:
                                    continue
                                part = np.einsum("ekm,ck->ecm", B.design(f[(L, p)], L, same), np.asarray(c))
                                blk = blk + np.einsum("ecm,ijm->ecij", part, cg.real_cg(l1, l2, L))
                            full[:, :, orb.shell_slice(za, s1), orb.shell_slice(zb, s2)] = blk
                    for k, e in enumerate(sel):
                        blocks[e] = full[k]
        # the non-canonical parts: transpose of the reverse bond's canonical block
        out = {c: {} for c in self.channels}
        for e in want:
            za, zb = int(zi[e]), int(zj[e])
            blk = blocks[e].copy()
            rev = blocks[g.reverse[e]]
            if za > zb:
                blk = np.swapaxes(rev, -1, -2)
            elif za == zb:
                for s1 in range(len(orb.ls(za))):
                    for s2 in range(s1):                 # (s1 > s2) of this bond = (s2, s1) of the reverse, transposed
                        a, b = orb.shell_slice(za, s1), orb.shell_slice(za, s2)
                        blk[:, a, b] = np.swapaxes(rev[:, b, a], -1, -2)
            key = (int(g.R[e, 0]), int(g.R[e, 1]), int(g.R[e, 2]), int(g.i[e]), int(g.j[e]))
            for c, name in enumerate(self.channels):
                out[name][key] = blk[c]
        return out
