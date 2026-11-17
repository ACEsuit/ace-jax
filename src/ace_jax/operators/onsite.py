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
from ..basis import cg
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


# ------------------------------------------------------------------ fast readout
class Readout(eqx.Module):
    """The coefficients of an `OnsiteModel` as dense tensors, per species z (in
    `species` order): for each (L, p), W (n_channel, Q, n_B) over the Q shell pairs
    that have that part, and the CG tensors placed in the atom's orbital block
    (both triangles) as sparse entries: value `Gv` at flat (shell pair, m) index
    `Gq` and flat orbital-pair index `Gij` (a dense (Q, 2L+1, n_orb, n_orb) is
    almost all zeros); `const` (n_channel, n_orb, n_orb), the isolated-atom block.
    block = const + sum_(L,p) scatter((W . B^(L,p))[Gq] Gv -> Gij): one contraction
    and one scatter per symmetry for all shell pairs and channels at once."""

    W: tuple
    Gq: tuple
    Gij: tuple
    Gv: tuple
    const: tuple
    species: tuple = eqx.field(static=True)
    lps: tuple = eqx.field(static=True)          # per species: the (L, p) of W / G

    def blocks(self, F, k, atoms):
        """(n_atoms, n_channel, n_orb, n_orb) for the atoms (indices into F) of species k."""
        jnp = jax.numpy
        const = self.const[k]
        nc, n = const.shape[0], const.shape[-1]
        flat = jnp.zeros((n * n, atoms.shape[0], nc), const.dtype)
        for lp, W, q, ij, v in zip(self.lps[k], self.W[k], self.Gq[k], self.Gij[k], self.Gv[k]):
            P = jnp.einsum("cqb,abm->qmac", W, F[lp][atoms])                 # (Q, 2L+1, n_atoms, n_channel)
            P = P.reshape(P.shape[0] * P.shape[1], *P.shape[2:])
            flat = flat + jax.ops.segment_sum(P[q] * v[:, None, None], ij, num_segments=n * n)
        return const[None] + jnp.moveaxis(flat, 0, -1).reshape(atoms.shape[0], nc, n, n)


def compile_readout(model):
    """`Readout` of an `OnsiteModel` (numpy work, once per model)."""
    orb, nc = model.orbitals, len(model.channels)
    Ws, Gq, Gij, Gv, consts, lps_all = [], [], [], [], [], []
    for z in orb.species:
        n = orb.n_orb(z)
        const = np.zeros((nc, n, n))
        by_lp = {}
        for s1, s2, l1, l2 in shell_pairs(orb, z):
            a, b = orb.shell_slice(z, s1), orb.shell_slice(z, s2)
            for L, p in parts_of(l1, l2, s1 == s2):
                c = model.coef.get((z, s1, s2, L))
                if c is None:
                    continue
                c = np.asarray(c)
                T = cg.real_cg(l1, l2, L)                                  # (i, j, M)
                g = np.zeros((2 * L + 1, n, n))
                g[:, a, b] = np.moveaxis(T, 2, 0)
                if s1 != s2:
                    g[:, b, a] = np.moveaxis(T, 2, 0).transpose(0, 2, 1)
                nB = model.basis.n(L, p)
                if L == 0:
                    const += c[:, -1][:, None, None] * g[0]
                if nB:
                    by_lp.setdefault((L, p), []).append((c[:, :nB], g))
        lps = tuple(sorted(by_lp))
        Ws.append(tuple(jax.numpy.asarray(np.stack([w for w, _ in by_lp[lp]], axis=1)) for lp in lps))
        dense = [np.stack([g for _, g in by_lp[lp]]).reshape(-1, n * n) for lp in lps]      # (Q (2L+1), n n)
        nz = [np.nonzero(d) for d in dense]
        Gq.append(tuple(jax.numpy.asarray(r, jax.numpy.int32) for r, _ in nz))
        Gij.append(tuple(jax.numpy.asarray(c, jax.numpy.int32) for _, c in nz))
        Gv.append(tuple(jax.numpy.asarray(d[r, c]) for d, (r, c) in zip(dense, nz)))
        consts.append(jax.numpy.asarray(const))
        lps_all.append(lps)
    return Readout(W=tuple(Ws), Gq=tuple(Gq), Gij=tuple(Gij), Gv=tuple(Gv), const=tuple(consts),
                   species=orb.species, lps=tuple(lps_all))


@jax.jit
def _readout(readout, F, atoms):
    return tuple(readout.blocks(F, k, a) for k, a in enumerate(atoms))


def _dense(basis, readout, rij, zi, zj, seg, atoms, n):
    """Features, then readout, as two compiled programs (the feature program is shared with
    `OnsiteModel.site_features` and the fit).  Open: on a 32-thread CPU the chain runs at
    150-200 us per atom although its stages take 42 and 5 us when timed alone, and 70 us on
    one thread; fusing the two into one program did not change it.  docs/dev/ham-results.md."""
    return _readout(readout, _features(basis, rij, zi, zj, seg, n), atoms)


class OnsitePredictor:
    """`OnsiteModel.predict` as one compiled program: neighbour list on the host,
    then features and readout on the device.  `dense` returns, per species of the
    orbital basis, the atom indices and their (n_atoms, n_channel, n_orb, n_orb) blocks."""

    def __init__(self, model):
        self.model, self.readout = model, compile_readout(model)

    def dense(self, positions, cell, numbers, pbc=True):
        numbers = np.asarray(numbers)
        m = self.model
        for z in set(int(z) for z in numbers):
            m.orbitals.ls(z)
        rij, zi, zj, seg, n = site_graph(m.basis, positions, cell, numbers, pbc)
        atoms = tuple(np.flatnonzero(numbers == z).astype(np.int32) for z in m.orbitals.species)
        if len(rij) == 0:                                    # no neighbour: one masked edge keeps the shapes
            z1 = np.zeros(1, np.int32)                       # at the cutoff, where every radial vanishes
            rij, zi, zj, seg = np.array([[m.basis.rcut, 0.0, 0.0]]), z1, z1, z1
        return atoms, _dense(m.basis, self.readout, rij, zi, zj, seg, atoms, n)

    def __call__(self, positions, cell, numbers, pbc=True):
        """As `OnsiteModel.predict`."""
        atoms, blocks = self.dense(positions, cell, numbers, pbc)
        out = {c: {} for c in self.model.channels}
        for a, b in zip(atoms, blocks):
            b = np.asarray(b)
            for k, i in enumerate(a):
                for c, name in enumerate(self.model.channels):
                    out[name][(0, 0, 0, int(i), int(i))] = b[k, c]
        return out
