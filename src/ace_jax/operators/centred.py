"""Equivariant coefficients on atoms: c[i, s, m] = C[z_i, s] . B^(l_s, p)[i, :, m].

Each atom carries a vector of coefficients of atom-centred functions, shell s of angular
momentum l with 2l+1 components that transform as the real harmonics Y_lm of `eval.harmonics`
(index m + l) and so have the parity p = (-1)^l.  One coefficient vector per (species, shell,
channel) on the site features of (l, p): the on-site block model (`onsite`) with a vector per
shell in place of a block per shell pair, on the same features.  The l = 0 shells have a
constant column: the coefficient of the isolated atom.  Nothing here knows what the functions
expand (an electron density in `ham.density`; a spin density is a second channel).

Layout of a structure's coefficients, everywhere (data, predictions, the fit's overlap):
one vector (n_total,) per channel, the atoms in order, within an atom the shells in the order
of the `OrbitalBasis`, within a shell m = -l..l: `offsets(shells, numbers)` gives the start of
each atom.  It is the order of one atom's orbitals in an `OperatorData` block.  `predict`
returns {channel: (n_total,)}; `CentredPredictor.dense` the ragged form per species (atom
indices and (n_atoms, n_channel, n_aux(z))), as `OnsitePredictor.dense`."""
import equinox as eqx
import jax
import numpy as np

from .features import SiteBasis
from .onsite import _features, design, site_graph
from .orbitals import OrbitalBasis
from .precision import highest


def parity(l):
    return 1 if l % 2 == 0 else -1


def offsets(shells, numbers):
    """(n_atoms + 1,): the start of each atom in a structure's coefficient vector, and the total."""
    return np.concatenate([[0], np.cumsum([shells.n_orb(z) for z in numbers])]).astype(np.int64)


def n_columns(basis, l):
    """Columns of a shell of angular momentum l: the features of (l, (-1)^l), and for l = 0 the constant."""
    return basis.n(l, parity(l)) + (l == 0)


def columns(basis, shells):
    """({(z, s): slice}, total): the parameters of every species and shell as one vector, species
    then shell (the columns of the overlap-metric design of `fit.fit_centred`)."""
    out, o = {}, 0
    for z in shells.species:
        for s, l in enumerate(shells.ls(z)):
            out[(z, s)] = slice(o, o + n_columns(basis, l)); o += n_columns(basis, l)
    return out, o


def flat(shells, numbers, atoms, values, channels):
    """{channel: (n_total,)} from the ragged per-species form (`CentredPredictor.dense`)."""
    off = offsets(shells, numbers)
    out = np.zeros((len(channels), off[-1]))
    for a, v in zip(atoms, values):
        v = np.asarray(v)
        for k, i in enumerate(a):
            out[:, off[i]:off[i + 1]] = v[k]
    return {c: out[k] for k, c in enumerate(channels)}


class CentredModel(eqx.Module):
    """coef[(z, s)]: (n_channel, n_B(l, p) [+ 1 for l = 0, the constant]), l the angular momentum
    of shell s of species z and p = (-1)^l; a shell without a key is predicted zero (no feature
    of its symmetry)."""

    basis: SiteBasis
    coef: dict
    shells: OrbitalBasis = eqx.field(static=True)
    channels: tuple = eqx.field(static=True)

    def site_features(self, positions, cell, numbers, pbc=True):
        rij, zi, zj, seg, n = site_graph(self.basis, positions, cell, numbers, pbc)
        if len(rij) == 0:                                    # no neighbours anywhere: all features vanish
            return {(L, p): np.zeros((n, self.basis.n(L, p), 2 * L + 1)) for L, p in self.basis.lps}
        return {k: np.asarray(v) for k, v in _features(self.basis, rij, zi, zj, seg, n).items()}

    def predict(self, positions, cell, numbers, pbc=True):
        """{channel: (n_total,)}: the coefficients of every atom (the module's layout)."""
        numbers = np.asarray(numbers)
        off = offsets(self.shells, numbers)                  # KeyError for a species without shells
        F = self.site_features(positions, cell, numbers, pbc)
        out = np.zeros((len(self.channels), off[-1]))
        for z in sorted(set(int(z) for z in numbers)):
            atoms = np.flatnonzero(numbers == z)
            for s, l in enumerate(self.shells.ls(z)):
                c = self.coef.get((z, s))
                if c is None:
                    continue
                v = np.einsum("amk,ck->cam", design(F.get((l, parity(l))), atoms, l), np.asarray(c))
                sl = self.shells.shell_slice(z, s)
                out[:, (off[atoms] + sl.start)[:, None] + np.arange(2 * l + 1)] = v
        return {c: out[k] for k, c in enumerate(self.channels)}


class Readout(eqx.Module):
    """The coefficients of a `CentredModel` as dense tensors, per species (in `species` order):
    for each l with features, W (n_channel, Q, n_B) over its Q shells and `cols` (Q (2l+1),),
    where component (shell, m) goes in the atom's vector; `const` (n_channel, n_aux), the
    isolated atom's coefficients (the constant columns of the l = 0 shells)."""

    W: tuple
    cols: tuple
    const: tuple
    species: tuple = eqx.field(static=True)
    lps: tuple = eqx.field(static=True)

    @highest
    def values(self, F, k, atoms):
        """(n_atoms, n_channel, n_aux) for the atoms (indices into F) of species k."""
        jnp = jax.numpy
        out = jnp.broadcast_to(self.const[k], (atoms.shape[0],) + self.const[k].shape)
        for lp, W, cols in zip(self.lps[k], self.W[k], self.cols[k]):
            P = jnp.einsum("cqb,abm->acqm", W, F[lp][atoms])
            out = out.at[:, :, cols].add(P.reshape(P.shape[0], P.shape[1], -1))     # on the constants
        return out


def compile_readout(model):
    """`Readout` of a `CentredModel` (numpy work, once per model)."""
    sh, nc = model.shells, len(model.channels)
    Ws, cols, consts, lps_all = [], [], [], []
    for z in sh.species:
        const, by_l = np.zeros((nc, sh.n_orb(z))), {}
        for s, l in enumerate(sh.ls(z)):
            c = model.coef.get((z, s))
            if c is None:
                continue
            c, sl, nB = np.asarray(c), sh.shell_slice(z, s), model.basis.n(l, parity(l))
            if l == 0:
                const[:, sl.start] = c[:, -1]
            if nB:
                by_l.setdefault(l, []).append((c[:, :nB], np.arange(sl.start, sl.stop)))
        ls = tuple(sorted(by_l))
        Ws.append(tuple(jax.numpy.asarray(np.stack([w for w, _ in by_l[l]], axis=1)) for l in ls))
        cols.append(tuple(jax.numpy.asarray(np.concatenate([i for _, i in by_l[l]]), jax.numpy.int32) for l in ls))
        consts.append(jax.numpy.asarray(const))
        lps_all.append(tuple((l, parity(l)) for l in ls))
    return Readout(W=tuple(Ws), cols=tuple(cols), const=tuple(consts), species=sh.species, lps=tuple(lps_all))


@jax.jit
def _readout(readout, F, atoms):
    return tuple(readout.values(F, k, a) for k, a in enumerate(atoms))


class CentredPredictor:
    """`CentredModel.predict` as compiled programs: neighbour list on the host, then the features
    (the program the fit and the on-site model use) and the readout on the device, the model's
    arrays passed per call.  `dense` returns, per species of the shells, the atom indices and
    their (n_atoms, n_channel, n_aux(z)) coefficients; calling it, {channel: (n_total,)}."""

    def __init__(self, model):
        self.model, self.readout = model, compile_readout(model)

    def dense(self, positions, cell, numbers, pbc=True):
        numbers = np.asarray(numbers)
        m = self.model
        for z in set(int(z) for z in numbers):
            m.shells.ls(z)
        rij, zi, zj, seg, n = site_graph(m.basis, positions, cell, numbers, pbc)
        atoms = tuple(np.flatnonzero(numbers == z).astype(np.int32) for z in m.shells.species)
        if len(rij) == 0:                                    # no neighbour: one masked edge keeps the shapes
            z1 = np.zeros(1, np.int32)                       # at the cutoff, where every radial vanishes
            rij, zi, zj, seg = np.array([[m.basis.rcut, 0.0, 0.0]]), z1, z1, z1
        return atoms, _readout(self.readout, _features(m.basis, rij, zi, zj, seg, n), atoms)

    def __call__(self, positions, cell, numbers, pbc=True):
        """As `CentredModel.predict`."""
        atoms, values = self.dense(positions, cell, numbers, pbc)
        return flat(self.model.shells, np.asarray(numbers), atoms, values, self.model.channels)
