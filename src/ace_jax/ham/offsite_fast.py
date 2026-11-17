"""`OffsiteModel.predict` as one compiled program.

The coefficients are contracted with the site features ONCE PER ATOM: for atom a and a
neighbour species, C_a[x; n, lm] is the linear map from the bond functions
u[n, lm] = phi_n(r) Y_lm(r) of any bond leaving a to the L parts x of its blocks.  A bond then
costs one matrix-vector product with its own u (two: the i group on its first atom, the j
group on its second, read through the reverse bond), i.e. a batched matrix product per atom
over its neighbours, plus the two-centre term and a sparse scatter of the parts into the
orbital block.  Nothing is pooled per bond."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..eval.harmonics import real_spherical_harmonics
from . import bond as B
from . import cg
from .offsite import canonical_pairs
from .onsite import _features, site_graph


class _Pack(eqx.Module):
    """One canonical species pair: per (L, p) the two-centre weights W0 (c, Q, n0), the coupled
    weights per group Wi, Wj (c, Q, n1, n_B) with their CG tensors, and the scatter of the
    (shell pair, m) parts into the (n_a, n_b) block."""

    W0: tuple
    Wi: tuple
    Wj: tuple
    T: tuple
    Gq: tuple
    Gij: tuple
    Gv: tuple
    low: jnp.ndarray                 # same species: 1 where the row's shell > the column's shell
    lps: tuple = eqx.field(static=True)
    groups: tuple = eqx.field(static=True)      # per lp: ((l, lam, pi), ...)
    shape: tuple = eqx.field(static=True)       # (n_a, n_b)


def _pack(model, za, zb):
    basis, orb, nc = model.bond, model.orbitals, len(model.channels)
    pairs = canonical_pairs(orb, za, zb)
    na, nb = orb.n_orb(za), orb.n_orb(zb)
    by = {}
    for s1, s2, l1, l2, same in pairs:
        p = 1 if (l1 + l2) % 2 == 0 else -1
        for L in cg.Ls(l1, l2):
            c = model.coef.get((za, zb, s1, s2, L))
            if c is not None:
                by.setdefault((L, p), []).append((s1, s2, l1, l2, same, np.asarray(c)))
    lps = tuple(sorted(by))
    W0s, Wis, Wjs, Ts, Gqs, Gijs, Gvs, grps = [], [], [], [], [], [], [], []
    for L, p in lps:
        two, cpl = B.layout(basis, L, p)
        k0 = basis.n_bond if two else 0
        items = by[(L, p)]
        W0 = np.zeros((nc, len(items), basis.n_bond))
        Wi = [np.zeros((nc, len(items), basis.n_cpl, nB)) for *_, nB in cpl]
        Wj = [np.zeros((nc, len(items), basis.n_cpl, nB)) for *_, nB in cpl]
        dense = np.zeros((len(items), 2 * L + 1, na * nb))
        k1 = basis.n_cpl * sum(nB for *_, nB in cpl)
        for q, (s1, s2, l1, l2, same, c) in enumerate(items):
            W0[:, q, :k0] = c[:, :k0]
            o = k0
            for g, (l, _lam, _pi, nB) in enumerate(cpl):
                w = c[:, o:o + basis.n_cpl * nB].reshape(nc, basis.n_cpl, nB)
                if same:                                  # tied: (i + (-1)^L (-1)^l j) / 2
                    Wi[g][:, q], Wj[g][:, q] = 0.5 * w, 0.5 * (-1) ** (L + l) * w
                else:
                    Wi[g][:, q] = w
                    Wj[g][:, q] = c[:, o + k1:o + k1 + basis.n_cpl * nB].reshape(nc, basis.n_cpl, nB)
                o += basis.n_cpl * nB
            blk = np.zeros((2 * L + 1, na, nb))
            blk[:, orb.shell_slice(za, s1), orb.shell_slice(zb, s2)] = np.moveaxis(cg.real_cg(l1, l2, L), 2, 0)
            dense[q] = blk.reshape(2 * L + 1, na * nb)
        dense = dense.reshape(-1, na * nb)
        r, cidx = np.nonzero(dense)
        W0s.append(jnp.asarray(W0)); Wis.append(tuple(jnp.asarray(w) for w in Wi)); Wjs.append(tuple(jnp.asarray(w) for w in Wj))
        Ts.append(tuple(jnp.asarray(cg.real_cg(l, lam, L)) for l, lam, _, _ in cpl))
        Gqs.append(jnp.asarray(r, jnp.int32)); Gijs.append(jnp.asarray(cidx, jnp.int32)); Gvs.append(jnp.asarray(dense[r, cidx]))
        grps.append(tuple((l, lam, pi) for l, lam, pi, _ in cpl))
    low = np.zeros((na, nb))
    if za == zb:
        sh = np.concatenate([[s] * (2 * l + 1) for s, l in enumerate(orb.ls(za))])
        low = (sh[:, None] > sh[None, :]).astype(float)
    return _Pack(W0=tuple(W0s), Wi=tuple(Wis), Wj=tuple(Wjs), T=tuple(Ts), Gq=tuple(Gqs), Gij=tuple(Gijs),
                 Gv=tuple(Gvs), low=jnp.asarray(low), lps=lps, groups=tuple(grps), shape=(na, nb))


def _radial(x, n):
    """`bond.radial` in JAX, x = r / rcut in [0, 1]."""
    if n == 0:
        return jnp.zeros((*x.shape, 0), x.dtype)
    t = 2 * x - 1
    T = [jnp.ones_like(t), t]
    for _ in range(2, n):
        T.append(2 * t * T[-1] - T[-2])
    return jnp.stack(T[:n], axis=-1) * ((1 - x * x) ** 2)[..., None]


def _canonical(pack, F, phi0, U, Y, usign, rev, n_cpl, l_cpl):
    """(n, K, c, n_a, n_b): the canonical blocks of one species pair on every (atom, slot)."""
    n, K = phi0.shape[:2]
    na, nb = pack.shape
    nlm = (l_cpl + 1) ** 2
    flat = None
    for k, (L, _p) in enumerate(pack.lps):
        W0 = pack.W0[k]
        nc, Q = W0.shape[0], W0.shape[1]
        yL = Y[..., L * L:(L + 1) ** 2]
        part = jnp.einsum("cqn,aKn,aKM->aKcqM", W0, phi0, yL)
        if n_cpl and pack.groups[k]:
            Ci = jnp.zeros((n, nc, Q, 2 * L + 1, n_cpl, nlm), phi0.dtype)
            Cj = jnp.zeros_like(Ci)
            for (l, lam, pi), Wi, Wj, T in zip(pack.groups[k], pack.Wi[k], pack.Wj[k], pack.T[k]):
                f = F[(lam, pi)]
                sl = slice(l * l, (l + 1) ** 2)
                Ci = Ci.at[..., sl].add(jnp.einsum("cqnb,abu,muM->acqMnm", Wi, f, T))
                Cj = Cj.at[..., sl].add(jnp.einsum("cqnb,abu,muM->acqMnm", Wj, f, T))
            part = part + jnp.einsum("acqMnm,aKnm->aKcqM", Ci, U)
            # the j group lives on the bond's second atom: evaluate it there, on the reverse bond
            # (whose u is this bond's with the sign (-1)^l), and read it back through `rev`
            Dj = jnp.einsum("acqMnm,aKnm->aKcqM", Cj, U * usign)
            part = part + Dj.reshape(n * K, nc, Q, 2 * L + 1)[rev].reshape(n, K, nc, Q, 2 * L + 1)
        P = jnp.moveaxis(part.reshape(n, K, nc, Q * (2 * L + 1)), -1, 0)          # (Q M, n, K, c)
        add = jax.ops.segment_sum(P[pack.Gq[k]] * pack.Gv[k][:, None, None, None], pack.Gij[k], num_segments=na * nb)
        flat = add if flat is None else flat + add
    if flat is None:
        return jnp.zeros((n, K, 1, na, nb), phi0.dtype)
    return jnp.moveaxis(flat, 0, -1).reshape(n, K, -1, na, nb)


def _blocks_impl(packs, F, rvec, mask, rev, rcut, shape_args):
    n_bond, n_cpl, l_cpl, lmax = shape_args
    n, K = mask.shape
    dist = jnp.sqrt(jnp.sum(rvec * rvec, axis=-1))
    x = jnp.clip(dist / rcut, 0.0, 1.0)
    phi0, phi1 = _radial(x, n_bond), _radial(x, n_cpl)
    Y = real_spherical_harmonics(rvec, max(lmax, 1))
    Ylm = Y[..., :(l_cpl + 1) ** 2]
    U = phi1[..., :, None] * Ylm[..., None, :] * mask[..., None, None]
    usign = jnp.asarray(np.concatenate([[(-1.0) ** l] * (2 * l + 1) for l in range(l_cpl + 1)]))
    phi0 = phi0 * mask[..., None]
    flat_rev = rev.reshape(-1)
    canon = {key: _canonical(pack, F, phi0, U, Y, usign, flat_rev, n_cpl, l_cpl) for key, pack in packs.items()}
    # every ordered species pair, on the device: the non-canonical parts are the reverse bond's, transposed
    out = {}
    for (za, zb), c in canon.items():
        back = jnp.swapaxes(c.reshape(n * K, *c.shape[2:])[flat_rev], -1, -2).reshape(n, K, c.shape[2], c.shape[4], c.shape[3])
        if za == zb:
            out[(za, zb)] = c + packs[(za, zb)].low * back
        else:
            out[(za, zb)], out[(zb, za)] = c, back
    return out


_blocks = jax.jit(_blocks_impl, static_argnums=6)       # (n_bond, n_cpl, l_cpl, lmax) size the program


class OffsitePredictor:
    """Compiled `OffsiteModel.predict`.  `dense` returns the bond graph and, per ordered species
    pair (za, zb), the bond indices and their blocks (n_bonds, n_channel, n_a, n_b)."""

    def __init__(self, model):
        self.model = model
        orb = model.orbitals
        self.packs = {(za, zb): _pack(model, za, zb) for za in orb.species for zb in orb.species if za <= zb}
        self.lmax = max([L for p in self.packs.values() for L, _ in p.lps] + [model.bond.l_cpl, 1])

    def device(self, F, rvec, mask, rev):
        """{(za, zb): (n_atoms, K, n_channel, n_a, n_b)} on the device, for every ordered species
        pair on every (atom, slot); a slot holds a block of that pair only where the species
        match.  For consumers that stay on the device (H(k), eigenvalues): no copy to the host."""
        b = self.model.bond
        return _blocks(self.packs, F, rvec, mask, rev, b.rcut_bond, (b.n_bond, b.n_cpl, b.l_cpl, self.lmax))

    def dense(self, positions, cell, numbers, pbc=True):
        m, basis = self.model, self.model.bond
        numbers = np.asarray(numbers)
        rij, zi, zj, seg, n = site_graph(basis.site, positions, cell, numbers, pbc)
        F = _features(basis.site, rij, zi, zj, seg, n)
        g = B.bond_graph(basis, positions, cell, numbers, pbc)
        # dense (atom, slot) layout: bonds are sorted by their first atom
        count = np.bincount(g.i, minlength=n)
        K = int(count.max()) if len(g.i) else 1
        start = np.concatenate([[0], np.cumsum(count)])
        slot = np.arange(len(g.i)) - start[g.i]
        flat = g.i * K + slot
        rvec = np.zeros((n, K, 3)); rvec[..., 0] = basis.rcut_bond            # padding: at the cutoff
        mask = np.zeros((n, K))
        rvec.reshape(-1, 3)[flat] = g.rij
        mask.reshape(-1)[flat] = 1.0
        rev = np.zeros(n * K, np.int32)
        rev[flat] = flat[g.reverse]
        canon = self.device(F, rvec, mask, rev.reshape(n, K))
        out = {}
        za_e, zb_e = numbers[g.i], numbers[g.j]
        for za in m.orbitals.species:
            for zb in m.orbitals.species:
                e = np.flatnonzero((za_e == za) & (zb_e == zb))
                if not len(e):
                    continue
                c = canon[(za, zb)]
                out[(za, zb)] = (e, np.asarray(c).reshape(n * K, *c.shape[2:])[flat[e]])
        return g, out

    def __call__(self, positions, cell, numbers, pbc=True):
        """As `OffsiteModel.predict`."""
        g, dense = self.dense(positions, cell, numbers, pbc)
        res = {c: {} for c in self.model.channels}
        for e, blk in dense.values():
            for k, b in zip(e, blk):
                key = (int(g.R[k, 0]), int(g.R[k, 1]), int(g.R[k, 2]), int(g.i[k]), int(g.j[k]))
                for c, name in enumerate(self.model.channels):
                    res[name][key] = b[c]
        return res
