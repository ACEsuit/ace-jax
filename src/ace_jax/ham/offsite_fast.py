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
    We: tuple                    # per lp: (c Q, G n_ebond n_env), the bond-environment groups side by side
    Te: tuple
    T: tuple
    Gq: tuple
    Gij: tuple
    Gv: tuple
    low: jnp.ndarray                 # same species: 1 where the row's shell > the column's shell
    lps: tuple = eqx.field(static=True)
    groups: tuple = eqx.field(static=True)      # per lp: ((l, lam, pi), ...)
    egroups: tuple = eqx.field(static=True)     # per lp: ((l, lam), ...) of the bond environment
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
    Wes, Tes, egrps = [], [], []
    for L, p in lps:
        two, cpl = B.layout(basis, L, p)
        k0 = basis.n_bond if two else 0
        items = by[(L, p)]
        W0 = np.zeros((nc, len(items), basis.n_bond))
        Wi = [np.zeros((nc, len(items), basis.n_cpl, nB)) for *_, nB in cpl]
        Wj = [np.zeros((nc, len(items), basis.n_cpl, nB)) for *_, nB in cpl]
        dense = np.zeros((len(items), 2 * L + 1, na * nb))
        k1 = basis.n_cpl * sum(nB for *_, nB in cpl)
        ecpl = B.env_layout(basis, L, p)
        ke = basis.n_ebond * basis.n_env
        We = [np.zeros((nc, len(items), basis.n_ebond, basis.n_env)) for _ in ecpl]
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
            o = k0 + (k1 if same else 2 * k1)             # the bond-environment columns follow
            for g, (l, _lam) in enumerate(ecpl):
                if same and (l - L) % 2:                  # an equal-shell block does not have this group
                    continue
                We[g][:, q] = c[:, o:o + ke].reshape(nc, basis.n_ebond, basis.n_env)
                o += ke
            blk = np.zeros((2 * L + 1, na, nb))
            blk[:, orb.shell_slice(za, s1), orb.shell_slice(zb, s2)] = np.moveaxis(cg.real_cg(l1, l2, L), 2, 0)
            dense[q] = blk.reshape(2 * L + 1, na * nb)
        dense = dense.reshape(-1, na * nb)
        r, cidx = np.nonzero(dense)
        W0s.append(jnp.asarray(W0)); Wis.append(tuple(jnp.asarray(w) for w in Wi)); Wjs.append(tuple(jnp.asarray(w) for w in Wj))
        Ts.append(tuple(jnp.asarray(cg.real_cg(l, lam, L)) for l, lam, _, _ in cpl))
        Gqs.append(jnp.asarray(r, jnp.int32)); Gijs.append(jnp.asarray(cidx, jnp.int32)); Gvs.append(jnp.asarray(dense[r, cidx]))
        grps.append(tuple((l, lam, pi) for l, lam, pi, _ in cpl))
        Wes.append(jnp.asarray(np.stack(We, axis=2).reshape(nc * len(items), -1)) if We else jnp.zeros((nc * len(items), 0)))
        Tes.append(tuple(jnp.asarray(cg.real_cg(l, lam, L)) for l, lam in ecpl))
        egrps.append(tuple(ecpl))
    low = np.zeros((na, nb))
    if za == zb:
        sh = np.concatenate([[s] * (2 * l + 1) for s, l in enumerate(orb.ls(za))])
        low = (sh[:, None] > sh[None, :]).astype(float)
    return _Pack(W0=tuple(W0s), Wi=tuple(Wis), Wj=tuple(Wjs), We=tuple(Wes), Te=tuple(Tes), T=tuple(Ts),
                 Gq=tuple(Gqs), Gij=tuple(Gijs), Gv=tuple(Gvs), low=jnp.asarray(low), lps=lps, groups=tuple(grps),
                 egroups=tuple(egrps), shape=(na, nb))


def _radial(x, n):
    """`bond.radial` in JAX, x = r / rcut in [0, 1]."""
    if n == 0:
        return jnp.zeros((*x.shape, 0), x.dtype)
    t = 2 * x - 1
    T = [jnp.ones_like(t), t]
    for _ in range(2, n):
        T.append(2 * t * T[-1] - T[-2])
    return jnp.stack(T[:n], axis=-1) * ((1 - x * x) ** 2)[..., None]


def _canonical(pack, Fa, Fb, ga, gb, rev, usign, n_cpl, l_cpl):
    """(n_a, K, c, n_orb_a, n_orb_b): the canonical blocks of one species pair on the slots of its
    layout.  `ga`: the bond geometry on this layout (first atoms of species a); `gb`: on the
    reverse layout (b -> a), where the features that live on the second atom are evaluated;
    `rev`: for each slot, the flat position of its reverse bond in that layout."""
    phi0, U, Y, phi2, side = ga
    n, K = phi0.shape[:2]
    na, nb = pack.shape
    nlm = (l_cpl + 1) ** 2
    Aenv = None
    if side is not None:                                  # both ends: the reverse bond's half
        sb = gb[4]
        Aenv = side + sb.reshape(-1, *sb.shape[2:])[rev].reshape(side.shape)
    flat = None
    for k, (L, _p) in enumerate(pack.lps):
        W0 = pack.W0[k]
        nc, Q = W0.shape[0], W0.shape[1]
        yL = Y[..., L * L:(L + 1) ** 2]
        part = jnp.einsum("cqn,aKn,aKM->aKcqM", W0, phi0, yL)
        if n_cpl and pack.groups[k]:
            Ub = gb[1]
            Ci = jnp.zeros((n, nc, Q, 2 * L + 1, n_cpl, nlm), phi0.dtype)
            Cj = jnp.zeros((Ub.shape[0], nc, Q, 2 * L + 1, n_cpl, nlm), phi0.dtype)
            for (l, lam, pi), Wi, Wj, T in zip(pack.groups[k], pack.Wi[k], pack.Wj[k], pack.T[k]):
                sl = slice(l * l, (l + 1) ** 2)
                Ci = Ci.at[..., sl].add(jnp.einsum("cqnb,abu,muM->acqMnm", Wi, Fa[(lam, pi)], T))
                Cj = Cj.at[..., sl].add(jnp.einsum("cqnb,abu,muM->acqMnm", Wj, Fb[(lam, pi)], T))
            part = part + jnp.einsum("acqMnm,aKnm->aKcqM", Ci, U)
            # the j group lives on the bond's second atom: evaluate it there, on the reverse bond
            # (whose u is this bond's with the sign (-1)^l), and read it back through `rev`
            Dj = jnp.einsum("acqMnm,aKnm->aKcqM", Cj, Ub * usign)
            part = part + Dj.reshape(-1, nc, Q, 2 * L + 1)[rev].reshape(n, K, nc, Q, 2 * L + 1)
        if pack.egroups[k]:
            # bond environment.  Per atom (its bonds' features fit the cache): the features
            # phi_n (Y_l (x) A^env_lam)^L of every group side by side, then ONE matrix product with
            # the coefficients of all shell pairs and channels.
            groups, Tes, We = pack.egroups[k], pack.Te[k], pack.We[k]

            def one(args, groups=groups, Tes=Tes, We=We, L=L):
                y, A, ph = args                                         # (K, nlm), (K, n_env, nlm_env), (K, n_ebond)
                Z = jnp.stack([jnp.einsum("Keu,KuM->KeM", A[..., lam * lam:(lam + 1) ** 2],
                                          jnp.einsum("Km,muM->KuM", y[:, l * l:(l + 1) ** 2], Te))
                               for (l, lam), Te in zip(groups, Tes)], axis=1)                   # (K, G, n_env, M)
                Phi = (ph[:, None, :, None, None] * Z[:, :, None, :, :]).reshape(y.shape[0], -1, 2 * L + 1)
                return jnp.einsum("xk,KkM->KxM", We, Phi)                # (K, c Q, M)

            part = part + jax.lax.map(one, (Y, Aenv, phi2)).reshape(n, K, nc, Q, 2 * L + 1)
        P = jnp.moveaxis(part.reshape(n, K, nc, Q * (2 * L + 1)), -1, 0)          # (Q M, n, K, c)
        add = jax.ops.segment_sum(P[pack.Gq[k]] * pack.Gv[k][:, None, None, None], pack.Gij[k], num_segments=na * nb)
        flat = add if flat is None else flat + add
    if flat is None:
        return jnp.zeros((n, K, 1, na, nb), phi0.dtype)
    return jnp.moveaxis(flat, 0, -1).reshape(n, K, -1, na, nb)


def _env_side(e_rij, e_live, rvec, mask, r_env, rmax, n_env, l_env):
    """(n, K, n_env, (l_env+1)^2): for each bond slot, the half of A^env pooled over the neighbours
    of its first atom (`bond._env_side` on the dense layouts).  One atom at a time: its
    (bonds x environment atoms) intermediates fit the cache, where all atoms at once is a pass
    over hundreds of MB per operation."""
    nlm = (l_env + 1) ** 2

    def one(args):
        d, live, r, m = args                                            # (Ke, 3), (Ke,), (K, 3), (K,)
        dist = jnp.sqrt(jnp.sum(d * d, axis=-1))
        e = jnp.where(live & (dist < r_env), (1 - jnp.clip(dist / r_env, 0, 1) ** 2) ** 2, 0.0)   # (Ke,)
        partner = jnp.sum((d[None, :, :] - r[:, None, :]) ** 2, axis=-1) < 1e-8                # (K, Ke)
        w = jnp.where(partner, 0.0, e[None, :]) * m[:, None]
        x = d[None, :, :] - 0.5 * r[:, None, :]                         # about the midpoint
        R = _radial(jnp.clip(jnp.sqrt(jnp.sum(x * x, axis=-1)) / rmax, 0.0, 1.0), n_env)
        Yx = real_spherical_harmonics(x, max(l_env, 1))[..., :nlm]
        return jnp.einsum("Ken,Key->Kny", w[..., None] * R, Yx)

    return jax.lax.map(one, (e_rij, e_live, rvec, mask))


def _blocks_impl(packs, F, lay, rcut, shape_args, env=None):
    """{(za, zb): (n_a, K_ab, c, n_orb_a, n_orb_b)} for every ordered species pair in `lay`.

    lay[(za, zb)] = (rows, rvec, mask, rev): the atoms of species za, their bonds to atoms of
    species zb in dense slots (padding at the cutoff), and for each slot the flat position of its
    reverse bond in lay[(zb, za)].  Each bond is evaluated once, in the layout of its canonical
    pair (za <= zb); the other orientation is that block transposed."""
    n_bond, n_cpl, l_cpl, lmax, n_env, l_env, n_ebond, l_ebond, r_env = shape_args
    usign = jnp.asarray(np.concatenate([[(-1.0) ** l] * (2 * l + 1) for l in range(l_cpl + 1)]))
    geo = {}
    for key, (rows, rvec, mask, _rev) in lay.items():
        dist = jnp.sqrt(jnp.sum(rvec * rvec, axis=-1))
        x = jnp.clip(dist / rcut, 0.0, 1.0)
        Y = real_spherical_harmonics(rvec, max(lmax, 1))
        U = _radial(x, n_cpl)[..., :, None] * Y[..., None, :(l_cpl + 1) ** 2] * mask[..., None, None]
        phi2 = side = None
        if n_env:
            e_rij, e_count = env
            e_live = jnp.arange(e_rij.shape[1])[None, :] < e_count[:, None]
            side = _env_side(e_rij[rows], e_live[rows], rvec, mask, r_env, r_env + 0.5 * rcut, n_env, l_env)
            phi2 = _radial(x, n_ebond) * mask[..., None]
        geo[key] = (_radial(x, n_bond) * mask[..., None], U, Y, phi2, side)
    out = {}
    for (za, zb), pack in packs.items():
        if (za, zb) not in lay:
            continue
        rows_a, _, _, rev_ab = lay[(za, zb)]
        rows_b, _, _, rev_ba = lay[(zb, za)]
        Fa = {k: v[rows_a] for k, v in F.items()}
        Fb = Fa if za == zb else {k: v[rows_b] for k, v in F.items()}
        c = _canonical(pack, Fa, Fb, geo[(za, zb)], geo[(zb, za)], rev_ab, usign, n_cpl, l_cpl)
        # the other orientation: the reverse bond's block, transposed
        nb_rows, Kb = lay[(zb, za)][2].shape
        back = jnp.swapaxes(c.reshape(-1, *c.shape[2:])[rev_ba], -1, -2).reshape(nb_rows, Kb, c.shape[2], c.shape[4], c.shape[3])
        if za == zb:
            out[(za, zb)] = c + pack.low * back
        else:
            out[(za, zb)], out[(zb, za)] = c, back
    return out


_blocks = jax.jit(_blocks_impl, static_argnums=4)       # the basis sizes fix the program


class PairLayouts:
    """The bonds of a structure in one dense layout per ordered species pair (host arrays).
    lay[(za, zb)] = (rows, rvec, mask, rev) as `_blocks_impl` takes them; `sel[(za, zb)]`: the bond
    indices (into the bond graph) of the pair; `flatpos[e]`: the flat (row, slot) position of bond e
    in its pair's layout; `bond[(za, zb)]`: (n_a, K_ab) bond index per slot, -1 for padding."""

    def __init__(self, g, numbers, species, rcut):
        numbers = np.asarray(numbers)
        present = [z for z in species if (numbers == z).any()]
        zi, zj = numbers[g.i], numbers[g.j]
        self.flatpos = np.zeros(len(g.i), np.int64)
        self.sel, self.bond, self.shape = {}, {}, {}
        tmp = {}
        for za in present:
            rows = np.flatnonzero(numbers == za)
            rowpos = np.full(len(numbers), -1, np.int64)
            rowpos[rows] = np.arange(len(rows))
            for zb in present:
                e = np.flatnonzero((zi == za) & (zj == zb))               # sorted by first atom, as the graph is
                r = rowpos[g.i[e]]
                count = np.bincount(r, minlength=len(rows))
                K = max(int(count.max()) if len(e) else 1, 1)
                start = np.concatenate([[0], np.cumsum(count)])
                pos = r * K + (np.arange(len(e)) - start[r])
                self.flatpos[e] = pos
                rvec = np.zeros((len(rows), K, 3)); rvec[..., 0] = rcut       # padding: at the cutoff
                mask = np.zeros((len(rows), K))
                bond = np.full(len(rows) * K, -1, np.int64)
                rvec.reshape(-1, 3)[pos] = g.rij[e]
                mask.reshape(-1)[pos] = 1.0
                bond[pos] = e
                tmp[(za, zb)] = (rows.astype(np.int32), rvec, mask)
                self.sel[(za, zb)], self.bond[(za, zb)], self.shape[(za, zb)] = e, bond.reshape(len(rows), K), (len(rows), K)
        self.lay = {}
        for (za, zb), (rows, rvec, mask) in tmp.items():
            rev = np.zeros(rows.shape[0] * mask.shape[1], np.int32)
            e = self.sel[(za, zb)]
            rev[self.flatpos[e]] = self.flatpos[g.reverse[e]]             # in the layout of (zb, za)
            self.lay[(za, zb)] = (rows, rvec, mask, rev)


class OffsitePredictor:
    """Compiled `OffsiteModel.predict`.  `dense` returns the bond graph and, per ordered species
    pair (za, zb), the bond indices and their blocks (n_bonds, n_channel, n_a, n_b)."""

    def __init__(self, model):
        self.model = model
        orb = model.orbitals
        self.packs = {(za, zb): _pack(model, za, zb) for za in orb.species for zb in orb.species if za <= zb}
        self.lmax = max([L for p in self.packs.values() for L, _ in p.lps] + [model.bond.l_cpl, model.bond.l_ebond, 1])

    def device(self, F, lay, env=None):
        """{(za, zb): (n_a, K_ab, n_channel, n_orb_a, n_orb_b)} on the device, one dense layout per
        ordered species pair (`PairLayouts.lay`, or the same built on the device).  For consumers
        that stay on the device (H(k), eigenvalues): no copy to the host."""
        b = self.model.bond
        if b.n_env and env is None:
            raise ValueError("this model has a bond environment: pass env = (neighbour vectors (n, Ke, 3), counts) within r_env")
        return _blocks(self.packs, F, lay, b.rcut_bond, self.static(), env if b.n_env else None)

    def static(self):
        b = self.model.bond
        return (b.n_bond, b.n_cpl, b.l_cpl, self.lmax, b.n_env, b.l_env, b.n_ebond, b.l_ebond, b.r_env)

    def dense(self, positions, cell, numbers, pbc=True):
        m, basis = self.model, self.model.bond
        numbers = np.asarray(numbers)
        rij, zi, zj, seg, n = site_graph(basis.site, positions, cell, numbers, pbc)
        F = _features(basis.site, rij, zi, zj, seg, n)
        g = B.bond_graph(basis, positions, cell, numbers, pbc)
        pl = PairLayouts(g, numbers, m.orbitals.species, basis.rcut_bond)
        blocks = self.device(F, pl.lay, B.env_lists(basis, positions, cell, pbc))
        out = {}
        for pair, e in pl.sel.items():
            if len(e):
                c = blocks[pair]
                out[pair] = (e, np.asarray(c).reshape(-1, *c.shape[2:])[pl.flatpos[e]])
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
