"""`OffsiteModel.predict` as one compiled program.

The coefficients are contracted with the site features ONCE PER ATOM: for atom a and a
neighbour species, C_a[x; n, lm] is the linear map from the bond functions
u[n, lm] = phi_n(r) Y_lm(r) of any bond leaving a to the L parts x of its blocks.  A bond then
costs one matrix-vector product with its own u (two: the i group on its first atom, the j
group on its second, read through the reverse bond), i.e. a batched matrix product per atom
over its neighbours, plus the two-centre term and a sparse scatter of the parts into the
orbital block.  Nothing is pooled per bond."""
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..eval.harmonics import real_spherical_harmonics
from . import bond as B
from ..basis import cg
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
        ne = basis.n_env * (B.env_embedding(basis).shape[1] if basis.n_env else 0)     # channels x radials
        ke = basis.n_ebond * ne
        We = [np.zeros((nc, len(items), basis.n_ebond, ne)) for _ in ecpl]
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
                We[g][:, q] = c[:, o:o + ke].reshape(nc, basis.n_ebond, ne)
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


def _take(x, rev, n, K):
    """Rows of the flat (row, slot) array `x` at the flat positions `rev`, as (n, K, ...)."""
    return x.reshape(-1, *x.shape[2:])[rev].reshape(n, K, *x.shape[2:])


def _usign(l_cpl):
    return jnp.asarray(np.concatenate([[(-1.0) ** l] * (2 * l + 1) for l in range(l_cpl + 1)]))


def _dj(pack, Fb, Ub, n_cpl, l_cpl):
    """Per (L, p) of the pack, (n_b, K, c, Q, 2L+1) or None: the part of the canonical blocks that
    lives on the bond's SECOND atom, evaluated there, on the reverse bond (whose u is the bond's
    with the sign (-1)^l).  The canonical layout reads it back through its reverse index."""
    out = []
    for k, (L, _p) in enumerate(pack.lps):
        if not (n_cpl and pack.groups[k]):
            out.append(None)
            continue
        nc, Q = pack.W0[k].shape[:2]
        Cj = jnp.zeros((Ub.shape[0], nc, Q, 2 * L + 1, n_cpl, (l_cpl + 1) ** 2), Ub.dtype)
        for (l, lam, pi), Wj, T in zip(pack.groups[k], pack.Wj[k], pack.T[k]):
            Cj = Cj.at[..., l * l:(l + 1) ** 2].add(jnp.einsum("cqnb,abu,muM->acqMnm", Wj, Fb[(lam, pi)], T))
        out.append(jnp.einsum("acqMnm,aKnm->aKcqM", Cj, Ub * _usign(l_cpl)))
    return tuple(out)


def _canonical(pack, Fa, ga, Aenv, dj, n_cpl, l_cpl):
    """(n_a, K, c, n_orb_a, n_orb_b): the canonical blocks of one species pair on the slots of its
    layout.  `ga` = (phi0, U, Y, phi2): the bond geometry on these rows; `Aenv`: the pooled bond
    environment of both ends; `dj`: `_dj` of the reverse layout, already gathered to these slots."""
    phi0, U, Y, phi2 = ga
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
            for (l, lam, pi), Wi, T in zip(pack.groups[k], pack.Wi[k], pack.T[k]):
                Ci = Ci.at[..., l * l:(l + 1) ** 2].add(jnp.einsum("cqnb,abu,muM->acqMnm", Wi, Fa[(lam, pi)], T))
            part = part + jnp.einsum("acqMnm,aKnm->aKcqM", Ci, U) + dj[k]
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


def _env_side(e_rij, e_live, e_ch, rvec, mask, r_env, rmax, n_env, l_env):
    """(n, K, channels n_env, (l_env+1)^2): for each bond slot, the half of A^env pooled over the neighbours
    of its first atom (`bond._env_side` on the dense layouts).  One atom at a time: its
    (bonds x environment atoms) intermediates fit the cache, where all atoms at once is a pass
    over hundreds of MB per operation."""
    nlm = (l_env + 1) ** 2

    def one(args):
        d, live, ch, r, m = args                                        # (Ke, 3), (Ke,), (Ke, C), (K, 3), (K,)
        dist = jnp.sqrt(jnp.sum(d * d, axis=-1))
        e = jnp.where(live & (dist < r_env), (1 - jnp.clip(dist / r_env, 0, 1) ** 2) ** 2, 0.0)   # (Ke,)
        partner = jnp.sum((d[None, :, :] - r[:, None, :]) ** 2, axis=-1) < 1e-8                # (K, Ke)
        w = jnp.where(partner, 0.0, e[None, :]) * m[:, None]
        x = d[None, :, :] - 0.5 * r[:, None, :]                         # about the midpoint
        R = _radial(jnp.clip(jnp.sqrt(jnp.sum(x * x, axis=-1)) / rmax, 0.0, 1.0), n_env)
        Yx = real_spherical_harmonics(x, max(l_env, 1))[..., :nlm]
        return jnp.einsum("Ken,ec,Key->Kcny", w[..., None] * R, ch, Yx).reshape(r.shape[0], -1, nlm)

    return jax.lax.map(one, (e_rij, e_live, e_ch, rvec, mask))


def _geo(rows, rvec, mask, env, rcut, shape_args):
    """The bond geometry of the rows of one layout: (phi0, U, Y, phi2, side)."""
    n_bond, n_cpl, l_cpl, lmax, n_env, l_env, n_ebond, l_ebond, r_env, env_emb = shape_args
    x = jnp.clip(jnp.sqrt(jnp.sum(rvec * rvec, axis=-1)) / rcut, 0.0, 1.0)
    Y = real_spherical_harmonics(rvec, max(lmax, 1))
    U = _radial(x, n_cpl)[..., :, None] * Y[..., None, :(l_cpl + 1) ** 2] * mask[..., None, None]
    phi2 = side = None
    if n_env:
        e_rij, e_count, e_z = env
        e_live = jnp.arange(e_rij.shape[1])[None, :] < e_count[:, None]
        e_ch = jnp.asarray(env_emb, rvec.dtype)[e_z[rows]]               # (rows, Ke, channels)
        side = _env_side(e_rij[rows], e_live[rows], e_ch, rvec, mask, r_env, r_env + 0.5 * rcut, n_env, l_env)
        phi2 = _radial(x, n_ebond) * mask[..., None]
    return _radial(x, n_bond) * mask[..., None], U, Y, phi2, side


def _back(c, rev, n, K):
    """The other orientation of the blocks `c`: the reverse bond's block, transposed."""
    return jnp.swapaxes(c.reshape(-1, *c.shape[2:])[rev], -1, -2).reshape(n, K, c.shape[2], c.shape[4], c.shape[3])


def _blocks_impl(packs, F, lay, rcut, shape_args, env=None):
    """{(za, zb): (n_a, K_ab, c, n_orb_a, n_orb_b)} for every ordered species pair in `lay`.

    lay[(za, zb)] = (rows, rvec, mask, rev): the atoms of species za, their bonds to atoms of
    species zb in dense slots (padding at the cutoff), and for each slot the flat position of its
    reverse bond in lay[(zb, za)].  Each bond is evaluated once, in the layout of its canonical
    pair (za <= zb); the other orientation is that block transposed."""
    n_cpl, l_cpl, n_env = shape_args[1], shape_args[2], shape_args[4]
    geo = {key: _geo(rows, rvec, mask, env, rcut, shape_args) for key, (rows, rvec, mask, _rev) in lay.items()}
    out = {}
    for (za, zb), pack in packs.items():
        if (za, zb) not in lay:
            continue
        rows_a, _, mask_a, rev_ab = lay[(za, zb)]
        rows_b, _, mask_b, rev_ba = lay[(zb, za)]
        ga, gb = geo[(za, zb)], geo[(zb, za)]
        n, K = mask_a.shape
        dj = tuple(None if d is None else _take(d, rev_ab, n, K)
                   for d in _dj(pack, {k: v[rows_b] for k, v in F.items()}, gb[1], n_cpl, l_cpl))
        Aenv = ga[4] + _take(gb[4], rev_ab, n, K) if n_env else None      # both ends: the reverse bond's half
        c = _canonical(pack, {k: v[rows_a] for k, v in F.items()}, ga[:4], Aenv, dj, n_cpl, l_cpl)
        back = _back(c, rev_ba, *mask_b.shape)
        if za == zb:
            out[(za, zb)] = c + pack.low * back
        else:
            out[(za, zb)], out[(zb, za)] = c, back
    return out


_blocks = jax.jit(_blocks_impl, static_argnums=4)       # the basis sizes fix the program


# ---- the same in chunks of rows, for a pool of threads.  XLA's own CPU threading gives little on
# this program (1.65x on 16 cores); independent calls from Python threads scale (the calls release
# the GIL).  Three passes, each over chunks of rows, each chunk complete in itself:
#   1. geometry of every layout, and `_dj` on the reverse layouts;
#   2. the canonical blocks (reading pass 1 of the reverse layout through the reverse index);
#   3. the other orientation (reading pass 2).
@partial(jax.jit, static_argnums=7)
def _pass1(pack, F, rows, rvec, mask, env, rcut, shape_args):
    g = _geo(rows, rvec, mask, env, rcut, shape_args)
    dj = () if pack is None else _dj(pack, {k: v[rows] for k, v in F.items()}, g[1], shape_args[1], shape_args[2])
    return g, dj


@partial(jax.jit, static_argnums=8)
def _pass2(pack, F, rows, ga, side_rev, dj_rev, rev, mask, shape_args):
    n, K = mask.shape
    dj = tuple(None if d is None else _take(d, rev, n, K) for d in dj_rev)
    Aenv = ga[4] + _take(side_rev, rev, n, K) if shape_args[4] else None
    return _canonical(pack, {k: v[rows] for k, v in F.items()}, ga[:4], Aenv, dj, shape_args[1], shape_args[2])


_pass3 = jax.jit(lambda c, rev, mask: _back(c, rev, *mask.shape))


def _chunks(lay_one, T, rcut):
    """Split one layout's rows into at most T chunks of one shape (the last padded with empty rows)."""
    rows, rvec, mask, rev = (np.asarray(x) for x in lay_one)
    n, K = mask.shape
    m = -(-n // T)
    out = []
    for i0 in range(0, n, m):
        k = min(m, n - i0)
        r, v, mk, rv = np.zeros(m, rows.dtype), np.zeros((m, K, 3), rvec.dtype), np.zeros((m, K), mask.dtype), np.zeros(m * K, rev.dtype)
        v[..., 0] = rcut
        r[:k], v[:k], mk[:k], rv[:k * K] = rows[i0:i0 + k], rvec[i0:i0 + k], mask[i0:i0 + k], rev[i0 * K:(i0 + k) * K]
        out.append((i0, k, r, v, mk, rv))
    return out


def _blocks_threaded(packs, F, lay, rcut, shape_args, env, pool, T, orient=True):
    """`_blocks` on the CPU with the rows of every layout in chunks on `pool` (T threads).
    Returns host arrays.  `orient=False`: passes 1 and 2 only, the canonical pairs."""
    ch = {key: _chunks(v, T, rcut) for key, v in lay.items()}
    canon = [key for key in packs if key in lay]
    rpack = {(zb, za): packs[(za, zb)] for za, zb in canon}                 # layouts that carry a `_dj`
    full = {}                                                             # (layout, name) -> host array over all rows

    def put(key, name, i0, k, x):
        x = np.asarray(x)
        with lock:
            if (key, name) not in full:
                full[(key, name)] = np.empty((lay[key][2].shape[0], *x.shape[1:]), x.dtype)
        full[(key, name)][i0:i0 + k] = x[:k]

    lock = threading.Lock()

    def one1(job):
        key, (i0, k, rows, rvec, mask, _rev) = job
        g, dj = _pass1(rpack.get(key), F, rows, rvec, mask, env, rcut, shape_args)
        if key in rpack:
            if g[4] is not None:
                put(key, "side", i0, k, g[4])
            for q, d in enumerate(dj):
                if d is not None:
                    put(key, q, i0, k, d)
        return g

    jobs = [(key, c) for key in lay for c in ch[key]]
    geo = dict(zip([(key, c[0]) for key, c in jobs], pool.map(one1, jobs)))

    def one2(job):
        (za, zb), (i0, k, rows, _rvec, mask, rev) = job
        pack, r = packs[(za, zb)], (zb, za)
        dj = tuple(full.get((r, q)) for q in range(len(pack.lps)))
        put((za, zb), "c", i0, k, _pass2(pack, F, rows, geo[((za, zb), i0)], full.get((r, "side")), dj, rev, mask, shape_args))

    list(pool.map(one2, [(key, c) for key in canon for c in ch[key]]))
    if not orient:
        return {key: full[(key, "c")] for key in canon}

    def one3(job):
        (za, zb), (i0, k, _rows, _rvec, mask, rev) = job              # a chunk of the layout (zb, za)
        put((zb, za), "back", i0, k, _pass3(full[((za, zb), "c")], rev, mask))

    list(pool.map(one3, [(key, c) for key in canon for c in ch[(key[1], key[0])]]))
    out = {}
    for za, zb in canon:
        c, back = full[((za, zb), "c")], full[((zb, za), "back")]
        if za == zb:
            out[(za, zb)] = c + np.asarray(packs[(za, zb)].low) * back
        else:
            out[(za, zb)], out[(zb, za)] = c, back
    return out


class PairLayouts:
    """The bonds of a structure in one dense layout per ordered species pair (host arrays).
    lay[(za, zb)] = (rows, rvec, mask, rev) as `_blocks_impl` takes them; `sel[(za, zb)]`: the bond
    indices (into the bond graph) of the pair; `flatpos[e]`: the flat (row, slot) position of bond e
    in its pair's layout; `bond[(za, zb)]`: (n_a, K_ab) bond index per slot, -1 for padding."""

    def __init__(self, g, numbers, species, rcut, k_step=1):
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
                K = -(-K // k_step) * k_step              # slots in steps: fewer shapes, fewer compilations
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

    def __init__(self, model, threads=1):
        """`threads` > 1 (CPU only): evaluate the rows in chunks on that many threads."""
        self.model, self.threads = model, int(threads)
        self._pool = ThreadPoolExecutor(self.threads) if self.threads > 1 else None
        orb = model.orbitals
        self.packs = {(za, zb): _pack(model, za, zb) for za in orb.species for zb in orb.species if za <= zb}
        self.lmax = max([L for p in self.packs.values() for L, _ in p.lps] + [model.bond.l_cpl, model.bond.l_ebond, 1])

    def device(self, F, lay, env=None):
        """{(za, zb): (n_a, K_ab, n_channel, n_orb_a, n_orb_b)} on the device, one dense layout per
        ordered species pair (`PairLayouts.lay`, or the same built on the device).  For consumers
        that stay on the device (H(k), eigenvalues): no copy to the host."""
        b = self.model.bond
        if b.n_env and env is None:
            raise ValueError("this model has a bond environment: pass env = (neighbour vectors (n, Ke, 3), counts, species indices (n, Ke)) within r_env")
        return _blocks(self.packs, F, lay, b.rcut_bond, self.static(), env if b.n_env else None)

    def blocks(self, F, lay, env=None):
        """`device`, or with `threads` > 1 on a CPU backend the same blocks as host arrays,
        evaluated in chunks of rows on the thread pool."""
        if self._pool is None or jax.default_backend() != "cpu":
            return self.device(F, lay, env)
        b = self.model.bond
        if b.n_env and env is None:
            raise ValueError("this model has a bond environment: pass env = (neighbour vectors (n, Ke, 3), counts, species indices (n, Ke)) within r_env")
        return _blocks_threaded(self.packs, F, lay, b.rcut_bond, self.static(), env if b.n_env else None, self._pool, self.threads)

    def static(self):
        b = self.model.bond
        emb = tuple(tuple(float(v) for v in row) for row in B.env_embedding(b)) if b.n_env else ()
        return (b.n_bond, b.n_cpl, b.l_cpl, self.lmax, b.n_env, b.l_env, b.n_ebond, b.l_ebond, b.r_env, emb)

    def dense(self, positions, cell, numbers, pbc=True):
        m, basis = self.model, self.model.bond
        numbers = np.asarray(numbers)
        rij, zi, zj, seg, n = site_graph(basis.site, positions, cell, numbers, pbc)
        F = _features(basis.site, rij, zi, zj, seg, n)
        g = B.bond_graph(basis, positions, cell, numbers, pbc)
        pl = PairLayouts(g, numbers, m.orbitals.species, basis.rcut_bond)
        env = B.env_lists(basis, positions, cell, pbc, numbers)
        if self._pool is not None and jax.default_backend() == "cpu":
            return g, self._dense_threaded(F, pl, g, env if basis.n_env else None)
        blocks = self.blocks(F, pl.lay, env)
        out = {}
        for pair, e in pl.sel.items():
            if len(e):
                c = blocks[pair]
                out[pair] = (e, np.asarray(c).reshape(-1, *c.shape[2:])[pl.flatpos[e]])
        return g, out

    def _dense_threaded(self, F, pl, g, env):
        """The blocks of every bond from the canonical pairs' layouts, gathered (and, for the
        other orientation, transposed) in chunks of bonds on the pool: no layout of the
        non-canonical orientation is ever built."""
        T = self.threads
        canon = _blocks_threaded(self.packs, F, pl.lay, self.model.bond.rcut_bond, self.static(), env, self._pool, T, orient=False)
        flat = {key: c.reshape(-1, *c.shape[2:]) for key, c in canon.items()}
        out, jobs = {}, []
        for (za, zb), e in pl.sel.items():
            if not len(e):
                continue
            cf = flat[(za, zb) if za <= zb else (zb, za)]
            shape = cf.shape[1:] if za <= zb else (cf.shape[1], cf.shape[3], cf.shape[2])
            blk = np.empty((len(e), *shape), cf.dtype)
            out[(za, zb)] = (e, blk)
            fwd, bwd = pl.flatpos[e], pl.flatpos[g.reverse[e]]
            low = np.asarray(self.packs[(za, za)].low) if za == zb else None
            step = -(-len(e) // T)
            jobs += [(cf, blk, fwd, bwd, low, za <= zb, i0, min(i0 + step, len(e))) for i0 in range(0, len(e), step)]

        def one(job):
            cf, blk, fwd, bwd, low, direct, i0, i1 = job
            if direct:
                np.take(cf, fwd[i0:i1], axis=0, out=blk[i0:i1], mode="clip")
                if low is not None:
                    blk[i0:i1] += low * np.swapaxes(np.take(cf, bwd[i0:i1], axis=0), -1, -2)
            else:
                blk[i0:i1] = np.swapaxes(np.take(cf, bwd[i0:i1], axis=0), -1, -2)

        list(self._pool.map(one, jobs))
        return out

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
