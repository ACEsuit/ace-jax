"""Prototype: exact, static restructurings of a loaded (folded) `ACEModel` that cut
its per-edge and A-assembly work (docs/ace-vs-pace-gap.md §4).  Not in src/.

    from ace_fast import make
    fast = make("prune+lblock+compact+pairfold", model)

Every variant gives the same E, F and virial as the unmodified model to
roundoff; `check` measures it.  The variants compose with "+":

  prune     drop what the A basis never reads: Rnl columns no A entry uses and
            Y_lm above the largest l any A entry uses.  A pure data transform
            (smaller `rnl_coefs`, `lmax`, remapped `aspec_r`), run by the stock
            ACEModel code.  ACEpotentials exports lmax from totaldegree, but at
            order 2 (Cantor small/medium) no product reaches it, and many Rnl
            columns are only in the order-1 / pair-degree budget of other
            species blocks.
  lblock    pool A per l-block: A_l = sum_k R_{.,l} (x) Y_l, (n, n_r(l), 2l+1),
            instead of the full n_rnl x n_Y outer product of which only the
            l(r) == l(y) diagonal blocks are ever read (4.5% of it on
            Cantor_medium).  aa_specs are remapped onto the block layout, so no
            column select follows.
  compact   ACE1's Rnl is block-sparse in the neighbour species: every column is
            nonzero for exactly one z_j (checked here).  Evaluate per edge only
            the M_l columns of its own z_j (spline gather 4 * sum M_l instead of
            4 * n_rnl) and expand by a one-hot over z_j, PACE's channel trick.
            Implies lblock.
  fm        (with lblock/compact) pool feature-major and hand _aa A = At.T.
  pairfold  fold the pair readout Wpair[:, z_i] into the pair spline table: the
            pair channel becomes one spline column per edge instead of n_pair.

The variants act on `site_energies_dense` (the dense layout the benchmark and
the LAMMPS bundle use); the sparse path is left alone.
"""
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.eval.model import ACEModel, pool_dense
from ace_jax.eval.radial import agnesi_normalized, env_poly2sx, spline_eval


def _l_of(y):
    return np.floor(np.sqrt(np.asarray(y))).astype(int)


def _owner(coefs):
    """(n_rnl,) the neighbour species z_j each Rnl column is nonzero for."""
    nzm = np.abs(np.asarray(coefs)).max(axis=2) > 0                 # (zi, zj, r)
    own = nzm.any(axis=0)                                           # (zj, r)
    if not ((own.sum(axis=0) <= 1).all() and all((nzm[i] == nzm[0]).all()
                                                   for i in range(nzm.shape[0]))):
        raise ValueError("Rnl is not block-sparse in z_j")
    return own.argmax(axis=0)


def prune(m):
    """Drop unused Rnl columns and Y_lm above the used lmax; columns sorted by
    (l, owner species, original index) so lblock/compact slice contiguously."""
    ar, ay = np.asarray(m.aspec_r), np.asarray(m.aspec_y)
    la = _l_of(ay)
    lr = {int(r): int(l) for r, l in zip(ar, la)}
    own = _owner(m.rnl_coefs)
    used = sorted(lr, key=lambda r: (lr[r], own[r], r))
    new = {r: i for i, r in enumerate(used)}
    return dataclasses.replace(
        m, rnl_coefs=m.rnl_coefs[..., np.asarray(used)], lmax=int(la.max()),
        aspec_r=jnp.asarray([new[int(r)] for r in ar], m.aspec_r.dtype))


def pairfold(m):
    """Pair channel: one spline column per edge, weights folded in (exact)."""
    pc = np.asarray(m.pair_coefs)                                   # (zi, zj, c, p)
    W = np.asarray(m.Wpair)                                         # (p, zi)
    folded = np.einsum("ijcp,pi->ijc", pc, W)[..., None]
    return dataclasses.replace(m, pair_coefs=jnp.asarray(folded, m.pair_coefs.dtype),
                               Wpair=jnp.ones((1, W.shape[1]), m.Wpair.dtype))


class ACEBlocked(ACEModel):
    """ACEModel whose dense energy pools A per l-block (optionally species-compact).
    blocks: per l, (l, column offset into the radial table, width w_l); with
    `compact` the pooled block is (NZ * w_l, 2l+1) through a one-hot over z_j."""
    blocks: tuple = eqx.field(static=True, default=())
    compact: bool = eqx.field(static=True, default=False)
    fm: bool = eqx.field(static=True, default=False)      # pool feature-major, A = At.T

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        n, K = mask.shape
        E = n * K
        r3, zi_, zj_, mk = (rij.reshape(E, 3), zi.reshape(E), zj.reshape(E), mask.reshape(E))
        r = jnp.linalg.norm(r3, axis=-1)
        x = agnesi_normalized(r, self.rnl_transform[zi_, zj_])
        env = env_poly2sx(x, self.rnl_envelope[zi_, zj_])
        x0, h, ns = self.rnl_grid
        R = jax.vmap(lambda xx, c: spline_eval(xx, c, x0, h, ns))(x, self.rnl_coefs[zi_, zj_])
        R = jnp.where(mk[:, None], R * env[:, None], 0.0)
        Y = real_spherical_harmonics(r3, self.lmax)
        Rpair = self.pair_radial(r3, zi_, zj_)
        hi = jax.lax.Precision.HIGHEST
        nz = self.E0.shape[0]
        oh = jax.nn.one_hot(zj_, nz, dtype=R.dtype) if self.compact else None
        As = []
        for l, off, w in self.blocks:
            Rl = R[:, off:off + w]
            if self.compact:
                Rl = (oh[:, :, None] * Rl[:, None, :]).reshape(E, nz * w)
            Yl = Y[:, l * l:(l + 1) ** 2]
            if self.fm:
                As.append(jnp.einsum("nkr,nky->ryn", Rl.reshape(n, K, -1), Yl.reshape(n, K, -1),
                                     precision=hi).reshape(-1, n))
            else:
                As.append(jnp.einsum("nkr,nky->nry", Rl.reshape(n, K, -1),
                                     Yl.reshape(n, K, -1), precision=hi).reshape(n, -1))
        A = jnp.concatenate(As, axis=0).T if self.fm else jnp.concatenate(As, axis=1)
        return self._readout_folded(A, pool_dense(Rpair.reshape(n, K, -1), mask), node_z)


def lblock(m, compact=False, fm=False):
    """`m` must be pruned (columns sorted by (l, owner)).  Returns ACEBlocked
    with aa_specs remapped onto the concatenated block layout."""
    ar, ay = np.asarray(m.aspec_r), np.asarray(m.aspec_y)
    la = _l_of(ay)
    own = _owner(m.rnl_coefs)
    nz = int(m.E0.shape[0])
    lr = {}
    for r, l in zip(ar, la):
        lr[int(r)] = int(l)
    ls = sorted(set(lr.values()))
    coefs = np.asarray(m.rnl_coefs)
    blocks, pos_of_rm, base, off = [], {}, 0, 0
    new_cols = []                                                  # compact table columns
    for l in ls:
        cols = [r for r in sorted(lr) if lr[r] == l]
        nm = 2 * l + 1
        if compact:
            per = {z: [r for r in cols if own[r] == z] for z in range(nz)}
            w = max(len(v) for v in per.values())
            tab = np.zeros(coefs.shape[:3] + (w,))
            for z, rs in per.items():
                for k, rr in enumerate(rs):
                    tab[:, z, :, k] = coefs[:, z, :, rr]
                    for mm in range(nm):
                        pos_of_rm[(rr, mm)] = base + ((z * w) + k) * nm + mm
            new_cols.append(tab)
            blocks.append((l, off, w))
            base += nz * w * nm
        else:
            w = len(cols)
            assert cols == list(range(cols[0], cols[0] + w)), "prune first"
            for k, rr in enumerate(cols):
                for mm in range(nm):
                    pos_of_rm[(rr, mm)] = base + k * nm + mm
            blocks.append((l, cols[0], w))
            base += w * nm
        off += w
    pos = np.asarray([pos_of_rm[(int(r), int(y) - l * l)] for r, y, l in zip(ar, ay, la)])
    specs = tuple(jnp.asarray(pos[np.asarray(g)], g.dtype) for g in m.aa_specs)
    fields = {f.name: getattr(m, f.name) for f in dataclasses.fields(m)}
    fields.update(aa_specs=specs, blocks=tuple(blocks), compact=compact, fm=fm)
    if compact:
        fields["rnl_coefs"] = jnp.asarray(np.concatenate(new_cols, axis=-1), m.rnl_coefs.dtype)
    return ACEBlocked(**fields)


def make(variant, m):
    if variant in ("", "baseline"):
        return m
    parts = set(variant.split("+"))
    unknown = parts - {"prune", "lblock", "compact", "pairfold", "fm"}
    if unknown:
        raise ValueError(f"unknown variant parts {unknown}")
    if not m.folded or m.radial_kind != "spline":
        raise ValueError("prototype needs a folded model with the spline radial")
    if "pairfold" in parts:
        m = pairfold(m)
    if parts & {"prune", "lblock", "compact"}:
        m = prune(m)
    if parts & {"lblock", "compact", "fm"}:
        m = lblock(m, compact="compact" in parts, fm="fm" in parts)
    return m


def check(base, fast, args):
    """Max |dE|/atom, |dF|, |dV| of fast against base on the dense E/F/V call."""
    f = jax.jit(lambda mm, *a: mm.energy_forces_virial_dense(*a))
    E0, F0, V0 = f(base, *args)
    E1, F1, V1 = f(fast, *args)
    n = args[-1].shape[0]
    return {"dE_per_atom": float(abs(E1 - E0)) / n, "max_dF": float(jnp.max(jnp.abs(F1 - F0))),
            "max_dV": float(jnp.max(jnp.abs(V1 - V0)))}
