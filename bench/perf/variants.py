"""Prototype PACEModel variants for the performance-gap investigation
(docs/pace-performance-gap.md).  Numerically these are the same model: each
variant is checked against the unmodified PACEModel on energies and forces.

Variants (combine with '+', e.g. "rec+pool+rev"):

  rec   SBessel radial basis by the Chebyshev recurrence for sin(kx): one sin
        and one cos per edge instead of K+1 sines (and their cosines in the
        backward pass), and one reciprocal instead of K+1 divisions.
  pool  "pool first" dense A basis: pool g_k(r) (x) Y_lm per (node, neighbour
        species) -- (n, C*K_rad, n_Y) -- and apply the radial coefficients
        crad[z_i, z_j] after pooling, per node.  No per-edge R_nl = crad . g
        (no (E, n_rad*(lmax+1), K_rad) crad gather) and the one-hot channel
        expansion is over K_rad = 8 columns instead of K_rad + n_rad*(lmax+1) = 26.
  fm    feature-major product basis: A as (n_A, n) for the AA products, so the
        gathers (and their scatter adjoints) move whole rows.  Effective only
        with `pool`, which produces A in that layout; a transposed node-major A
        is laid out node-major again by XLA's layout assignment (measured).
  rev   forces by a gather over reverse edges instead of a scatter-add:
        F_i = sum_k g[i,k] - sum_k g[idx[i,k], rev[i,k]], with rev[i,k] the slot
        of edge (j -> i) in row j (a full neighbour list is symmetric).  rev is
        a property of the neighbour list, computed once per list (here on the
        host, `reverse_slots`).
"""
import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.eval.pace_model import PACEModel
from ace_jax.eval.pace_radial import PI, cutoff_func_poly, radbase


def _sbessel_rec(r, rc, K):
    """`pace_radial._sbessel` with sin(k x), k = 1..K+1, from one sin/cos by
    sin((k+1)x) = 2 cos(x) sin(kx) - sin((k-1)x)."""
    x = r * PI / rc
    xs = jnp.where(x == 0, 1.0, x)
    s1, c2 = jnp.sin(xs), 2.0 * jnp.cos(xs)
    inv = 1.0 / xs
    s = [jnp.zeros_like(xs), s1]
    for _ in range(2, K + 2):
        s.append(c2 * s[-1] - s[-2])
    sinc = [None] + [jnp.where(x == 0, 1.0, s[k] * inv * (1.0 / k)) for k in range(1, K + 2)]
    rc15 = rc ** 1.5

    def f(n):
        pre = ((-1) ** n * math.sqrt(2) * PI * (n + 1) * (n + 2)
               / math.sqrt((n + 1) ** 2 + (n + 2) ** 2))
        return pre / rc15 * (sinc[n + 1] + sinc[n + 2])
    g = [f(0)]
    d_prev = 1.0
    for n in range(1, K):
        en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = 1 - en / d_prev
        g.append((f(n) + math.sqrt(en / d_prev) * g[-1]) / math.sqrt(dn))
        d_prev = dn
    return jnp.stack(g, axis=-1)


def radbase_rec(r, name, inner, lam, rc, dcut, cut_in, dcut_in, K):
    """`pace_radial.radbase` with the recurrence SBessel; other bases unchanged."""
    if name != "SBessel":
        return radbase(r, name, inner, lam, rc, dcut, cut_in, dcut_in, K)
    if inner == "zbl":
        one = jnp.ones_like(dcut_in)
        cut_in = jnp.where(dcut_in == 0, one, 0 * one)
    inside = (r > cut_in - dcut_in) & (r < rc)
    rs = jnp.where(inside, r, 0.5 * rc)
    g = _sbessel_rec(rs, rc, K)
    if inner in ("distance", "zbl"):
        g = g * (1.0 - cutoff_func_poly(rs, cut_in, dcut_in))[..., None]
    return jnp.where(inside[..., None], g, 0.0)


class FastPACE(PACEModel):
    use_rec: bool = eqx.field(static=True, default=False)
    use_pool: bool = eqx.field(static=True, default=False)
    use_fm: bool = eqx.field(static=True, default=False)
    W: jax.Array = None        # (NZ_i, NZ_j, n_a, K_rad): A_a = sum_k W g_k Y_y(a), per channel
    sel_y: jax.Array = None    # (n_Y, n_a) one-hot: Y columns -> A entries

    def _radbase(self, *a):
        return (radbase_rec if self.use_rec else radbase)(*a)

    def edge_a_factors(self, rij, zi, zj, mask=None):
        bp, valid, r, rij_s = self._geometry(rij, zi, zj, mask)
        lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
        g = self._radbase(r, self.radbasename, self.inner_cutoff_type, lam, rc, dcut,
                          cin, dcin, self.nradbase)
        R = jnp.einsum("ek,enlk->enl", g, self.crad[zi, zj])
        R = R.reshape(g.shape[0], R.shape[1] * R.shape[2])
        cols = jnp.where(valid[:, None], jnp.concatenate([g, R], axis=-1), 0.0)
        return cols, real_spherical_harmonics(rij_s, self.lmax)

    def edge_g_factors(self, rij, zi, zj, mask=None):
        """(g_k, Y_lm) per edge: only the K_rad radial basis columns."""
        bp, valid, r, rij_s = self._geometry(rij, zi, zj, mask)
        lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
        g = self._radbase(r, self.radbasename, self.inner_cutoff_type, lam, rc, dcut,
                          cin, dcin, self.nradbase)
        g = jnp.where(valid[:, None], g, 0.0)
        return g, real_spherical_harmonics(rij_s, self.lmax)

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        if not self.use_pool:
            return super().site_energies_dense(rij, zi, zj, mask, node_z)
        n, K = mask.shape
        flat = lambda a: a.reshape(n * K, *a.shape[2:])
        g, Y = self.edge_g_factors(flat(rij), flat(zi), flat(zj), flat(mask))
        A = self.pool_first_dense(g.reshape(n, K, -1), Y.reshape(n, K, -1), zj, node_z,
                                  transposed=self.use_fm)
        seg = jnp.repeat(jnp.arange(n), K)
        core = self._edge_core(flat(rij), flat(zi), flat(zj), flat(mask))
        if self.use_fm:          # A is already (C*n_a, n): feature-major by construction
            return self._node_energies_t(A, *core, seg, n, node_z)
        return self._node_energies(A, *core, seg, n, node_z)

    def pool_first_dense(self, g, Y, zj, node_z, transposed=False):
        """A (n, C * n_a) = sum_k W[z_i, z_j] g_k (x) Y_y, pooled before W;
        (C * n_a, n) when `transposed`."""
        n, K, kr = g.shape
        C = self.nz
        hi = jax.lax.Precision.HIGHEST
        oh = jax.nn.one_hot(zj, C, dtype=g.dtype)                          # (n, K, C)
        gc = (oh[..., :, None] * g[..., None, :]).reshape(n, K, C * kr)    # (n, K, C*kr)
        Ag = jnp.einsum("nkg,nky->ngy", gc, Y, precision=hi)               # (n, C*kr, nY)
        Agy = jnp.matmul(Ag, self.sel_y, precision=hi)                     # (n, C*kr, n_a)
        Agy = Agy.reshape(n, C, kr, -1)
        if transposed:
            A = jnp.einsum("nmka,nmak->man", Agy, self.W[node_z], precision=hi)
            return A.reshape(-1, n)
        A = jnp.einsum("nmka,nmak->nma", Agy, self.W[node_z], precision=hi)
        return A.reshape(n, -1)

    # ------------------------------------------------------------ fm product basis
    def _node_energies(self, A, cr, d, dcin, segment_ids, n_nodes, node_z):
        if not self.use_fm:
            return super()._node_energies(A, cr, d, dcin, segment_ids, n_nodes, node_z)
        return self._node_energies_t(A.T, cr, d, dcin, segment_ids, n_nodes, node_z)

    def _node_energies_t(self, At, cr, d, dcin, segment_ids, n_nodes, node_z):
        """`_node_energies` on At (C*n_a, n): feature-major, so the product
        gathers read whole rows and their scatter adjoints add whole rows."""
        AA = jnp.concatenate([jnp.prod(At[s.T], axis=0) for s in self.aa_specs], axis=0)
        ct = self.ctilde_real().reshape(self.n_aa, -1)                      # (n_aa, NZ*P)
        rho_all = jnp.matmul(ct.T, AA, precision=jax.lax.Precision.HIGHEST)  # (NZ*P, n)
        rho_all = rho_all.reshape(self.nz, self.ndensity, n_nodes)
        rho = rho_all[node_z, :, jnp.arange(n_nodes)]                       # (n, P)
        EF = self._embedding(rho, node_z)
        rho_core = jax.ops.segment_sum(cr, segment_ids, num_segments=n_nodes)
        if self.inner_cutoff_type == "zbl":
            raise NotImplementedError("fm prototype: distance inner cutoff only")
        rc_, drc_ = self.rho_core_cut[node_z, 0], self.rho_core_cut[node_z, 1]
        E = EF * cutoff_func_poly(rho_core, rc_, drc_) + rho_core
        return E + self.E0[node_z]

    # ------------------------------------------------------------ rev forces
    def energy_forces_virial_dense_rev(self, rij, zi, zj, idx, rev, mask, node_z):
        """`energy_forces_virial_dense` with the force assembly as a gather over
        reverse edges (see module docstring) instead of a scatter-add."""
        def total(r, eps):
            sym = 0.5 * (eps + eps.T)
            return jnp.sum(self.site_energies_dense(r + r @ sym, zi, zj, mask, node_z))

        eps0 = jnp.zeros((3, 3), rij.dtype)
        E, (g_r, g_eps) = jax.value_and_grad(total, argnums=(0, 1))(rij, eps0)
        g_r = jnp.where(mask[..., None], g_r, 0.0)
        back = g_r[idx, rev]                                                # (n, K, 3)
        F = g_r.sum(axis=1) - jnp.where(mask[..., None], back, 0.0).sum(axis=1)
        return E, F, -g_eps


def reverse_slots(idx, rij, count, tol=1e-8):
    """rev (n, K): slot of the edge j -> i in row j = idx[i, k], matched by
    rij[j, rev] == -rij[i, k] (periodic images of one pair are distinct edges).
    Host numpy, O(E log E); raises if any live edge has no reverse."""
    idx, rij, count = np.asarray(idx), np.asarray(rij), np.asarray(count)
    n, K = idx.shape
    live = np.arange(K)[None, :] < count[:, None]
    i = np.repeat(np.arange(n), K).reshape(n, K)[live]
    k = np.tile(np.arange(K), n).reshape(n, K)[live]
    j = idx[live]
    d = rij[live]
    q = np.round(d / 1e-6).astype(np.int64)
    # forward key (i, j, d) and the reverse edge's key seen from its partner (j, i, -d)
    fwd = np.lexsort((q[:, 2], q[:, 1], q[:, 0], j, i))
    bwd = np.lexsort((-q[:, 2], -q[:, 1], -q[:, 0], i, j))
    rev = np.zeros((n, K), np.int32)
    # the e-th smallest forward key equals the e-th smallest reversed key
    rev[i[bwd], k[bwd]] = k[fwd]
    ok = (idx[j, rev[i, k]] == i) & (np.abs(rij[j, rev[i, k]] + d).max(axis=1) < tol)
    if not ok.all():
        raise ValueError(f"{(~ok).sum()} edges without a matched reverse")
    return rev


def make(name, model):
    parts = set(name.split("+"))
    unknown = parts - {"rec", "pool", "rev", "fm"}
    if unknown:
        raise ValueError(f"unknown variant parts {unknown}")
    fields = {f.name: getattr(model, f.name) for f in dataclasses.fields(model)
              if f.name in PACEModel.__dataclass_fields__}
    W = sel_y = None
    if "pool" in parts:
        W, sel_y = pool_first_tables(model)
    return FastPACE(**fields, use_rec="rec" in parts, use_pool="pool" in parts,
                    use_fm="fm" in parts, W=W, sel_y=sel_y)


def pool_first_tables(model):
    """W (NZ, NZ, n_a, K_rad) and sel_y (n_Y, n_a) for `pool_first_dense`.

    Local A entry a pairs radial column c(a) (c < K_rad: g_c itself; else R_nl,
    c = K_rad + n*(lmax+1) + l) with harmonic y(a).  R_nl = sum_k crad[zi,zj,n,l,k] g_k,
    so A_a = sum_k W[zi, zj, a, k] (g_k Y_y(a)) with W the delta or crad row."""
    kr = model.nradbase
    nl = model.crad.shape[3]
    ar = np.asarray(model.aspec_r)
    ay = np.asarray(model.aspec_y)
    crad = np.asarray(model.crad)                                           # (NZ, NZ, nrad, nl, kr)
    NZ = crad.shape[0]
    W = np.zeros((NZ, NZ, len(ar), kr))
    for a, c in enumerate(ar):
        if c < kr:
            W[:, :, a, c] = 1.0
        else:
            nn, l = divmod(int(c) - kr, nl)
            W[:, :, a, :] = crad[:, :, nn, l, :]
    ny = (model.lmax + 1) ** 2
    sel_y = np.zeros((ny, len(ay)))
    sel_y[ay, np.arange(len(ay))] = 1.0
    dt = model.E0.dtype
    return jnp.asarray(W, dt), jnp.asarray(sel_y, dt)
