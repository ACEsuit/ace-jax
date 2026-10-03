"""PACE C-tilde potential (pacemaker `.yace`) as an Equinox module.

Mirrors ML-PACE `ACECTildeEvaluator::compute_atom` (ace_evaluator.cpp:146-536,
lammps-user-pace @ 99aa6e6) with analytic radials instead of its spline tables.
Neighbour species enter PACE's A as an explicit channel; here edges stay narrow
and are pooled by (node, neighbour species) -- segment id node*NZ + zj -- so per-
edge work does not grow with the number of elements.  See docs/dev/pace-yace-spec.md.

A is built pool-first (`EdgeSiteModel.pool_first_dense` / `pool_first_sparse`):
g_k (x) Y_lm is pooled per (node, neighbour species) and crad applied per node
afterwards, so the per-edge R_nl is never formed; A comes out feature-major
(C * n_a, n) for the product basis.  docs/dev/pace-performance-gap.md sections 7-8.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .harmonics import real_spherical_harmonics
from .edge_model import EdgeSiteModel, check_edge_a_kind, product_basis_t, with_edge_a_kind
from .pace_build import build_basis
from .pace_io import parse_yace
from .pace_radial import cutoff_func_poly, fexp, fexp_shifted_scaled, pace_zbl, radbase, radcore


# SBessel g_k by one sin per k and a constant matrix (`pace_radial._sbessel_mm`)
# from this nradbase up, else the rotation recurrence.  Measured on an A100 in
# float64 (docs/dev/ace-vs-pace-gap.md 4.3): -6% (8192 atoms) and -21% (131072) at
# nradbase 13, -6% at 15, -1% at 11, but +11-19% slower at 4-9.
SBESSEL_MATMUL_MIN_K = 12


class PACEModel(EdgeSiteModel):
    # trainable leaves (what write_yace serialises)
    crad: jax.Array            # (NZ, NZ, nradmax, lmax+1, K)
    radparams: jax.Array       # (NZ, NZ, 5): lambda, rcut, dcut, rcut_in, dcut_in
    core: jax.Array            # (NZ, NZ, 2): prehc, lambdahc
    ctilde_complex: jax.Array  # (n_terms, P)
    fs_params: jax.Array       # (NZ, 2P): w0, m0, w1, m1, ...
    rho_core_cut: jax.Array    # (NZ, 2)
    E0: jax.Array              # (NZ,)
    # integer structure (leaves, never differentiated)
    Zf: jax.Array              # (NZ,) atomic numbers as floats, for ZBL
    aspec_r: jax.Array         # local A entry -> radial column ([g_k | R_nl] per edge)
    aspec_y: jax.Array         # local A entry -> harmonic column (l*l + l + m)
    aa_specs: tuple
    T_rows: jax.Array
    T_cols: jax.Array
    T_vals: jax.Array
    # static
    radbasename: str = eqx.field(static=True)
    inner_cutoff_type: str = eqx.field(static=True)
    npoti: tuple = eqx.field(static=True)
    ndensity: int = eqx.field(static=True)
    nradbase: int = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    n_a_local: int = eqx.field(static=True)
    n_aa: int = eqx.field(static=True)
    # A-basis form (EdgeSiteModel.edge_a): inert for PACE, whose energy path is
    # pool-first (uses_edge_a = False); kept so `with_edge_a_kind` / `load(...,
    # edge_a_kind=)` accept either model family.
    edge_a_kind: str = eqx.field(static=True, default="gather")
    a_sel_r: jax.Array = None
    a_sel_y: jax.Array = None
    # pool-first structure (static, set by load_yace; W itself is built from crad
    # in the trace by pool_first_weights, so crad stays trainable)
    pf_col: jax.Array = None     # (n_a,) radial column of each local A entry
    pf_sel_y: jax.Array = None   # (n_Y, n_a) one-hot: Y columns -> A entries
    # SBessel evaluation form, fixed per model by load_yace (SBESSEL_MATMUL_MIN_K)
    sbessel_form: str = eqx.field(static=True, default="rotation")

    uses_edge_a = False

    @property
    def nz(self):
        return self.E0.shape[0]

    def ctilde_real(self):
        """(n_AA, NZ, P): T applied to the complex coefficients (cheap, per call)."""
        flat = jax.ops.segment_sum(self.T_vals[:, None] * self.ctilde_complex[self.T_cols],
                                   self.T_rows, num_segments=self.n_aa * self.nz)
        return flat.reshape(self.n_aa, self.nz, self.ndensity)

    # ------------------------------------------------------------ per edge
    def _geometry(self, rij, zi, zj, mask):
        """Per-edge bond parameters, validity (r < bond rcut, and mask), and a
        safe r / rij for invalid edges so no branch sees r = 0 or r >= rcut."""
        r2 = jnp.sum(rij * rij, axis=-1)
        bp = self.radparams[zi, zj]
        valid = r2 < bp[:, 1] * bp[:, 1]
        if mask is not None:
            valid = valid & mask
        r = jnp.sqrt(jnp.where(valid, r2, 1.0)) * jnp.where(valid, 1.0, 0.5 * bp[:, 1])
        rij_s = jnp.where(valid[:, None], rij, jnp.stack([r, 0 * r, 0 * r], -1))
        return bp, valid, r, rij_s

    def edge_basis_factors(self, rij, zi, zj, mask=None):
        """(g_k, Y_lm) per edge: the fixed radial basis only, zero on invalid edges."""
        bp, valid, r, rij_s = self._geometry(rij, zi, zj, mask)
        lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
        g = radbase(r, self.radbasename, self.inner_cutoff_type, lam, rc, dcut,
                    cin, dcin, self.nradbase, self.sbessel_form)          # (E, K)
        return jnp.where(valid[:, None], g, 0.0), real_spherical_harmonics(rij_s, self.lmax)

    @property
    def pool_first_sel_y(self):
        return self.pf_sel_y

    def pool_first_weights(self):
        """W (NZ, NZ, n_a, K_rad) from crad, in the trace (crad stays trainable):
        column c < K_rad is g_c itself (a delta row); else R_nl with
        c = K_rad + n*(lmax+1) + l, i.e. the crad[:, :, n, l, :] row."""
        kr, nl = self.nradbase, self.crad.shape[3]
        c = self.pf_col
        is_g = c < kr
        nn, l = jnp.divmod(jnp.maximum(c - kr, 0), nl)
        rows = self.crad[:, :, nn, l, :]                                   # (NZ, NZ, n_a, kr)
        delta = jax.nn.one_hot(jnp.where(is_g, c, 0), kr, dtype=self.crad.dtype)
        return jnp.where(is_g[None, None, :, None], delta[None, None], rows)

    def _edge_core(self, rij, zi, zj, mask):
        """Core repulsion / ZBL per edge, and the zbl switch coordinate d."""
        bp, valid, r, _ = self._geometry(rij, zi, zj, mask)
        lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
        if self.inner_cutoff_type == "zbl":
            cr = pace_zbl(r, self.Zf[zi], self.Zf[zj], rc, dcut, self.core[zi, zj, 0])
        else:
            cr = radcore(r, self.core[zi, zj, 0], self.core[zi, zj, 1], rc, cin, dcin,
                         self.inner_cutoff_type)
        cr = jnp.where(valid, cr, 0.0)
        d = jnp.where(valid, r - (cin - dcin), jnp.inf)                   # zbl switch coordinate
        return cr, d, dcin

    # ------------------------------------------------------------ per node
    def _embedding(self, rho, node_z):
        w, m = self.fs_params[node_z, 0::2], self.fs_params[node_z, 1::2]
        ss = jnp.asarray([p == "FinnisSinclairShiftedScaled" for p in self.npoti])[node_z]
        F = jnp.where(ss[:, None], fexp_shifted_scaled(rho, m), fexp(rho, m))
        return jnp.sum(w * F, axis=-1)

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        """Edge-list layout: A pool-first by segment_sum (`pool_first_sparse`)."""
        g, Y = self.edge_basis_factors(rij, zi, zj, mask)
        At = self.pool_first_sparse(g, Y, segment_ids, zj, node_z, n_nodes)
        return self._node_energies_t(At, *self._edge_core(rij, zi, zj, mask), segment_ids,
                                     n_nodes, node_z)

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        """Dense (n, K) layout: A pool-first by a batched outer product
        (`pool_first_dense`); the per-edge core / ZBL scalars on the flattened edges."""
        n, K = mask.shape
        flat = lambda a: a.reshape(n * K, *a.shape[2:])
        g, Y = self.edge_basis_factors(flat(rij), flat(zi), flat(zj), flat(mask))
        At = self.pool_first_dense(g.reshape(n, K, -1), Y.reshape(n, K, -1), zj, node_z)
        seg = jnp.repeat(jnp.arange(n), K)
        return self._node_energies_t(At, *self._edge_core(flat(rij), flat(zi), flat(zj),
                                                          flat(mask)), seg, n, node_z)

    def _node_energies_t(self, At, cr, d, dcin, segment_ids, n_nodes, node_z):
        """Site energies from feature-major A (NZ*n_a, n): the product gathers
        read whole rows and their adjoints add whole rows."""
        AA = product_basis_t(At, self.aa_specs)
        ct = self.ctilde_real().reshape(self.n_aa, -1)                      # (n_aa, NZ*P)
        rho_all = jnp.matmul(ct.T, AA, precision=jax.lax.Precision.HIGHEST)  # (NZ*P, n)
        rho = rho_all.reshape(self.nz, self.ndensity, n_nodes)[node_z, :, jnp.arange(n_nodes)]
        return self._energy_tail(rho, cr, d, dcin, segment_ids, n_nodes, node_z)

    def _energy_tail(self, rho, cr, d, dcin, segment_ids, n_nodes, node_z):
        """Everything after rho (n, P): embedding, the core-repulsion switch (ZBL
        by the nearest neighbour's distance, else by rho_core), and E0."""
        EF = self._embedding(rho, node_z)
        rho_core = jax.ops.segment_sum(cr, segment_ids, num_segments=n_nodes)
        if self.inner_cutoff_type == "zbl":
            dmin = jax.ops.segment_min(d, segment_ids, num_segments=n_nodes)
            has = jnp.isfinite(dmin)
            is_min = jnp.isfinite(d) & (d == dmin[segment_ids])
            dc = jax.ops.segment_max(jnp.where(is_min, dcin, -jnp.inf), segment_ids,
                                     num_segments=n_nodes)
            dc_s = jnp.where(has, dc, 1.0)
            f = jnp.where(has, cutoff_func_poly(dc_s - jnp.where(has, dmin, 0.0), dc_s, dc_s), 1.0)
            E = EF * f + rho_core * (1.0 - f)
        else:
            rc_, drc_ = self.rho_core_cut[node_z, 0], self.rho_core_cut[node_z, 1]
            E = EF * cutoff_func_poly(rho_core, rc_, drc_) + rho_core
        return E + self.E0[node_z]

    # ------------------------------------------------------------ EdgeSiteModel hooks
    @property
    def a_channels(self):
        """A carries one neighbour-species channel per element (PACE's mu_j)."""
        return self.nz

    def pad_cutoff(self):
        """Padded edges sit at the largest bond cutoff; they are masked too."""
        return jnp.max(self.radparams[..., 1])

    def pool_first_widths(self):
        """(n_b, n_Y): the per-edge fixed radial basis and harmonic widths
        (`estimate_a_bytes` for the pool-first form)."""
        return self.nradbase, (self.lmax + 1) ** 2

    def product_basis_width(self):
        """Per-node product-basis width, n_AA + sum_k k n_spec_k: the gathered
        A factors of every order and AA (`estimate_a_bytes`)."""
        return self.n_aa + sum(int(s.size) for s in self.aa_specs)

    def edge_a_widths(self):
        """(radial columns [g_k | R_nl], harmonic columns) of the per-edge form;
        inert for PACE's energy path (pool-first)."""
        nrad, nl = self.crad.shape[2], self.crad.shape[3]
        return self.nradbase + nrad * nl, nl * nl

    # ------------------------------------------------------------ not applicable
    def _no_b(self, *a, **k):
        raise NotImplementedError(
            "PACE models have no B-basis: site_basis / site_descriptors / edge_jacobian "
            "are only defined for ACEModel")
    site_basis = site_descriptors = edge_jacobian = edge_jacobian_dense = compact_basis = _no_b


def load_yace(path, dtype=jnp.float64, edge_a_kind="gather"):
    check_edge_a_kind(edge_a_kind)
    spec, a = parse_yace(path)
    b = build_basis(a)
    A = lambda x: jnp.asarray(x, dtype)
    I = lambda x: jnp.asarray(x, jnp.int32)
    ny = (int(a["lmax"]) + 1) ** 2
    sel = np.zeros((ny, len(b["a_y"])))
    sel[np.asarray(b["a_y"]), np.arange(len(b["a_y"]))] = 1.0
    model = PACEModel(
        crad=A(a["crad"]), radparams=A(a["radparams"]), core=A(a["core"]),
        ctilde_complex=A(a["ctilde_complex"]), fs_params=A(a["fs_params"]),
        rho_core_cut=A(a["rho_core_cut"]), E0=A(a["E0"]), Zf=A(a["Z"]),
        aspec_r=I(b["a_rad"]), aspec_y=I(b["a_y"]), aa_specs=tuple(I(s) for s in b["aa_specs"]),
        T_rows=I(b["T_rows"]), T_cols=I(b["T_cols"]), T_vals=A(b["T_vals"]),
        radbasename=a["radbasename"], inner_cutoff_type=a["inner_cutoff_type"],
        npoti=a["npoti"], ndensity=int(a["ndensity"]), nradbase=int(a["nradbase"]),
        lmax=int(a["lmax"]), n_a_local=int(b["n_a_local"]), n_aa=int(b["n_aa"]),
        pf_col=I(b["a_rad"]), pf_sel_y=A(sel),
        sbessel_form=("matmul" if a["radbasename"] == "SBessel"
                      and int(a["nradbase"]) >= SBESSEL_MATMUL_MIN_K else "rotation"))
    model = with_edge_a_kind(model, edge_a_kind)
    meta = {"elements": [int(z) for z in a["Z"]], "rcut": float(a["radparams"][..., 1].max()),
            "format": "yace"}
    return model, meta, spec
