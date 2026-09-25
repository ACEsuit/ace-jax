"""PACE C-tilde potential (pacemaker `.yace`) as an Equinox module.

Mirrors ML-PACE `ACECTildeEvaluator::compute_atom` (ace_evaluator.cpp:146-536,
lammps-user-pace @ 99aa6e6) with analytic radials instead of its spline tables.
Neighbour species enter PACE's A as an explicit channel; here edges stay narrow
and are pooled by (node, neighbour species) -- segment id node*NZ + zj -- so per-
edge work does not grow with the number of elements.  See docs/pace-yace-spec.md.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .harmonics import real_spherical_harmonics
from .edge_model import EdgeSiteModel, check_edge_a_kind, with_edge_a_kind
from .pace_build import build_basis
from .pace_io import parse_yace
from .pace_radial import cutoff_func_poly, fexp, fexp_shifted_scaled, pace_zbl, radbase, radcore


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
    # A-basis form (EdgeSiteModel.edge_a); one-hot selectors only for "matmul"
    edge_a_kind: str = eqx.field(static=True, default="gather")
    a_sel_r: jax.Array = None
    a_sel_y: jax.Array = None

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

    def edge_a_rows(self, rij, zi, zj, mask=None):
        """Per-edge A rows [g_k | R_nl] x Y_lm, zero for invalid edges."""
        bp, valid, r, rij_s = self._geometry(rij, zi, zj, mask)
        lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
        g = radbase(r, self.radbasename, self.inner_cutoff_type, lam, rc, dcut,
                    cin, dcin, self.nradbase)                             # (E, K)
        R = jnp.einsum("ek,enlk->enl", g, self.crad[zi, zj])
        R = R.reshape(g.shape[0], R.shape[1] * R.shape[2])      # explicit: E may be 0
        cols = jnp.concatenate([g, R], axis=-1)
        Y = real_spherical_harmonics(rij_s, self.lmax)
        return jnp.where(valid[:, None], self.edge_a(cols, Y), 0.0)

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
        cr, d, dcin = self._edge_core(rij, zi, zj, mask)
        A = self.pool_edge_a(rij, zi, zj, segment_ids * self.nz + zj, n_nodes * self.nz, mask)
        A = A.reshape(n_nodes, self.nz * self.n_a_local)
        AA = jnp.concatenate([jnp.prod(A[:, s], axis=-1) for s in self.aa_specs], axis=-1)
        rho_all = (AA @ self.ctilde_real().reshape(self.n_aa, -1)).reshape(
            n_nodes, self.nz, self.ndensity)
        rho = rho_all[jnp.arange(n_nodes), node_z]
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

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        n, K = mask.shape
        seg = jnp.repeat(jnp.arange(n), K)
        return self.site_energies(rij.reshape(n * K, 3), zi.reshape(-1), zj.reshape(-1),
                                  seg, n, node_z, mask.reshape(-1))

    # ------------------------------------------------------------ EdgeSiteModel hooks
    def pad_cutoff(self):
        """Padded edges sit at the largest bond cutoff; they are masked too."""
        return jnp.max(self.radparams[..., 1])

    def edge_a_widths(self):
        """(radial columns [g_k | R_nl], harmonic columns)."""
        nrad, nl = self.crad.shape[2], self.crad.shape[3]
        return self.nradbase + nrad * nl, nl * nl

    # ------------------------------------------------------------ not applicable
    def _no_b(self, *a, **k):
        raise NotImplementedError(
            "PACE models have no B-basis: site_basis / site_descriptors / edge_jacobian "
            "are only defined for ACEModel")
    site_basis = site_descriptors = edge_jacobian = compact_basis = _no_b


def load_yace(path, dtype=jnp.float64, edge_a_kind="gather"):
    check_edge_a_kind(edge_a_kind)
    spec, a = parse_yace(path)
    b = build_basis(a)
    A = lambda x: jnp.asarray(x, dtype)
    I = lambda x: jnp.asarray(x, jnp.int32)
    model = PACEModel(
        crad=A(a["crad"]), radparams=A(a["radparams"]), core=A(a["core"]),
        ctilde_complex=A(a["ctilde_complex"]), fs_params=A(a["fs_params"]),
        rho_core_cut=A(a["rho_core_cut"]), E0=A(a["E0"]), Zf=A(a["Z"]),
        aspec_r=I(b["a_rad"]), aspec_y=I(b["a_y"]), aa_specs=tuple(I(s) for s in b["aa_specs"]),
        T_rows=I(b["T_rows"]), T_cols=I(b["T_cols"]), T_vals=A(b["T_vals"]),
        radbasename=a["radbasename"], inner_cutoff_type=a["inner_cutoff_type"],
        npoti=a["npoti"], ndensity=int(a["ndensity"]), nradbase=int(a["nradbase"]),
        lmax=int(a["lmax"]), n_a_local=int(b["n_a_local"]), n_aa=int(b["n_aa"]))
    model = with_edge_a_kind(model, edge_a_kind)
    meta = {"elements": [int(z) for z in a["Z"]], "rcut": float(a["radparams"][..., 1].max()),
            "format": "yace"}
    return model, meta, spec
