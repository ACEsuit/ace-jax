"""Equivariant site features B^(L, p) for all L from ONE A / AA evaluation: the linear-ACE
feature provider (`eval.features.SiteFeatures`).

The one-particle basis is a tensor radial times harmonics, pooled over each atom's neighbours
(`eval.features.pooled_a`), and its products are `eval.product.product_basis` on the A list and
AA order of an `EquivariantSet`, so one AA vector per atom serves every (L, p):
B^(L, p)[b, m] = sum_k A2B[b, m, k] AA_k, a sparse contraction each.  Every shell pair and every
operator channel then reads the same features, which is what keeps the cost of a structure
independent of the number of orbitals.  No model is held: the radial is a
`eval.radial.TensorRadial` (or a `RadialTable` of it), built from `basis.radial_init`."""
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..basis import radial_init as ri
from ..basis.coupling import couple_equivariant_set
from ..basis.prior import smoothness_prior
from ..basis.spec import build_spec
from ..eval.features import pooled_a
from ..eval.product import product_basis
from ..eval.radial import TensorRadial, cubic_bspline_coefs, spline_eval_pairs
from .precision import highest


class RadialTable(eqx.Module):
    """A radial (r, zi, zj) -> (E, n_rnl) as one cubic B-spline per species pair on a uniform grid
    in r over [r_min, rcut]: no transcendental per edge.  Outside the grid r is clamped (the
    radial vanishes at rcut; below r_min it is held constant).  `tabulate` makes it."""
    coefs: jax.Array                                 # (NZ, NZ, n + 2, n_rnl)
    grid: tuple = eqx.field(static=True)             # (r_min, h, n nodes)

    def __call__(self, r, zi, zj):
        x0, h, n = self.grid
        return spline_eval_pairs(jnp.clip(r, x0, x0 + h * (n - 1)), self.coefs, zi, zj, x0, h, n)


def tabulate(radial, n_species, rcut, n_intervals=4096, r_min=0.5):
    """`RadialTable` of a radial callable.  The end curvatures are the radial's own (finite
    differences on a finer grid), so the error is O(h^4) up to the ends; with 4096 intervals on
    the analytic tensor radial the features agree with the untabulated ones to ~1e-9 relative
    (tests/test_operators_features.py)."""
    n = n_intervals + 1
    r = np.linspace(r_min, rcut, n)
    h = r[1] - r[0]
    d = 1e-3 * h
    rows = []
    for a in range(n_species):
        row = []
        for b in range(n_species):
            za, zb = np.full(n, a, np.int32), np.full(n, b, np.int32)
            y = np.asarray(radial(jnp.asarray(r), za, zb), np.float64)
            ends = []
            for r0, sgn in ((r_min, 1.0), (rcut, -1.0)):             # one-sided second differences
                pts = jnp.asarray(r0 + sgn * d * np.arange(4))
                v = np.asarray(radial(pts, za[:4], zb[:4]), np.float64)
                ends.append(h * h * (2 * v[0] - 5 * v[1] + 4 * v[2] - v[3]) / (d * d))
            row.append(cubic_bspline_coefs(y, end_d2=tuple(ends)))
        rows.append(row)
    return RadialTable(coefs=jnp.asarray(np.asarray(rows)), grid=(float(r_min), float(h), int(n)))


class SiteBasis(eqx.Module):
    """radial: (r, zi, zj) -> (E, n_rnl), a `TensorRadial` or a `RadialTable`; aspec_r, aspec_y:
    the radial column and harmonic of each A; aa_specs: the products, one index array per
    correlation order; lps: the (L, parity) present; per (L, parity) the COO coupling as
    flat-index arrays."""

    radial: eqx.Module
    aspec_r: jax.Array
    aspec_y: jax.Array
    aa_specs: tuple
    flat: tuple                 # per lp: row * (2L+1) + m, int32 (nnz,)
    cols: tuple                 # per lp: AA column, int32 (nnz,)
    vals: tuple                 # per lp: float64 (nnz,)
    lps: tuple = eqx.field(static=True)
    n_B: tuple = eqx.field(static=True)
    elements: tuple = eqx.field(static=True)
    rcut: float = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    solid: bool = eqx.field(static=True, default=True)     # solid (r^l Y_lm) or spherical harmonics
    gamma: tuple = eqx.field(static=True, default=None)    # per lp: the smoothness prior of each feature

    def smoothness(self, L, p):
        """(n(L, p),) the algebraic smoothness prior gamma of each feature of symmetry (L, p)
        (`basis.prior.smoothness_prior` on the (n, l) of its one-particle functions, n the radial
        degree: the neighbour element folded into the radial index of a categorical basis, and
        the channel of an element embedding, are not counted)."""
        return np.asarray(self.gamma[self.lps.index((L, p))], float) if (L, p) in self.lps else np.zeros(0)

    def n(self, L, p):
        """Number of features of symmetry (L, p); 0 if the basis has none."""
        return self.n_B[self.lps.index((L, p))] if (L, p) in self.lps else 0

    def species_index(self, numbers):
        idx = {z: k for k, z in enumerate(self.elements)}
        try:
            return np.asarray([idx[int(z)] for z in numbers], np.int32)
        except KeyError as e:
            raise KeyError(f"the site basis has no element Z = {e.args[0]} (it has {self.elements})") from None

    @highest
    def aa(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """(n_nodes, n_AA): the pooled A basis and its products, once."""
        A = pooled_a(self.radial, rij, zi, zj, segment_ids, n_nodes, self.aspec_r, self.aspec_y, self.lmax, self.solid, mask=mask)
        return product_basis(A, self.aa_specs)

    @highest
    def from_aa(self, AA):
        """{(L, p): (n, n_B, 2L+1)} from an AA matrix (n, n_AA)."""
        out = {}
        for (L, p), nB, flat, cols, vals in zip(self.lps, self.n_B, self.flat, self.cols, self.vals):
            k = 2 * L + 1
            B = jax.ops.segment_sum((AA[:, cols] * vals).T, flat, num_segments=nB * k)
            out[(L, p)] = B.T.reshape(AA.shape[0], nB, k)
        return out

    def features(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """{(L, p): (n_nodes, n_B, 2L+1)}; zi, zj are species indices (`species_index`), one per edge."""
        return self.from_aa(self.aa(rij, zi, zj, segment_ids, n_nodes, mask))

    def with_table(self, n_intervals=4096):
        """This basis with its radial tabulated in r (`RadialTable`): the CPU evaluation form."""
        if isinstance(self.radial, RadialTable) or getattr(self.radial, "radial_kind", "") == "spline_factorised":
            return self                                 # the element-embedding radial is one small spline table already
        return dataclasses.replace(self, radial=tabulate(self.radial, len(self.elements), self.rcut, n_intervals))


def _all_bodies(bb):
    """Every body: the coupling finds which carry a function of each (L, parity)."""
    return True


def _coupled(radial, lmax_feat, mb, Rnl, Ylm, zs, rcut, solid, coupling_cache_dir, channels=1, natural_parity=False):
    """The `SiteBasis` of a radial and a many-body spec: every (L, +-1), L <= lmax_feat, that has a
    function; natural_parity: only (L, (-1)^L), what a model of vectors of harmonics reads
    (`centred`), so that the other parities are neither coupled nor evaluated."""
    lps = [(L, p) for L in range(lmax_feat + 1) for p in (1, -1) if not natural_parity or p == (-1) ** L]
    es = couple_equivariant_set(lps, mb, list(Rnl), list(Ylm), cache_dir=coupling_cache_dir)
    lps = [lp for lp in lps if es.A2B[lp].shape[0]]
    k = lambda lp: 2 * lp[0] + 1
    return SiteBasis(
        radial=radial,
        aspec_r=jnp.asarray([a[0] for a in es.aspec], jnp.int32),
        aspec_y=jnp.asarray([a[1] for a in es.aspec], jnp.int32),
        aa_specs=tuple(jnp.asarray(g, jnp.int32) for g in es.aa_specs),
        flat=tuple(jnp.asarray(es.A2B[lp].rows * k(lp) + es.A2B[lp].ms, jnp.int32) for lp in lps),
        cols=tuple(jnp.asarray(es.A2B[lp].cols, jnp.int32) for lp in lps),
        vals=tuple(jnp.asarray(es.A2B[lp].vals, jnp.float64) for lp in lps),
        lps=tuple(lps), n_B=tuple(int(es.A2B[lp].shape[0]) for lp in lps),
        elements=tuple(int(z) for z in zs), rcut=float(rcut), lmax=int(max(l for l, _ in Ylm)), solid=solid,
        gamma=tuple(tuple(float(g) for g in smoothness_prior([[((n - 1) // channels + 1, l) for n, l in bb] for bb in es.nnll[lp]]))
                    for lp in lps))


def random_embedding(n_elements, d, seed=0):
    """(n_elements, d) random element embedding, rows of unit norm: independent Gaussian entries
    from a seeded generator.  The usual TrACE choice: the elements need distinct directions, not
    meaningful ones (the coefficients are fitted on top)."""
    rows = np.random.default_rng(seed).normal(size=(n_elements, d))
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def build_embedding_site_basis(elements, order, totaldegree, lmax_feat, embedding="random", *, rcut, rows=None, d_max=None,
                               wL=1.5, reduction="pca", normalise=True, block_rule="symmetric", coupling_cache_dir=None, seed=0,
                               widths=None):
    """A `SiteBasis` with an element embedding (the species-compressed linear ACE, "TrACE";
    `basis.model.build_embedding_model`'s one-particle basis): the neighbour species enters the
    radial through a frozen embedding row, R_(n'k)l(r, Z1, Z2) = Q_n'(x) emb[Z2, k], so the
    number of features grows with the channel width d (d_max), not with the number of elements.
    Bodies are channel-diagonal; order nu has min(d_max, C(S+nu-1, nu)) channels.

    embedding: "random" (the default: `random_embedding` with `seed`), a JSON artefact path (keys
    Z, emb), a (Z, table) pair reduced by PCA, or "identity" (needs
    d_max >= number of elements; then channel k is "neighbours of element k", the categorical
    basis on these radials); rows: an explicit (S, d) block instead.  rcut is shared by every
    pair (ace1's uniform cutoffs).  Radials: Jacobi(4, 4) in the Agnesi coordinate, splined on
    100 nodes, one table for all species; spherical harmonics.  block_rule: "symmetric" counts
    the degree on the radial index alone (the default here), "ace1" as the categorical ace1_model.

    widths: the channels of each correlation order, instead of min(d_max, C(S+nu-1, nu)).  Order 1
    is linear in the neighbour density of each element, so fewer than S channels there lose
    information that no readout recovers (on a four-element alloy the on-site blocks went from 51
    meV to an eV; docs/dev/ham-results.md): compress the higher orders, widths = (S, d, ...).  With
    an embedding of fewer than max(widths) columns the first widths[nu-1] of them serve order nu."""
    from ..basis import radial_ace1 as ra
    from ..basis.embedding import embedding_rows, embedding_widths, read_embedding
    from ..basis.spec import build_embedding_spec
    zs = ri.resolve_elements(elements)
    S = len(zs)
    widths = embedding_widths(S, order, d_max=d_max) if widths is None else [int(w) for w in widths]
    d = max(widths)
    if rows is not None:
        emb = np.asarray(rows, np.float64)
        if emb.shape != (S, d):
            raise ValueError(f"rows must be ({S}, {d}) for these elements and widths, got {emb.shape}")
    elif embedding is None:
        raise ValueError("pass an embedding ('random', a path, (Z, table) or 'identity') or explicit rows")
    elif isinstance(embedding, str) and embedding == "random":
        emb = random_embedding(S, d, seed)
    elif isinstance(embedding, str) and embedding == "identity":
        if d < S:
            raise ValueError(f"embedding='identity' needs one channel per element: d_max={d_max} gives {d} for {S} elements")
        emb = np.eye(S, d)
    else:
        Z, table = read_embedding(embedding)[:2] if isinstance(embedding, (str, bytes)) or hasattr(embedding, "__fspath__") else embedding
        emb = embedding_rows(table, Z, zs, d=d, reduction=reduction, normalise=normalise)
    sp = build_embedding_spec(S, order, totaldegree, widths, wL=wL, maxl=lmax_feat, block_rule=block_rule, admit=_all_bodies)
    cut = ra.uniform_cutoffs(zs, rcut)
    single = ra.jacobi_spline_table(max(n for n, _ in sp.r1), 4.0, 4.0)[:, [n - 1 for n, _ in sp.r1]]
    z1 = jnp.zeros((1, 1, 1, 1))
    tr = TensorRadial(
        rnl_transform=jnp.asarray(ra.transform_table(S, cut, 2, 4)), rnl_envelope=jnp.asarray(ra.tensor_envelope_table(S)),
        pair_envelope=jnp.asarray(ra.pair_envelope_table(S, cut)), rnl_coefs=z1, rnl_Wnlq=z1,
        polys_A=jnp.zeros(1), polys_B=jnp.zeros(1), polys_C=jnp.zeros(1),
        rnl_coefs_single=jnp.asarray(single), rnl_embedding=jnp.asarray(emb),
        rnl_emb_nidx=jnp.asarray(sp.nidx), rnl_emb_kidx=jnp.asarray(sp.kidx),
        radial_kind="spline_factorised", rnl_grid=(-1.0, 2.0 / (ra.SPLINE_NODES - 1), ra.SPLINE_NODES))
    return _coupled(tr, lmax_feat, sp.mb, sp.rspec, sp.Ylm, zs, cut[2], False, coupling_cache_dir, channels=sp.d)


def build_site_basis(elements, order, totaldegree, lmax_feat, *, rcut, wL=1.5, coupling_cache_dir=None,
                     nmax_by_order=None, lmax_by_order=None, natural_parity=False, **radial):
    """A `SiteBasis` on the TotalDegree basis of `basis.model.build_model` (the same
    radials and one-particle basis, built here without the model), with every (L, +-1), L <= lmax_feat, that has a
    function.  lmax_feat = 2 * (largest orbital l).  `radial`: `build_model`'s
    r0, rin, radial_mode, seed, n_q_factor, rnl_basis.

    nmax_by_order / lmax_by_order (`spec.build_spec`'s caps): radial degree and l allowed in a
    body of each correlation order.  An operator block is to leading order linear in the
    neighbour density, each L part with its own radial profile, so the basis that pays is
    many radial functions at order 1 for l <= lmax_feat and few at higher orders
    (docs/dev/ham-results.md); a plain total-degree bound gives high l too few radials.  With
    caps, radials and harmonics the bodies do not use are not evaluated.

    natural_parity: keep only the features of (L, (-1)^L) (`_coupled`); at order 1 there are no
    others."""
    caps = None if nmax_by_order is None and lmax_by_order is None else (nmax_by_order, lmax_by_order)
    known = {"r0", "rin", "radial_mode", "seed", "n_q_factor", "rnl_basis"}
    if set(radial) - known:
        raise TypeError(f"build_site_basis: unknown arguments {sorted(set(radial) - known)}")
    rnl_basis = radial.pop("rnl_basis", "poly")
    if "radial_mode" in radial:
        radial["mode"] = radial.pop("radial_mode")
    zs = ri.resolve_elements(elements)
    mb, Rnl, Ylm = build_spec(len(zs), order, totaldegree, wL, caps=caps, admit=_all_bodies)
    t = ri.tensor_radial_init(zs, Rnl, rcut=rcut, **{"mode": "onehot", **radial})
    pair_env = ri.pair_radial_init(zs, 1, rcut=rcut, r0=radial.get("r0"), rin=radial.get("rin", 0.0))["pair_envelope"]
    tr = TensorRadial(
        rnl_transform=jnp.asarray(t["rnl_transform"]), rnl_envelope=jnp.asarray(t["rnl_envelope"]),
        pair_envelope=jnp.asarray(pair_env), rnl_coefs=jnp.zeros((1, 1, 1, 1)), rnl_Wnlq=jnp.asarray(t["rnl_Wnlq"]),
        polys_A=jnp.asarray(t["polys_A"]), polys_B=jnp.asarray(t["polys_B"]), polys_C=jnp.asarray(t["polys_C"]),
        radial_kind="analytic", rnl_basis=rnl_basis, rnl_grid=(0.0, 1.0, 2))
    # the categorical radial index carries the neighbour element, n = (n' - 1) NZ + z': the prior counts n'
    return _coupled(tr, lmax_feat, mb, Rnl, Ylm, zs, rcut, True, coupling_cache_dir, channels=len(zs), natural_parity=natural_parity)
