"""ACE model as an Equinox module: A -> AA -> B -> site energy.

Design notes that the plan fixes and this code must not quietly undo:

* **Edge-vector core.** `site_energies` takes edge vectors, species indices and
  segment ids -- never positions and a cell.  The LAMMPS contract has no cell
  (ghost atoms carry periodicity), so a positions+cell core could not serve it.
  `energy_from_positions` is the thin wrapper for callers that do have
  positions, matching lammps-jax's `energy_fn(positions, species, graph)`.

* **Pooling is swappable.** `pool_sparse` (segment_sum over an edge list) and
  `pool_dense` (masked sum over a fixed (n, K) neighbour axis) are
  interchangeable; everything downstream is per-node and layout-agnostic.

* **Radial coefficients are a live array leaf, not static.**  For the splined
  branch Wnlq is already folded into the spline coefficients by Julia's
  `splinify`, so the coefficients occupy Wnlq's place in the parameter tree.
  Keeping them a leaf is what keeps trainable radials reachable; the
  analytic branch will carry a true Wnlq alongside.

* **Precision is explicit.**  Nothing here calls `jax.config.update`.  Use
  `with jax.default_matmul_precision("highest")` (or the `highest_precision`
  helper) -- the TF32 default costs ~400x accuracy on this descriptor.
"""

from contextlib import contextmanager

import equinox as eqx
import jax
import jax.numpy as jnp

from .edge_model import (EdgeSiteModel, calibrate_edge_a, one_hot_selector,  # noqa: F401
                         with_edge_a_kind)
from .harmonics import real_solid_harmonics, real_spherical_harmonics
from .radial import (agnesi_normalized, env_ace1_poly1sr, env_poly1sr,
                     env_poly2sx, poly_recursion, spline_eval, spline_eval_pairs)


@contextmanager
def highest_precision():
    """Force true f32/f64 matmuls.  On Ampere+ the TF32 default silently costs
    ~400x accuracy (1.17e-3 vs 2.93e-6 on this descriptor)."""
    with jax.default_matmul_precision("highest"):
        yield


def pool_sparse(edge_feats, segment_ids, n_nodes, mask=None):
    """Sum edge features into nodes over a sparse edge list."""
    if mask is not None:
        edge_feats = jnp.where(mask[:, None], edge_feats, 0.0)
    return jax.ops.segment_sum(edge_feats, segment_ids, num_segments=n_nodes,
                               indices_are_sorted=False)


def pool_dense(edge_feats, mask):
    """Sum over a fixed neighbour axis: edge_feats (n_nodes, K, F), mask (n_nodes, K)."""
    return jnp.sum(jnp.where(mask[..., None], edge_feats, 0.0), axis=1)


class ACEModel(EdgeSiteModel):
    # ---- array leaves (parameters) ----
    # radial: exactly one branch is populated per basis, chosen by radial_kind.
    # Both are live array leaves (never static): for the splined branch Julia's
    # `splinify` has already folded Wnlq into the coefficients, so they occupy
    # Wnlq's place in the parameter tree; the analytic branch carries a true
    # trainable Wnlq, which is what training would need.
    rnl_coefs: jax.Array          # spline:   (NZ, NZ, ncoef, n_rnl)
    pair_coefs: jax.Array         # spline:   (NZ, NZ, ncoef, n_pair)
    rnl_Wnlq: jax.Array           # analytic: (NZ, NZ, n_rnl, n_q)
    pair_Wnlq: jax.Array          # analytic: (NZ, NZ, n_pair, n_q)
    polys_A: jax.Array            # analytic: (n_q,)
    polys_B: jax.Array
    polys_C: jax.Array
    pair_polys_A: jax.Array
    pair_polys_B: jax.Array
    pair_polys_C: jax.Array
    rnl_transform: jax.Array      # (NZ, NZ, 7)
    pair_transform: jax.Array     # (NZ, NZ, 7)
    rnl_envelope: jax.Array       # (NZ, NZ, 5)
    pair_envelope: jax.Array      # (NZ, NZ, 3)
    A2B: jax.Array                # (n_B, n_AA) dense
    # A2B is extremely sparse -- typically one nonzero per column, 0.07%
    # occupied at 1429 basis functions -- so the dense contraction does
    # n_B * n_AA multiply-adds per node where nnz would do.  These triplets
    # drive the sparse path; see `a2b_sparse`.
    a2b_rows: jax.Array
    a2b_cols: jax.Array
    a2b_vals: jax.Array
    WB: jax.Array                 # (n_B, NZ)
    Wpair: jax.Array              # (n_pair, NZ)
    E0: jax.Array                 # (NZ,)
    # ---- static structure ----
    # index arrays are int32 leaves, not static fields: marking a JAX array
    # static warns and is a mistake; integer leaves are simply not differentiated
    aspec_r: jax.Array
    aspec_y: jax.Array
    aa_specs: tuple                               # per-order (n_v, order) int arrays
    lmax: int = eqx.field(static=True)
    ysolid: bool = eqx.field(static=True)
    a2b_sparse: bool = eqx.field(static=True)
    radial_kind: str = eqx.field(static=True)        # "spline" | "analytic"
    pair_radial_kind: str = eqx.field(static=True)
    pair_envelope_kind: str = eqx.field(static=True)  # "ace1_poly1sr" | "poly1sr"
    rnl_grid: tuple = eqx.field(static=True)      # (x0, h, n)
    pair_grid: tuple = eqx.field(static=True)
    elements: tuple = eqx.field(static=True)
    # Defaulted fields must come last (dataclass ordering).
    edge_a_kind: str = eqx.field(static=True, default="gather")   # "gather" | "matmul"
    # factorised radial (frozen embedding + uniform cutoffs): one species-
    # independent spline table plus the (NZ, d) embedding, instead of an
    # (NZ, NZ, ncoef, n_rnl) table that is O(S^2) to store and to gather
    rnl_coefs_single: jax.Array = None            # (ncoef, n1)
    rnl_embedding: jax.Array = None               # (NZ, d)
    rnl_emb_nidx: jax.Array = None                # (n_rnl,) -> column of the table
    rnl_emb_kidx: jax.Array = None                # (n_rnl,) -> embedding channel
    # one-hot selectors for the "matmul" form; None on the gather path so models
    # that never use it do not carry the arrays
    a_sel_r: jax.Array = None                     # (n_rnl, n_A)
    a_sel_y: jax.Array = None                     # (n_ylm, n_A)
    # C-tilde readout (PACE's ctilde): ctilde = A2B^T @ WB, (n_AA, NZ).  When
    # `folded`, site energies contract AA directly against it and the A2B
    # contraction never runs.  `site_basis` keeps using A2B, since descriptors
    # and fitting need B itself.  See `fold_readout`.
    ctilde: jax.Array = None
    folded: bool = eqx.field(static=True, default=False)
    # The pair readout Wpair[:, z] folded into the pair radial table (`fold_pair`,
    # part of `lean`): the energy is unchanged, but the pair channel is no longer
    # Apair, so the basis methods (descriptors, Jacobians, fitting rows) refuse.
    energy_only: bool = eqx.field(static=True, default=False)
    # l-blocked dense A (`block_dense`, part of `lean`): per used l, (l, radial
    # column offset, width).  () keeps the n_rnl x n_Y outer product (`pool_a_dense`).
    # With `blk_compact` each edge evaluates only its own z_j's columns, from the
    # compacted table `blk_rnl_coefs`, expanded by a one-hot over z_j.
    # `blk_aa_specs` index the concatenated blocks; `aa_specs` stay in A-basis
    # order for the sparse layout and the basis methods.
    blk: tuple = eqx.field(static=True, default=())
    blk_compact: bool = eqx.field(static=True, default=False)
    blk_aa_specs: tuple = None
    blk_rnl_coefs: jax.Array = None                  # (NZ, NZ, ncoef, sum_l w_l)

    # -------------------------------------------------- edge embeddings
    def _radial_one(self, r, zi, zj, kind, trans, coefs, grid, Wnlq, ABC, env):
        """One radial basis, either branch.  `env` is the already-evaluated
        envelope; both branches multiply by it identically."""
        x = agnesi_normalized(r, trans[zi, zj])
        if kind == "spline":
            x0, h, n = grid
            val = spline_eval_pairs(x, coefs, zi, zj, x0, h, n)     # never coefs[zi, zj]
        elif kind == "analytic":
            P = poly_recursion(x, *ABC)                        # (E, n_q)
            val = jnp.einsum("eq,enq->en", P, Wnlq[zi, zj])    # (E, n_rnl)
        else:
            raise ValueError(f"unknown radial_kind {kind!r}")
        return val * env[:, None]

    def radial(self, rij, zi, zj):
        """Rnl and the pair radial for each edge.  rij (E,3), zi/zj (E,) species indices."""
        r = jnp.linalg.norm(rij, axis=-1)
        # many-body envelope is applied in transformed coordinates
        env = env_poly2sx(agnesi_normalized(r, self.rnl_transform[zi, zj]),
                          self.rnl_envelope[zi, zj])
        if self.radial_kind == "spline_factorised":
            # Rnl[e, i] = P[e, n'(i)] * emb[zj[e], k(i)]
            # The spline table is species-independent, so there is no
            # (E, ncoef, n_rnl) gather here at all -- that gather is what makes
            # many-element models memory-bandwidth bound.
            x = agnesi_normalized(r, self.rnl_transform[zi, zj])
            x0, h, nsp = self.rnl_grid
            P = jax.vmap(lambda xx: spline_eval(xx, self.rnl_coefs_single,
                                                x0, h, nsp))(x)          # (E, n1)
            Rnl = (P[:, self.rnl_emb_nidx]
                   * self.rnl_embedding[zj][:, self.rnl_emb_kidx]) * env[:, None]
        else:
            Rnl = self._radial_one(r, zi, zj, self.radial_kind, self.rnl_transform,
                                   self.rnl_coefs, self.rnl_grid, self.rnl_Wnlq,
                                   (self.polys_A, self.polys_B, self.polys_C), env)
        # pair envelope is a function of r, and its form differs by model family
        pe = self.pair_envelope[zi, zj]
        envp = (env_ace1_poly1sr(r, pe) if self.pair_envelope_kind == "ace1_poly1sr"
                else env_poly1sr(r, pe))
        Rpair = self._radial_one(r, zi, zj, self.pair_radial_kind, self.pair_transform,
                                 self.pair_coefs, self.pair_grid, self.pair_Wnlq,
                                 (self.pair_polys_A, self.pair_polys_B, self.pair_polys_C),
                                 envp)
        return Rnl, Rpair

    def angular(self, rij):
        # ace1_model uses SPHERICAL harmonics (ace1_compat.jl:407, Ytype=:spherical);
        # ace_model defaults to :solid.  The earlier prototype used ace_model, so the
        # production path differs from it here -- hence the exported flag.
        #
        # Pure JAX, not sphericart-jax: the latter lowers to an FFI custom call,
        # which the LAMMPS bundle would then have to resolve at run time.
        return (real_solid_harmonics(rij, self.lmax) if self.ysolid
                else real_spherical_harmonics(rij, self.lmax))

    # -------------------------------------------------- many-body
    def edge_features(self, rij, zi, zj):
        """Per-edge (A-basis rows, pair rows).  Layout-agnostic: the caller
        pools these however its neighbour-list layout dictates.  The A-basis
        product and its two forms live in `EdgeSiteModel.edge_a`."""
        Rnl, Rpair = self.radial(rij, zi, zj)
        return self.edge_a(Rnl, self.angular(rij)), Rpair

    # -------------------------------------------------- EdgeSiteModel hooks
    a_channels = 1          # species enter through the radial, not a channel

    def _edge_factors_and_pair(self, rij, zi, zj, mask=None):
        """(Rnl, Ylm, Rpair) per edge, Rnl zeroed on masked edges; one radial pass."""
        Rnl, Rpair = self.radial(rij, zi, zj)
        if mask is not None:
            Rnl = jnp.where(mask[:, None], Rnl, 0.0)
        return Rnl, self.angular(rij), Rpair

    def edge_a_factors(self, rij, zi, zj, mask=None):
        """The two per-edge factors of the A basis: (Rnl, Ylm)."""
        Rnl, Y, _ = self._edge_factors_and_pair(rij, zi, zj, mask)
        return Rnl, Y

    def pad_cutoff(self):
        """Padded edges sit at the pair cutoff, where every envelope vanishes."""
        return jnp.max(self.pair_envelope[..., 0])        # traceable: no float()

    def edge_a_widths(self):
        """(radial columns, harmonic columns) the A-basis selectors index."""
        n_rnl = (self.rnl_emb_nidx.shape[0] if self.radial_kind == "spline_factorised"
                 else self.rnl_coefs.shape[-1] if self.radial_kind == "spline"
                 else self.rnl_Wnlq.shape[-2])
        return n_rnl, (self.lmax + 1) ** 2

    def _check_basis(self):
        if self.energy_only:
            raise ValueError("an energy-only (lean) ACEModel has its pair readout folded into "
                             "the pair radial, so it has no basis: use the model `lean` was "
                             "applied to for descriptors, Jacobians or fitting")

    def require_full(self, what="this"):
        """Raise unless this is a full model: not `energy_only` (`fold_pair`) and
        not l-blocked (`block_dense`).  A lean model holds the radial twice
        (`rnl_coefs` for the sparse layout, `blk_rnl_coefs` for the dense one) and
        its pair channel is the pair energy, so editing its radials or weights
        would desynchronise the layouts.  Edit the full model, then re-apply
        `lean`."""
        if self.energy_only or self.blk:
            raise ValueError(f"{what} needs the full ACEModel, not its lean evaluation form "
                             "(ace_jax.eval.model.lean): edit the full model and re-apply lean")

    def _aa(self, A, specs=None):
        specs = self.aa_specs if specs is None else specs
        return jnp.concatenate([jnp.prod(A[:, g], axis=-1) for g in specs], axis=-1)

    def _from_pooled(self, A, Apair):
        AA = self._aa(A)
        if self.a2b_sparse:
            # gather the nnz contributing columns and scatter into basis rows.
            # There is usually exactly one nonzero per column, so this replaces
            # an (n_B x n_AA) matmul with an nnz-length gather.
            contrib = AA[:, self.a2b_cols] * self.a2b_vals          # (n_nodes, nnz)
            B = jax.ops.segment_sum(contrib.T, self.a2b_rows,
                                    num_segments=self.A2B.shape[0]).T
        else:
            B = AA @ self.A2B.T
        return B, Apair

    def _readout_folded(self, A, Apair, node_z, specs=None):
        e = jnp.einsum("ia,ai->i", self._aa(A, specs), self.ctilde[:, node_z])
        e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
        return e + self.E0[node_z]

    def site_basis(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """Sparse (edge-list) pooling -- the layout lammps-jax exports."""
        self._check_basis()
        edge_A, Rpair = self.edge_features(rij, zi, zj)
        return self._from_pooled(pool_sparse(edge_A, segment_ids, n_nodes, mask),
                                 pool_sparse(Rpair, segment_ids, n_nodes, mask))

    def site_basis_dense(self, rij, zi, zj, mask):
        """Dense (n, K) pooling -- matscipy-neighbours' `neighbour_matrix` form,
        which maps onto ET's own (maxneigs, nnodes, nfeat) layout and needs no
        scatter.  rij (n,K,3), zi/zj (n,K), mask (n,K)."""
        self._check_basis()
        n, K = mask.shape
        flat = lambda a: a.reshape(n * K, *a.shape[2:])
        edge_A, Rpair = self.edge_features(flat(rij), flat(zi), flat(zj))
        un = lambda a: a.reshape(n, K, -1)
        return self._from_pooled(pool_dense(un(edge_A), mask), pool_dense(un(Rpair), mask))

    def site_descriptors(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        """Per-site descriptors, (n_nodes, (n_B + n_pair) * NZ).

        Parity target is `ACEpotentials.site_descriptors` (src/descriptor.jl).
        Layout is species-blocked exactly as `get_basis_inds` /
        `get_pairbasis_inds` define it (src/models/ace.jl:544-566): the centre
        species selects which block is populated and the rest are zero, which is
        what the readout contracts against.

        The Julia version is marked "RETIRING THIS FOR NOW BECAUSE IT IS HIGHLY
        INEFFICIENT" because it recomputes per site.  This one takes the whole
        batch from a single forward pass -- the same pass the energy uses.
        """
        B, Apair = self.site_basis(rij, zi, zj, segment_ids, n_nodes, mask)
        n_B, n_pair, nz_count = B.shape[1], Apair.shape[1], self.WB.shape[1]
        out = jnp.zeros((n_nodes, (n_B + n_pair) * nz_count), B.dtype)
        rows = jnp.arange(n_nodes)
        # scatter each site's blocks into the slot its centre species owns
        b_off = node_z * n_B
        p_off = nz_count * n_B + node_z * n_pair
        out = out.at[rows[:, None], b_off[:, None] + jnp.arange(n_B)].set(B)
        out = out.at[rows[:, None], p_off[:, None] + jnp.arange(n_pair)].set(Apair)
        return out

    # -------------------------------------------------- compact basis + Jacobian
    def compact_basis(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """Per-site (B, Apair) concatenated, (n_nodes, n_B + n_pair) -- the
        species-agnostic vector the GP kernel acts on.  `site_descriptors` is
        this scattered into species blocks."""
        B, Apair = self.site_basis(rij, zi, zj, segment_ids, n_nodes, mask)
        return jnp.concatenate([B, Apair], axis=1)

    def edge_jacobian(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """Compact basis X (n_nodes, D) and its edge Jacobian
        J (n_edges, D, 3) = dX[segment_ids[e]] / d rij[e].

        Hybrid push, the way ET._jacobian_X works: the per-edge derivative of
        the edge features is analytic (jacfwd over a 3-vector), then pushed
        through the per-node map A -> AA -> B with one jacrev per node.  A
        naive vmap-VJP over basis rows is 6-8x slower on CPU and 7.5-13x (f64) /
        12-23x (f32) on GPU
        (docs/plans/jax_ace_port_plan.md, decision gate 3).  Padded edges
        (mask False) contribute zero rows.
        """
        self._check_basis()
        def feats(r1, z1, z2):
            eA, Rp = self.edge_features(r1[None, :], z1[None], z2[None])
            return eA[0], Rp[0]

        edge_A, Rpair = jax.vmap(feats)(rij, zi, zj)
        dA, dRp = jax.vmap(jax.jacfwd(feats))(rij, zi, zj)     # (E, n_A, 3), (E, n_pair, 3)
        A = pool_sparse(edge_A, segment_ids, n_nodes, mask)

        def node_B(A_i):
            AA = jnp.concatenate([jnp.prod(A_i[g], axis=-1) for g in self.aa_specs])
            return AA @ self.A2B.T

        J_BA = jax.vmap(jax.jacrev(node_B))(A)                  # (n_nodes, n_B, n_A)
        J_B = jnp.einsum("ebm,emc->ebc", J_BA[segment_ids], dA) # (E, n_B, 3)
        J = jnp.concatenate([J_B, dRp], axis=1)                 # (E, D, 3)
        if mask is not None:
            J = jnp.where(mask[:, None, None], J, 0.0)
        X = jnp.concatenate([jax.vmap(node_B)(A),
                             pool_sparse(Rpair, segment_ids, n_nodes, mask)], axis=1)
        return X, J

    def edge_jacobian_dense(self, rij, zi, zj, mask):
        """`edge_jacobian` for the dense (n, K) neighbour layout, returning
        X (n, D) and J (n*K, D, 3) in node-major edge order (edge e = n*K + k).

        Same hybrid push, but the per-node dB/dA (n_B, n_A) is contracted
        against each node's own K edges with a batched matmul instead of
        being gathered onto every edge first: the gathered form is
        (E, n_B, n_A) -- 14 GB at 1348 basis functions x 72 A-functions on
        an 18k-edge batch -- while this needs only (n, n_B, n_A) + the output.
        """
        self._check_basis()
        n, K = mask.shape
        flat = lambda a: a.reshape(n * K, *a.shape[2:])

        def feats(r1, z1, z2):
            eA, Rp = self.edge_features(r1[None, :], z1[None], z2[None])
            return eA[0], Rp[0]

        edge_A, Rpair = jax.vmap(feats)(flat(rij), flat(zi), flat(zj))
        dA, dRp = jax.vmap(jax.jacfwd(feats))(flat(rij), flat(zi), flat(zj))   # (nK, n_A, 3), (nK, n_pair, 3)
        A = pool_dense(edge_A.reshape(n, K, -1), mask)                          # (n, n_A)

        def node_B(A_i):
            AA = jnp.concatenate([jnp.prod(A_i[g], axis=-1) for g in self.aa_specs])
            return AA @ self.A2B.T

        # forward mode: n_A tangents (72 at the Cantor basis) instead of n_B
        # cotangents (1348) pulled back through n_AA products -- jacrev here
        # materialises (n, n_B, n_AA) intermediates, 11 GB per batch
        J_BA = jax.vmap(jax.jacfwd(node_B))(A)                                  # (n, n_B, n_A)
        J_B = jnp.einsum("nbm,nkmc->nkbc", J_BA, dA.reshape(n, K, -1, 3))       # (n, K, n_B, 3)
        J = jnp.concatenate([J_B.reshape(n * K, -1, 3), dRp], axis=1)          # (nK, D, 3)
        J = jnp.where(flat(mask)[:, None, None], J, 0.0)
        X = jnp.concatenate([jax.vmap(node_B)(A), pool_dense(Rpair.reshape(n, K, -1), mask)], axis=1)
        return X, J

    def pair_radial(self, rij, zi, zj):
        """The pair radial alone, (E, n_pair): the second output of `radial`
        without the many-body Rnl."""
        r = jnp.linalg.norm(rij, axis=-1)
        pe = self.pair_envelope[zi, zj]
        envp = (env_ace1_poly1sr(r, pe) if self.pair_envelope_kind == "ace1_poly1sr"
                else env_poly1sr(r, pe))
        return self._radial_one(r, zi, zj, self.pair_radial_kind, self.pair_transform,
                                self.pair_coefs, self.pair_grid, self.pair_Wnlq,
                                (self.pair_polys_A, self.pair_polys_B, self.pair_polys_C),
                                envp)

    def pair_features_dense(self, rij, zi, zj, mask):
        """The pair-density channels only: per-site Xpair (n, n_pair) and the
        per-edge Jacobian (n*K, n_pair, 3), node-major -- exactly the pair slices
        of `edge_jacobian_dense`'s X and J, with no many-body basis evaluated."""
        self._check_basis()
        n, K = mask.shape
        flat = lambda a: a.reshape(n * K, *a.shape[2:])
        f = lambda r1, z1, z2: self.pair_radial(r1[None, :], z1[None], z2[None])[0]
        Rp = jax.vmap(f)(flat(rij), flat(zi), flat(zj))                     # (nK, n_pair)
        dRp = jax.vmap(jax.jacfwd(f))(flat(rij), flat(zi), flat(zj))       # (nK, n_pair, 3)
        dRp = jnp.where(flat(mask)[:, None, None], dRp, 0.0)
        return pool_dense(Rp.reshape(n, K, -1), mask), dRp

    def _readout(self, B, Apair, node_z):
        e = jnp.einsum("ib,bi->i", B, self.WB[:, node_z])
        e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
        return e + self.E0[node_z]

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        """Per-site energies (n_nodes,).  `node_z` is the centre species index per node."""
        if self.folded:
            Rnl, Y, Rpair = self._edge_factors_and_pair(rij, zi, zj, mask)
            A = self.pool_a_sparse(Rnl, Y, segment_ids, 0, n_nodes)
            return self._readout_folded(A, pool_sparse(Rpair, segment_ids, n_nodes, mask),
                                        node_z)
        return self._readout(*self.site_basis(rij, zi, zj, segment_ids, n_nodes, mask), node_z)

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        if self.blk:
            return self._site_energies_blocked(rij, zi, zj, mask, node_z)
        if self.folded:
            # A by the batched outer product over each node's K slots (no
            # per-edge column gather); the pair channel is a plain masked sum
            n, K = mask.shape
            flat = lambda a: a.reshape(n * K, *a.shape[2:])
            Rnl, Y, Rpair = self._edge_factors_and_pair(flat(rij), flat(zi), flat(zj),
                                                        flat(mask))
            un = lambda a: a.reshape(n, K, -1)
            A = self.pool_a_dense(un(Rnl), un(Y), None)
            return self._readout_folded(A, pool_dense(un(Rpair), mask), node_z)
        return self._readout(*self.site_basis_dense(rij, zi, zj, mask), node_z)

    def _site_energies_blocked(self, rij, zi, zj, mask, node_z):
        """The dense energy with A pooled per l-block (`block_dense`):
        A_l = sum_k R_{.,l} (x) Y_l, (w_l (2l+1)) per node, instead of the full
        n_rnl x n_Y outer product of which only the l(R) = l(Y) diagonal blocks
        are read; no column select follows.  Pooled feature-major and handed to
        the product basis transposed: without that, XLA's layout for the
        order-3/4 product adjoint regresses (docs/ace-vs-pace-gap.md 4.1)."""
        n, K = mask.shape
        E = n * K
        r3, zi_, zj_, mk = rij.reshape(E, 3), zi.reshape(E), zj.reshape(E), mask.reshape(E)
        if self.blk_compact:
            r = jnp.linalg.norm(r3, axis=-1)
            env = env_poly2sx(agnesi_normalized(r, self.rnl_transform[zi_, zj_]),
                              self.rnl_envelope[zi_, zj_])
            R = self._radial_one(r, zi_, zj_, "spline", self.rnl_transform, self.blk_rnl_coefs,
                                 self.rnl_grid, None, None, env)
            Rpair = self.pair_radial(r3, zi_, zj_)
        else:
            R, Rpair = self.radial(r3, zi_, zj_)
        R = jnp.where(mk[:, None], R, 0.0)
        Y = self.angular(r3)
        nz = self.E0.shape[0]
        oh = jax.nn.one_hot(zj_, nz, dtype=R.dtype) if self.blk_compact else None
        hi = jax.lax.Precision.HIGHEST
        At = []
        for l, off, w in self.blk:
            Rl = R[:, off:off + w]
            if self.blk_compact:
                Rl = (oh[:, :, None] * Rl[:, None, :]).reshape(E, nz * w)
            Yl = Y[:, l * l:(l + 1) ** 2]
            At.append(jnp.einsum("nkr,nky->ryn", Rl.reshape(n, K, -1), Yl.reshape(n, K, -1),
                                 precision=hi).reshape(-1, n))
        A = jnp.concatenate(At, axis=0).T
        return self._readout_folded(A, pool_dense(Rpair.reshape(n, K, -1), mask), node_z,
                                    self.blk_aa_specs)


# ------------------------------------------------------------------ readout fold
def fold_readout(model):
    """Return `model` with the linear readout folded through A2B.

    e_i = WB[:,z] . (A2B AA_i)  ==  (A2B^T WB[:,z]) . AA_i, so ctilde = A2B^T WB
    is computed once here and the A2B contraction -- the largest isolated stage at
    production basis size -- and its adjoint never run.  Exact to roundoff
    (relative ~1e-16; 1e-12 absolute on the fitted test fixtures, ~1e-10 absolute
    on a random-weight 2849-function fixture with E ~= -6e5 eV -- see
    `bench/results.md` "Lever rows"; tests/test_fold.py holds it to 1e-12 on the
    committed fixtures).  This is the same fold as PACE's ctilde basis.

    `WB` is retained on the model after folding -- `site_basis`/`site_descriptors`
    still use it via `A2B` -- but it is no longer on the energy path once folded:
    a gradient w.r.t. `WB` on a folded model is zero. Anyone replacing `WB` on a
    folded model must re-fold it (`dataclasses.replace(model, WB=..., folded=False)`
    then `fold_readout(...)`), or `ctilde` stays stale and site energies keep
    using the old weights.
    """
    import dataclasses
    if model.folded:
        return model
    with highest_precision():                 # TF32 would corrupt ctilde on Ampere+
        ctilde = model.A2B.T @ model.WB       # (n_AA, NZ)
    return dataclasses.replace(model, ctilde=ctilde, folded=True)


# ------------------------------------------------------------------ lean evaluation form
# Exact, static data transforms of a loaded (folded) model that drop per-edge work
# the energy never reads (docs/ace-vs-pace-gap.md sections 1, 3.2 and 4.1).  They
# are for evaluation only -- ACECalculator and export_lammps apply `lean` -- and
# never for fitting, which needs the full basis: `load` returns the full model.
import dataclasses as _dc  # noqa: E402

import numpy as _np  # noqa: E402


def _l_of_y(y):
    return _np.floor(_np.sqrt(_np.asarray(y))).astype(int)


def _reselect(model):
    """Rebuild the one-hot A selectors after the radial / harmonic widths changed."""
    if model.edge_a_kind != "matmul":
        return model
    m = _dc.replace(model, edge_a_kind="gather", a_sel_r=None, a_sel_y=None)
    return with_edge_a_kind(m, "matmul")


def _rnl_owner(model):
    """(n_rnl,) the neighbour species each R_nl column is nonzero for, or None
    when R_nl is not block-sparse in z_j.  ACE1's splined R_nl always is: every
    column is nonzero for exactly one z_j, whatever z_i (checked here, not
    assumed), and so do learned radials splined by `to_spline`: learning keeps
    the zero rows exactly zero, and a zero row tabulates to exact zeros.  Spline
    tables only; the other radial kinds return None (`lean` splines an analytic
    radial first)."""
    if model.radial_kind != "spline":
        return None
    nzm = _np.abs(_np.asarray(model.rnl_coefs)).max(axis=2) > 0         # (zi, zj, r)
    own = nzm.any(axis=0)                                               # (zj, r)
    if (own.sum(axis=0) > 1).any() or not (nzm == nzm[:1]).all():
        return None
    return own.argmax(axis=0)


def prune_columns(model):
    """Drop the R_nl columns no A entry reads, and the Y_lm above the largest l
    any A entry uses.  Exact: the A basis (hence B, Apair and the energy) is
    unchanged, only never-read columns go.

    ACEpotentials exports lmax from the total degree, but a product of order nu
    needs nu factors whose l couple to 0 within the degree budget, so no A entry
    reaches it (lmax 4 -> 2 on Cantor_medium); and about 40% of the R_nl columns
    are only in other species blocks' budgets (99 -> 58).  Columns are re-sorted
    by (l, neighbour species, original index), so each l is one contiguous block
    (`block_dense`).  Returns `model` itself when nothing changes."""
    ar, ay = _np.asarray(model.aspec_r), _np.asarray(model.aspec_y)
    la = _l_of_y(ay)
    lmax = int(la.max()) if la.size else 0
    l_of = {}
    for r, l in zip(ar.tolist(), la.tolist()):
        l_of.setdefault(r, set()).add(l)
    own = _rnl_owner(model)
    order = sorted(l_of, key=lambda r: (min(l_of[r]), 0 if own is None else int(own[r]), r))
    n_rnl = model.edge_a_widths()[0]
    if order == list(range(n_rnl)) and lmax == model.lmax:
        return model
    cols = _np.asarray(order)
    new = _np.zeros(n_rnl, _np.int64)
    new[cols] = _np.arange(len(cols))
    kw = {}
    if model.radial_kind == "spline":
        kw["rnl_coefs"] = model.rnl_coefs[..., cols]
    elif model.radial_kind == "analytic":
        kw["rnl_Wnlq"] = model.rnl_Wnlq[..., cols, :]
    elif model.radial_kind == "spline_factorised":
        kw.update(rnl_emb_nidx=model.rnl_emb_nidx[cols], rnl_emb_kidx=model.rnl_emb_kidx[cols])
    else:
        raise ValueError(f"unknown radial_kind {model.radial_kind!r}")
    return _reselect(_dc.replace(model, **kw, lmax=lmax,
                                 aspec_r=jnp.asarray(new[ar], model.aspec_r.dtype)))


def fold_pair(model):
    """Fold the pair readout into the pair radial: one pair column per edge
    instead of n_pair, e_pair,i = sum_j sum_p Rpair_p(r_ij) Wpair[p, z_i] with
    the sum over p done once here, on the (linear) spline or Wnlq coefficients.
    The same fold as `fold_readout`, for the pair channel.  Exact to roundoff.

    The result is `energy_only`: its pair channel is now the pair energy, not
    Apair, so the basis methods refuse.  Needs a folded model (the unfolded
    energy materialises the basis).
    Edit the full model, never this one: a lean model holds the radial twice
    and its pair channel is the readout, so replacing rnl_coefs, Wnlq, Wpair,
    WB or ctilde on it changes one layout and not the other (`require_full`
    guards the radial helpers).  Change the full model and re-apply `lean`."""
    if not model.folded:
        raise ValueError("fold_pair needs a folded model (fold_readout first)")
    if model.energy_only:
        return model
    dt = model.Wpair.dtype
    W = _np.asarray(model.Wpair, _np.float64)                             # (n_pair, NZ)
    if model.pair_radial_kind == "spline":
        pc = _np.asarray(model.pair_coefs, _np.float64)                   # (zi, zj, c, p)
        kw = {"pair_coefs": jnp.asarray(_np.einsum("ijcp,pi->ijc", pc, W)[..., None],
                                        model.pair_coefs.dtype)}
    elif model.pair_radial_kind == "analytic":
        pw = _np.asarray(model.pair_Wnlq, _np.float64)                    # (zi, zj, p, q)
        kw = {"pair_Wnlq": jnp.asarray(_np.einsum("ijpq,pi->ijq", pw, W)[:, :, None, :],
                                       model.pair_Wnlq.dtype)}
    else:
        raise ValueError(f"unknown pair_radial_kind {model.pair_radial_kind!r}")
    return _dc.replace(model, **kw, Wpair=jnp.ones((1, W.shape[1]), dt), energy_only=True)


def block_dense(model):
    """The l-blocked dense A (`ACEModel._site_energies_blocked`), species-compact
    when R_nl is block-sparse in z_j and there is more than one species.

    Per edge the compact form evaluates only the w_l columns of its own z_j (a
    (NZ, NZ, ncoef, sum w_l) table, zero-padded to the widest species) and
    expands them by a one-hot over z_j: PACE's neighbour-species channel, which
    ACE1 folds into the radial index.  On Cantor_medium that is 12 live columns
    of 58 per edge.  Applies `prune_columns` first (the blocks need each l's
    columns contiguous).  Needs a folded model.  Returns the pruned model
    unblocked when a radial column is used at more than one l.
    Edit the full model, never this one: a lean model holds the radial twice
    and its pair channel is the readout, so replacing rnl_coefs, Wnlq, Wpair,
    WB or ctilde on it changes one layout and not the other (`require_full`
    guards the radial helpers).  Change the full model and re-apply `lean`."""
    if not model.folded:
        raise ValueError("block_dense needs a folded model (fold_readout first)")
    if model.blk:
        return model
    m = prune_columns(model)
    ar, ay = _np.asarray(m.aspec_r), _np.asarray(m.aspec_y)
    la = _l_of_y(ay)
    l_of = {}
    for r, l in zip(ar.tolist(), la.tolist()):
        if l_of.setdefault(r, l) != l:
            return m
    own = _rnl_owner(m)
    nz = int(m.E0.shape[0])
    compact = own is not None and nz > 1
    coefs = _np.asarray(m.rnl_coefs) if compact else None
    blocks, pos_of, tabs, base, off = [], {}, [], 0, 0
    for l in sorted(set(l_of.values())):
        cols = [r for r in sorted(l_of) if l_of[r] == l]
        nm = 2 * l + 1
        if compact:
            per = [[r for r in cols if own[r] == z] for z in range(nz)]
            w = max(len(v) for v in per)
            tab = _np.zeros(coefs.shape[:3] + (w,), coefs.dtype)
            for z, rs in enumerate(per):
                for k, r in enumerate(rs):
                    tab[:, z, :, k] = coefs[:, z, :, r]
                    for mm in range(nm):
                        pos_of[(r, mm)] = base + (z * w + k) * nm + mm
            tabs.append(tab)
            blocks.append((l, off, w))
            off += w
        else:
            w = len(cols)
            if cols != list(range(cols[0], cols[0] + w)):         # prune_columns sorts by l
                return m
            for k, r in enumerate(cols):
                for mm in range(nm):
                    pos_of[(r, mm)] = base + k * nm + mm
            blocks.append((l, cols[0], w))
        base += (nz if compact else 1) * w * nm
    pos = _np.asarray([pos_of[(int(r), int(y) - l * l)] for r, y, l in zip(ar, ay, la)])
    specs = tuple(jnp.asarray(pos[_np.asarray(g)], g.dtype) for g in m.aa_specs)
    return _dc.replace(m, blk=tuple(blocks), blk_compact=compact, blk_aa_specs=specs,
                       blk_rnl_coefs=(jnp.asarray(_np.concatenate(tabs, axis=-1), m.rnl_coefs.dtype)
                                      if compact else None))


def _splined(model, spline_tol):
    """`to_spline(model, tol=spline_tol)` when a radial is analytic, else `model`."""
    if spline_tol is None or "analytic" not in (model.radial_kind, model.pair_radial_kind):
        return model
    from .splinify import to_spline
    return to_spline(model, tol=spline_tol)[0]


def _wraps(model):
    """A wrapper model (e.g. an FSModel(base, ...)) that evaluates through an
    ACEModel's basis: it has `.base` and `with_base(new_base)`."""
    return hasattr(model, "base") and callable(getattr(model, "with_base", None))


def lean_keep_basis(model, spline_tol=1e-8):
    """The basis-preserving part of `lean`, for models that read the basis:
    `to_spline` (analytic radials only; agrees to `spline_tol`, see `lean`) and
    `prune_columns` (exact: B and Apair unchanged).  Never `fold_pair` or
    `block_dense`, so `site_basis`, `site_basis_dense`, `_readout` (WB, Wpair)
    and the descriptors keep working.  A wrapper model's `lean` applies this to
    its `.base`.  Anything that is not an ACEModel is returned as given.
    Needs the full model (`require_full`)."""
    if not isinstance(model, ACEModel):
        return model
    model.require_full("lean_keep_basis")
    return prune_columns(_splined(model, spline_tol))


def lean(model, spline_tol=1e-8):
    """The evaluation form of a folded ACEModel: `prune_columns`, `fold_pair`
    and the l-blocked dense A (`block_dense`).  Exact to roundoff in E, F and the
    virial for a splined model; 1.1-3.3x faster forces on the benchmark models
    (docs/ace-vs-pace-gap.md section 8).

    An analytic radial (a learned one, `rnl_Wnlq` / `pair_Wnlq`) is first
    splined by `to_spline(model, tol=spline_tol)`, so it gets the spline
    gather and, when each R_nl column belongs to one neighbour species (ACE1's
    pattern, which learned radials keep), the species-compact blocks.  That
    step is an approximation: the lean form of an analytic model agrees with
    the full one to about `spline_tol` (relative, per radial), not to roundoff.
    spline_tol=None keeps the analytic radial (exact, no compaction).  For
    evaluation and export only: fitting keeps the analytic model, and a UQ
    variance should come from the full model.

    A wrapper model (`.base` and `with_base`, e.g. FSModel(base, ...)) returns
    `model.with_base(lean_keep_basis(model.base, spline_tol))`: the wrapper
    reads the basis, so only the basis-preserving transforms apply.

    Energy only (see `fold_pair`): keep the original for descriptors and
    fitting.  Anything that is not a folded ACEModel (a PACEModel, an unfolded
    model) is returned as given, as is a model that is already lean.
    Edit the full model, never this one: a lean model holds the radial twice
    and its pair channel is the readout, so replacing rnl_coefs, Wnlq, Wpair,
    WB or ctilde on it changes one layout and not the other (`require_full`
    guards the radial helpers).  Change the full model and re-apply `lean`."""
    if _wraps(model):
        return model.with_base(lean_keep_basis(model.base, spline_tol))
    if not isinstance(model, ACEModel) or not model.folded or model.energy_only:
        return model
    return block_dense(fold_pair(prune_columns(_splined(model, spline_tol))))
