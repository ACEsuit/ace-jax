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

from .harmonics import real_solid_harmonics, real_spherical_harmonics
from .radial import (agnesi_normalized, env_ace1_poly1sr, env_poly1sr,
                     env_poly2sx, poly_recursion, spline_eval)


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


class ACEModel(eqx.Module):
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

    # -------------------------------------------------- edge embeddings
    def _radial_one(self, r, zi, zj, kind, trans, coefs, grid, Wnlq, ABC, env):
        """One radial basis, either branch.  `env` is the already-evaluated
        envelope; both branches multiply by it identically."""
        x = agnesi_normalized(r, trans[zi, zj])
        if kind == "spline":
            x0, h, n = grid
            val = jax.vmap(lambda xx, c: spline_eval(xx, c, x0, h, n))(x, coefs[zi, zj])
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
        pools these however its neighbour-list layout dictates.

        Two algebraically identical forms of the A-basis product, selected by
        `edge_a_kind`; they agree to bit-identity on values and gradients.

          "gather"  A = Rnl[:, aspec_r] * Ylm[:, aspec_y]
          "matmul"  A = (Rnl @ Sr) * (Ylm @ Sy),  Sr/Sy one-hot

        They differ only in the reverse pass: the gather's adjoint is an axis-1
        scatter whose cost per slot grows with buffer length, while the matmul's
        adjoint is a matmul and is flat.  Neither wins everywhere -- on an M3 Pro
        the matmul wins throughout and by up to 11x, on a Xeon the gather wins
        below ~200k edge slots -- so this is a choice to calibrate, not a
        constant to hardcode.  See docs/findings/FINDINGS_apple_scaling.md and
        `calibrate_edge_a`.
        """
        Rnl, Rpair = self.radial(rij, zi, zj)
        Ylm = self.angular(rij)
        if self.edge_a_kind == "matmul":
            edge_A = (Rnl @ self.a_sel_r) * (Ylm @ self.a_sel_y)
        else:
            edge_A = Rnl[:, self.aspec_r] * Ylm[:, self.aspec_y]
        return edge_A, Rpair

    def _aa(self, A):
        return jnp.concatenate([jnp.prod(A[:, g], axis=-1) for g in self.aa_specs], axis=-1)

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

    def _readout_folded(self, A, Apair, node_z):
        e = jnp.einsum("ia,ai->i", self._aa(A), self.ctilde[:, node_z])
        e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
        return e + self.E0[node_z]

    def site_basis(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """Sparse (edge-list) pooling -- the layout lammps-jax exports."""
        edge_A, Rpair = self.edge_features(rij, zi, zj)
        return self._from_pooled(pool_sparse(edge_A, segment_ids, n_nodes, mask),
                                 pool_sparse(Rpair, segment_ids, n_nodes, mask))

    def site_basis_dense(self, rij, zi, zj, mask):
        """Dense (n, K) pooling -- matscipy-neighbours' `neighbour_matrix` form,
        which maps onto ET's own (maxneigs, nnodes, nfeat) layout and needs no
        scatter.  rij (n,K,3), zi/zj (n,K), mask (n,K)."""
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

    def _readout(self, B, Apair, node_z):
        e = jnp.einsum("ib,bi->i", B, self.WB[:, node_z])
        e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
        return e + self.E0[node_z]

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        """Per-site energies (n_nodes,).  `node_z` is the centre species index per node."""
        if self.folded:
            edge_A, Rpair = self.edge_features(rij, zi, zj)
            return self._readout_folded(pool_sparse(edge_A, segment_ids, n_nodes, mask),
                                        pool_sparse(Rpair, segment_ids, n_nodes, mask),
                                        node_z)
        return self._readout(*self.site_basis(rij, zi, zj, segment_ids, n_nodes, mask), node_z)

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        if self.folded:
            n, K = mask.shape
            flat = lambda a: a.reshape(n * K, *a.shape[2:])
            edge_A, Rpair = self.edge_features(flat(rij), flat(zi), flat(zj))
            un = lambda a: a.reshape(n, K, -1)
            return self._readout_folded(pool_dense(un(edge_A), mask),
                                        pool_dense(un(Rpair), mask), node_z)
        return self._readout(*self.site_basis_dense(rij, zi, zj, mask), node_z)

    # -------------------------------------------------- energy / forces / virial
    def energy_forces_virial(self, rij, zi, zj, senders, receivers, n_nodes,
                             node_z, mask=None):
        """E, F, V from edge vectors alone -- no cell needed.

        Virial by the symmetric-displacement trick, after mace-jax
        `modules/utils.py::compute_forces_and_stress` (MIT).  There the strain is
        applied to positions *and* cell, hence to the edge shifts; because
        rij = r[j] - r[i] + S@cell, both halves transform the same way and the
        whole thing collapses to rij -> rij + rij @ eps.  That keeps the virial a
        function of the edge vectors, consistent with the core.

        Sign follows Julia (AtomsCalculatorsUtilities sitepotentials/assembly.jl:6,
        `site_virial = -sum(dv_i * r_i')`), i.e. V = -dE/d(eps); verified against
        the exported reference rather than argued from the algebra.
        """
        def total(r, eps):
            sym = 0.5 * (eps + eps.T)
            return jnp.sum(self.site_energies(r + r @ sym, zi, zj, senders,
                                              n_nodes, node_z, mask))

        eps0 = jnp.zeros((3, 3), rij.dtype)
        E, (g_r, g_eps) = jax.value_and_grad(total, argnums=(0, 1))(rij, eps0)
        F = (jnp.zeros((n_nodes, 3), rij.dtype)
             .at[senders].add(g_r).at[receivers].add(-g_r))
        return E, F, -g_eps

    # -------------------------------------------------- positions wrapper
    def energy_from_positions(self, positions, node_z, senders, receivers,
                              edge_mask=None, shifts=None):
        """lammps-jax-shaped entry point.  Padded edges are placed at the cutoff,
        where the envelope vanishes and the gradient stays defined."""
        n_nodes = positions.shape[0]
        rij = positions[receivers] - positions[senders]
        if shifts is not None:
            rij = rij + shifts
        if edge_mask is not None:
            rcut = float(jnp.max(self.pair_envelope[..., 0]))
            pad = jnp.asarray([1.0, 0.0, 0.0], positions.dtype) * rcut
            rij = jnp.where(edge_mask[:, None], rij, pad)
        zi, zj = node_z[senders], node_z[receivers]
        return self.site_energies(rij, zi, zj, senders, n_nodes, node_z, edge_mask)


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


# ------------------------------------------------------------------ edge_A kind
def with_edge_a_kind(model, kind):
    """Return `model` using the other A-basis form.  Values and gradients are
    unchanged (bit-identically, measured); only the reverse-pass cost differs."""
    import dataclasses
    if kind not in ("gather", "matmul"):
        raise ValueError(f'kind must be "gather" or "matmul", got {kind!r}')
    if kind == model.edge_a_kind:
        return model
    if kind == "gather":
        return dataclasses.replace(model, edge_a_kind="gather", a_sel_r=None, a_sel_y=None)
    n_rnl = (model.rnl_emb_nidx.shape[0] if model.radial_kind == "spline_factorised"
             else model.rnl_coefs.shape[-1] if model.radial_kind == "spline"
             else model.rnl_Wnlq.shape[-2])
    n_ylm = (model.lmax + 1) ** 2
    dt = model.WB.dtype
    sel = lambda idx, w: jnp.zeros((w, idx.shape[0]), dt).at[
        idx, jnp.arange(idx.shape[0])].set(1)
    return dataclasses.replace(model, edge_a_kind="matmul",
                               a_sel_r=sel(model.aspec_r, n_rnl),
                               a_sel_y=sel(model.aspec_y, n_ylm))


def calibrate_edge_a(model, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None, reps=5):
    """Time both A-basis forms at THESE shapes and return (best_model, timings).

    Calibrate; do not guess.  The crossover is a property of the XLA backend, the
    dtype and the edge-buffer length, not of the platform name -- on an M3 Pro the
    matmul form wins throughout, on a Xeon the gather wins below ~200k edge slots.
    Timing the forward *and* reverse pass matters: the forward gather is cheap and
    flat, and the entire difference is in the adjoint.
    """
    import time
    out = {}
    for kind in ("gather", "matmul"):
        m = with_edge_a_kind(model, kind)
        fn = jax.jit(lambda mm, r: jnp.sum(jax.grad(
            lambda rr: jnp.sum(mm.site_energies(rr, zi, zj, segment_ids, n_nodes,
                                                node_z, mask))
        )(r)))
        jax.block_until_ready(fn(m, rij))                      # warm up / compile
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(m, rij))
            best = min(best, time.perf_counter() - t0)
        out[kind] = best * 1e3                                 # ms
    return with_edge_a_kind(model, min(out, key=out.get)), out
