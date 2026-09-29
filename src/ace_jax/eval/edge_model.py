"""What every edge-vector site-energy model shares (ACEModel, PACEModel).

A model here maps edge vectors rij -- never positions and a cell -- to per-site
energies.  Everything downstream of that contract is model-independent and lives
in `EdgeSiteModel`, so the two model families cannot drift apart:

* the A-basis product in its two interchangeable forms (`edge_a`), and the
  helpers that switch and calibrate between them;
* energy, forces and virial from one `value_and_grad` over the edge vectors;
* the lammps-jax positions entry point.

A subclass provides `site_energies`, `site_energies_dense`, `pad_cutoff()`
(where padded edges are parked), `edge_a_widths()` (row counts of the one-hot
selectors), `edge_a_factors(rij, zi, zj, mask)` (the two per-edge factors of the
A basis, zero on invalid edges) and `a_channels` (1, or the number of
neighbour-species channels A carries), plus the
fields `aspec_r`, `aspec_y`, `edge_a_kind`, `a_sel_r`, `a_sel_y` and `E0`.

A model whose A basis is built pool-first (`pool_first_dense` /
`pool_first_sparse`; PACE) provides instead `pool_first_weights()`,
`pool_first_sel_y` (and `pool_first_widths()`, `product_basis_width()` for
`estimate_a_bytes`), and sets `uses_edge_a = False`: the `edge_a` form is then
inert and `ACECalculator` does not calibrate it.
"""
import dataclasses
import time

import equinox as eqx
import jax
import jax.numpy as jnp

EDGE_A_KINDS = ("gather", "matmul")

CHUNK_NODES = 16384   # dense rows per block: peak memory ~ block, not N (spec, component 2)


def one_hot_selector(idx, width, dtype):
    """One-hot (width, n) matrix S with X @ S == X[:, idx]."""
    return jnp.zeros((width, idx.shape[0]), dtype).at[idx, jnp.arange(idx.shape[0])].set(1)


def check_edge_a_kind(kind):
    if kind not in EDGE_A_KINDS:
        raise ValueError(f'edge_a_kind must be "gather" or "matmul", got {kind!r}')


class EdgeSiteModel(eqx.Module):
    """Base class: no fields of its own, only the shared behaviour."""

    # False for a model whose energy path is pool-first (A never formed per edge),
    # so the `edge_a` form, and its calibration, do not apply.
    uses_edge_a = True

    # -------------------------------------------------- A-basis product
    def edge_a(self, R, Y):
        """Per-edge A-basis rows from radial columns R and harmonics Y.

        Two algebraically identical forms, selected by `edge_a_kind`; they agree
        to bit-identity on values and (in f64) on gradients.

          "gather"  A = R[:, aspec_r] * Y[:, aspec_y]
          "matmul"  A = (R @ Sr) * (Y @ Sy),  Sr/Sy one-hot

        They differ only in the reverse pass: the gather's adjoint is an axis-1
        scatter whose cost per slot grows with buffer length, while the matmul's
        adjoint is a matmul and is flat.  Neither wins everywhere -- on an M3 Pro
        the matmul wins throughout and by up to 11x, on a Xeon the gather wins
        below ~200k edge slots, and on an A4500 the gather's two scatters were
        66-84% of a PACE force call -- so this is a choice to calibrate, not a
        constant to hardcode.  See docs/findings/FINDINGS_apple_scaling.md and
        `calibrate_edge_a`.
        """
        if self.edge_a_kind == "matmul":
            # the one-hot products only select, so they must not round: at default
            # precision an f32 matmul on Ampere+ is TF32 (10-bit mantissa)
            hi = jax.lax.Precision.HIGHEST
            return (jnp.matmul(R, self.a_sel_r, precision=hi)
                    * jnp.matmul(Y, self.a_sel_y, precision=hi))
        return R[:, self.aspec_r] * Y[:, self.aspec_y]

    # -------------------------------------------------- A basis, two layouts
    def pool_a_sparse(self, cols, Y, seg, channel, n_nodes):
        """A (n_nodes, C * n_A) from edge-list factors: per-edge rows in the chosen
        form (`edge_a`), summed per (node, channel).  Memory ~ E * n_A."""
        C = self.a_channels
        rows = self.edge_a(cols, Y)
        A = jax.ops.segment_sum(rows, seg * C + channel, num_segments=n_nodes * C)
        return A.reshape(n_nodes, C * rows.shape[1])

    def pool_a_dense(self, cols, Y, channel):
        """A (n, C * n_A) from padded (n, K, .) factors, per node as a batched
        outer product sum_k cols_k (x) Y_k, then the used entries selected.  No
        per-edge column gather, so no scatter in the reverse pass (3-13x faster
        forces measured on A4500/A100/H100).  Memory ~ n * K * (C * n_cols + n_Y)
        + n * C * n_cols * n_Y, so it grows with channels and neighbour padding --
        which is why the sparse layout stays."""
        n, K, nc = cols.shape
        ny = Y.shape[-1]
        C = self.a_channels
        if C > 1:                                    # neighbour-species channel
            oh = jax.nn.one_hot(channel, C, dtype=cols.dtype)
            cols = (oh[..., :, None] * cols[..., None, :]).reshape(n, K, C * nc)
        A_full = jnp.einsum("nkr,nky->nry", cols, Y,
                            precision=jax.lax.Precision.HIGHEST).reshape(n, -1)
        mu = jnp.arange(C, dtype=self.aspec_r.dtype)[:, None]
        sel = ((mu * nc + self.aspec_r[None, :]) * ny + self.aspec_y[None, :]).reshape(-1)
        return A_full[:, sel]

    # -------------------------------------------------- pool-first A (feature-major)
    def pool_first_dense(self, b, Y, zj, node_z):
        """A_t (C*n_a, n) = sum_k W[z_i, z_j] b_k (x) Y_y, pooled BEFORE W.

        Every model here has a radial that is linear in per-pair coefficients
        over a fixed per-edge basis b (PACE: g_k with crad; ACE: polynomials,
        spline or factorised table with its weights), so the per-edge R_nl is
        never formed: pool b (x) Y per (node, neighbour-species channel), then
        apply W per node.  Feature-major output so the product-basis gathers read
        whole rows.  b (n, K, n_b), Y (n, K, n_Y), zj (n, K) neighbour channel.
        docs/pace-performance-gap.md #4, #5."""
        n, K, nb = b.shape
        C = self.a_channels
        hi = jax.lax.Precision.HIGHEST
        if C > 1:
            oh = jax.nn.one_hot(zj, C, dtype=b.dtype)
            b = (oh[..., :, None] * b[..., None, :]).reshape(n, K, C * nb)
        Ag = jnp.einsum("nkg,nky->ngy", b, Y, precision=hi)                  # (n, C*nb, nY)
        Agy = jnp.matmul(Ag, self.pool_first_sel_y, precision=hi)            # (n, C*nb, n_a)
        Agy = Agy.reshape(n, C, nb, -1)
        W = self.pool_first_weights()                                        # (NZ, C, n_a, nb)
        At = jnp.einsum("nmka,nmak->man", Agy, W[node_z], precision=hi)     # (C, n_a, n)
        return At.reshape(C * At.shape[1], n)

    def pool_first_sparse(self, b, Y, seg, zj, node_z, n_nodes):
        """`pool_first_dense` for an edge list: pool b (x) Y per (node, channel)
        by segment_sum, then W per node.  Same output layout, (C*n_a, n_nodes)."""
        C = self.a_channels
        hi = jax.lax.Precision.HIGHEST
        nb = b.shape[1]
        outer = (b[:, :, None] * Y[:, None, :]).reshape(b.shape[0], nb * Y.shape[1])  # explicit: E may be 0
        Ag = jax.ops.segment_sum(outer, seg * C + (zj if C > 1 else 0),
                                 num_segments=n_nodes * C)
        Ag = Ag.reshape(n_nodes, C, nb, -1)
        Agy = jnp.matmul(Ag, self.pool_first_sel_y, precision=hi)             # (n, C, nb, n_a)
        W = self.pool_first_weights()
        At = jnp.einsum("nmka,nmak->man", Agy, W[node_z], precision=hi)     # (C, n_a, n)
        return At.reshape(C * At.shape[1], n_nodes)

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

    def site_energies_dense_blocked(self, rij, zi, zj, mask, node_z, chunk):
        """`site_energies_dense` evaluated in row blocks of at most `chunk`:
        per-row energies (n,), identical to the unblocked call.

        At or below one block this IS the unblocked call (no lax.map, no
        checkpoint).  Above it, rows are split into nb equal blocks of B <= chunk
        (padded with parked, fully masked rows, dropped from the result) and
        evaluated with lax.map under jax.checkpoint, so the backward pass
        recomputes one block at a time: peak memory scales with the block, not
        with n.  Exact, because a site energy depends only on its own row."""
        n = mask.shape[0]
        nb = max(1, -(-n // chunk))
        if nb == 1:
            return self.site_energies_dense(rij, zi, zj, mask, node_z)
        B = -(-n // nb)
        pad_n = nb * B - n
        park = jnp.asarray([1.0, 0.0, 0.0], rij.dtype) * self.pad_cutoff()

        def blocks(a, fill):
            if pad_n:
                a = jnp.concatenate([a, jnp.broadcast_to(jnp.asarray(fill, a.dtype),
                                                         (pad_n,) + a.shape[1:])])
            return a.reshape((nb, B) + a.shape[1:])

        xs = (blocks(rij, park), blocks(zi, 0), blocks(zj, 0), blocks(mask, False),
              blocks(node_z, 0))
        block = jax.checkpoint(lambda a: self.site_energies_dense(*a))
        return jax.lax.map(block, xs).reshape(-1)[:n]

    def energy_forces_virial_dense(self, rij, zi, zj, idx, mask, node_z, chunk=CHUNK_NODES,
                                   rev=None, return_edge_grad=False):
        """`energy_forces_virial` for the dense layout: rij (n, K, 3), zi / zj /
        idx (neighbour index) / mask (n, K).  Same strain trick for the virial.

        Rows are evaluated in blocks of `chunk` (lax.map; jax.checkpoint so the
        backward pass recomputes a block rather than storing all of them): peak
        memory scales with the block.  A site energy depends only on its own row,
        so blocks are independent; padding rows are fully masked.

        `rev` (n, K), from `nlist.reverse_slots`, is optional and independent of
        chunking: it only changes how the per-edge gradient `g_r` is turned into
        forces, after the (possibly chunked) energy above has produced it.

        `return_edge_grad=True` returns (E, g_r, V) instead: g_r (n, K, 3) is
        dE/drij, zero on masked slots, for a caller that assembles the forces
        itself (the skin list's step, `calc.skin.step`, which gathers in its own
        slot space); `rev` is then ignored."""
        n = mask.shape[0]

        def total(r, eps):
            sym = 0.5 * (eps + eps.T)
            return jnp.sum(self.site_energies_dense_blocked(r + r @ sym, zi, zj, mask,
                                                            node_z, chunk))

        eps0 = jnp.zeros((3, 3), rij.dtype)
        E, (g_r, g_eps) = jax.value_and_grad(total, argnums=(0, 1))(rij, eps0)
        g_r = jnp.where(mask[..., None], g_r, 0.0)
        if return_edge_grad:
            return E, g_r, -g_eps
        if rev is None:
            F = (jnp.zeros((n, 3), rij.dtype).at[jnp.arange(n)].add(g_r.sum(axis=1))
                 .at[idx.reshape(-1)].add(-g_r.reshape(-1, 3)))
        else:
            # F_i = sum_k g[i,k] - sum_k g[j, rev[i,k]], j = idx[i,k]: the edge
            # j -> i sits at slot rev[i,k] of row j (a full list is symmetric),
            # so the force assembly is a gather, not a scatter-add.  Padded slots
            # have idx = rev = 0; g_r there is already zeroed by the mask above,
            # and `back` at those slots is masked out again below, so g_r[0, 0]
            # (a live value, generally nonzero) never leaks into F -- and for a
            # live edge, g_r[idx, rev] reads a live slot of row j, because the
            # match found by reverse_slots guarantees the reverse edge is live.
            back = g_r[idx, rev]
            F = g_r.sum(axis=1) - jnp.where(mask[..., None], back, 0.0).sum(axis=1)
        return E, F, -g_eps

    # -------------------------------------------------- positions wrapper
    def energy_from_positions(self, positions, node_z, senders, receivers,
                              edge_mask=None, shifts=None):
        """lammps-jax-shaped entry point.  Padded edges are placed at
        `pad_cutoff()`, where every radial vanishes and the gradient stays
        defined (and they are masked as well)."""
        n_nodes = positions.shape[0]
        rij = positions[receivers] - positions[senders]
        if shifts is not None:
            rij = rij + shifts
        if edge_mask is not None:
            pad = jnp.asarray([1.0, 0.0, 0.0], positions.dtype) * self.pad_cutoff()
            rij = jnp.where(edge_mask[:, None], rij, pad)
        zi, zj = node_z[senders], node_z[receivers]
        return self.site_energies(rij, zi, zj, senders, n_nodes, node_z, edge_mask)


# ------------------------------------------------------------------ layouts
LAYOUTS = ("sparse", "dense")


def estimate_a_bytes(model, layout, n_nodes, n_edges, max_neighbours, itemsize):
    """Peak bytes of the A-basis stage (energy + forces; pool-first: the whole
    dense call), for choosing a layout.

    Fitted to measured peaks on A4500/A100/H100 (c_ace, 4k and 14k atoms):
    dense  ~ 2 n K (C n_cols + n_Y) + 3 n C n_cols n_Y   (factors + A_full)
    sparse ~ 2 E (n_cols + n_Y + n_A)                    (factors + per-edge rows)
    with C neighbour-species channels.  Dense grows with C and with the (n, K)
    padding; sparse only with the edge count.

    A pool-first model (`uses_edge_a` False; PACE) never forms per-edge columns
    or A rows: per edge it holds the fixed basis b (n_b wide, one-hot over the C
    channels) and Y; per node the pooled Ag = b (x) Y, A, the (n, C, n_b, n_a)
    Agy = Ag @ sel_y that W[z_i] contracts, and the product basis (n_P =
    `product_basis_width()`: the gathered A factors of every order and AA), so
    with n_b, n_Y = `pool_first_widths()`
    dense  ~ 2 n K (C n_b + n_Y) + per_node
    sparse ~ 2 E (n_b + n_Y + n_b n_Y) + per_node
    per_node = n C (n_b n_Y + n_a) + n C n_b n_a + n n_P / 4
    Refitted to XLA's compiled temp size of the dense E/F/V call on an A100
    (six PACE models x 1k/4k/8k atoms, float64,
    bench/perf/results/microbench_memory_float64.json): measured / estimate
    0.68-1.06, so the estimate errs high by at most 1.5x.  XLA's CPU backend
    keeps about twice as much live (fewer fusions), so on the CPU this is low
    by about 2x.
    """
    C, n_a = model.a_channels, int(model.aspec_r.shape[0])
    if not model.uses_edge_a:
        nb, ny = model.pool_first_widths()
        per_node = (n_nodes * C * (nb * ny + n_a) + n_nodes * C * nb * n_a
                    + n_nodes * model.product_basis_width() // 4)
        if layout == "dense":
            return itemsize * (2 * n_nodes * max_neighbours * (C * nb + ny) + per_node)
        if layout == "sparse":
            return itemsize * (2 * n_edges * (nb + ny + nb * ny) + per_node)
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    nc, ny = model.edge_a_widths()
    if layout == "dense":
        return itemsize * (2 * n_nodes * max_neighbours * (C * nc + ny) + 3 * n_nodes * C * nc * ny)
    if layout == "sparse":
        return itemsize * 2 * n_edges * (nc + ny + n_a)
    raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")


# ------------------------------------------------------------------ edge_A kind
def with_edge_a_kind(model, kind):
    """Return `model` using the other A-basis form.  Values and gradients are
    unchanged (bit-identically, measured); only the reverse-pass cost differs."""
    check_edge_a_kind(kind)
    if kind == model.edge_a_kind:
        return model
    if kind == "gather":
        return dataclasses.replace(model, edge_a_kind="gather", a_sel_r=None, a_sel_y=None)
    n_r, n_y = model.edge_a_widths()
    dt = model.E0.dtype
    return dataclasses.replace(model, edge_a_kind="matmul",
                               a_sel_r=one_hot_selector(model.aspec_r, n_r, dt),
                               a_sel_y=one_hot_selector(model.aspec_y, n_y, dt))


def calibrate_edge_a(model, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None, reps=5):
    """Time both A-basis forms at THESE shapes and return (best_model, timings).

    Calibrate; do not guess.  The crossover is a property of the XLA backend, the
    dtype and the edge-buffer length, not of the platform name -- on an M3 Pro the
    matmul form wins throughout, on a Xeon the gather wins below ~200k edge slots.
    Timing the forward *and* reverse pass matters: the forward gather is cheap and
    flat, and the entire difference is in the adjoint.

    `ACECalculator(edge_a_kind="auto")` does this automatically.  For a lammps-jax
    export -- traced once at fixed buffer sizes -- call it on a representative
    structure at the export's dtype and edge capacity, and export the returned
    model.
    """
    out = {}
    for kind in EDGE_A_KINDS:
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
