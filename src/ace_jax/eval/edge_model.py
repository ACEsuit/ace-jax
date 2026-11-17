"""What every edge-vector site-energy model shares (ACEModel, PACEModel).

A model here maps edge vectors rij -- never positions and a cell -- to per-site
energies.  Everything downstream of that contract is model-independent and lives
in `EdgeSiteModel`, so the two model families cannot drift apart:

* the A basis on the model's own arrays (`edge_a` and the pooling methods; the
  arithmetic is `product`'s), and the calibration between its two forms;
* energy, forces and virial from one `value_and_grad` over the edge vectors,
  the total a compensated sum of the site energies (`total_energy`);
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
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from . import product
from .product import (EDGE_A_KINDS, LAYOUTS, check_edge_a_kind, estimate_a_bytes,  # noqa: F401  (their old home)
                      one_hot_selector, product_basis, product_basis_dot, product_basis_t, with_edge_a_kind)

CHUNK_NODES = 16384   # dense rows per block: peak memory ~ block, not N (spec, component 2)
# On a GPU, 4096: the block's temp is ~120 KB/row at o4d20 (the DAG product basis's
# order-2 cotangent and the per-pair radials), so 16384 rows held 4.1 GB, 4096 holds
# 1.1 GB, at the same step time within the spread over block sizes (A100, float64:
# 145 vs 154 ms at 32768 atoms, 432 vs 393 ms at 97336; docs/dev/gpu-memory-results.md).
# Other backends keep CHUNK_NODES (not measured there).
GPU_CHUNK_NODES = 4096


def default_chunk():
    """The dense row-block size `energy_forces_virial_dense` uses when given none:
    GPU_CHUNK_NODES on the GPU backend, else CHUNK_NODES.  Resolved at trace time."""
    return GPU_CHUNK_NODES if jax.default_backend() == "gpu" else CHUNK_NODES


def _two_sum(a, b):
    """s, e with s = fl(a + b) and s + e == a + b exactly (Knuth's TwoSum)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _sum2(x):
    """(hi, lo) in x.dtype with hi + lo ~ sum(x): a pairwise TwoSum tree carrying
    each level's rounding error alongside (Ogita, Rump & Oishi's Sum2 in pairwise
    order), as accurate as a pairwise sum in twice the working precision.
    log2(n) unrolled levels, jit-compatible."""
    s = x.reshape(-1)
    n = s.shape[0]
    if n == 0:
        return jnp.zeros((), x.dtype), jnp.zeros((), x.dtype)
    s = jnp.pad(s, (0, (1 << (n - 1).bit_length()) - n))
    c = jnp.zeros_like(s)
    while s.shape[0] > 1:
        s, e = _two_sum(s[0::2], s[1::2])
        c = c[0::2] + c[1::2] + e
    return s[0], c[0]


@jax.custom_jvp
def compensated_sum(x):
    """sum(x), the JAX analogue of `math.fsum` for an energy total: `_sum2`
    rounded once, within ~1 ulp of the exact sum on realistic data.  The
    derivative is jnp.sum's (custom_jvp): d/dx is exactly ones."""
    hi, lo = _sum2(x)
    return hi + lo


@compensated_sum.defjvp
def _compensated_sum_jvp(primals, tangents):
    return compensated_sum(primals[0]), jnp.sum(tangents[0])


def _acc_dtype(dtype):
    return jnp.float64 if jax.config.jax_enable_x64 else dtype


@jax.custom_jvp
def total_energy(e):
    """The total of site energies `e`: `compensated_sum`, returned in float64
    where JAX has it.  float32 sites stay float32 arrays (the compensated pair is
    formed in float32 and only the two scalars are widened: no float64 array in a
    float32 graph, cf. test_pace_model), so a float32 model gets a float64 total
    accurate to about float32^2.

    Why (docs/dev/energy-sum-results.md): a float32 total is quantised at its own
    ulp (0.03 eV at 2k Si atoms, 1 eV at 60k), which stalls an Armijo line search
    at fmax 0.2-0.3 eV/A; with a float64 total the energy noise is the per-site
    float32 rounding instead.  In float64 a plain XLA sum was already within 1 ulp
    on CPU (up to ~12 ulp on a GPU); compensated, it is correctly rounded in
    practice, so the total no longer depends on atom order or layout.  The
    derivative is jnp.sum's, so forces are bitwise those of a plain sum."""
    hi, lo = _sum2(e)
    acc = _acc_dtype(e.dtype)
    return hi.astype(acc) + lo.astype(acc)


@total_energy.defjvp
def _total_energy_jvp(primals, tangents):
    (e,), (t,) = primals, tangents
    return total_energy(e), jnp.sum(t).astype(_acc_dtype(e.dtype))


class EdgeSiteModel(eqx.Module):
    """Base class: no fields of its own, only the shared behaviour."""

    # False for a model whose energy path is pool-first (A never formed per edge),
    # so the `edge_a` form, and its calibration, do not apply.
    uses_edge_a = True

    # -------------------------------------------------- A basis (`product`, on this model's arrays)
    def edge_a(self, R, Y):
        """Per-edge A-basis rows from radial columns R and harmonics Y, in the form
        `edge_a_kind` selects (`product.edge_a`: "gather" or "matmul", identical values)."""
        return product.edge_a(R, Y, self.aspec_r, self.aspec_y, self.edge_a_kind, self.a_sel_r, self.a_sel_y)

    def pool_a_sparse(self, cols, Y, seg, channel, n_nodes):
        """A (n_nodes, C * n_A) from edge-list factors (`product.pool_a_sparse` of `edge_a`'s rows)."""
        return product.pool_a_sparse(self.edge_a(cols, Y), seg, channel, n_nodes, self.a_channels)

    def pool_a_dense(self, cols, Y, channel):
        """A (n, C * n_A) from padded (n, K, .) factors (`product.pool_a_dense`)."""
        return product.pool_a_dense(cols, Y, channel, self.aspec_r, self.aspec_y, self.a_channels)

    def pool_first_dense(self, b, Y, zj, node_z):
        """A_t (C*n_a, n), pooled before the radial weights (`product.pool_first_dense`)."""
        return product.pool_first_dense(b, Y, zj, node_z, self.pool_first_weights(), self.pool_first_sel_y,
                                        self.a_channels)

    def pool_first_sparse(self, b, Y, seg, zj, node_z, n_nodes):
        """`pool_first_dense` for an edge list (`product.pool_first_sparse`)."""
        return product.pool_first_sparse(b, Y, seg, zj, node_z, n_nodes, self.pool_first_weights(),
                                         self.pool_first_sel_y, self.a_channels)

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
            return total_energy(self.site_energies(r + r @ sym, zi, zj, senders,
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

    def energy_forces_virial_dense(self, rij, zi, zj, idx, mask, node_z, chunk=None,
                                   rev=None, return_edge_grad=False):
        """`energy_forces_virial` for the dense layout: rij (n, K, 3), zi / zj /
        idx (neighbour index) / mask (n, K).  Same strain trick for the virial.

        Rows are evaluated in blocks of `chunk` (None: `default_chunk()`; lax.map;
        jax.checkpoint so the backward pass recomputes a block rather than storing
        all of them): peak memory scales with the block.  A site energy depends only on its own row,
        so blocks are independent; padding rows are fully masked.

        `rev` (n, K), from `nlist.reverse_slots`, is optional and independent of
        chunking: it only changes how the per-edge gradient `g_r` is turned into
        forces, after the (possibly chunked) energy above has produced it.

        `return_edge_grad=True` returns (E, g_r, V) instead: g_r (n, K, 3) is
        dE/drij, zero on masked slots, for a caller that assembles the forces
        itself (the skin list's step, `calc.skin.step`, which gathers in its own
        slot space); `rev` is then ignored."""
        n = mask.shape[0]
        chunk = default_chunk() if chunk is None else chunk

        def total(r, eps):
            sym = 0.5 * (eps + eps.T)
            return total_energy(self.site_energies_dense_blocked(r + r @ sym, zi, zj, mask,
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
