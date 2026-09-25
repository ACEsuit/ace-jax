"""What every edge-vector site-energy model shares (ACEModel, PACEModel).

A model here maps edge vectors rij -- never positions and a cell -- to per-site
energies.  Everything downstream of that contract is model-independent and lives
in `EdgeSiteModel`, so the two model families cannot drift apart:

* the A-basis product in its two interchangeable forms (`edge_a`), and the
  helpers that switch and calibrate between them;
* energy, forces and virial from one `value_and_grad` over the edge vectors;
* the lammps-jax positions entry point.

A subclass provides `site_energies`, `pad_cutoff()` (where padded edges are
parked) and `edge_a_widths()` (row counts of the one-hot selectors), plus the
fields `aspec_r`, `aspec_y`, `edge_a_kind`, `a_sel_r`, `a_sel_y` and `E0`.
"""
import dataclasses
import time

import equinox as eqx
import jax
import jax.numpy as jnp

EDGE_A_KINDS = ("gather", "matmul")


def one_hot_selector(idx, width, dtype):
    """One-hot (width, n) matrix S with X @ S == X[:, idx]."""
    return jnp.zeros((width, idx.shape[0]), dtype).at[idx, jnp.arange(idx.shape[0])].set(1)


def check_edge_a_kind(kind):
    if kind not in EDGE_A_KINDS:
        raise ValueError(f'edge_a_kind must be "gather" or "matmul", got {kind!r}')


class EdgeSiteModel(eqx.Module):
    """Base class: no fields of its own, only the shared behaviour."""

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
