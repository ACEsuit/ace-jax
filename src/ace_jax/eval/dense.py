"""The energy of a dense neighbour layout whose rows hold more slots than the model needs: the core shared
by the JAX-MD adapter (`ace_jax.jaxmd.potential`, one structure, a JAX-MD list) and the batched MD path
(`ace_jax.batch.energy_fn`, ace-jax's own lists, under jax.vmap).

Both hand over rows of cutoff + skin slots (rij_s, j_s) with `valid_s` marking the slots within the
model's cutoff.  `dense_energy` then compacts the valid slots into K model slots (`calc.skin.compact`,
the skin list's step does the same), parks the rest where every radial vanishes, poisons energy and
forces with NaN on any overflow (never truncation), and evaluates the site energies in row blocks.

The evaluation model is whatever `ACECalculator` resolved: the JAX model, or the kernel
(`eval.ffi.KernelModel`, `ffi=True`: the C++ one on the CPU, the CUDA one on a GPU).  The kernel's FFI call
has a batching rule that flattens a vmapped batch into rows (`eval.ffi._edge_kernel`), so the batched path
runs it as one call (threaded on the CPU, one launch on the GPU).
"""
import jax.numpy as jnp

from .edge_model import total_energy


def dense_energy(model, rij_s, j_s, valid_s, node_z, K=None, chunk=None, bad=None, live=None):
    """Total energy of one structure's dense rows.

    rij_s (N, Ks, 3) edge vectors and j_s (N, Ks) neighbour indices of the list slots; valid_s (N, Ks)
    the slots that are real AND within the model's cutoff; node_z (N,) species indices.  `K` (int)
    compacts the valid slots of each row into K model slots (a row with more than K is an overflow);
    None evaluates all Ks slots, masked.  `chunk`: rows per dense block (`site_energies_dense_blocked`,
    lax.map + checkpoint above one block); None, one block.  `bad` (a device bool) makes energy and
    forces NaN (a list overflow, an invalid species); `live` (N,) drops padding atoms' site energies.
    Returns (E, overflow): overflow is bad or a row over K."""
    from ..calc.skin import compact
    bad = jnp.zeros((), bool) if bad is None else jnp.asarray(bad, bool)
    if K is None:
        rij, j, mask = rij_s, j_s, valid_s
    else:
        order, mask, csum = compact(valid_s, K)
        rij = jnp.take_along_axis(rij_s, order[..., None], axis=1)
        j = jnp.where(mask, jnp.take_along_axis(j_s, order, axis=1), 0)
        bad = bad | jnp.any(csum[:, -1] > K)
    # parked slots: a constant vector where every radial vanishes, so their gradient is zero and never
    # NaN; an overflow poisons E and F (r * NaN), never a silently truncated row
    park = (jnp.asarray([1.0, 0.0, 0.0]) * model.pad_cutoff()).astype(rij.dtype)
    rij = jnp.where(mask[..., None], rij, park) * jnp.where(bad, jnp.nan, 1.0).astype(rij.dtype)
    zi = jnp.broadcast_to(node_z[:, None], j.shape)
    e = (model.site_energies_dense(rij, zi, node_z[j], mask, node_z) if chunk is None
         else model.site_energies_dense_blocked(rij, zi, node_z[j], mask, node_z, int(chunk)))
    if live is not None:
        e = jnp.where(live, e, 0.0)
    return total_energy(e), bad
