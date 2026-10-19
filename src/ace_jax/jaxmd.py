"""JAX-MD adapter: an ace-jax model as a JAX-MD energy function.

    pot = jaxmd.potential("si.npz", displacement, species=numbers, K=K)
    neighbor_fn = partition.neighbor_list(displacement, box, jaxmd.cutoff("si.npz"), dr_threshold=1.0,
                                          format=partition.Dense)
    nbrs = jaxmd.allocate(neighbor_fn, R)   # not neighbor_fn.allocate: see `allocate`
    init, apply = simulate.nve(pot.force, shift, dt)      # or pot.energy

`pot.energy(R, neighbor, **kwargs)` follows JAX-MD's contract, so its
integrators, `quantity.force` and `quantity.stress` (the `box=` keyword reaches
`displacement_fn`) work unchanged, and everything -- the neighbour list update,
the model, the integrator -- stays on the device inside one jit.  The model is
`ACECalculator`'s evaluation model (lean, splined, DAG product basis as the
calculator's options choose), so the two agree to roundoff.

A JAX-MD list holds every pair within cutoff + dr_threshold, so up to
((rc + skin) / rc)^3 more slots than the model needs (1.6x at rc 5.5 A, skin 1 A,
times JAX-MD's capacity_multiplier).  With `K` the in-cutoff slots are compacted
into (N, K) rows before the model (`calc.skin.compact`, as the skin list's step
does); without it the model runs on the whole list, masked at rc.  A row with more
than K in-cutoff neighbours makes the energy and forces NaN: never truncation.
`capacity` sizes K from a structure; `overflow` is the explicit check.  Install
with the `jaxmd` extra (`pip install "ace-jax[jaxmd]"`).

With a JAX-MD list, edge vectors come from `displacement_fn`, so periodic images
are JAX-MD's minimum image: every periodic box width must exceed 2 (rc +
dr_threshold) (`min_image_ok`); below that JAX-MD misses images without an error.

"""
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .calc.skin import compact, round_k
from .eval.edge_model import total_energy


def _jax_md(module):
    """A jax_md submodule, or an ImportError that says what to install."""
    import importlib
    try:
        return importlib.import_module(f"jax_md.{module}")
    except ImportError as e:
        raise ImportError('ace_jax.jaxmd needs jax-md: pip install "ace-jax[jaxmd]"') from e


def _partition():
    return _jax_md("partition")


def cutoff(model, meta=None):
    """The model's cutoff rc (Angstrom), the `r_cutoff` for a neighbour list: from the
    model file's metadata (`model` a path), or from `meta` (a loaded model), without
    building the evaluation model.  It is `potential(...).cutoff`."""
    if meta is None:
        import os
        if not isinstance(model, (str, os.PathLike)):
            raise ValueError("cutoff(model) reads a model file; for a loaded model pass its meta")
        from .eval.io import load
        meta = load(model)[1]
    return float(meta["rcut"])


def eval_model(model, meta=None, **calc_kw):
    """(eval_model, meta, cutoff): the model `ACECalculator(model, meta, **calc_kw)`
    evaluates, so the adapter takes every calculator option (lean, spline_tol,
    radial_table, aa_dag, energy_reference, dtype)."""
    calc = _calc(model, meta, **calc_kw)
    return calc.eval_model, calc.meta, calc.cutoff


def _calc(model, meta=None, **calc_kw):
    from .calc.point import ACECalculator
    return ACECalculator(model, meta, **calc_kw)


def block_rows(model, n, k, itemsize):
    """Rows per dense block for n atoms and k slots: the rule of the lammps-jax
    bundles (`export.lammps.block_rows_for` on the full `model`, as export_lammps
    sizes it: estimate_a_bytes underestimates the lean widths), within this
    device's dense budget and block target; MIN_BLOCK_ROWS when none fits.  Above
    one block the rows run under lax.map + checkpoint, so memory follows the block,
    not n.  Si, 32768 atoms, RTX 4000 Ada, float64, against fixed 16384-row blocks:
    o4d20 368 -> 308 ms/step, and the model's GPU memory 10.2 -> 1.3 GB (measured
    with an O(N K) neighbour list: JAX-MD's own cell list adds ~4 GB of build
    temporaries); the checkpoint recompute makes small models slower (o3d10 43 ->
    48 ms, as the bundles: lammps-jax 50 ms).  An int `chunk` to `potential` trades
    back.  docs/dev/jax-md-results.md."""
    from .calc.point import dense_budget_bytes, device_memory_bytes
    from .export.lammps import MIN_BLOCK_ROWS, block_rows_for, block_target_elems
    return (block_rows_for(model, n, k, dense_budget_bytes(), itemsize,
                           target=block_target_elems(device_memory_bytes())) or MIN_BLOCK_ROWS)


def _edges_dense(R, idx, displacement_fn, kw):
    """rij (N, Ks, 3) = r_j - r_i over a JAX-MD Dense list, with live (N, Ks): pad index N."""
    N = R.shape[0]
    live = idx < N
    j = jnp.where(live, idx, 0)
    d = lambda Ra, Rb: displacement_fn(Ra, Rb, **kw)
    rij = jax.vmap(jax.vmap(d, (0, None)), (0, 0))(R[j], R)   # d(R_j, R_i) = r_j - r_i
    return rij, j, live


class Potential(NamedTuple):
    energy: Callable      # energy(R, neighbor, species=None, **kw) -> E (eV)
    force: Callable       # force(R, neighbor, species=None, **kw) -> F (N, 3), Cartesian eV/A
    cutoff: float         # the model's rc: the r_cutoff of the JAX-MD list
    K: int | None         # compact slots per atom (None: the whole list)
    overflow: Callable    # overflow(R, neighbor, **kw) -> () bool: list overflow or a row over K


def potential(model, displacement_fn, species, meta=None, K=None, chunk="auto", **calc_kw):
    """`Potential(energy, force, cutoff, K, overflow)` for JAX-MD.

    `model` is a path or a loaded model (then pass `meta`).  `species` is the atomic
    numbers (N,), fixed for the run (a `species=` keyword per call overrides it; a
    traced one with an element the model lacks makes energy and forces NaN, a
    concrete one raises).
    `cutoff` is the model's rc, the `r_cutoff` to build the JAX-MD list with.
    `K` (int) compacts the list to K in-cutoff slots per atom (Dense format only);
    None evaluates every list slot.  K sized for rc + skin (`capacity(..., rc +
    skin)`) cannot overflow between list rebuilds; sized for rc ("tight": fewer
    model slots, ~1.5x faster forces) it can, as atoms move, and `overflow` says
    when (energy and forces are then NaN, never truncated).  `chunk` is the rows
    per dense block: "auto" sizes it at trace time from N and the slots as the
    lammps-jax bundles do (`block_rows`); an int fixes it.  Other keywords go to
    ACECalculator, which builds the evaluation model.

    `energy` is JAX-MD's energy-function contract: give it to any integrator,
    `quantity.force` or `quantity.stress` (stress is JAX-MD's sign, minus ASE's).
    `force` is -grad(energy) with respect to the positions, as JAX-MD's
    `quantity.force(energy)`: Cartesian, also under
    `space.periodic_general(fractional_coordinates=True)` (JAX-MD's `transform`
    passes the gradient through), for the integrators that take a force
    function.  (An explicit assembly of dE/drij into forces measured no faster:
    the full reverse pass was 5-6% faster on an A100.)

    Supports partition.Dense and partition.Sparse (a full list; without K: K is
    refused for it); OrderedSparse, a half list, is refused.  An overflowed list
    makes energy and forces NaN, whatever its format.  Remaining keywords of a call
    (`box=` for `space.periodic_general`) are passed to `displacement_fn`."""
    if "cutoff" in calc_kw:   # the list is built with the model's rc (`cutoff`); a different one would not match
        raise ValueError("potential() uses the model's cutoff (jaxmd.cutoff(model)); do not pass cutoff=")
    calc = _calc(model, meta, **calc_kw)
    m, meta, rc, full = calc.eval_model, calc.meta, calc.cutoff, calc.model
    lut = np.full(119, -1, np.int32)
    for i, z in enumerate(meta["elements"]):
        lut[int(z)] = i
    lut_d = jnp.asarray(lut)

    def node_z_of(numbers):
        """(species indices, bad): bad (a device bool) poisons E and F when a traced
        `numbers` holds an element the model lacks; a concrete one raises instead."""
        if isinstance(numbers, jax.core.Tracer):
            z = lut_d[jnp.clip(numbers, 0, 118)]
            return jnp.maximum(z, 0), jnp.any(z < 0) | jnp.any((numbers < 0) | (numbers > 118))
        numbers = np.asarray(numbers)
        z = lut[numbers]
        if (z < 0).any():
            raise ValueError(f"elements {sorted({int(v) for v in numbers[z < 0]})} not in the model "
                             f"{meta['elements']}")
        return jnp.asarray(z), jnp.zeros((), bool)

    node_z0 = node_z_of(species)
    poison = lambda r, bad: r * jnp.where(bad, jnp.nan, 1.0).astype(r.dtype)
    park = jnp.asarray([1.0, 0.0, 0.0]) * m.pad_cutoff()

    def rows(R, neighbor, kw):
        """(rij_s, j_s, valid_s, bad0): the list's edges, which are within rc, and
        whether the list itself overflowed."""
        rij_s, j_s, live_s = _edges_dense(R, neighbor.idx, displacement_fn, kw)
        bad0 = jnp.asarray(getattr(neighbor, "did_buffer_overflow", False), bool)
        return rij_s, j_s, live_s & (jnp.sum(rij_s * rij_s, axis=-1) < rc * rc), bad0

    def overflow(R, neighbor, species=None, **kw):
        """Whether energy and forces at R are invalid (NaN): the neighbour list
        overflowed, a row has more than K neighbours within rc, or a traced
        `species` holds an element the model lacks.  A device scalar, for R itself:
        an overflow earlier in a chunk of steps shows as non-finite positions
        instead, so check both after each chunk, then rebuild the list
        (`did_buffer_overflow`) or the potential with a larger K."""
        bad_species = jnp.zeros((), bool) if species is None else node_z_of(species)[1]
        if not dense(neighbor):
            return jnp.asarray(neighbor.did_buffer_overflow, bool) | bad_species
        _, _, valid_s, bad0 = rows(jax.lax.stop_gradient(R), neighbor, kw)
        return bad0 | bad_species | (False if K is None else jnp.any(jnp.sum(valid_s, axis=1) > K))

    def graph(R, neighbor, kw, bad_species):
        """Dense list -> (rij, j, mask) on the model's rows; NaN when the list
        or a row overflowed, or the species are invalid."""
        rij_s, j_s, valid_s, bad0 = rows(R, neighbor, kw)
        bad0 = bad0 | bad_species
        if K is None:
            rij, j, mask, bad = rij_s, j_s, valid_s, bad0
        else:
            order, mask, csum = compact(valid_s, K)
            rij = jnp.take_along_axis(rij_s, order[..., None], axis=1)
            j = jnp.where(mask, jnp.take_along_axis(j_s, order, axis=1), 0)
            bad = bad0 | jnp.any(csum[:, -1] > K)
        # parked slots: a constant vector where every radial vanishes, so their
        # gradient is zero and never NaN; an overflowed row poisons E and F
        rij = jnp.where(mask[..., None], rij, park.astype(rij.dtype))
        return poison(rij, bad), j, mask

    def site_energies_dense(rij, j, mask, node_z):
        zi = jnp.broadcast_to(node_z[:, None], j.shape)
        b = (block_rows(full, j.shape[0], j.shape[1], jnp.dtype(rij.dtype).itemsize)
             if chunk == "auto" else int(chunk))
        return m.site_energies_dense_blocked(rij, zi, node_z[j], mask, node_z, b)

    def dense(neighbor):
        """True for a Dense JAX-MD list, False for a (full) Sparse one."""
        partition = _partition()
        if partition.is_sparse(neighbor.format):
            if neighbor.format is partition.OrderedSparse:
                raise ValueError("OrderedSparse is a half list; use partition.Sparse or Dense")
            if K is not None:
                raise ValueError("K compacts Dense rows; a partition.Sparse list is evaluated whole: "
                                 "build the potential without K, or use partition.Dense")
            return False
        return True

    def energy(R, neighbor, species=None, **kw):
        node_z, bad_species = node_z0 if species is None else node_z_of(species)
        with jax.default_matmul_precision("highest"):
            if not dense(neighbor):
                bad = bad_species | jnp.asarray(neighbor.did_buffer_overflow, bool)
                return energy_sparse(R, neighbor.idx, node_z, bad, kw)
            rij, j, mask = graph(R, neighbor, kw, bad_species)
            return total_energy(site_energies_dense(rij, j, mask, node_z))

    def force(R, neighbor, species=None, **kw):
        return -jax.grad(energy)(R, neighbor, species=species, **kw)

    def energy_sparse(R, idx, node_z, bad, kw):
        N = R.shape[0]
        i, j = idx[0], idx[1]                       # JAX-MD: idx[0] the centre, pad N
        live = i < N
        i, j = jnp.where(live, i, 0), jnp.where(live, j, 0)
        rij = jax.vmap(lambda a, b: displacement_fn(a, b, **kw))(R[j], R[i])
        mask = live & (jnp.sum(rij * rij, axis=-1) < rc * rc)
        rij = poison(jnp.where(mask[:, None], rij, park.astype(rij.dtype)), bad)   # never truncation
        return total_energy(m.site_energies(rij, node_z[i], node_z[j], i, N, node_z, mask))

    return Potential(energy, force, rc, K, overflow)


def min_image_ok(cell, cutoff):
    """Whether JAX-MD's minimum image holds every pair within `cutoff` (rc +
    dr_threshold) of this cell (rows: lattice vectors, ASE's convention): every
    perpendicular width exceeds 2 cutoff.  Below that JAX-MD silently misses
    images, so check before a run on a small or sheared cell."""
    cell = np.asarray(cell, float)
    vol = abs(np.linalg.det(cell))
    widths = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    return min(widths) > 2 * cutoff


def allocate(neighbor_fn, R, **kw):
    """`neighbor_fn.allocate(R, **kw)` with its `cell_size` a Python float.  JAX-MD
    keeps cell_size, a jax array, in a static (metadata) field; JAX 0.11 compares
    metadata when a jitted function sees a new pytree, and an array there raises
    ("metadata fields ... hashable") as soon as a *reallocated* list reaches a
    function compiled for the old one -- the usual overflow-and-reallocate loop."""
    nb = neighbor_fn.allocate(R, **kw)
    cs = getattr(nb, "cell_size", None)
    return nb.set(cell_size=float(cs)) if cs is not None and not isinstance(cs, float) else nb


def capacity(R, neighbor, displacement_fn, cutoff, step=4, **kw):
    """K for `potential`: the largest in-cutoff neighbour count over a Dense list
    at positions R, rounded up
    to `step` (`calc.skin.round_k`).  Add headroom (e.g. `round_k(1.2 * k)`) for
    liquids or compressing cells."""
    if neighbor.idx.ndim != 2 or neighbor.idx.shape[0] != R.shape[0]:
        raise ValueError("capacity needs a partition.Dense list: K compacts Dense rows")
    rij, _, live = _edges_dense(R, neighbor.idx, displacement_fn, kw)
    cnt = jnp.sum(live & (jnp.sum(rij * rij, axis=-1) < cutoff * cutoff), axis=1)
    return round_k(int(jnp.max(cnt)), step)
