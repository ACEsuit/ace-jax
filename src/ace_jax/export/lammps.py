"""ace-jax model -> lammps-jax bundle (`pair_style jax/kk`).

lammps-jax calls energy_fn(positions, species, graph) with LAMMPS's packed edge
buffer (senders = owned centre, receivers = neighbour, possibly a ghost;
edge_mask marks real edges) and species = type - 1.  Per-atom energies are
returned; lammps-jax masks ghost rows.  Both EdgeSiteModel layouts work: sparse
uses the buffer as is; dense regroups it into (n, k_dense) slots inside the
exported function, independent of edge order, and returns NaN energies if an
atom has more than k_dense neighbours (never a silent truncation).

Needs lammps-jax (not on PyPI yet): `uv pip install -e path/to/lammps-jax`.
Only `export_lammps` imports it; `make_energy_fn` is plain JAX.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..eval.edge_model import LAYOUTS, estimate_a_bytes


def make_energy_fn(model, n_species, layout, k_dense=None, type_map=None, n_rows=None):
    """type_map[t] is the model species of LAMMPS type t+1 (default: identity).

    n_rows (dense only): evaluate rows < n_rows only -- owned atoms, which
    LAMMPS numbers first -- padding the rest with zero energy, and return NaN
    (energy and, via the multiplicative overflow below, forces too) if a
    sender is >= n_rows or a slot overflows k_dense.  Capped at the actual
    row count, so n_rows >= the buffer's row count is a no-op.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if layout == "dense" and not k_dense:
        raise ValueError("the dense layout needs k_dense (max neighbours per atom)")

    def energy_fn(positions, species, graph):
        n = positions.shape[0]
        nr = n if n_rows is None else min(n_rows, n)
        m = graph.edge_mask
        s = jnp.where(m, graph.senders, 0)
        r = jnp.where(m, graph.receivers, 0)
        node_z = jnp.clip(species, 0, n_species - 1)
        if type_map is not None:
            node_z = jnp.asarray(type_map, jnp.int32)[node_z]
        pad = jnp.asarray([1.0, 0.0, 0.0], positions.dtype) * model.pad_cutoff()
        rij = jnp.where(m[:, None], positions[r] - positions[s], pad)
        if layout == "sparse":
            return model.site_energies(rij, node_z[s], node_z[r], s, n, node_z, m)
        # dense: rank each edge within its centre's group, whatever the order,
        # restricted to rows < nr (owned atoms, when n_rows narrows the buffer)
        in_range = m & (s < nr)
        key = jnp.where(in_range, s, nr)                      # out of range sorts last
        order = jnp.argsort(key, stable=True)
        ks = key[order]
        counts = jax.ops.segment_sum(jnp.ones_like(ks), ks, num_segments=nr + 1)
        starts = jnp.cumsum(counts) - counts
        slot_sorted = jnp.arange(ks.shape[0]) - starts[ks]
        slot = jnp.zeros_like(slot_sorted).at[order].set(slot_sorted)
        ok = in_range & (slot < k_dense)
        row = jnp.where(ok, s, nr)                            # out of range: dropped
        col = jnp.where(ok, slot, 0)
        rd = jnp.broadcast_to(pad, (nr, k_dense, 3)).at[row, col].set(rij, mode="drop")
        idx = jnp.zeros((nr, k_dense), jnp.int32).at[row, col].set(r, mode="drop")
        md = jnp.zeros((nr, k_dense), bool).at[row, col].set(True, mode="drop")
        zr = node_z[:nr]
        e = model.site_energies_dense(rd, jnp.broadcast_to(zr[:, None], idx.shape),
                                      node_z[idx], md, zr)
        overflow = jnp.any(m & ((s >= nr) | (slot >= k_dense)))
        if nr < n:
            e = jnp.concatenate([e, jnp.zeros((n - nr,), e.dtype)])
        # multiply, not where: jnp.where sends no cotangent to e when overflow
        # is set, so the energy would be NaN but the forces silently zero
        return e * jnp.where(overflow, jnp.nan, 1.0)

    return energy_fn


def export_lammps(model, meta, path, *, max_atoms, max_edges, k_dense=None,
                  dtype="float64", layout="auto", type_elements=None, max_owned=None):
    """Write a lammps-jax JSON bundle for `model`; returns the bundle dict.

    type_elements: atomic numbers in LAMMPS type order (type 1 first).  LAMMPS
    hands the model species = type - 1, and a model's own element order (a
    .yace lists its elements as fitted) need not match; default: the model's.

    max_owned: dense-layout row capacity (owned atoms only, LAMMPS numbers them
    first).  When given, the dense energy function evaluates only rows < it
    (see make_energy_fn's n_rows); always recorded in the bundle, including for
    the sparse layout, which has no row concept and ignores it otherwise.

    layout="auto" picks dense when k_dense is given and estimate_a_bytes for the
    dense row capacity (max_owned if given, else max_atoms) fits
    ace_jax.calc.point.dense_budget_bytes(), else sparse.
    """
    from lammps_jax.export import export_model
    from ..calc.point import dense_budget_bytes
    model_z = [int(z) for z in meta["elements"]]
    type_elements = model_z if type_elements is None else [int(z) for z in type_elements]
    missing = sorted(set(type_elements) - set(model_z))
    if missing:
        raise ValueError(f"LAMMPS types {missing} are not in the model's elements {model_z}")
    type_map = [model_z.index(z) for z in type_elements]
    n_species = len(type_elements)
    if layout == "auto":
        itemsize = np.dtype(dtype).itemsize
        n_rows = max_owned if max_owned is not None else max_atoms
        fits = k_dense and estimate_a_bytes(model, "dense", n_rows, max_edges, k_dense,
                                            itemsize) <= dense_budget_bytes()
        layout = "dense" if fits else "sparse"
    energy_fn = make_energy_fn(model, n_species, layout, k_dense,
                               None if type_map == list(range(len(model_z))) else type_map,
                               n_rows=max_owned if layout == "dense" else None)
    bundle = export_model(energy_fn=energy_fn, path=path, max_atoms=max_atoms,
                          max_edges=max_edges, cutoff=float(meta["rcut"]), unit_style="metal",
                          precision=dtype, n_species=n_species)
    bundle["ace_jax"] = {"layout": layout, "elements": model_z, "type_elements": type_elements,
                         "k_dense": int(k_dense) if layout == "dense" else None,
                         # not "max_owned": lammps-jax reads that key from anywhere in
                         # the file and would take it as its own contract's
                         "owned_rows": int(max_owned) if max_owned is not None else None}
    Path(path).write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
