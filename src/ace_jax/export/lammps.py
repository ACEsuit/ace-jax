"""ace-jax model -> lammps-jax bundle (`pair_style jax/kk`).

lammps-jax calls energy_fn(positions, species, graph), species = type - 1, and
per-atom energies are returned (lammps-jax masks ghost rows).  Three layouts:

* "sparse", "dense": graph is LAMMPS's packed edge buffer (senders = owned
  centre, receivers = neighbour, possibly a ghost; edge_mask marks real edges;
  lammps-jax packs only pairs within the contract cutoff, every step).  Sparse
  uses it as is; dense regroups it into (n, k_dense) slots inside the exported
  function (an argsort per step), independent of edge order.
* "matrix": graph is lammps-jax's neighbour matrix (export_model's
  max_neighbors / max_owned contract, lammps-jax >= 4a7f4fb): the LAMMPS full
  list itself, slot-major (max_neighbors, rows) int32, row i = atom i (owned
  atoms first), filled from slot 0, padding = max_atoms, with num_neighbors
  (rows,).  It is copied only when LAMMPS rebuilds its list, so it holds the
  skin pairs (rcut < r <= rcut + skin), which the model drops itself; there is
  no per-step packing.  If k_dense < max_neighbors the in-cutoff pairs of each
  row are compacted into k_dense model slots (a cumsum and a search per row,
  as calc/skin.py does); otherwise the model runs at the list's width.

Overflow is never a silent truncation: a dense row with more than k_dense
neighbours, or a matrix row with more than k_dense in-cutoff pairs, gives NaN
energies and forces; a LAMMPS list row wider than max_neighbors aborts the run
in lammps-jax ("neighbor capacity exceeded").

Needs lammps-jax (not on PyPI yet): `uv pip install -e path/to/lammps-jax`.
Only `export_lammps` imports it; `make_energy_fn` is plain JAX.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..eval.edge_model import LAYOUTS, estimate_a_bytes

BUNDLE_LAYOUTS = LAYOUTS + ("matrix",)

# Dense rows per block in the bundle (lax.map + jax.checkpoint above one block;
# the unblocked program at or below it).  Unblocked, XLA temp memory grows with
# the row count and the product-basis gather's adjoint scatter slows per row:
# in LAMMPS on an A100, PACE Cantor at 131k atoms ran 0.88M atom-steps/s
# unblocked against 1.12M blocked, and dense ACE at 131k ran out of memory
# unblocked.  Not the calculator's CHUNK_NODES = 16384: with
# max_owned ~ 1.1 N a 16k-atom system would split into two blocks and pay the
# checkpoint recompute, 1.5x slower; 32k rows keeps it one block and is as fast
# as or faster than 64k at scale.  docs/perf-lammps-large-n.md.
BUNDLE_BLOCK_ROWS = 32768


def matrix_supported():
    """Whether the installed lammps-jax exports the neighbour-matrix input
    layout (export_model(max_neighbors=...); lammps-jax 4a7f4fb and later)."""
    import inspect
    try:
        from lammps_jax import export
    except ImportError:
        return False
    return (hasattr(export, "LammpsNeighborMatrix")
            and "max_neighbors" in inspect.signature(export.export_model).parameters)


def neighbour_capacity(atoms, rcut, skin=1.0, slots="skin", margin=8, owned=1.1,
                       list_headroom=0.5):
    """lammps-jax buffer sizes for `atoms` (a periodic ase.Atoms, the structure
    the run starts from).  Returns max_atoms (owned + ghost shell within
    rcut + skin of each face, estimated from the cell-vector lengths, x1.1),
    max_owned (owned x `owned`), k_list / k_cut (largest coordination within
    rcut + skin / rcut), max_neighbors (neighbour-matrix list slots: LAMMPS
    copies its whole rcut + skin list at every rebuild, and a wider row aborts
    the run, so max(k_list + margin, ceil((1 + list_headroom) * k_list))), k_dense
    (model slots; the matrix compacts the in-cutoff pairs into them) and
    max_edges (max_owned * k_dense).

    The list needs more headroom than the model slots: when a structure
    compresses, the rcut + skin count grows as the rcut count does, from a
    larger base.  On the benchmark deck SiGe medium's widest list row grew
    from 34 to 45 by the first rebuild (k_list + 8 = 42 aborted), while its
    rcut slots never overflowed (docs/perf-lammps-large-n.md).

    slots="skin" (default, always safe between list rebuilds): k_dense =
    k_list + margin.  No atom can gain more neighbours within rcut than its
    rcut + skin list holds, so this covers structures that compress during the
    run (the benchmark's random-weight models push Cantor's coordination within
    rcut from 42 to 50 in 250 steps; docs/perf-lammps-large-n.md).

    slots="cutoff": k_dense = k_cut + margin, ~1.5x faster on Cantor (fewer
    model slots).  Safe for stable MD with a fitted model -- a thermalised
    crystal or liquid whose coordination within rcut stays within `margin` of
    the starting structure's.  If it does not, an atom overflows its slots and
    the step's energy and forces are NaN (the run fails loudly; never a
    truncation)."""
    if slots not in ("skin", "cutoff"):
        raise ValueError(f"slots must be 'skin' or 'cutoff', got {slots!r}")
    from ..eval import sparse_graph
    L = np.linalg.norm(atoms.cell.array, axis=1)
    ghost = float(np.prod((L + 2 * (rcut + skin)) / L))
    n = len(atoms)

    def k_max(c):
        g = sparse_graph(atoms.positions, atoms.cell.array, atoms.pbc, c)
        return int(np.bincount(g.senders, minlength=n).max())

    k_list = k_max(rcut + skin)
    k_cut = k_max(rcut)
    max_owned = int(np.ceil(owned * n))
    k_dense = (k_list if slots == "skin" else k_cut) + margin
    max_neighbors = max(k_list + margin, int(np.ceil((1 + list_headroom) * k_list)))
    return {"max_atoms": int(np.ceil(n * ghost * 1.1)), "max_owned": max_owned,
            "k_list": k_list, "k_cut": k_cut, "max_neighbors": max_neighbors,
            "k_dense": k_dense, "max_edges": max_owned * k_dense}


def _matrix_energy_fn(model, n_species, k_dense, type_map, rcut):
    """Energy function over lammps-jax's neighbour matrix (module docstring)."""
    rc2 = float(rcut) ** 2

    def energy_fn(positions, species, graph):
        n = positions.shape[0]
        nb = graph.neighbors.T                                 # (rows, list slots)
        rows, K = nb.shape
        node_z = jnp.clip(species, 0, n_species - 1)
        if type_map is not None:
            node_z = jnp.asarray(type_map, jnp.int32)[node_z]
        pad = jnp.asarray([1.0, 0.0, 0.0], positions.dtype) * model.pad_cutoff()
        live = (jnp.arange(K)[None, :] < graph.num_neighbors[:, None]) & (nb < n)
        j = jnp.where(live, nb, 0)                             # padding indexes max_atoms
        d = positions[j] - positions[:rows, None, :]
        r2 = jnp.sum(d * d, axis=-1)
        valid = live & (r2 <= rc2)                             # skin pairs out
        # a NaN distance fails both tests, and the model zeroes a NaN pair: flag
        # it, so a NaN position gives NaN energies instead of a dropped pair
        broken = jnp.any(live & jnp.isnan(r2))
        if k_dense is None or k_dense >= K:
            md, idx = valid, j
            rd = jnp.where(valid[..., None], d, pad)
            overflow = broken
        else:
            # in-cutoff slots first, in list order: model slot k takes the
            # (k+1)-th valid entry, found by a search in the row's running count
            csum = jnp.cumsum(valid.astype(jnp.int32), axis=1)
            cnt = csum[:, -1]
            want = jnp.arange(1, k_dense + 1, dtype=jnp.int32)
            order = jax.vmap(lambda c: jnp.searchsorted(c, want, side="left"))(csum)
            order = jnp.minimum(order, K - 1)
            md = want[None, :] <= cnt[:, None]
            idx = jnp.where(md, jnp.take_along_axis(j, order, axis=1), 0)
            rd = jnp.where(md[..., None], jnp.take_along_axis(d, order[..., None], axis=1), pad)
            overflow = jnp.any(cnt > k_dense) | broken
        zr = node_z[:rows]
        e = model.site_energies_dense_blocked(rd, jnp.broadcast_to(zr[:, None], idx.shape),
                                              node_z[idx], md, zr, BUNDLE_BLOCK_ROWS)
        if rows < n:
            e = jnp.concatenate([e, jnp.zeros((n - rows,), e.dtype)])
        # multiply, not where: a where would leave the forces finite (make_energy_fn)
        return e * jnp.where(overflow, jnp.nan, 1.0)

    return energy_fn


def make_energy_fn(model, n_species, layout, k_dense=None, type_map=None, n_rows=None,
                   rcut=None):
    """type_map[t] is the model species of LAMMPS type t+1 (default: identity).

    layout="matrix" takes lammps-jax's neighbour matrix as the graph (module
    docstring) and needs rcut, the cutoff that drops the list's skin pairs
    (the bundle's contract cutoff).  Its rows are the matrix's (max_owned);
    k_dense (default: the list's width) below the list width compacts the
    in-cutoff pairs into k_dense model slots per step.  n_rows is ignored.

    n_rows (dense only): evaluate rows < n_rows only -- owned atoms, which
    LAMMPS numbers first -- padding the rest with zero energy, and return NaN
    (energy and, via the multiplicative overflow below, forces too) if a
    sender is >= n_rows or a slot overflows k_dense.  Capped at the actual
    row count, so n_rows >= the buffer's row count is a no-op.

    Dense rows are evaluated in blocks of BUNDLE_BLOCK_ROWS (read at trace
    time) when there are more than that; see `site_energies_dense_blocked`.
    """
    if layout not in BUNDLE_LAYOUTS:
        raise ValueError(f"layout must be one of {BUNDLE_LAYOUTS}, got {layout!r}")
    if layout == "dense" and not k_dense:
        raise ValueError("the dense layout needs k_dense (max neighbours per atom)")
    if layout == "matrix":
        if rcut is None:
            raise ValueError("the matrix layout needs rcut: the list holds skin pairs")
        return _matrix_energy_fn(model, n_species, k_dense, type_map, rcut)

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
        e = model.site_energies_dense_blocked(rd, jnp.broadcast_to(zr[:, None], idx.shape),
                                              node_z[idx], md, zr, BUNDLE_BLOCK_ROWS)
        overflow = jnp.any(m & ((s >= nr) | (slot >= k_dense)))
        if nr < n:
            e = jnp.concatenate([e, jnp.zeros((n - nr,), e.dtype)])
        # multiply, not where: jnp.where sends no cotangent to e when overflow
        # is set, so the energy would be NaN but the forces silently zero
        return e * jnp.where(overflow, jnp.nan, 1.0)

    return energy_fn


def export_lammps(model, meta, path, *, max_atoms, max_edges=None, k_dense=None,
                  dtype="float64", layout="auto", type_elements=None, max_owned=None,
                  lean=True, max_neighbors=None):
    """Write a lammps-jax JSON bundle for `model`; returns the bundle dict.

    Capacities: max_atoms (owned + ghost positions); max_edges (sparse / dense:
    the packed edge buffer, pairs within rcut); k_dense (dense / matrix: model
    slots per atom); max_neighbors (matrix only: list slots per row, default
    k_dense -- LAMMPS copies its rcut + skin list whole, and a wider row aborts
    the run).  `neighbour_capacity` sizes all of them for a structure; its
    slots="cutoff" gives tight model slots (k_dense < max_neighbors).

    type_elements: atomic numbers in LAMMPS type order (type 1 first).  LAMMPS
    hands the model species = type - 1, and a model's own element order (a
    .yace lists its elements as fitted) need not match; default: the model's.

    max_owned: dense / matrix row capacity (owned atoms only, LAMMPS numbers
    them first).  When given, the dense energy function evaluates only rows < it
    (see make_energy_fn's n_rows) and the matrix has that many rows (lammps-jax's
    max_owned); always recorded in the bundle, including for the sparse layout,
    which has no row concept and ignores it otherwise.

    layout="auto" picks a dense-family layout when k_dense is given and
    estimate_a_bytes for one dense block (the row capacity -- max_owned if
    given, else max_atoms -- capped at BUNDLE_BLOCK_ROWS) fits
    ace_jax.calc.point.dense_budget_bytes(), else sparse.  The dense-family
    layout is "matrix" when the installed lammps-jax supports it
    (`matrix_supported`), else "dense".

    lean (default True): export `ace_jax.eval.model.lean(model)`, the exact
    evaluation form with the dead per-edge work removed (docs/ace-vs-pace-gap.md);
    recorded as `ace_jax.lean` (False for a model it does not apply to, e.g. PACE).
    """
    from lammps_jax.export import export_model

    from ..eval.model import lean as _lean
    from ..calc.point import dense_budget_bytes
    model_z = [int(z) for z in meta["elements"]]
    type_elements = model_z if type_elements is None else [int(z) for z in type_elements]
    missing = sorted(set(type_elements) - set(model_z))
    if missing:
        raise ValueError(f"LAMMPS types {missing} are not in the model's elements {model_z}")
    type_map = [model_z.index(z) for z in type_elements]
    n_species = len(type_elements)
    if layout not in BUNDLE_LAYOUTS + ("auto",):
        raise ValueError(f"layout must be 'auto' or one of {BUNDLE_LAYOUTS}, got {layout!r}")
    if layout == "matrix" and not (k_dense or max_neighbors):
        raise ValueError("the matrix layout needs k_dense or max_neighbors (slots per atom)")
    if layout in ("sparse", "dense") and max_edges is None:
        raise ValueError(f"the {layout} layout needs max_edges (the packed edge buffer)")
    if layout == "auto":
        itemsize = np.dtype(dtype).itemsize
        # rows run in BUNDLE_BLOCK_ROWS blocks, so one block's temporaries bound memory
        n_rows = min(max_owned if max_owned is not None else max_atoms, BUNDLE_BLOCK_ROWS)
        k = min(k_dense, max_neighbors or k_dense) if k_dense else 0
        edges = n_rows * k if max_edges is None else min(max_edges, n_rows * k)
        fits = k and estimate_a_bytes(model, "dense", n_rows, edges, k,
                                      itemsize) <= dense_budget_bytes()
        matrix = matrix_supported()
        if not matrix and max_edges is None:
            raise ValueError("layout='auto' needs max_edges: this lammps-jax has no "
                             "neighbour-matrix layout")
        layout = ("matrix" if matrix else "dense") if fits else "sparse"
        if layout == "sparse" and max_edges is None:
            raise ValueError("layout='auto' fell back to sparse, which needs max_edges")
    if layout == "matrix":
        max_neighbors = int(max_neighbors or k_dense)
        k_dense = min(int(k_dense or max_neighbors), max_neighbors)
    # layout="auto" above sized one block on the full model, as without lean:
    # estimate_a_bytes fits the full dense path, and on the lean widths it
    # underestimates the blocked path's compiled temp (measured 6.6-6.9x actual /
    # estimate on CPU, against 3.0-5.1x full), so it could pick dense and OOM
    if lean:
        model = _lean(model)
    rcut = float(meta["rcut"])
    energy_fn = make_energy_fn(model, n_species, layout, k_dense,
                               None if type_map == list(range(len(model_z))) else type_map,
                               n_rows=max_owned if layout == "dense" else None, rcut=rcut)
    graph = ({"max_neighbors": max_neighbors, "max_owned": max_owned} if layout == "matrix"
             else {"max_edges": max_edges})
    bundle = export_model(energy_fn=energy_fn, path=path, max_atoms=max_atoms, cutoff=rcut,
                          unit_style="metal", precision=dtype, n_species=n_species, **graph)
    bundle["ace_jax"] = {"layout": layout, "elements": model_z, "type_elements": type_elements,
                         "k_dense": int(k_dense) if layout in ("dense", "matrix") else None,
                         # not "max_owned": lammps-jax reads that key from anywhere in
                         # the file and would take it as its own contract's
                         "owned_rows": int(max_owned) if max_owned is not None else None,
                         "lean": bool(getattr(model, "energy_only", False))}
    Path(path).write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
