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
from ..eval.splinify import AUTO

BUNDLE_LAYOUTS = LAYOUTS + ("matrix",)

# Dense rows per block in the bundle (lax.map + jax.checkpoint above one block;
# the unblocked program at or below it).  Unblocked, XLA temp memory grows with
# the row count and the product-basis gather's adjoint scatter slows per row:
# in LAMMPS on an A100, PACE Cantor at 131k atoms ran 0.88M atom-steps/s
# unblocked against 1.12M blocked, and dense ACE at 131k ran out of memory
# unblocked.  Not the calculator's CHUNK_NODES = 16384: with
# max_owned ~ 1.1 N a 16k-atom system would split into two blocks and pay the
# checkpoint recompute, 1.5x slower; 32k rows keeps it one block and is as fast
# as or faster than 64k at scale.  docs/dev/perf-lammps-large-n.md.
BUNDLE_BLOCK_ROWS = 32768
# The smallest block layout="auto" halves down to before it gives up on the dense
# family for sparse: blocked dense at a few thousand rows still beats the sparse
# edge list many times over (o4d16 Si on an A100: sparse 15x slower than dense).
MIN_BLOCK_ROWS = 1024


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


def lammps_jax_version():
    """The installed lammps-jax as '<version>[+<git commit>]', or None."""
    import importlib.metadata
    try:
        dist = importlib.metadata.distribution("lammps_jax")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        commit = json.loads(dist.read_text("direct_url.json") or "{}").get(
            "vcs_info", {}).get("commit_id")
    except (ValueError, AttributeError):
        commit = None
    return dist.version + (f"+{commit[:7]}" if commit else "")


def matrix_prep_bytes(rows, max_neighbors, k_dense, itemsize):
    """Rough bytes of the matrix layout's pre-processing, which runs over every
    row at once (not in BUNDLE_BLOCK_ROWS blocks): per list slot the neighbour
    index, the edge vector and its cotangent, r2, masks and the running count;
    per model slot the compacted vector and its cotangent, indices and mask."""
    return int(rows) * (int(max_neighbors) * (6 * itemsize + 14)
                        + int(k_dense) * (6 * itemsize + 13))


def block_rows_for(model, rows, k_dense, budget, itemsize, max_edges=None, prep=0, block_rows=None):
    """Rows per dense block for a bundle of `rows` rows and `k_dense` slots: the
    largest of BUNDLE_BLOCK_ROWS, halved down to MIN_BLOCK_ROWS (each capped at
    `rows`), whose estimate_a_bytes plus `prep` (bytes not run in blocks, e.g.
    matrix_prep_bytes) fits `budget`; None when none does.  An explicit
    `block_rows` is the only candidate."""
    cands, b = [], min(int(rows), BUNDLE_BLOCK_ROWS)
    while True:
        cands.append(b)
        if b <= MIN_BLOCK_ROWS:
            break
        b = max(MIN_BLOCK_ROWS, b // 2)
    for b in ([int(block_rows)] if block_rows else cands):
        edges = b * k_dense if max_edges is None else min(max_edges, b * k_dense)
        if estimate_a_bytes(model, "dense", b, edges, k_dense, itemsize) + prep <= budget:
            return b
    return None


def neighbour_capacity(atoms, rcut, skin=1.0, slots="skin", margin=8, owned=1.1,
                       list_headroom=0.5):
    """lammps-jax buffer sizes for `atoms` (a periodic ase.Atoms, the structure
    the run starts from).  Returns max_atoms (owned + ghost shell rcut + skin
    deep perpendicular to each face -- the face spacings, 1/|reciprocal row|,
    so sheared cells are covered -- x1.1),
    max_owned (owned x `owned`), k_list / k_cut (largest coordination within
    rcut + skin / rcut), max_neighbors (neighbour-matrix list slots: LAMMPS
    copies its whole rcut + skin list at every rebuild, and a wider row aborts
    the run, so max(k_list + margin, ceil((1 + list_headroom) * k_list))), k_dense
    (model slots; the matrix compacts the in-cutoff pairs into them) and
    max_edges (max_owned * k_dense).

    The list needs more headroom than the model slots: when a structure
    compresses, the rcut + skin count grows as the rcut count does, from a
    larger base.  list_headroom = 0.5 comes from ONE observed overflow: on the
    benchmark deck SiGe medium's widest list row grew from 34 to 45 by the
    first rebuild (k_list + 8 = 42 aborted), while its rcut slots never
    overflowed (docs/dev/perf-lammps-large-n.md).  It is a guess, not a bound;
    for stable MD list_headroom=0 (with margin >= 8) is cheaper.

    slots="skin" (default, always safe between list rebuilds): k_dense =
    k_list + margin.  No atom can gain more neighbours within rcut than its
    rcut + skin list holds, so this covers structures that compress during the
    run (the benchmark's random-weight models push Cantor's coordination within
    rcut from 42 to 50 in 250 steps; docs/dev/perf-lammps-large-n.md).

    slots="cutoff": k_dense = k_cut + margin, 1.2-1.4x faster on Cantor (fewer
    model slots).  Safe for stable MD with a fitted model -- a thermalised
    crystal or liquid whose coordination within rcut stays within `margin` of
    the starting structure's.  If it does not, an atom overflows its slots and
    the step's energy and forces are NaN (the run fails loudly; never a
    truncation)."""
    if slots not in ("skin", "cutoff"):
        raise ValueError(f"slots must be 'skin' or 'cutoff', got {slots!r}")
    from ..eval import sparse_graph
    h = 1.0 / np.linalg.norm(atoms.cell.reciprocal(), axis=1)   # face spacings
    ghost = float(np.prod((h + 2 * (rcut + skin)) / h))
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


def _matrix_energy_fn(model, n_species, k_dense, type_map, rcut, block_rows):
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
                                              node_z[idx], md, zr, block_rows or BUNDLE_BLOCK_ROWS)
        if rows < n:
            e = jnp.concatenate([e, jnp.zeros((n - rows,), e.dtype)])
        # multiply, not where: a where would leave the forces finite (make_energy_fn)
        return e * jnp.where(overflow, jnp.nan, 1.0)

    return energy_fn


def make_energy_fn(model, n_species, layout, k_dense=None, type_map=None, n_rows=None,
                   rcut=None, block_rows=None):
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

    Dense rows are evaluated in blocks of block_rows (default BUNDLE_BLOCK_ROWS,
    read at trace time) when there are more than that; see
    `site_energies_dense_blocked`.
    """
    if layout not in BUNDLE_LAYOUTS:
        raise ValueError(f"layout must be one of {BUNDLE_LAYOUTS}, got {layout!r}")
    if layout == "dense" and not k_dense:
        raise ValueError("the dense layout needs k_dense (max neighbours per atom)")
    if layout == "matrix":
        if rcut is None:
            raise ValueError("the matrix layout needs rcut: the list holds skin pairs")
        return _matrix_energy_fn(model, n_species, k_dense, type_map, rcut, block_rows)

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
                                              node_z[idx], md, zr, block_rows or BUNDLE_BLOCK_ROWS)
        overflow = jnp.any(m & ((s >= nr) | (slot >= k_dense)))
        if nr < n:
            e = jnp.concatenate([e, jnp.zeros((n - nr,), e.dtype)])
        # multiply, not where: jnp.where sends no cotangent to e when overflow
        # is set, so the energy would be NaN but the forces silently zero
        return e * jnp.where(overflow, jnp.nan, 1.0)

    return energy_fn


def export_lammps(model, meta, path, *, max_atoms, max_edges=None, k_dense=None,
                  dtype="float64", layout="auto", type_elements=None, max_owned=None,
                  lean=True, spline_tol=AUTO, spline_intervals=None, max_neighbors=None,
                  radial_table=None, device_memory=None, block_rows=None):
    """Write a lammps-jax JSON bundle for `model`; returns the bundle dict.

    Capacities: max_atoms (owned + ghost positions); max_edges (sparse / dense:
    the packed edge buffer, pairs within rcut); k_dense (dense / matrix: model
    slots per atom, for the matrix default max_neighbors); max_neighbors
    (matrix only, required: list slots per row -- LAMMPS copies its rcut + skin
    list whole, so a k_dense sized for pairs within rcut is too small, and a
    wider row aborts the run).  `neighbour_capacity` sizes all of them for a
    structure; its slots="cutoff" gives tight model slots (k_dense <
    max_neighbors).

    type_elements: atomic numbers in LAMMPS type order (type 1 first).  LAMMPS
    hands the model species = type - 1, and a model's own element order (a
    .yace lists its elements as fitted) need not match; default: the model's.

    max_owned: dense / matrix row capacity (owned atoms only, LAMMPS numbers
    them first).  When given, the dense energy function evaluates only rows < it
    (see make_energy_fn's n_rows) and the matrix has that many rows (lammps-jax's
    max_owned); always recorded in the bundle, including for the sparse layout,
    which has no row concept and ignores it otherwise.

    device_memory: bytes XLA can allocate on the device LAMMPS will run on (with
    XLA's default preallocation, 75% of the GPU's memory).  The export need not
    run there, so this is a setting of the bundle; None queries the exporting
    process's device (`calc.point.dense_budget_bytes`: CPU_DENSE_BUDGET_BYTES
    where it reports none).  The dense-family budget is DENSE_BUDGET_FRACTION of it.

    Dense and matrix rows run in blocks of `block_rows` rows (lax.map +
    jax.checkpoint), so one block's temporaries bound memory.  Default:
    BUNDLE_BLOCK_ROWS for an explicit layout without device_memory, else
    `block_rows_for`: the largest of BUNDLE_BLOCK_ROWS halved down to
    MIN_BLOCK_ROWS whose estimate fits the budget.

    layout="auto" picks a dense-family layout when k_dense is given and some
    block fits (estimate_a_bytes for that many rows, of the row capacity --
    max_owned if given, else max_atoms), else sparse.  The dense-family layout
    is "matrix" when max_neighbors is given, the installed lammps-jax supports it
    (`matrix_supported`) and a block plus the matrix's unblocked pre-processing
    (`matrix_prep_bytes`) fits; else "dense".  Recorded as `ace_jax.block_rows`,
    `ace_jax.device_memory` and `ace_jax.dense_budget`.  A LAMMPS plugin
    older than the Python package rejects a matrix bundle: rebuild the plugin,
    or export layout="dense".  The bundle records the exporting lammps-jax as
    `ace_jax.lammps_jax`.

    lean (default True): export `ace_jax.eval.model.lean(model, spline_tol,
    spline_intervals)`, the evaluation form with the dead per-edge work removed
    (docs/dev/ace-vs-pace-gap.md).  spline_tol="auto" (default) first splines a
    learned analytic tensor radial (`radial_learned`) at 1e-10, which is not
    roundoff: energies agree with lean=False to up to ~1e-9 relative and forces
    to up to ~2.3e-8 of max|F| on the benchmark models
    (docs/dev/learned-radial-splining.md).  Other analytic models (ACEpotentials
    `ace_model` exports, built bases) stay exact; a float spline_tol opts
    them in, None never splines.
    Recorded from what lean actually did (looking through a wrapper's `.base`):
    `ace_jax.lean` (False when lean returned the model as given, e.g. PACE or an
    unfolded model), `ace_jax.spline_tol` and `ace_jax.spline_intervals`
    ({radial: n}), both None when nothing was splined.

    radial_table (None, the default: off; True: 4000 intervals; an int: that
    many): tabulate the exported model's radial stage in r after `lean`
    (`eval.model.with_radial_table`; ACE R_nl and the pair radial, PACE g_k),
    for every layout.  A CPU speed-up and an approximation (at 4000 intervals
    energies to ~1e-11 relative, forces to ~1e-8 of max|F|); exactly zero
    beyond each pair's cutoff, so the matrix layout's skin pairs still drop
    out.  Recorded as `ace_jax.radial_table` (n_intervals, r_min, r_max,
    max_rel_err, max_rel_deriv_err; None when off), next to
    `ace_jax.spline_tol`.
    """
    from lammps_jax.export import export_model

    from ..eval.model import lean as _lean
    from ..eval.model import splining, with_radial_table
    from ..calc.point import DENSE_BUDGET_FRACTION, dense_budget_bytes
    model_z = [int(z) for z in meta["elements"]]
    type_elements = model_z if type_elements is None else [int(z) for z in type_elements]
    missing = sorted(set(type_elements) - set(model_z))
    if missing:
        raise ValueError(f"LAMMPS types {missing} are not in the model's elements {model_z}")
    type_map = [model_z.index(z) for z in type_elements]
    n_species = len(type_elements)
    if layout not in BUNDLE_LAYOUTS + ("auto",):
        raise ValueError(f"layout must be 'auto' or one of {BUNDLE_LAYOUTS}, got {layout!r}")
    if layout == "matrix" and not max_neighbors:
        raise ValueError("the matrix layout needs max_neighbors (list slots per row, for "
                         "the rcut + skin list; see neighbour_capacity)")
    if layout in ("sparse", "dense") and max_edges is None:
        raise ValueError(f"the {layout} layout needs max_edges (the packed edge buffer)")
    from ..calc.point import device_memory_bytes
    itemsize = np.dtype(dtype).itemsize
    if device_memory is None:
        budget, device_memory = dense_budget_bytes(), device_memory_bytes()
    else:
        budget = DENSE_BUDGET_FRACTION * float(device_memory)
    rows = max_owned if max_owned is not None else max_atoms
    sized = layout == "auto" or (layout in ("dense", "matrix") and (device_memory is not None or block_rows))
    if layout == "auto":
        k = min(k_dense, max_neighbors or k_dense) if k_dense else 0
        b_mat = (block_rows_for(model, rows, k, budget, itemsize, max_edges,
                                matrix_prep_bytes(rows, max_neighbors, k, itemsize), block_rows)
                 if k and max_neighbors and matrix_supported() else None)
        b_den = (block_rows_for(model, rows, k, budget, itemsize, max_edges, 0, block_rows)
                 if k and max_edges is not None and not b_mat else None)
        if b_mat:
            layout, block_rows = "matrix", b_mat
        elif b_den:
            layout, block_rows = "dense", b_den
        elif max_edges is not None:
            layout, block_rows = "sparse", None
        else:
            raise ValueError("layout='auto' needs max_edges unless the matrix layout is "
                             "chosen (max_neighbors given, supported, and fitting)")
    elif layout in ("dense", "matrix") and not block_rows:
        k = min(k_dense or max_neighbors, max_neighbors or k_dense or 0)
        block_rows = ((block_rows_for(model, rows, k, budget, itemsize, max_edges if layout == "dense" else None,
                                      matrix_prep_bytes(rows, max_neighbors, k, itemsize) if layout == "matrix" else 0)
                       or MIN_BLOCK_ROWS) if sized else BUNDLE_BLOCK_ROWS)
    if layout == "matrix":
        max_neighbors = int(max_neighbors)
        k_dense = min(int(k_dense or max_neighbors), max_neighbors)
    # layout="auto" above sized one block on the full model, as without lean:
    # estimate_a_bytes fits the full dense path, and on the lean widths it
    # underestimates the blocked path's compiled temp (measured 6.6-6.9x actual /
    # estimate on CPU, against 3.0-5.1x full), so it could pick dense and OOM
    splined, leaned = None, False
    if lean:
        full = model
        model = _lean(full, spline_tol, spline_intervals)
        splined = splining(full, model, spline_tol)
        leaned = model is not full or bool(getattr(model, "energy_only", False))
    model, rtab = with_radial_table(model, radial_table)
    rcut = float(meta["rcut"])
    energy_fn = make_energy_fn(model, n_species, layout, k_dense,
                               None if type_map == list(range(len(model_z))) else type_map,
                               n_rows=max_owned if layout == "dense" else None, rcut=rcut,
                               block_rows=block_rows if layout in ("dense", "matrix") else None)
    graph = ({"max_neighbors": max_neighbors, "max_owned": max_owned} if layout == "matrix"
             else {"max_edges": max_edges})
    bundle = export_model(energy_fn=energy_fn, path=path, max_atoms=max_atoms, cutoff=rcut,
                          unit_style="metal", precision=dtype, n_species=n_species, **graph)
    bundle["ace_jax"] = {"layout": layout, "elements": model_z, "type_elements": type_elements,
                         "k_dense": int(k_dense) if layout in ("dense", "matrix") else None,
                         # not "max_owned": lammps-jax reads that key from anywhere in
                         # the file and would take it as its own contract's
                         "owned_rows": int(max_owned) if max_owned is not None else None,
                         "lean": leaned,
                         # what the analytic radials were splined to (None: not splined)
                         "spline_tol": splined["spline_tol"] if splined else None,
                         "spline_intervals": splined["n_intervals"] if splined else None,
                         # the radial stage tabulated in r (None: analytic / as lean left it)
                         "radial_table": rtab,
                         # dense / matrix rows per block, and the memory they were sized for
                         "block_rows": int(block_rows) if layout in ("dense", "matrix") else None,
                         "device_memory": int(device_memory) if device_memory else None,
                         "dense_budget": int(budget),
                         "lammps_jax": lammps_jax_version()}
    Path(path).write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
