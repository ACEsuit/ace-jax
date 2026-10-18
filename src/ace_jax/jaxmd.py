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

EXPERIMENTAL: `neighbor_list` is the alternative: a `ShiftList` built by
matscipy-neighbours' XLA FFI (CPU or CUDA, inside jit), carrying each pair's image
shift, so any cell works (smaller than the cutoff, triclinic, slab), as does NPT.
It needs a matscipy-neighbours build with its XLA FFI (libAtoms/matscipy-neighbours#10,
not yet released), and its API may change until that is released:

    nf = jaxmd.neighbor_list(box, jaxmd.cutoff(model), skin=1.0, fractional_coordinates=True)
    nbrs = nf.allocate(R)                    # host-sized capacity, then built on the device
    ... nbrs = nbrs.update(state.position) in the jitted loop (CUDA, planned build), or
    state, nbrs, info = jax.jit(lambda s, b: jaxmd.run(apply, s, b, 1000))(state, nbrs)
    # potential(model, None, species, K=K): the list gives the cell and image shifts

On CUDA the list is built by matscipy-neighbours' planned build (experimental:
`xla.plan` fixes the cell grid and workspace at allocation; the build is enqueued
on XLA's stream with no host synchronisation and is CUDA-graph compatible), so a
plain `nbrs = nbrs.update(R)` every step stays on the device: 512 Si atoms, 300
NVE steps, 0 host syncs and 3 one-byte reads (the per-chunk overflow checks), as
JAX-MD's own list; with the regular build, 452 syncs.  Without the planned build
(CPU, or a build without it) drive the list with `run(apply, state, nbrs,
n_steps, check_every=10)`, which checks once per block of steps (4096 atoms,
1000 K, shared A4500: 27 ms/step per-step against 6.5 with `run`).  A rebuild
costs 5.9 ms against JAX-MD's 43 ms (32768 atoms, shared A4500).
"""
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .calc.skin import compact, round_k
from .eval.edge_model import CHUNK_NODES, total_energy


def _jax_md(module):
    """A jax_md submodule, or an ImportError that says what to install."""
    import importlib
    try:
        return importlib.import_module(f"jax_md.{module}")
    except ImportError as e:
        raise ImportError('ace_jax.jaxmd needs jax-md: pip install "ace-jax[jaxmd]"') from e


def _partition():
    return _jax_md("partition")


def _xla():
    """matscipy_neighbours.xla (the XLA FFI build), or an ImportError that says what it needs."""
    try:
        from matscipy_neighbours import xla
        xla._register()
    except ImportError as e:
        raise ImportError("jaxmd.neighbor_list needs matscipy-neighbours built with its XLA FFI "
                          "(libAtoms/matscipy-neighbours#10, not yet released; CMake option "
                          f"JAX_FFI_INCLUDE_DIR): {e}") from e
    return xla


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
    from .calc.point import ACECalculator
    calc = ACECalculator(model, meta, **calc_kw)
    return calc.eval_model, calc.meta, calc.cutoff


def _edges_dense(R, idx, displacement_fn, kw):
    """rij (N, Ks, 3) = r_j - r_i over a JAX-MD Dense list, with live (N, Ks): pad index N."""
    N = R.shape[0]
    live = idx < N
    j = jnp.where(live, idx, 0)
    d = lambda Ra, Rb: displacement_fn(Ra, Rb, **kw)
    rij = jax.vmap(jax.vmap(d, (0, None)), (0, 0))(R[j], R)   # d(R_j, R_i) = r_j - r_i
    return rij, j, live


# ------------------------------------------------------------------ shift list
def _cell(box):
    """Lattice vectors as rows (ASE's convention) from a JAX-MD box: a (3, 3) box has
    them as columns (`space.periodic_general`); a scalar or (3,) box is orthorhombic
    (`space.periodic`)."""
    box = jnp.asarray(box)
    return box.T if box.ndim == 2 else jnp.eye(3, dtype=box.dtype) * box


def _inv3(a):
    """3x3 inverse in closed form (adjugate / det): elementwise ops XLA fuses, where
    jnp.linalg.inv is a solver call (on a GPU ~1 ms of fixed latency per MD step)."""
    c = jnp.stack([jnp.cross(a[1], a[2]), jnp.cross(a[2], a[0]), jnp.cross(a[0], a[1])], axis=1)
    return c / jnp.dot(a[0], c[:, 0])


class ShiftList(eqx.Module):
    """EXPERIMENTAL (needs matscipy-neighbours#10).  A fixed-capacity neighbour list built by matscipy-neighbours on the device
    (an XLA FFI call, inside jit) that carries each pair's periodic image shift S,
    so a cell may be smaller than 2 (rc + skin), triclinic or partly periodic --
    no minimum-image limit.  `update` rebuilds it under `lax.cond` when atoms or
    the cell have moved far enough, as LAMMPS's `neigh_modify every 1 check yes`.

    Positions may be wrapped back into the cell between builds (JAX-MD's shift_fn
    does): edge vectors are r_ij = (f_j - f_i + S') @ cell from the *current*
    fractional coordinates f and cell, with S' = S - w_j + w_i and w the integer
    wraps since the build (round(f - f_ref): true displacements are below half a
    cell).  So forces and the cell derivative (stress) are exact between builds.

    Rebuild when 2 max|u| + |inv(cell_ref) @ cell - 1|_F (rc + skin) > skin: u the
    unwrapped displacements since the build, the second term bounding how far a
    deformation of the cell (NPT) moves any listed pair.  Ks is the capacity per
    row; a row with more than Ks pairs within rc + skin sets `overflow`, which
    stays set and makes the potential NaN (never truncation): allocate again.

    Non-periodic directions: wraps are ignored along them (an atom cannot cross a
    boundary that is not periodic), so with `space.periodic_general`, which wraps
    every direction, a slab is correct only while no atom crosses the box along the
    non-periodic direction: give it enough vacuum, or use Cartesian positions and
    `space.free()`'s shift."""
    idx: jax.Array            # (N, Ks) int32 neighbour index (unspecified past count)
    shift: jax.Array          # (N, Ks, 3) int32 image shift S of the slot
    count: jax.Array          # (N,) int32 pairs within rc + skin (may exceed Ks)
    ref_frac: jax.Array       # (N, 3) fractional coordinates at the build
    ref_cell: jax.Array       # (3, 3) cell (rows) at the build
    box: jax.Array            # the JAX-MD box at the build (used when a call passes no box=)
    overflow: jax.Array       # () bool, sticky
    cutoff: float = eqx.field(static=True)        # rc: pairs the model reads
    skin: float = eqx.field(static=True)
    pbc: tuple = eqx.field(static=True)
    fractional: bool = eqx.field(static=True)     # R in fractional coordinates (periodic_general)
    plan: tuple | None = eqx.field(static=True, default=None)   # the device-only CUDA build's plan

    @property
    def did_buffer_overflow(self):
        return self.overflow

    def frac(self, R, cell):
        return R if self.fractional else R @ _inv3(cell)

    def wraps(self, f):
        """Integer wraps since the build, zero along non-periodic directions."""
        per = jnp.asarray(self.pbc, f.dtype)
        return jnp.round(jax.lax.stop_gradient(f) - self.ref_frac) * per

    def edges(self, R, box=None, perturbation=None):
        """rij (N, Ks, 3) = r_j - r_i + image, j (N, Ks), live (N, Ks) at positions R.
        `perturbation` as JAX-MD's displacement functions take it (what
        `quantity.stress` and the NPT integrators differentiate): a scalar, (3,) or
        (3, 3), applied as space.raw_transform(perturbation, rij)."""
        box = self.box if box is None else box
        cell = _cell(box)
        f = self.frac(R, cell)
        w = self.wraps(f)
        live = jnp.arange(self.idx.shape[1])[None, :] < self.count[:, None]
        j = jnp.where(live, self.idx, 0)
        S = self.shift.astype(f.dtype) - w[j] + w[:, None, :]
        if self.fractional:                          # JAX-MD's transform: gradients stay Cartesian
            rij = _jax_md("space").transform(box, f[j] - f[:, None, :] + S)
        else:
            rij = R[j] - R[:, None, :] + S @ cell
        if perturbation is not None:                 # scalar (NPT, isotropic), vector or matrix
            rij = _jax_md("space").raw_transform(perturbation, rij)
        return rij, j, live

    def usage(self, R, box=None):
        """How much of the skin positions R (and box) have used since the build:
        2 max|u| + |inv(cell_ref) @ cell - 1|_F (rc + skin).  The list holds every
        pair within rc while this is at most `skin`.  Pure XLA (no FFI)."""
        box = self.box if box is None else box
        cell = _cell(box)
        f = self.frac(R, cell)
        u = (f - self.ref_frac - self.wraps(f)) @ cell
        strain = jnp.linalg.norm(_inv3(self.ref_cell) @ cell - jnp.eye(3, dtype=cell.dtype))
        return 2 * jnp.sqrt(jnp.max(jnp.sum(u * u, axis=1), initial=0.0)) + strain * (self.cutoff + self.skin)

    def update(self, R, box=None, **kw):
        """The list for positions R (and box): itself, or rebuilt on the device.
        With the planned CUDA build (`plan` set) this belongs in every step: the
        rebuild check stays inside the step's CUDA graph.  With the regular build
        (CPU, or no plan) its conditional holds a foreign call that XLA does not put
        in a graph, so it reads the predicate on the host each call: drive the list
        with `run`, which checks once per block of steps."""
        box = self.box if box is None else box
        return jax.lax.cond(self.usage(R, box) > self.skin,
                            lambda: self._build(self.frac(R, _cell(box)), box), lambda: self)

    def rebuilt(self, R, box=None):
        """The list built afresh at positions R (and box)."""
        box = self.box if box is None else box
        return self._build(self.frac(R, _cell(box)), box)

    def _build(self, f, box):
        """Build on the device: matscipy-neighbours' FFI.  On CUDA with a `plan`, its
        planned build (enqueued on XLA's stream with no host synchronisation, and
        CUDA-graph compatible, so a rebuild inside `update`'s conditional stays on
        the device); otherwise the regular build (host round trips per build)."""
        xla = _xla()
        cell = _cell(box)
        cart = f @ cell                              # float32 or float64: the search runs in float64
        c64, rcs, K = cell.astype(cart.dtype), self.cutoff + self.skin, self.idx.shape[1]

        def regular(cart, c64):
            idx, shift, count, ov = xla.neighbour_matrix(cart, c64, rcs, self.pbc, K)
            return idx, shift, count, ov > 0

        def planned(cart, c64):
            idx, shift, count, st = xla.neighbour_matrix_planned(cart, c64, rcs, self.pbc, K, self.plan)
            return idx, shift, count, (st[0] > 0) | (st[1] > 0)

        if self.plan is None:
            idx, shift, count, bad = regular(cart, c64)
        else:
            idx, shift, count, bad = jax.lax.platform_dependent(cart, c64, cuda=planned, default=regular)
        return ShiftList(idx, shift, count, f, cell, jnp.asarray(box, self.box.dtype),
                         self.overflow | bad, self.cutoff, self.skin, self.pbc, self.fractional, self.plan)


class RunInfo(NamedTuple):
    rebuilds: jax.Array   # () int32: list builds at block starts
    redos: jax.Array      # () int32: blocks redone because an atom outran the skin inside them


def run(apply_fn, state, neighbor, n_steps, check_every=10, box_of=None, **apply_kw):
    """EXPERIMENTAL (a `ShiftList` needs matscipy-neighbours#10).
    (state, neighbor, RunInfo): `n_steps` steps of `apply_fn(state, neighbor=nbrs,
    **apply_kw)` (a JAX-MD integrator's apply) with a `ShiftList`, checking the list
    once per block of `check_every` steps instead of every step.  Jittable;
    n_steps a multiple of check_every (both static).

    Why: a conditional holding the regular FFI build is not a CUDA graph, so XLA
    reads its predicate on the host -- a host round trip per `update` call.  With
    the planned CUDA build (`neighbor_list(planned=None)`, the default where
    available) that does not happen and per-step `update` is the better choice
    (`run`'s block conditionals cost one host sync each).  Here the steps of
    a block contain no FFI call (only the pure-XLA `usage` check), so XLA keeps them
    in a CUDA graph, and the two conditionals run once per block.

    Exact, not LAMMPS's `neigh_modify every N` (which counts "dangerous builds"):
    each block starts by rebuilding when its usage plus the last block's growth
    (x1.5) would exceed the skin; within the block the worst usage is tracked, and
    if it exceeded the skin the block is redone from its start with a fresh list.
    A redone block that still exceeds it (an atom moved more than skin / 2 in
    check_every steps) sets the list's overflow flag: energies are then NaN; use
    a smaller check_every or a larger skin.

    `box_of(state)` gives the box for NPT (e.g. `simulate.npt_box`); None: the
    list's own box (NVE, NVT)."""
    N = int(check_every)
    if n_steps % N:
        raise ValueError(f"n_steps={n_steps} is not a multiple of check_every={N}")
    box = (lambda s: None) if box_of is None else box_of

    def inner(s, nb, used):
        def step(i, c):
            s, worst = c
            s = apply_fn(s, neighbor=nb, **apply_kw)
            return s, jnp.maximum(worst, nb.usage(s.position, box(s)))
        return jax.lax.fori_loop(0, N, step, (s, used))

    def block(c, _):
        s0, nb, growth, builds, redos = c
        used = nb.usage(s0.position, box(s0))
        pre = used + 1.5 * growth > nb.skin
        nb = jax.lax.cond(pre, lambda: nb.rebuilt(s0.position, box(s0)), lambda: nb)
        used0 = jnp.where(pre, nb.usage(s0.position, box(s0)), used)
        s1, worst = inner(s0, nb, used0)

        def redo():
            nb2 = nb.rebuilt(s0.position, box(s0))
            u2 = nb2.usage(s0.position, box(s0))
            s2, w2 = inner(s0, nb2, u2)
            return s2, eqx.tree_at(lambda b: b.overflow, nb2, nb2.overflow | (w2 > nb2.skin)), w2, u2
        bad = worst > nb.skin
        s1, nb, worst, start = jax.lax.cond(bad, redo, lambda: (s1, nb, worst, used0))
        # the next block's growth estimate: this block's, from the list it ended with
        return (s1, nb, jnp.maximum(worst - start, 0.0), builds + pre.astype(jnp.int32) + bad.astype(jnp.int32),
                redos + bad.astype(jnp.int32)), None

    z = jnp.zeros((), jnp.int32)
    (state, neighbor, _, builds, redos), _ = jax.lax.scan(
        block, (state, neighbor, jnp.zeros((), neighbor.ref_frac.dtype), z, z), None, length=n_steps // N)
    return state, neighbor, RunInfo(builds, redos)


class ShiftListFns(NamedTuple):
    allocate: Callable    # allocate(R, box=None) -> ShiftList (host-sized capacity; not jittable)
    update: Callable      # update(R, neighbor, box=None) -> ShiftList (jittable)


def neighbor_list(box, cutoff, skin=1.0, pbc=True, fractional_coordinates=False, capacity_multiplier=1.25,
                  planned=None):
    """EXPERIMENTAL: needs matscipy-neighbours built with its XLA FFI
    (libAtoms/matscipy-neighbours#10, not yet released); the API may change.
    `ShiftListFns` for a `ShiftList`: JAX-MD's `partition.neighbor_list` shape
    (`allocate`, then `nbrs.update(R)` inside jit), built by matscipy-neighbours'
    XLA FFI (`matscipy_neighbours.xla`, a matscipy-neighbours build with
    JAX_FFI_INCLUDE_DIR; CPU and CUDA) instead of JAX-MD's cell list.

    `box` as JAX-MD's (see `_cell`); `cutoff` the model's rc (`Potential.cutoff`);
    `skin` as JAX-MD's dr_threshold.  `pbc` a bool or three.  The list's capacity
    is the largest count within rc + skin at allocation times capacity_multiplier.
    float32 or float64 positions (the list keeps their dtype; the search itself
    runs in float64 inside matscipy-neighbours).

    `planned` (None: when available): on CUDA, build with matscipy-neighbours'
    experimental planned build (`xla.plan` at allocation fixes the cell grid and
    workspace): no host synchronisation, so `nbrs.update(R)` every step stays on
    the device (its conditional is part of the step's CUDA graph).  False: the
    regular build.  True: require it.  CPU always uses the regular build.  If it is
    not available on a GPU, the regular build is used and a RuntimeWarning says so
    (`nbrs.plan is None`): drive that list with `run`."""
    pbc = tuple(bool(b) for b in np.broadcast_to(pbc, 3))
    cutoff, skin = float(cutoff), float(skin)

    def allocate(R, box=box, extra_capacity=0):
        from .eval.nlist import _neighbour_list
        xla = _xla()                                 # fail early, with what to install
        cell = np.asarray(_cell(box), float)
        f = np.asarray(R, float) if fractional_coordinates else np.asarray(R, float) @ np.linalg.inv(cell)
        i = _neighbour_list(f @ cell, cell, pbc, cutoff + skin)[0]
        k = int(np.bincount(i, minlength=len(f)).max(initial=0))
        Ks = round_k(int(np.ceil(capacity_multiplier * k)) + extra_capacity)
        n = len(f)
        plan = None
        if planned is not False:
            try:
                plan = xla.plan(n, cell, cutoff + skin, pbc, Ks, dtype=jnp.asarray(R).dtype)
            except (ImportError, ValueError, RuntimeError, OSError) as e:
                if planned:                          # asked for explicitly: say why not
                    raise
                if jax.default_backend() == "gpu":
                    import warnings
                    warnings.warn(f"matscipy-neighbours' planned build is not available ({e}); using the "
                                  "regular build, which reads back to the host on each update: drive the "
                                  "list with jaxmd.run", RuntimeWarning, stacklevel=2)
        dt = jnp.asarray(R).dtype                    # the positions' dtype throughout
        empty = ShiftList(jnp.zeros((n, Ks), jnp.int32), jnp.zeros((n, Ks, 3), jnp.int32),
                          jnp.zeros((n,), jnp.int32), jnp.asarray(f, dt), jnp.asarray(cell, dt), jnp.asarray(box, dt),
                          jnp.zeros((), bool), cutoff, skin, pbc, bool(fractional_coordinates), plan)
        return jax.jit(lambda e, f, b: e._build(f, b))(empty, jnp.asarray(f, dt), jnp.asarray(box, dt))

    return ShiftListFns(allocate, lambda R, neighbor, box=None: neighbor.update(R, box))


class Potential(NamedTuple):
    energy: Callable      # energy(R, neighbor, species=None, **kw) -> E (eV)
    force: Callable       # force(R, neighbor, species=None, **kw) -> F (N, 3), Cartesian eV/A
    cutoff: float         # the model's rc: the r_cutoff of the JAX-MD list
    K: int | None         # compact slots per atom (None: the whole list)
    overflow: Callable    # overflow(R, neighbor, **kw) -> () bool: list overflow or a row over K


def potential(model, displacement_fn, species, meta=None, K=None, chunk=CHUNK_NODES, **calc_kw):
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
    when (energy and forces are then NaN, never truncated).  Other keywords go to
    ACECalculator, which builds the evaluation model.

    `energy` is JAX-MD's energy-function contract: give it to any integrator,
    `quantity.force` or `quantity.stress` (stress is JAX-MD's sign, minus ASE's).
    `force` is -grad(energy) with respect to the positions, as JAX-MD's
    `quantity.force(energy)`: Cartesian, also under
    `space.periodic_general(fractional_coordinates=True)` (JAX-MD's `transform`
    passes the gradient through), for the integrators that take a force
    function.  (An explicit assembly of dE/drij into forces measured no faster:
    the full reverse pass was 5-6% faster on an A100.)

    Supports partition.Dense, partition.Sparse (a full list; without K: K is
    refused for it), and a `ShiftList` (`neighbor_list`, experimental:
    matscipy-neighbours on the device, any cell; then `displacement_fn` is unused
    and may be None); OrderedSparse, a half list, is refused.  An overflowed list
    makes energy and forces NaN, whatever its format.  Remaining keywords of a call (`box=` for `space.periodic_general`)
    are passed to `displacement_fn`, or to the ShiftList's edge vectors."""
    if "cutoff" in calc_kw:   # the list is built with the model's rc (`cutoff`); a different one would not match
        raise ValueError("potential() uses the model's cutoff (jaxmd.cutoff(model)); do not pass cutoff=")
    m, meta, rc = eval_model(model, meta, **calc_kw)
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
        if isinstance(neighbor, ShiftList):
            rij_s, j_s, live_s = neighbor.edges(R, kw.get("box"), kw.get("perturbation"))
            bad0 = neighbor.overflow | jnp.any(neighbor.count > neighbor.idx.shape[1])
        else:
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
        """Dense or shift list -> (rij, j, mask) on the model's rows; NaN when the list
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
        return m.site_energies_dense_blocked(rij, zi, node_z[j], mask, node_z, chunk)

    def dense(neighbor):
        """True for a Dense JAX-MD list or a ShiftList, False for a (full) Sparse one."""
        if isinstance(neighbor, ShiftList):
            return True
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
    """K for `potential`: the largest in-cutoff neighbour count over a Dense (or
    `ShiftList`, then displacement_fn may be None) list at positions R, rounded up
    to `step` (`calc.skin.round_k`).  Add headroom (e.g. `round_k(1.2 * k)`) for
    liquids or compressing cells."""
    if isinstance(neighbor, ShiftList):
        rij, _, live = neighbor.edges(R, kw.get("box"), kw.get("perturbation"))
    else:
        if neighbor.idx.ndim != 2 or neighbor.idx.shape[0] != R.shape[0]:
            raise ValueError("capacity needs a partition.Dense list (or a ShiftList): K compacts Dense rows")
        rij, _, live = _edges_dense(R, neighbor.idx, displacement_fn, kw)
    cnt = jnp.sum(live & (jnp.sum(rij * rij, axis=-1) < cutoff * cutoff), axis=1)
    return round_k(int(jnp.max(cnt)), step)
