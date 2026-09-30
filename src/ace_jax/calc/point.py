"""ASE calculator: energies, forces, stress and site descriptors.

Builds the neighbour list with matscipy-neighbours, evaluates the edge-vector
core, and returns energy, forces and stress.  Nothing here knows about cells
beyond handing them to the neighbour list -- the strain derivative acts on edge
vectors, so the same code path serves LAMMPS, where there is no cell at all.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..eval.edge_model import LAYOUTS, calibrate_edge_a, check_edge_a_kind, with_edge_a_kind
from ..eval.model import highest_precision
from ..eval.model import lean as lean_form
from ..eval.model import splining
from ..eval.splinify import AUTO
from ..eval.nlist import backend as nlist_backend
from ..eval.nlist import dense_from_sparse, dense_graph, have_matscipy_neighbours, sparse_graph
from . import skin as skin_list
from .skin import round_k as _round_k

# Below this many edges "auto" keeps the gather form: compile time dominates and
# the forms differ little.  Above it, the adjoint of the gather (an atomic
# scatter) and the one-hot matmul trade places by backend and dtype -- on an
# A4500 the matmul is 3x faster for f32 forces and 1.7x slower for f64.
AUTO_MIN_EDGES = 20_000

# layout="auto": the dense (n, K) layout -- A per node by a batched outer product,
# 3-13x faster forces on A4500/A100/H100 -- when the neighbour padding is
# efficient (edges / (n * K) >= MIN_DENSE_FILL); otherwise the sparse edge list.
# Memory is not a criterion: the dense model runs in blocks of CHUNK_NODES rows,
# so its peak is bounded per block.  (dense_budget_bytes is for export_lammps,
# whose layout="auto" sizes one bundle block against it.)
MIN_DENSE_FILL = 0.5
DENSE_BUDGET_FRACTION = 0.5
CPU_DENSE_BUDGET_BYTES = 4 * 2**30


def _edge_bucket(n_edges, floor=64):
    """Sparse edge-list length: the next power of two (at least `floor`), so the
    compiled shape changes only when the edge count crosses a doubling."""
    return max(floor, 1 << max(0, int(n_edges) - 1).bit_length())


def dense_budget_bytes():
    """Bytes the dense layout may use: a fraction of the accelerator's limit, or
    CPU_DENSE_BUDGET_BYTES where the backend reports none."""
    import jax
    stats = jax.devices()[0].memory_stats() or {}
    limit = stats.get("bytes_limit")
    return DENSE_BUDGET_FRACTION * limit if limit else CPU_DENSE_BUDGET_BYTES


class ACECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress",
                              "site_descriptors"]

    def __init__(self, model, meta=None, cutoff=None, dtype=None, edge_a_kind="auto",
                 layout="auto", skin=1.0, lean=True, spline_tol=AUTO,
                 spline_intervals=None, **kw):
        """`ACECalculator("si_fitted.npz")` is the intended form: cutoff,
        species and dtype all come from the file.  A pre-loaded (model, meta)
        pair is still accepted, which is what the validation tests use.

        `edge_a_kind` picks the A-basis form (see `EdgeSiteModel.edge_a`):
        "gather", "matmul", or "auto" (default), which calibrates both on the
        actual neighbour list once per power-of-two edge-count bucket and keeps
        the faster, using the gather below `AUTO_MIN_EDGES` edges.  It does not
        apply to a pool-first model (PACE, `uses_edge_a` False), which is used
        as given and reports `last_edge_a_kind` None.

        `layout` picks the neighbour layout: "sparse" (edge list), "dense" (padded
        (n, K) per-node blocks; A by a batched outer product), or "auto" (default):
        dense when the padding is efficient (see MIN_DENSE_FILL), else sparse.
        `edge_a_kind` applies to the sparse layout.

        `skin` (A, default 1.0) is the Verlet skin of the dense layout (see
        `calc.skin`): the neighbour list is built for cutoff + skin and reused,
        one compiled step per call, until an atom has moved skin / 2 or the
        cell, pbc or species change.  skin=0 rebuilds the list every call.
        `last_timing["rebuilds"]` counts the calls that built a neighbour list,
        and `last_timing["nlist_s"]` is this call's build time (0 on reuse).

        `lean` (default True) evaluates energies, forces and stress with
        `ace_jax.eval.model.lean(model)`: exact to roundoff, with the per-edge
        work the energy never reads removed (docs/ace-vs-pace-gap.md).
        spline_tol="auto" (default) first splines a learned analytic tensor
        radial (`radial_learned`, set by the radial learner) at 1e-10
        (`to_spline`).  That is not roundoff: energies agree with lean=False to
        up to ~1e-9 relative and forces to up to ~2.3e-8 of max|F| on the
        benchmark models (docs/learned-radial-splining.md).  Other analytic
        models -- Julia `ace_model` exports, Python-authored models -- stay
        exact; a float spline_tol (e.g. 1e-10) opts them in, None never
        splines.  spline_intervals pins the grid.  `calc.splined` (and `last_timing["spline_tol"]`) says
        what was splined, None when nothing was.  The spline is cached on the
        radial's content, and its grid is bucketed, so radial swaps usually
        keep the compiled step.  It is
        `eval_model`; `model` stays the model as given, and descriptors use it.
        A PACE or unfolded model is evaluated as given either way.  Setting
        `calc.model` recomputes the lean form on the host (a device-to-host copy
        of the model's arrays): negligible for MD, but a per-step cost if the
        model is swapped every step."""
        if edge_a_kind != "auto":
            check_edge_a_kind(edge_a_kind)
        if layout != "auto" and layout not in LAYOUTS:
            raise ValueError(f"layout must be 'auto' or one of {LAYOUTS}, got {layout!r}")
        if not skin >= 0:
            raise ValueError(f"skin must be >= 0, got {skin!r}")
        super().__init__(**kw)
        from ..eval.api import _resolve
        model, meta = _resolve(model, meta, dtype)
        self._skin_jit = None                 # (static model part, cutoff, compiled step)
        self._lean = bool(lean)
        self._spline_tol = spline_tol
        self._spline_intervals = spline_intervals
        self.model = model                    # (the setter resets what derives from it)
        self.meta = meta
        self.edge_a_kind = edge_a_kind
        self.last_edge_a_kind = None
        self.layout = layout
        self.last_layout = None
        self.cutoff = float(cutoff if cutoff is not None else meta["rcut"])
        self._z2i = {int(z): i for i, z in enumerate(meta["elements"])}
        self.dtype = dtype
        self.last_timing = None
        self._k_hint = None                   # dense row capacity, learnt from the first call
        self._nl_gpu = True                   # build the dense graph on the GPU when possible
        self.skin = skin                      # (the setter drops any skin list)
        self._rebuilds = 0                    # calls that built a neighbour list
        self._lut = skin_list.species_lut(meta["elements"])
        # compiled entry points: run eagerly, a call dispatches thousands of ops
        # one by one (a flat ~0.6 s per call on an A100).  The model is an
        # argument, so each edge_a form is its own cache entry; n_nodes is static.
        import equinox as eqx
        self._efv_dense = eqx.filter_jit(
            lambda m, rij, zi, zj, idx, mask, nz: m.energy_forces_virial_dense(
                rij, zi, zj, idx, mask, nz))
        self._efv_sparse = eqx.filter_jit(
            lambda m, rij, zi, zj, s, r, n, nz, emask: m.energy_forces_virial(
                rij, zi, zj, s, r, n, nz, emask))

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        """A new model takes effect on the next call: what was derived from the
        old one (its edge_a forms, the skin list and the step bound to its
        weights) is dropped.  The compiled step is kept while the structure
        (the static part) is unchanged, so new weights do not retrace.  ASE's
        cached results are cleared too, as they are for a parameter change."""
        if hasattr(self, "_model"):
            self.reset()
        self._model = model
        self._eval_model = (lean_form(model, self._spline_tol, self._spline_intervals)
                            if self._lean else model)
        self._splined = splining(model, self._eval_model, self._spline_tol)
        self._by_kind = {}                    # form -> model in that form
        self._by_bucket = {}                  # edge bucket -> calibrated form
        self._skin_state = None               # the skin list in use (calc.skin.SkinState)
        self._skin_step = None                # f(u, arrays, K), bound to this model's weights

    @property
    def splined(self):
        """What the lean form splined, None when nothing: {"spline_tol",
        "radials", "n_intervals"} (`eval.model.splining`)."""
        return self._splined

    @property
    def eval_model(self):
        """The model energies, forces and stress are evaluated with: `lean(model)`
        (or `model` itself with lean=False).  Energy only: no descriptors."""
        return self._eval_model

    @property
    def skin(self):
        return self._skin

    @skin.setter
    def skin(self, skin):
        if not skin >= 0:
            raise ValueError(f"skin must be >= 0, got {skin!r}")
        self._skin = float(skin)
        self._skin_state = None               # built for the old skin (or none)

    def _step(self):
        """The skin step bound to the current model's weights, compiled once per
        (model structure, cutoff) and built on first use."""
        if self._skin_step is None:
            import equinox as eqx
            params, static = eqx.partition(self.eval_model, eqx.is_array)
            key = (static, self.cutoff)
            if self._skin_jit is None or not bool(self._skin_jit[:2] == key):
                self._skin_jit = (*key, skin_list.jitted_step(static, self.cutoff))
            f = self._skin_jit[2]
            self._skin_step = lambda u, arrays, K: f(params, u, arrays, K=K)
        return self._skin_step

    def _species_index(self, numbers):
        try:
            return np.array([self._z2i[int(z)] for z in numbers], np.int32)
        except KeyError as e:
            raise ValueError(
                f"element Z={e.args[0]} not in model elements {self.meta['elements']}") from e

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        import jax.numpy as jnp

        pos, cell, pbc = (self.atoms.get_positions(), self.atoms.get_cell().array,
                          self.atoms.get_pbc())
        dtype = np.dtype(self.dtype or jnp.zeros(()).dtype)
        out = None
        if self.skin > 0 and self.layout != "sparse":
            out = self._skin_calculate(pos, cell, pbc, self.atoms.get_atomic_numbers(), dtype)
        if out is None:
            out = self._rebuild_calculate(pos, cell, pbc, dtype)
        E, F, V, timing = out
        self.last_timing = {**timing, "rebuilds": self._rebuilds,
                            "spline_tol": self._splined["spline_tol"] if self._splined else None}
        E = float(E)
        self.results["energy"] = E
        self.results["free_energy"] = E
        self.results["forces"] = np.asarray(F)
        vol = self.atoms.get_volume()
        if vol > 0:
            # ASE stress is -virial/volume, Voigt-ordered
            s = -np.asarray(V) / vol
            self.results["stress"] = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])

    def _skin_calculate(self, pos, cell, pbc, numbers, dtype):
        """E, F, V from the skin list (built or reused), or None when
        layout="auto" prefers sparse for this structure."""
        import time

        import jax

        t_nl = 0.0
        st = self._skin_state
        rebuilt = not skin_list.valid(st, pos, cell, pbc, numbers, self.skin)
        if rebuilt:
            st, dt = self._skin_build(pos, cell, pbc, numbers, dtype, st)
            t_nl += dt
        if self.layout == "auto" and self._layout_from(
                len(pos), st.n_edges, max(st.k_max, 1), dtype) == "sparse":
            return None                        # the rebuild path counts this call
        n = len(pos)
        t0 = time.perf_counter()
        u = jax.device_put(st.displacements(pos, dtype))
        E, F, V, drift, overflow, k_max, n_edges = skin_list.unpack(
            self._step()(u, st.arrays, K=st.K), n)
        if drift or overflow:
            # an atom moved skin / 2 (valid() checked in float64; this is the
            # model dtype), or gained neighbours inside the cutoff past K
            t1 = time.perf_counter()
            K_hint = _round_k(k_max + 4) if overflow else st.K
            st, dt = self._skin_build(pos, cell, pbc, numbers, dtype, st, K_hint=K_hint)
            t_nl += dt
            rebuilt = True
            t0 += time.perf_counter() - t1
            u = jax.device_put(st.displacements(pos, dtype))       # zero: the list is new
            E, F, V, drift, overflow, k_max, n_edges = skin_list.unpack(
                self._step()(u, st.arrays, K=st.K), n)
            if drift or overflow:
                raise RuntimeError("skin list invalid straight after a rebuild")
        t_model = time.perf_counter() - t0
        self._rebuilds += rebuilt
        self.last_layout = "dense"
        self.last_n_edges = n_edges
        # nlist_s: skin list builds this call (0 on reuse); model_s: the compiled
        # step, with the displacements' H2D and the packed result's D2H
        return E, F, V, {"nlist_s": t_nl, "model_s": t_model, "nlist_backend": st.backend}

    def _skin_build(self, pos, cell, pbc, numbers, dtype, prev, K_hint=None):
        """A new SkinState (kept on self), reusing the capacities learnt by
        `prev` so the compiled shapes stay put; returns (state, seconds)."""
        import time

        import jax
        t0 = time.perf_counter()
        gpu = jax.default_backend() == "gpu" and self._nl_gpu and have_matscipy_neighbours()
        kw = dict(dtype=dtype, K_hint=max(K_hint or 0, prev.K if prev else 0))
        args = (pos, cell, pbc, numbers, self._lut, self.cutoff, self.skin,
                prev.K_skin if prev else None)
        try:
            st = skin_list.build(*args, device="cuda" if gpu else None, **kw)
        except Exception as e:                          # no CUDA backend in this build
            if not gpu or isinstance(e, ValueError):
                raise
            self._nl_gpu = False
            st = skin_list.build(*args, device=None, **kw)
        jax.block_until_ready(st.arrays)
        self._skin_state = st
        return st, time.perf_counter() - t0

    def _rebuild_calculate(self, pos, cell, pbc, dtype):
        """E, F, V with a neighbour list built for this call (skin=0, sparse)."""
        import time

        import jax
        import jax.numpy as jnp

        self._rebuilds += 1
        t0 = time.perf_counter()
        n = len(pos)
        node_z = jnp.asarray(self._species_index(self.atoms.get_atomic_numbers()))
        dg, g, backend = self._native_dense(pos, cell, pbc, n, dtype), None, "neighbour_matrix"
        if dg is None:                                  # sparse list (also learns K)
            g = sparse_graph(pos, cell, pbc, self.cutoff)
            backend = nlist_backend()
            layout = self._layout_for(g, dtype)
            if layout == "dense":
                counts = np.bincount(np.asarray(g.senders), minlength=n)
                self._k_hint = _round_k(int(counts.max(initial=0)))
                dg = dense_from_sparse(g, self.cutoff, max_neighbours=self._k_hint)
        layout = "dense" if dg is not None else "sparse"
        self.last_layout = layout
        if layout == "dense":
            self.last_n_edges = int(np.asarray(dg.count).sum())
            # one dtype whichever builder ran (regroup vs neighbour_matrix): no retrace
            idx = jnp.asarray(dg.idx, dtype=jnp.int32)
            count = jnp.asarray(dg.count, dtype=jnp.int32)
            args = (jnp.asarray(dg.rij, dtype=dtype), jnp.broadcast_to(node_z[:, None], idx.shape),
                    node_z[idx], idx, jnp.arange(idx.shape[1])[None, :] < count[:, None], node_z)
            jax.block_until_ready(args)
            t1 = time.perf_counter()
            with highest_precision():
                E, F, V = jax.block_until_ready(self._efv_dense(self.eval_model, *args))
        else:
            # pad the edge list to a power-of-two bucket: MD changes the edge
            # count every few steps, and each new length was a new compiled shape
            n_e = len(g.senders)
            bucket = _edge_bucket(n_e)
            park = np.array([float(self.eval_model.pad_cutoff()), 0.0, 0.0])
            rij = jnp.asarray(np.concatenate([np.asarray(g.rij), np.tile(park, (bucket - n_e, 1))]),
                              dtype=dtype)
            send = jnp.asarray(np.concatenate([np.asarray(g.senders), np.zeros(bucket - n_e, int)]),
                               dtype=jnp.int32)
            recv = jnp.asarray(np.concatenate([np.asarray(g.receivers), np.zeros(bucket - n_e, int)]),
                               dtype=jnp.int32)
            emask = jnp.arange(bucket) < n_e
            self.last_n_edges = n_e
            model = self._model_for(rij, node_z[send], node_z[recv], send, g.n_nodes, node_z)
            args = (rij, node_z[send], node_z[recv], send, recv)
            jax.block_until_ready(args)
            t1 = time.perf_counter()
            with highest_precision():
                E, F, V = jax.block_until_ready(
                    self._efv_sparse(model, *args, int(g.n_nodes), node_z, emask))
        t2 = time.perf_counter()
        # nlist_s: neighbour list + layout + host->device; model_s: the compiled call
        return E, F, V, {"nlist_s": t1 - t0, "model_s": t2 - t1, "nlist_backend": backend}

    def _native_dense(self, pos, cell, pbc, n, dtype):
        """The dense graph straight from matscipy_neighbours' neighbour_matrix,
        on the GPU when JAX runs there (zero-copy via DLPack), with the K learnt
        by an earlier call.  None when that path does not apply: no
        matscipy_neighbours, sparse layout, K not learnt yet, an atom outgrew K
        (the sparse list then re-learns it), or auto-layout now prefers sparse."""
        if self.layout == "sparse" or not self._k_hint or not have_matscipy_neighbours():
            return None
        import jax
        device = "cuda" if (jax.default_backend() == "gpu" and self._nl_gpu) else None
        try:
            dg = dense_graph(pos, cell, pbc, self.cutoff, self._k_hint, device=device)
        except ValueError:                              # capacity exceeded
            return None
        except Exception:                               # no CUDA backend in this build
            if device is None:
                raise
            self._nl_gpu = False
            return self._native_dense(pos, cell, pbc, n, dtype)
        if self.layout == "auto":
            count = np.asarray(dg.count)
            K = max(int(count.max(initial=0)), 1)
            if self._layout_from(n, int(count.sum()), K, dtype) == "sparse":
                return None
        return dg

    def _layout_for(self, g, dtype):
        """"dense" or "sparse" for this neighbour list (see `layout`)."""
        if self.layout != "auto":
            return self.layout
        n, n_edges = g.n_nodes, len(g.senders)
        K = max(int(np.bincount(np.asarray(g.senders), minlength=n).max(initial=0)), 1)
        return self._layout_from(n, n_edges, K, dtype)

    def _layout_from(self, n, n_edges, K, dtype):
        if self.layout != "auto":
            return self.layout
        return "sparse" if n_edges / (n * K) < MIN_DENSE_FILL else "dense"

    def _in_form(self, kind):
        if kind not in self._by_kind:
            self._by_kind[kind] = with_edge_a_kind(self.eval_model, kind)
        return self._by_kind[kind]

    def _model_for(self, rij, zi, zj, send, n_nodes, node_z):
        """The model in the A-basis form to use for this neighbour list."""
        if not self.eval_model.uses_edge_a:   # pool-first (PACE): the form is inert
            self.last_edge_a_kind = None
            return self.eval_model
        n_edges = int(rij.shape[0])
        if self.edge_a_kind != "auto":
            kind = self.edge_a_kind
        elif n_edges < AUTO_MIN_EDGES:
            kind = "gather"
        else:
            bucket = 1 << max(n_edges - 1, 0).bit_length()
            if bucket not in self._by_bucket:
                with highest_precision():
                    best, _ = calibrate_edge_a(self.eval_model, rij, zi, zj, send, n_nodes,
                                               node_z, reps=3)
                self._by_bucket[bucket] = best.edge_a_kind
                self._by_kind.setdefault(best.edge_a_kind, best)
            kind = self._by_bucket[bucket]
        self.last_edge_a_kind = kind
        return self._in_form(kind)

    def get_site_descriptors(self, atoms=None, domain=None):
        """Per-site descriptors for `atoms`, alongside energy/forces/stress.

        Computed on demand rather than in `calculate`, since a plain energy call
        should not pay for them."""
        from ..eval.api import site_descriptors
        at = atoms if atoms is not None else self.atoms
        return site_descriptors(self.model, at.get_positions(),
                                at.get_atomic_numbers(), at.get_cell().array,
                                at.get_pbc(), meta=self.meta, cutoff=self.cutoff,
                                dtype=self.dtype, domain=domain)
