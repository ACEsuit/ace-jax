"""ASE calculator: energies, forces, stress and site descriptors.

Builds the neighbour list with matscipy-neighbours, evaluates the edge-vector
core, and returns energy, forces and stress.  Nothing here knows about cells
beyond handing them to the neighbour list -- the strain derivative acts on edge
vectors, so the same code path serves LAMMPS, where there is no cell at all.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..eval.edge_model import (LAYOUTS, calibrate_edge_a, check_edge_a_kind,
                               estimate_a_bytes, with_edge_a_kind)
from ..eval.model import highest_precision
from ..eval.nlist import backend as nlist_backend
from ..eval.nlist import dense_from_sparse, dense_graph, have_matscipy_neighbours, sparse_graph

# Below this many edges "auto" keeps the gather form: compile time dominates and
# the forms differ little.  Above it, the adjoint of the gather (an atomic
# scatter) and the one-hot matmul trade places by backend and dtype -- on an
# A4500 the matmul is 3x faster for f32 forces and 1.7x slower for f64.
AUTO_MIN_EDGES = 20_000

# layout="auto": the dense (n, K) layout -- A per node by a batched outer product,
# 3-13x faster forces on A4500/A100/H100 -- when its estimated memory fits the
# budget and the neighbour padding is efficient (edges / (n * K) >= MIN_DENSE_FILL);
# otherwise the sparse edge list, whose memory grows only with the edge count.
MIN_DENSE_FILL = 0.5
DENSE_BUDGET_FRACTION = 0.5
CPU_DENSE_BUDGET_BYTES = 4 * 2**30


def _edge_bucket(n_edges, floor=64):
    """Sparse edge-list length: the next power of two (at least `floor`), so the
    compiled shape changes only when the edge count crosses a doubling."""
    return max(floor, 1 << max(0, int(n_edges) - 1).bit_length())


def _round_k(k, step=4):
    """Dense row capacity: the largest neighbour count rounded up, so a small
    change in the count keeps the compiled shape (and the neighbour_matrix K)."""
    return max(step, -(-int(k) // step) * step)


def dense_budget_bytes():
    """Bytes the dense layout may use: a fraction of the accelerator's limit, or
    CPU_DENSE_BUDGET_BYTES where the backend reports none."""
    import jax
    stats = jax.devices()[0].memory_stats() or {}
    limit = stats.get("bytes_limit")
    return DENSE_BUDGET_FRACTION * limit if limit else CPU_DENSE_BUDGET_BYTES


class ACECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress",
                              "site_descriptors", "forces_std"]

    def __init__(self, model, meta=None, cutoff=None, dtype=None, edge_a_kind="auto",
                 layout="auto", posterior=None,
                 forces_std_every_call=False, **kw):
        """`ACECalculator("si_fitted.npz")` is the intended form: cutoff,
        species and dtype all come from the file.  A pre-loaded (model, meta)
        pair is still accepted, which is what the validation tests use.

        `edge_a_kind` picks the A-basis form (see `EdgeSiteModel.edge_a`):
        "gather", "matmul", or "auto" (default), which calibrates both on the
        actual neighbour list once per power-of-two edge-count bucket and keeps
        the faster, using the gather below `AUTO_MIN_EDGES` edges.

        `layout` picks the neighbour layout: "sparse" (edge list), "dense" (padded
        (n, K) per-node blocks; A by a batched outer product), or "auto" (default):
        dense when `estimate_a_bytes` fits `dense_budget_bytes()` and the padding
        is efficient (see MIN_DENSE_FILL), else sparse.  `edge_a_kind` applies to
        the sparse layout.

        `posterior` (a `posterior.npz` from `fit --uq ard`, with `model` the matching
        `model.npz` FILE) adds `results["forces_std"]`: the per-atom force std, shape (N,).
        By default (posteriors fitted with `--ard-variance sandwich`) it is lambda times the
        configuration-clustered sandwich std, lambda * sqrt(sum_c phi_c A^-1 M A^-1 phi_c^T);
        for `--ard-variance kappa` or a schema-1 posterior.npz it is the tempered epistemic
        std, kappa * sqrt(sum_c phi_c A^-1 phi_c^T).  By default it is computed only
        when requested (`calc.get_property("forces_std", atoms)`, which reuses the cached
        E/F/stress): a design-row rebuild plus an L^2 solve per step is not a silent MD cost.
        `forces_std_every_call=True` adds it to every calculation.  posterior= requires
        `jax.config.update("jax_enable_x64", True)` (RuntimeError otherwise).

        Memory: forces_std builds the force design rows of the WHOLE cell, about N*3*L*8 bytes
        (N atoms padded, L = (n_B + n_pair) * NZ columns) -- 7 GB for N = 100k at L = 3k -- on
        the JAX device, besides the posterior's L^2 factor.  The edge Jacobian is node-chunked
        (`linear_rows_chunked`), the rows themselves are not: size cells to fit them."""
        model_path = model
        if posterior is not None:
            import jax
            if not jax.config.jax_enable_x64:
                # A^-1 at cond(S) ~ 1e13 is meaningless in float32; enabling x64 here would silently
                # change every other JAX computation in the process
                raise RuntimeError("posterior= (forces_std) needs float64: call "
                                   "jax.config.update('jax_enable_x64', True) before creating the calculator")
        if edge_a_kind != "auto":
            check_edge_a_kind(edge_a_kind)
        if layout != "auto" and layout not in LAYOUTS:
            raise ValueError(f"layout must be 'auto' or one of {LAYOUTS}, got {layout!r}")
        super().__init__(**kw)
        from ..eval.api import _resolve
        model, meta = _resolve(model, meta, dtype)
        self.model = model
        self.meta = meta
        self.edge_a_kind = edge_a_kind
        self.last_edge_a_kind = None
        self.layout = layout
        self.last_layout = None
        self._by_kind = {}                    # form -> model in that form
        self._by_bucket = {}                  # edge bucket -> calibrated form
        self.cutoff = float(cutoff if cutoff is not None else meta["rcut"])
        self._z2i = {int(z): i for i, z in enumerate(meta["elements"])}
        self.dtype = dtype
        self.last_timing = None
        self._k_hint = None                   # dense row capacity, learnt from the first call
        self._nl_gpu = True                   # build the dense graph on the GPU when possible
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
        self.posterior = None
        self.forces_std_every_call = bool(forces_std_every_call)
        if posterior is not None:
            from ..eval import load as _load_fit_model
            from ..fit.ard import ARDPosterior
            from ..fit.inducing import GPConfig
            if not isinstance(model_path, (str, bytes)) and not hasattr(model_path, "__fspath__"):
                raise ValueError("posterior= needs the model FILE path (the design rows use the fit model)")
            post = ARDPosterior.load(posterior)
            NZ = len(meta["elements"])
            L = (meta["n_B"] + meta["n_pair"]) * NZ
            if (len(post.mean) != L or post.meta.get("n_B") != meta["n_B"]
                    or post.meta.get("NZ") != NZ):
                raise ValueError(f"posterior {posterior} does not match the model: "
                                 f"basis {len(post.mean)} vs {L}")
            p_els = [int(e) for e in post.meta.get("elements", [])]
            m_els = [int(e) for e in meta["elements"]]
            if p_els != m_els:                           # order-sensitive: columns are per species index
                raise ValueError(f"posterior {posterior} does not match the model: "
                                 f"elements {p_els} vs {m_els}")
            # the posterior must belong to THIS model's readout: its mean is the fit's coefficients in
            # the `rows._place` layout (species-major WB blocks, then Wpair blocks), bit-identical when
            # one run wrote both files
            z = np.load(model_path)
            if "WB" not in z.files or "Wpair" not in z.files:
                raise ValueError(f"posterior= needs the model.npz written by the same fit; {model_path} "
                                 f"has no WB/Wpair readout")
            coef = np.concatenate([np.asarray(z["WB"]).T.ravel(), np.asarray(z["Wpair"]).T.ravel()])
            mean = np.asarray(post.mean, np.float64)
            if coef.shape != mean.shape or not np.allclose(
                    coef, mean, rtol=1e-10, atol=1e-14 * max(float(np.abs(mean).max(initial=0.0)), 1e-300)):
                raise ValueError(f"posterior {posterior} does not match the model: its mean is not the "
                                 f"model's coefficients (a posterior from a different fit?)")
            # the L x L Cholesky factor and the sandwich factor: host->device once, not on every call
            import jax.numpy as jnp
            post = post._replace(chol=jnp.asarray(post.chol, jnp.float64))
            if post.Q is not None:
                post = post._replace(Q=jnp.asarray(post.Q, jnp.float64))
            self.posterior = post
            self._fit_model = _load_fit_model(model_path)[0]
            self._fit_cfg = GPConfig(r0=1.0, rcut=float(meta["rcut"]), n_B=meta["n_B"],
                                     n_pair=meta["n_pair"], NZ=NZ, C=1)

    def _species_index(self, numbers):
        try:
            return np.array([self._z2i[int(z)] for z in numbers], np.int32)
        except KeyError as e:
            raise ValueError(
                f"element Z={e.args[0]} not in model elements {self.meta['elements']}") from e

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if (self.posterior is not None and not system_changes and "forces" in self.results
                and set(properties) <= {"forces_std"}):
            super().calculate(atoms, properties, system_changes)   # E/F/stress cached: std only
            self.results["forces_std"] = self._forces_std()
            return
        super().calculate(atoms, properties, system_changes)
        import time

        import jax
        import jax.numpy as jnp

        t0 = time.perf_counter()
        pos, cell, pbc = (self.atoms.get_positions(), self.atoms.get_cell().array,
                          self.atoms.get_pbc())
        n = len(pos)
        node_z = jnp.asarray(self._species_index(self.atoms.get_atomic_numbers()))
        dtype = np.dtype(self.dtype or jnp.zeros(()).dtype)
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
                E, F, V = jax.block_until_ready(self._efv_dense(self.model, *args))
        else:
            # pad the edge list to a power-of-two bucket: MD changes the edge
            # count every few steps, and each new length was a new compiled shape
            n_e = len(g.senders)
            bucket = _edge_bucket(n_e)
            park = np.array([float(self.model.pad_cutoff()), 0.0, 0.0])
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
        self.last_timing = {"nlist_s": t1 - t0, "model_s": t2 - t1, "nlist_backend": backend}
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
        if self.posterior is not None and (self.forces_std_every_call or "forces_std" in properties):
            self.results["forces_std"] = self._forces_std()

    def _forces_std(self):
        """Tempered ARD per-atom force std (posterior.npz), from node-chunked design rows."""
        import jax

        from ..fit.data import Config, build_dataset
        from ..fit.rows import chunked_rows_fn
        at = self.atoms
        c = Config(at.get_positions(), at.get_atomic_numbers(), at.get_cell().array, at.get_pbc(),
                   None, None, None, 1.0, 1.0, 1.0)
        ds = build_dataset([c], self.meta, np.zeros(len(self.meta["elements"])), 1)
        b = jax.tree.map(lambda a: a[0], ds)
        if b.nbr.shape[1] == 0 or not bool(np.asarray(b.nbr_mask).any()):
            return np.zeros(len(at))                     # no neighbours: forces are identically zero
        # F stays on the device (one copy of the Ncap*3*L rows): sigma of every padded node (the
        # padding rows are zero), then the mask on the (Ncap,) result -- no host copies of F
        with highest_precision():
            if getattr(self, "_rows_fn", None) is None:        # compiled once per cell shape (MD)
                self._rows_fn = chunked_rows_fn(self._fit_model, self._fit_cfg)
            F = self._rows_fn(b).F
            s = self.posterior.forces_std(F)
        return s[np.asarray(b.node_mask)]

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
        if n_edges / (n * K) < MIN_DENSE_FILL:
            return "sparse"
        need = estimate_a_bytes(self.model, "dense", n, n_edges, K, np.dtype(dtype).itemsize)
        return "dense" if need <= dense_budget_bytes() else "sparse"

    def _in_form(self, kind):
        if kind not in self._by_kind:
            self._by_kind[kind] = with_edge_a_kind(self.model, kind)
        return self._by_kind[kind]

    def _model_for(self, rij, zi, zj, send, n_nodes, node_z):
        """The model in the A-basis form to use for this neighbour list."""
        n_edges = int(rij.shape[0])
        if self.edge_a_kind != "auto":
            kind = self.edge_a_kind
        elif n_edges < AUTO_MIN_EDGES:
            kind = "gather"
        else:
            bucket = 1 << max(n_edges - 1, 0).bit_length()
            if bucket not in self._by_bucket:
                with highest_precision():
                    best, _ = calibrate_edge_a(self.model, rij, zi, zj, send, n_nodes,
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
