"""ASE calculator: energies, forces, stress and site descriptors.

Builds the neighbour list with matscipy-neighbours, evaluates the edge-vector
core, and returns energy, forces and stress.  Nothing here knows about cells
beyond handing them to the neighbour list -- the strain derivative acts on edge
vectors, so the same code path serves LAMMPS, where there is no cell at all.
"""

import math

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..eval.edge_model import LAYOUTS, calibrate_edge_a, check_edge_a_kind, with_edge_a_kind
from ..eval.model import highest_precision
from ..eval.model import lean as lean_form
from ..eval.model import splining, with_radial_table
from ..eval.splinify import AUTO
from ..eval.nlist import backend as nlist_backend
from ..eval.nlist import dense_from_sparse, dense_graph, have_matscipy_neighbours, sparse_graph
from . import skin as skin_list

ENERGY_REFERENCES = ("absolute", "E0")
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


POSTERIOR_PROPS = ("forces_std", "forces_cov", "forces_q", "forces_q_mahal", "forces_group",
                   "forces_support")


class ACECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress",
                              "site_descriptors", *POSTERIOR_PROPS]

    def __init__(self, model, meta=None, cutoff=None, dtype=None, edge_a_kind="auto",
                 layout="auto", skin=1.0, lean=True, spline_tol=AUTO,
                 spline_intervals=None, radial_table=None, posterior=None,
                 forces_std_every_call=False, energy_reference="absolute", **kw):
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
        work the energy never reads removed (docs/dev/ace-vs-pace-gap.md).
        spline_tol="auto" (default) first splines a learned analytic tensor
        radial (`radial_learned`, set by the radial learner) at 1e-10
        (`to_spline`).  That is not roundoff: energies agree with lean=False to
        up to ~1e-9 relative and forces to up to ~2.3e-8 of max|F| on the
        benchmark models (docs/dev/learned-radial-splining.md).  Other analytic
        models -- ACEpotentials `ace_model` exports, built bases -- stay
        exact; a float spline_tol (e.g. 1e-10) opts them in, None never
        splines.  spline_intervals pins the grid.  `calc.splined` (and `last_timing["spline_tol"]`) says
        what was splined, None when nothing was.  The spline is cached on the
        radial's content, and its grid is bucketed, so radial swaps usually
        keep the compiled step.  It is
        `eval_model`; `model` stays the model as given, and descriptors use it.
        A PACE or unfolded model is evaluated as given either way.  Setting
        `calc.model` recomputes the lean form on the host (a device-to-host copy
        of the model's arrays): negligible for MD, but a per-step cost if the
        model is swapped every step.

        `radial_table` (None, the default: off; True: 4000 intervals; an int:
        that many) tabulates the evaluation model's radial stage in r
        (`eval.model.with_radial_table`, after `lean`; ACE: R_nl with its
        transform and envelope, and the pair radial; PACE: g_k), one cubic
        B-spline per species pair on [0.5 A, rcut], exactly zero beyond each
        pair's cutoff.  A CPU speed-up (1.1-1.4x on one core), and an
        approximation: at 4000 intervals energies agree to ~1e-11 relative and
        forces to ~1e-8 of max|F| (tests/test_radial_table.py).  Applies with
        lean=False too (to the model as given).  `calc.radial_table` is the
        table's info (n_intervals, r_min, r_max, max_rel_err,
        max_rel_deriv_err), None when off; `last_timing["radial_table"]` its
        n_intervals.  Cached on the radial's content, so a readout-only
        `calc.model` swap does not rebuild it.

        `posterior` (a `posterior.npz` from `fit --uq ard`, with `model` the matching
        `model.npz` FILE) adds the per-atom force uncertainty.  For a revision-2 (schema-3)
        posterior, each atom has a 3x3 shape V -- the centred delete-one-cluster (PRESS)
        jackknife covariance of its force (`--ard-variance kappa`: phi A^-1 phi^T) -- and a
        Mondrian group g that selects the calibrated scales: `forces_std` = lam_rms[g] sqrt(tr V),
        (N,); `forces_cov` = lam_rms[g]^2 V, (N, 3, 3); `forces_q`, the conformal radius at the
        fit's coverage (q[g] sqrt(tr V / 3) iso, the ellipsoid's largest semi-axis aniso);
        `forces_q_mahal` = q[g] (aniso); `forces_group` = g; and `forces_support` (on request).
        Schema-1/2 posteriors serve only the scalar `forces_std` (kappa times the epistemic std,
        or lambda times an uncentred configuration-clustered sandwich std).  Energy and stress
        carry no uncertainty.  By default the uncertainty is computed only when requested
        (`calc.get_property("forces_std", atoms)` or any of the above, which reuses the cached
        E/F/stress): a design-row rebuild plus an L^2 solve per step is not a silent MD cost.
        `forces_std_every_call=True` adds it to every calculation.  posterior= requires
        `jax.config.update("jax_enable_x64", True)` (RuntimeError otherwise).

        `energy_reference` is "absolute" (default: the model's energy, isolated-atom
        energies E0 included) or "E0": the energy relative to the isolated atoms,
        sum_i (E_i - E0[z_i]).  With "E0" the per-atom constant (~-160 eV for Si) is
        never added to the site energies, so the total of a large cell stays small and
        keeps resolution: an energy-based line search (ASE PreconLBFGS's Armijo test)
        otherwise stops resolving energy decreases once they fall below the ulp of
        |E| ~ 160 eV x N, at fmax ~ 1e-4..1e-5 eV/A on 10^4-10^5 atoms
        (docs/dev/energy-sum-results.md).  Forces and stress are unchanged.
        `results["e0_offset"]` is sum_i E0[z_i] (math.fsum; 0.0 for "absolute"), so the
        absolute energy is `energy + e0_offset`.

        Memory: forces_std builds the force design rows of the WHOLE cell, about N*3*L*8 bytes
        (N atoms padded, L = (n_B + n_pair) * NZ columns) -- 7 GB for N = 100k at L = 3k -- on
        the JAX device, besides the posterior's L^2 factor.  The edge Jacobian is node-chunked
        (`linear_rows_chunked`), the rows themselves are not: size cells to fit them.  The rows come
        from the FULL `model` (never the energy-only `eval_model`), whatever `lean` is."""
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
        if not skin >= 0:
            raise ValueError(f"skin must be >= 0, got {skin!r}")
        if energy_reference not in ENERGY_REFERENCES:
            raise ValueError(f"energy_reference must be one of {ENERGY_REFERENCES}, "
                             f"got {energy_reference!r}")
        super().__init__(**kw)
        self._energy_reference = energy_reference   # fixed: the evaluation model is built for it
        from ..eval.api import _resolve
        model, meta = _resolve(model, meta, dtype)
        self._skin_jit = None                 # (static model part, cutoff, compiled step)
        self._lean = bool(lean)
        self._spline_tol = spline_tol
        self._spline_intervals = spline_intervals
        self._radial_table_opt = radial_table
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
            # a joint-E0 posterior (meta e0_cols) carries NZ E0 columns after the readout: the model
            # file folds their shift into E0, and they are zero on every force row (ARDPosterior pads)
            if (len(post.mean) - post.n_e0 != L or post.meta.get("n_B") != meta["n_B"]
                    or post.meta.get("NZ") != NZ):
                raise ValueError(f"posterior {posterior} does not match the model: "
                                 f"basis {len(post.mean) - post.n_e0} vs {L}")
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
            mean = np.asarray(post.mean, np.float64)[:len(post.mean) - post.n_e0]
            if coef.shape != mean.shape or not np.allclose(
                    coef, mean, rtol=1e-10, atol=1e-14 * max(float(np.abs(mean).max(initial=0.0)), 1e-300)):
                raise ValueError(f"posterior {posterior} does not match the model: its mean is not the "
                                 f"model's coefficients (a posterior from a different fit?)")
            # the L x L Cholesky factor and the sandwich factor: host->device once, not on every call
            import jax.numpy as jnp
            post = post._replace(chol=jnp.asarray(post.chol, jnp.float64))
            if post.Q is not None:
                post = post._replace(Q=jnp.asarray(post.Q, jnp.float64))
            if post.R is not None:
                post = post._replace(R=jnp.asarray(post.R, jnp.float64))
            self.posterior = post
            # design rows from the full model as given (calc.model; the lean eval_model is energy-only);
            # a non-float64 dtype would degrade sigma, so the fit model is then re-read in float64
            self._fit_model = (self.model if dtype is None or np.dtype(dtype) == np.float64
                               else _load_fit_model(model_path)[0])
            self._fit_cfg = GPConfig(r0=1.0, rcut=float(meta["rcut"]), n_B=meta["n_B"],
                                     n_pair=meta["n_pair"], NZ=NZ, C=1)

    @property
    def energy_reference(self):
        """"absolute" or "E0", fixed at construction (see __init__)."""
        return self._energy_reference

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        """A new model takes effect on the next call: what was derived from the
        old one (its edge_a forms, the skin list and the step bound to its
        weights) is dropped.  The compiled step is kept while the structure
        (the static part) is unchanged, so new weights do not retrace.  ASE's
        cached results are cleared too, as they are for a parameter change.

        Refused when a posterior is attached: the posterior, its design-row model and the device
        factors belong to the model FILE given at construction, so forces_std would silently
        describe the old model.  Build a new ACECalculator(model, posterior=...) instead."""
        if hasattr(self, "_model"):
            if getattr(self, "posterior", None) is not None:
                raise ValueError("cannot swap calc.model on a calculator with posterior=: forces_std "
                                 "would still describe the model it was built with; build a new "
                                 "ACECalculator(model_file, posterior=posterior_file) instead")
            self.reset()
        self._model = model
        self._eval_model = (lean_form(model, self._spline_tol, self._spline_intervals)
                            if self._lean else model)
        self._splined = splining(model, self._eval_model, self._spline_tol)
        self._eval_model, self._radial_table = with_radial_table(self._eval_model,
                                                                 self._radial_table_opt)
        self._e0 = np.asarray(model.E0, np.float64)
        if self._energy_reference == "E0":        # site energies without the isolated atoms
            import equinox as eqx
            import jax.numpy as jnp
            self._eval_model = eqx.tree_at(lambda m: m.E0, self._eval_model,
                                           jnp.zeros_like(self._eval_model.E0))
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
    def radial_table(self):
        """The radial table's info (`splinify.radial_table`), None when off."""
        return self._radial_table

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
        if (self.posterior is not None and not system_changes and "forces" in self.results
                and set(properties) <= set(POSTERIOR_PROPS)):
            super().calculate(atoms, properties, system_changes)   # E/F/stress cached: posterior only
            self.results.update(self._posterior_quantities(set(properties)))
            return
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
                            "spline_tol": self._splined["spline_tol"] if self._splined else None,
                            "radial_table": (self._radial_table["n_intervals"]
                                             if self._radial_table else None)}
        E = float(E)
        self.results["energy"] = E
        self.results["free_energy"] = E
        self.results["e0_offset"] = (
            math.fsum(self._e0[self._species_index(self.atoms.get_atomic_numbers())])
            if self._energy_reference == "E0" else 0.0)
        self.results["forces"] = np.asarray(F)
        vol = self.atoms.get_volume()
        if vol > 0:
            # ASE stress is -virial/volume, Voigt-ordered
            s = -np.asarray(V) / vol
            self.results["stress"] = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        if self.posterior is not None:
            want = {p for p in properties if p in POSTERIOR_PROPS}
            if self.forces_std_every_call:
                want.add("forces_std")
            if want:
                self.results.update(self._posterior_quantities(want))

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

    def _served_set(self):
        """The posterior properties that share one pass over the force rows and the shape V: forces_std,
        plus forces_cov / forces_q / forces_group (schema 3) and forces_q_mahal (aniso).  Any one of them
        requested computes and caches all of them (cleared with the other results on a system change)."""
        post = self.posterior
        out = {"forces_std"}
        if post.group_table is not None:
            out |= {"forces_cov", "forces_q", "forces_group"}
            if post.force_shape == "aniso":
                out.add("forces_q_mahal")
        return out

    def _posterior_quantities(self, which):
        """Posterior-served per-atom properties as a dict: the requested ones (subset of POSTERIOR_PROPS)
        together with every other property of `_served_set` (one rows + shape pass serves them all):
        forces_std (N,), forces_cov (N,3,3), forces_q (N,), forces_q_mahal (N,; aniso only),
        forces_group (N,) int; forces_support (dict) only when requested (it is the expensive one).
        Calibrated ARD std (lam x the shape, or kappa x the posterior std for --ard-variance kappa /
        schema 1) from node-chunked design rows of the full fit model."""

        from ..fit.ard import _NEED3
        from ..fit.rows import chunked_rows_fn
        post = self.posterior
        which = set(which)
        if which & {"forces_cov", "forces_q", "forces_q_mahal", "forces_group"} and post.group_table is None:
            raise ValueError(_NEED3)
        if "forces_q_mahal" in which and post.force_shape != "aniso":
            from ase.calculators.calculator import PropertyNotImplementedError
            raise PropertyNotImplementedError("forces_q_mahal needs an aniso posterior (--uq ard "
                                              "--force-shape aniso)")
        if "forces_support" in which and post.support is None:
            raise ValueError("this posterior has no support reference (fitted with --no-ard-support, or "
                             "predates it): refit with --uq ard without --no-ard-support")
        shared = self._served_set() if which - {"forces_support"} else set()
        at = self.atoms
        n = len(at)
        ds, b, live = self._one_config_dataset(at)
        groups = (np.asarray(post.groups_of(b))[live].astype(np.int64)
                  if post.group_consts is not None else None)
        out = {}
        if "forces_support" in which:
            out["forces_support"] = self._support(ds, live, b)
        if not shared:
            return out
        if b.nbr.shape[1] == 0 or not bool(np.asarray(b.nbr_mask).any()):
            # no neighbours: forces are identically zero, so is every spread
            F = np.zeros((n, 3, len(np.asarray(post.mean))))
        else:
            with highest_precision():
                if getattr(self, "_rows_fn", None) is None:    # compiled once per cell shape (MD)
                    self._rows_fn = chunked_rows_fn(self._fit_model, self._fit_cfg)
                # padding rows are zero; F[live] drops the padded nodes
                F = self._rows_fn(b).F
                if not bool(np.all(live)):
                    F = F[live]
        if post.group_table is not None:
            out.update({k: np.asarray(v) for k, v in
                        post.served(F, groups, shared - {"forces_group"}).items()})
            out["forces_group"] = groups
        else:
            out["forces_std"] = np.asarray(post.forces_std(F, groups))
        return out

    def _support(self, ds, live, b):
        """Covariate-shift support of each atom against the posterior's reference (fit/support.py), from
        the same site_features descriptors the fit stage used (they depend on the model only)."""
        from ..fit.support import support_check
        post = self.posterior
        alpha = float(np.asarray(post.group_table["alpha"]))
        return support_check(post.support, self._site_X(ds, b), np.asarray(b.node_z)[live], 1 - alpha)

    def _one_config_dataset(self, at):
        """(ds, b, live): the one-config dataset of atoms, its batch, and the live-node mask."""
        import jax

        from ..fit.data import Config, build_dataset
        c = Config(at.get_positions(), at.get_atomic_numbers(), at.get_cell().array, at.get_pbc(),
                   None, None, None, 1.0, 1.0, 1.0)
        ds = build_dataset([c], self.meta, np.zeros(len(self.meta["elements"])), 1)
        b = jax.tree.map(lambda a: a[0], ds)
        return ds, b, np.asarray(b.node_mask)

    def _site_X(self, ds, b):
        """Live-atom site descriptors (the ones the support reference uses) of one-config dataset ds."""
        from ..fit.inducing import site_features
        live = np.asarray(b.node_mask)
        with highest_precision():
            return np.asarray(site_features(self._fit_model, self._fit_cfg, ds)[0])[0][live]

    def support_descriptors(self, atoms):
        """(X, Z) site descriptors and species indices of the live atoms; for `aj calibrate`."""
        ds, b, live = self._one_config_dataset(atoms)
        return self._site_X(ds, b), np.asarray(b.node_z)[live]

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
