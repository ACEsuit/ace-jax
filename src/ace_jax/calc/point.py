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
from ..eval.nlist import dense_from_sparse, sparse_graph

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
                 layout="auto", **kw):
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
        the sparse layout."""
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
        # compiled entry points: run eagerly, a call dispatches thousands of ops
        # one by one (a flat ~0.6 s per call on an A100).  The model is an
        # argument, so each edge_a form is its own cache entry; n_nodes is static.
        import equinox as eqx
        self._efv_dense = eqx.filter_jit(
            lambda m, rij, zi, zj, idx, mask, nz: m.energy_forces_virial_dense(
                rij, zi, zj, idx, mask, nz))
        self._efv_sparse = eqx.filter_jit(
            lambda m, rij, zi, zj, s, r, n, nz: m.energy_forces_virial(rij, zi, zj, s, r, n, nz))

    def _species_index(self, numbers):
        try:
            return np.array([self._z2i[int(z)] for z in numbers], np.int32)
        except KeyError as e:
            raise ValueError(
                f"element Z={e.args[0]} not in model elements {self.meta['elements']}") from e

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        import jax.numpy as jnp

        g = sparse_graph(self.atoms.get_positions(), self.atoms.get_cell().array,
                         self.atoms.get_pbc(), self.cutoff)
        node_z = jnp.asarray(self._species_index(self.atoms.get_atomic_numbers()))
        rij = jnp.asarray(g.rij, dtype=self.dtype)
        send = jnp.asarray(g.senders)
        recv = jnp.asarray(g.receivers)
        layout = self._layout_for(g, rij.dtype)
        self.last_layout = layout
        if layout == "dense":
            d = dense_from_sparse(g, self.cutoff)
            idx = jnp.asarray(d.idx)
            with highest_precision():
                E, F, V = self._efv_dense(
                    self.model, jnp.asarray(d.rij, dtype=rij.dtype),
                    jnp.broadcast_to(node_z[:, None], idx.shape), node_z[idx], idx,
                    jnp.asarray(d.mask), node_z)
        else:
            model = self._model_for(rij, node_z[send], node_z[recv], send, g.n_nodes, node_z)
            with highest_precision():
                E, F, V = self._efv_sparse(
                    model, rij, node_z[send], node_z[recv], send, recv, int(g.n_nodes), node_z)
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

    def _layout_for(self, g, dtype):
        """"dense" or "sparse" for this neighbour list (see `layout`)."""
        if self.layout != "auto":
            return self.layout
        n, n_edges = g.n_nodes, len(g.senders)
        K = max(int(np.bincount(np.asarray(g.senders), minlength=n).max(initial=0)), 1)
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
