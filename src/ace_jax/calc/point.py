"""ASE calculator: energies, forces, stress and site descriptors.

Builds the neighbour list with matscipy-neighbours, evaluates the edge-vector
core, and returns energy, forces and stress.  Nothing here knows about cells
beyond handing them to the neighbour list -- the strain derivative acts on edge
vectors, so the same code path serves LAMMPS, where there is no cell at all.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..eval.edge_model import calibrate_edge_a, check_edge_a_kind, with_edge_a_kind
from ..eval.model import highest_precision
from ..eval.nlist import sparse_graph

# Below this many edges "auto" keeps the gather form: compile time dominates and
# the forms differ little.  Above it, the adjoint of the gather (an atomic
# scatter) and the one-hot matmul trade places by backend and dtype -- on an
# A4500 the matmul is 3x faster for f32 forces and 1.7x slower for f64.
AUTO_MIN_EDGES = 20_000


class ACECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress",
                              "site_descriptors"]

    def __init__(self, model, meta=None, cutoff=None, dtype=None, edge_a_kind="auto", **kw):
        """`ACECalculator("si_fitted.npz")` is the intended form: cutoff,
        species and dtype all come from the file.  A pre-loaded (model, meta)
        pair is still accepted, which is what the validation tests use.

        `edge_a_kind` picks the A-basis form (see `EdgeSiteModel.edge_a`):
        "gather", "matmul", or "auto" (default), which calibrates both on the
        actual neighbour list once per power-of-two edge-count bucket and keeps
        the faster, using the gather below `AUTO_MIN_EDGES` edges."""
        if edge_a_kind != "auto":
            check_edge_a_kind(edge_a_kind)
        super().__init__(**kw)
        from ..eval.api import _resolve
        model, meta = _resolve(model, meta, dtype)
        self.model = model
        self.meta = meta
        self.edge_a_kind = edge_a_kind
        self.last_edge_a_kind = None
        self._by_kind = {}                    # form -> model in that form
        self._by_bucket = {}                  # edge bucket -> calibrated form
        self.cutoff = float(cutoff if cutoff is not None else meta["rcut"])
        self._z2i = {int(z): i for i, z in enumerate(meta["elements"])}
        self.dtype = dtype

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
        model = self._model_for(rij, node_z[send], node_z[recv], send, g.n_nodes, node_z)
        with highest_precision():
            E, F, V = model.energy_forces_virial(
                rij, node_z[send], node_z[recv], send, recv, g.n_nodes, node_z)
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
