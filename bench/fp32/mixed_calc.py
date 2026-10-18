"""An experimental ASE calculator for "float32 compute, float64 accumulation" (OpenMM-style mixed precision,
suggested by the lammps-jax developer): the site basis B_i is evaluated in float32 (bond vectors formed in
float64, then cast, as ACECalculator does), the readout sum_k c_k B_ik and the sum over sites are accumulated in
float64.  Forces and stress by autodiff of that energy w.r.t. positions and a symmetric strain.

    mode "f64acc": float32 B, float64 readout and total     (the idea under test)
    mode "f32acc": float32 B, float32 readout and total     (the control: plain float32)
    mode "f64":    everything float64                       (the layout check against ACECalculator)

Single-species models only (the readout is WB[:, 0] | Wpair[:, 0], site_descriptors' column order).  Unfolded,
unlean model (load(fold=False)): the lean form folds the readout into the product basis, where there is no
separate readout to accumulate.  docs/dev/fp32-results.md.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402

from ace_jax.eval import load, with_dtype  # noqa: E402
from ace_jax.eval.nlist import sparse_graph  # noqa: E402


class MixedACE(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, path, mode="f64acc", **kw):
        super().__init__(**kw)
        m64, meta, z = load(path, fold=False)
        if len(meta["elements"]) != 1:
            raise ValueError("MixedACE: single-species models only")
        self.mode, self.rcut = mode, float(meta["rcut"])
        self.model = m64 if mode == "f64" else with_dtype(m64, jnp.float32)
        self.bdt = jnp.float64 if mode == "f64" else jnp.float32
        self.adt = jnp.float32 if mode == "f32acc" else jnp.float64        # accumulation dtype
        self.c = jnp.asarray(np.concatenate([np.asarray(z["WB"])[:, 0], np.asarray(z["Wpair"])[:, 0]]), self.adt)
        self.E0 = float(np.asarray(z["E0"]).ravel()[0])
        self._f = {}

    def _fn(self, n, cap):
        if (n, cap) not in self._f:
            model, bdt, adt, c, rcut = self.model, self.bdt, self.adt, self.c, self.rcut

            def energy(pos, eps, shifts, send, recv, mask):
                d = jnp.eye(3) + 0.5 * (eps + eps.T)
                p, s = pos @ d, shifts @ d
                rij = p[recv] - p[send] + s                                     # float64 bond vectors
                rij = jnp.where(mask[:, None], rij, jnp.array([0.5 * rcut, 0.0, 0.0]))
                zi = jnp.zeros(cap, jnp.int32)
                B = model.site_descriptors(rij.astype(bdt), zi, zi, send, n, jnp.zeros(n, jnp.int32), mask)
                site = B.astype(adt) @ c                                         # the readout, in adt
                return jnp.sum(site).astype(jnp.float64)                         # the total, in adt
            with jax.default_matmul_precision("highest"):
                self._f[(n, cap)] = jax.jit(jax.value_and_grad(energy, argnums=(0, 1)))
        return self._f[(n, cap)]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos, cell, pbc = self.atoms.get_positions(), self.atoms.get_cell().array, self.atoms.get_pbc()
        g = sparse_graph(pos, cell, pbc, self.rcut)
        n_e = len(g.senders)
        cap = 1 << max(n_e - 1, 0).bit_length()
        pad = lambda a, v: np.concatenate([np.asarray(a), np.full((cap - n_e,) + np.shape(a)[1:], v, np.asarray(a).dtype)])
        args = (jnp.asarray(pos), jnp.zeros((3, 3)), jnp.asarray(pad(g.shifts, 0.0)),
                jnp.asarray(pad(g.senders, 0)), jnp.asarray(pad(g.receivers, 0)), jnp.asarray(np.arange(cap) < n_e))
        e, (gpos, geps) = self._fn(len(pos), cap)(*args)
        E = float(e) + len(pos) * self.E0
        self.results["energy"] = self.results["free_energy"] = E
        self.results["forces"] = -np.asarray(gpos)
        vol = self.atoms.get_volume() if all(pbc) else 0.0
        if vol > 0:
            s = np.asarray(geps) / vol                                            # ASE stress: (1/V) dE/d(strain)
            self.results["stress"] = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
