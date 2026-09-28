"""ACECalculator with a jitted, fixed-shape evaluation for MD.

The plain calculator hands a different-length edge list to the model at every
step, so JAX re-traces each time (~0.8 s/step for 64 atoms). Here the edges are
padded up to a bucket (a multiple of `bucket`) with dummy self-edges on atom 0
at r = rcut + 1. Every tensor radial and pair envelope, together with its
slope, is exactly zero beyond the cutoff, so the padding adds nothing to E, F
or the virial. Each bucket compiles once.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ace_jax.calc.point import ACECalculator
from ace_jax.eval.model import highest_precision
from ace_jax.eval.nlist import sparse_graph


@partial(jax.jit, static_argnames=("n_nodes",))
def _efv(model, rij, zi, zj, send, recv, node_z, n_nodes):
    return model.energy_forces_virial(rij, zi, zj, send, recv, n_nodes, node_z)


class PaddedACECalculator(ACECalculator):
    def __init__(self, *args, bucket=1024, **kw):
        super().__init__(*args, **kw)
        self.bucket = int(bucket)

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        from ase.calculators.calculator import Calculator, all_changes
        Calculator.calculate(self, atoms, properties, system_changes or all_changes)
        g = sparse_graph(self.atoms.get_positions(), self.atoms.get_cell().array,
                         self.atoms.get_pbc(), self.cutoff)
        n_e = len(g.senders)
        cap = max(self.bucket, -(-n_e // self.bucket) * self.bucket)
        pad = cap - n_e
        rij = np.concatenate([g.rij, np.tile([[self.cutoff + 1.0, 0.0, 0.0]], (pad, 1))])
        send = np.concatenate([g.senders, np.zeros(pad, g.senders.dtype)])
        recv = np.concatenate([g.receivers, np.zeros(pad, g.receivers.dtype)])
        node_z = jnp.asarray(self._species_index(self.atoms.get_atomic_numbers()))
        send, recv = jnp.asarray(send), jnp.asarray(recv)
        with highest_precision():
            E, F, V = _efv(self.model, jnp.asarray(rij, dtype=self.dtype), node_z[send], node_z[recv],
                           send, recv, node_z, int(g.n_nodes))
        E = float(E)
        self.results["energy"] = E
        self.results["free_energy"] = E
        self.results["forces"] = np.asarray(F)
        vol = self.atoms.get_volume()
        if vol > 0:
            s = -np.asarray(V) / vol
            self.results["stress"] = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
