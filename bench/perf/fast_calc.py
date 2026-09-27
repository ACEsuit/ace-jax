"""Prototype calculator path for MD-style repeated calls (docs/pace-performance-gap.md).

`SkinDenseCalculator` is `ACECalculator(layout="dense")` with what LAMMPS does
and the ASE calculator does not:

* a Verlet skin: the (n, K_skin) neighbour matrix is built for cutoff + skin and
  reused until an atom has moved more than skin / 2 since the build (LAMMPS:
  `neighbor 1.0 bin`, `neigh_modify every 1 delay 0 check yes`);
* each call does ONE host->device copy (positions) and ONE compiled call that
  forms rij = x[j] - x[i] + shift, compacts the skin list to the K slots inside
  the cutoff (ML-PACE's ComputeNeigh does the same), and evaluates E, F, V;
* species indices by a lookup table, not a Python loop over atoms.

    PYTHONPATH=bench:src python bench/perf/e2e.py pace_Cantor_medium Cantor 8192
"""
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import all_changes

from ace_jax.calc.point import ACECalculator, _round_k
from ace_jax.eval.model import highest_precision
from ace_jax.eval.nlist import dense_graph, have_matscipy_neighbours, sparse_graph


def _step(model, x, idx_s, shift_s, live_s, node_z, rc, K):
    """Skin list -> compacted (n, K) dense graph -> E, F, V; plus overflow flag."""
    n = x.shape[0]
    rij_s = x[idx_s] - x[:, None, :] + shift_s
    r2 = jnp.sum(rij_s * rij_s, axis=-1)
    valid = live_s & (r2 < rc * rc)
    # in-cutoff slots first, in their original order: output slot s takes the
    # (s+1)-th valid entry, found by a search in the row's running count (gathers
    # only; a per-row argsort was ~1 ms at 8k atoms on an A100)
    csum = jnp.cumsum(valid.astype(jnp.int32), axis=1)
    want = jnp.arange(1, K + 1, dtype=jnp.int32)
    order = jax.vmap(lambda c: jnp.searchsorted(c, want, side="left"))(csum)
    order = jnp.minimum(order, idx_s.shape[1] - 1)
    idx = jnp.take_along_axis(idx_s, order, axis=1)
    mask = jnp.take_along_axis(valid, order, axis=1)
    rij = jnp.take_along_axis(rij_s, order[..., None], axis=1)
    pad = jnp.asarray([1.0, 0.0, 0.0], x.dtype) * model.pad_cutoff()
    rij = jnp.where(mask[..., None], rij, pad)
    zi = jnp.broadcast_to(node_z[:, None], idx.shape)
    E, F, V = model.energy_forces_virial_dense(rij, zi, node_z[idx], idx, mask, node_z)
    overflow = jnp.any(jnp.sum(valid, axis=1) > K)
    return E, F, V, overflow, jnp.max(jnp.sum(valid, axis=1))


class SkinDenseCalculator(ACECalculator):
    def __init__(self, model, meta=None, skin=1.0, dtype=None, **kw):
        super().__init__(model, meta, dtype=dtype, layout="dense", edge_a_kind="gather", **kw)
        self.skin = float(skin)
        self.n_rebuilds = 0
        self._nl = None
        self._jstep = eqx.filter_jit(_step)
        lut = np.full(119, -1, np.int32)
        for z, i in self._z2i.items():
            lut[z] = i
        self._lut = lut

    def _need_rebuild(self, pos, cell, pbc, numbers):
        nl = self._nl
        if nl is None or not np.array_equal(nl["cell"], cell) or not np.array_equal(
                nl["pbc"], pbc) or not np.array_equal(nl["numbers"], numbers):
            return True
        d2 = np.max(np.sum((pos - nl["pos"]) ** 2, axis=1), initial=0.0)
        return d2 > (0.5 * self.skin) ** 2

    def _rebuild(self, pos, cell, pbc, numbers, dtype, K_cut=None):
        t0 = time.perf_counter()
        rcs = self.cutoff + self.skin
        Ks = (self._nl or {}).get("K_skin")
        if Ks is None:
            g = sparse_graph(pos, cell, pbc, rcs)
            Ks = _round_k(int(np.bincount(np.asarray(g.senders), minlength=len(pos)).max()))
        gpu = jax.default_backend() == "gpu" and have_matscipy_neighbours()
        dg = dense_graph(pos, cell, pbc, rcs, Ks, device="cuda" if gpu else None)
        z = self._lut[numbers]
        if (z < 0).any():
            raise ValueError(f"elements {sorted(set(numbers[z < 0]))} not in the model")
        node_z = jnp.asarray(z)
        idx = jnp.asarray(dg.idx, jnp.int32)
        count = jnp.asarray(dg.count, jnp.int32)
        live = jnp.arange(idx.shape[1])[None, :] < count[:, None]
        xp = jnp.asarray(pos, dtype)
        rij = jnp.asarray(dg.rij, dtype)
        shift = jnp.where(live[..., None], rij - (xp[idx] - xp[:, None, :]), 0.0)
        if K_cut is None:
            r2 = jnp.sum(rij * rij, -1)
            K_cut = _round_k(int(jnp.max(jnp.sum(live & (r2 < self.cutoff ** 2), 1))))
        self._nl = {"pos": pos.copy(), "cell": cell.copy(), "pbc": np.array(pbc),
                    "numbers": numbers.copy(), "idx": idx, "shift": shift, "live": live,
                    "node_z": node_z, "K_skin": Ks, "K": int(K_cut)}
        jax.block_until_ready(shift)
        self.n_rebuilds += 1
        self.last_rebuild_s = time.perf_counter() - t0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super(ACECalculator, self).calculate(atoms, properties, system_changes)
        t0 = time.perf_counter()
        pos, cell, pbc = (self.atoms.positions, self.atoms.cell.array, self.atoms.pbc)
        numbers = self.atoms.numbers
        dtype = np.dtype(self.dtype or jnp.zeros(()).dtype)
        rebuilt = self._need_rebuild(pos, cell, pbc, numbers)
        if rebuilt:
            self._rebuild(pos, cell, pbc, numbers, dtype)
        nl = self._nl
        x = jnp.asarray(pos, dtype)
        t1 = time.perf_counter()
        with highest_precision():
            E, F, V, ovf, kmax = self._jstep(self.model, x, nl["idx"], nl["shift"], nl["live"],
                                             nl["node_z"], self.cutoff, nl["K"])
            if bool(ovf):                  # an atom gained neighbours inside the cutoff
                self._rebuild(pos, cell, pbc, numbers, dtype, K_cut=_round_k(int(kmax) + 4))
                nl = self._nl
                E, F, V, ovf, kmax = self._jstep(self.model, x, nl["idx"], nl["shift"],
                                                 nl["live"], nl["node_z"], self.cutoff, nl["K"])
        E = float(E)
        F = np.asarray(F)
        V = np.asarray(V)
        t2 = time.perf_counter()
        self.last_timing = {"prep_s": t1 - t0, "model_and_d2h_s": t2 - t1, "rebuilt": rebuilt}
        self.results["energy"] = self.results["free_energy"] = E
        self.results["forces"] = F
        vol = self.atoms.get_volume()
        if vol > 0:
            s = -V / vol
            self.results["stress"] = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
