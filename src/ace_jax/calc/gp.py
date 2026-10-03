"""ASE calculator over the fitted hybrid GP: mixture-mean E/F/stress and the
mixture standard deviations as `energy_std` / `forces_std`."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..eval import sparse_graph
from ..fit.data import Config, build_dataset
from ..fit.hypers import from_array
from ..fit.objective import posterior
from ..fit.predict import _e0_offset, _predict_fn
from ..fit.stats import sufficient_statistics


# Neighbour slots are padded to a quarter-octave bucket, round(2^(j/4)) (at least 8),
# not the exact maximum count, so structures (and MD steps) whose counts differ a little
# reuse one compiled predictor: <= 19% padding, ~4 buckets per doubling (multiples of 8
# gave 7 compiles over si_tiny's 39-76).  Padded slots sit at the cutoff, where the
# envelope vanishes: the result is unchanged.
SLOT_BUCKETS = tuple(sorted({int(round(2 ** (j / 4))) for j in range(12, 61)}))   # 8 .. 32768


def slot_bucket(k):
    """The padded neighbour-slot count for a maximum of k neighbours."""
    return next((b for b in SLOT_BUCKETS if b >= k), -(-k // SLOT_BUCKETS[-1]) * SLOT_BUCKETS[-1])


class FittedGP(NamedTuple):
    prob: object
    draws: np.ndarray
    posteriors: list


def fit_posteriors(prob, ds_train, draws):
    posts = []
    for d in np.asarray(draws):
        theta = from_array(jnp.asarray(d))
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds_train)
        posts.append(posterior(theta, st, prob))
    return FittedGP(prob, np.asarray(draws), posts)


class GPCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress", "energy_std", "forces_std"]

    @classmethod
    def from_file(cls, path, **kw):
        """A calculator from the gp_model.npz that `ace-jax fit` writes."""
        from ..fit.pipeline.export import load_gp_model
        return cls(*load_gp_model(path), **kw)

    def __init__(self, fitted, meta, deriv_dtc=True, **kw):
        """deriv_dtc=False drops the derivative-DTC term from forces_std (SoR-only force
        variance): cheaper, and needed when a big cell's (n, K, d, 3) arrays do not fit."""
        super().__init__(**kw)
        self.deriv_dtc = bool(deriv_dtc)
        self.fitted, self.meta = fitted, meta
        self._E0 = np.asarray(fitted.prob.model.E0)
        # one jitted predictor per calculator, reused across draws and calls (theta/mu/L
        # are arguments): run eagerly, the derivative-DTC dispatches op by op (~5x slower)
        # and the node-chunked rows' fori_loop would recompile per draw and per call
        self._predict = _predict_fn(fitted.prob, True, self.deriv_dtc)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        prob = self.fitted.prob
        cfg = prob.cfg
        c = Config(np.asarray(atoms.positions), np.asarray(atoms.numbers), np.asarray(atoms.cell.array),
                   np.asarray(atoms.pbc), None, None, None, 1.0, 1.0, 1.0)
        k = int(np.bincount(sparse_graph(c.positions, c.cell, c.pbc, cfg.rcut).senders,
                            minlength=len(c.numbers)).max())
        ds = build_dataset([c], self.meta, self._E0, 1, rcut=cfg.rcut, node_chunk=cfg.node_chunk,
                           k_cap=slot_bucket(max(k, 1)))
        batch = jax.tree.map(lambda a: a[0], ds)
        live = np.asarray(batch.node_mask)
        Es, Fs, Vs, Ev, Fv = [], [], [], [], []
        for d, (mu, L) in zip(self.fitted.draws, self.fitted.posteriors):
            theta = from_array(jnp.asarray(d))
            Em, Ev_, Fm, Fv_, Vm, _ = self._predict(theta, mu, L, batch)
            Es.append(float(Em[0]) + float(_e0_offset(prob, ds)[0, 0]))
            Ev.append(float(Ev_[0])); Fs.append(np.asarray(Fm)[live]); Fv.append(np.asarray(Fv_)[live])
            Vs.append(np.asarray(Vm)[0])
        Es, Ev, Fs, Fv, Vs = map(np.asarray, (Es, Ev, Fs, Fv, Vs))
        self.results["energy"] = float(Es.mean())
        self.results["forces"] = Fs.mean(0)
        self.results["stress"] = -Vs.mean(0) / atoms.get_volume()
        self.results["energy_std"] = float(np.sqrt(Ev.mean() + Es.var()))
        self.results["forces_std"] = np.sqrt(Fv.mean(0) + Fs.var(0))
