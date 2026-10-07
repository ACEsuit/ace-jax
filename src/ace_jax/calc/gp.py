"""ASE calculator over the fitted hybrid GP: mixture-mean E/F/stress and the
mixture standard deviations as `energy_std` / `forces_std`.  With posterior= (the
posterior.npz of `fit --uq ard-gp`) it serves the calibrated force UQ instead:
forces_std / forces_cov / forces_q / forces_q_mahal / forces_group, as ACECalculator does
for --uq ard, from the joint rows [B | k(B, B_M)] at the posterior's theta."""
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
    implemented_properties = ["energy", "forces", "stress", "energy_std", "forces_std",
                              "forces_cov", "forces_q", "forces_q_mahal", "forces_group"]

    @classmethod
    def from_file(cls, path, **kw):
        """A calculator from the gp_model.npz that `ace-jax fit` writes."""
        from ..fit.pipeline.export import load_gp_model
        return cls(*load_gp_model(path), **kw)

    def __init__(self, fitted, meta, deriv_dtc=True, posterior=None, **kw):
        """deriv_dtc=False drops the derivative-DTC term from forces_std (SoR-only force
        variance): cheaper, and needed when a big cell's (n, K, d, 3) arrays do not fit.
        posterior: the posterior.npz of `fit --uq ard-gp` that wrote this gp_model.npz (its mean must be the
        model's); the calibrated force UQ then replaces the mixture forces_std."""
        super().__init__(**kw)
        self.posterior = None
        if posterior is not None:
            self._attach_posterior(fitted, posterior)
        self.deriv_dtc = bool(deriv_dtc)
        self.fitted, self.meta = fitted, meta
        self._E0 = np.asarray(fitted.prob.model.E0)
        # one jitted predictor per calculator, reused across draws and calls (theta/mu/L
        # are arguments): run eagerly, the derivative-DTC dispatches op by op (~5x slower)
        # and the node-chunked rows' fori_loop would recompile per draw and per call
        # with a posterior the calibrated force UQ replaces the mixture forces_std: skip the derivative DTC
        # (its (n, K, d, 3) arrays are the big-cell memory cost); energy_std does not depend on it
        self._predict = _predict_fn(fitted.prob, True, self.deriv_dtc and self.posterior is None)

    def _attach_posterior(self, fitted, path):
        from ..fit.ard import ARDPosterior, rows_fn_for
        post = ARDPosterior.load(path)
        if post.prior_root.M == 0:
            raise ValueError(f"posterior {path} is a linear --uq ard posterior: serve it with "
                             "ACECalculator(model.npz, posterior=...)")
        if post.group_table is None:
            raise ValueError(f"posterior {path} has no schema-3 group table: refit with --uq ard-gp")
        mu = np.asarray(fitted.posteriors[0][0])
        mean = np.asarray(post.mean)
        if (post.prior_root.width != len(mu) or len(fitted.posteriors) != 1
                or not np.allclose(mean, mu, rtol=0.0, atol=1e-12 * max(np.abs(mu).max(), 1e-300))):
            raise ValueError(f"posterior {path} does not belong to this gp_model.npz: its mean is not the model's "
                             "(serve the posterior.npz written by the same fit --uq ard-gp)")
        self.posterior = post
        self._post_theta = from_array(jnp.asarray(post.gp_theta))
        self._post_rows = rows_fn_for(fitted.prob, self._post_theta)

    def _served(self, batch, live):
        """The calibrated force UQ of the live atoms of one batch: one rows + shape pass serves them all."""
        from ..eval import highest_precision
        post = self.posterior
        groups = np.asarray(post.groups_of(batch))[live].astype(np.int64)
        which = {"forces_std", "forces_cov", "forces_q"} | ({"forces_q_mahal"} if post.force_shape == "aniso" else set())
        with highest_precision():
            V = post.atom_shape(np.asarray(self._post_rows(batch).F)[live])
        out = {k: np.asarray(v) for k, v in post.served_from_V(V, groups, which).items()}
        out["forces_group"] = groups
        return out

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
        if all(L is not None for _, L in self.fitted.posteriors):   # an ard-gp gp_model.npz has no L
            self.results["energy_std"] = float(np.sqrt(Ev.mean() + Es.var()))
            self.results["forces_std"] = np.sqrt(Fv.mean(0) + Fs.var(0))
        if self.posterior is not None:       # the calibrated --uq ard-gp force UQ replaces the mixture std
            self.results.update(self._served(batch, live))
