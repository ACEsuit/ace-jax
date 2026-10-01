"""SPIKE: sandwich (misspecification-robust) force variance on the ARD posterior, bench365.

The ARD posterior N(c, A^-1) of the linear ACE model is tempered today by one scalar kappa^2.
Under misspecification the covariance of the fitted c is the sandwich A^-1 M A^-1 (Huber-White;
Mueller 2013 for its Bayesian use), M = sum_i rho_i^2 psi_i psi_i^T over the noise-weighted
training rows psi_i with residuals rho_i -- kappa^2 A^-1 is its homoscedastic special case, and
the covariance of POPS's per-point corrections dtheta_i = A^-1 psi_i rho_i / h_i is the same form
with weights rho_i^2 / h_i^2 (h_i the leverage).  Meat variants:
  hc0   rho^2                      hc3   rho^2 / (1 - h)^2          pops  rho^2 / max(h, h_floor)^2
  atom  per-atom clusters (sum over an atom's 3 force rows; E / V rows alone)
  cfg   per-config clusters (all rows of a config)
Per-atom force variance at x: sum_c u_c^T M u_c, u_c = A^-1 phi_c(x); A^-1 is rebuilt in float64
from the linear statistics and the run's ARD hyperparameters (posterior.npz is float32).

    python sandwich_spike.py <data_dir> <run_dir>      -> <run_dir>/sandwich.npz
"""
import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                                             # noqa: E402
import numpy as np                                                                 # noqa: E402
from ase.io import read                                                            # noqa: E402
from jax.scipy.linalg import cho_solve, solve_triangular                          # noqa: E402

from ace_jax.eval import highest_precision                                          # noqa: E402
from ace_jax.fit.ard import (ARDEvidence, ard_posterior, body_order_columns,      # noqa: E402
                             joint_ard_stats)
from ace_jax.fit.data import Config, build_dataset                                  # noqa: E402
from ace_jax.fit.pipeline import FitConfig, load_fit_data                           # noqa: E402
from ace_jax.fit.pipeline.problem import build_problem                              # noqa: E402
from ace_jax.fit.rows import chunked_rows_fn, linear_rows                           # noqa: E402
from ace_jax.fit.stats import linear_statistics                                     # noqa: E402

data, run = sys.argv[1], sys.argv[2]
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:7.0f} s]", *a, flush=True)
VARIANTS = ("hc0", "hc3", "pops", "atom", "cfg")

cfg = FitConfig(model=f"{data}/cantor_embed_d16_deg10.npz", energy_key="mace_energy", force_key="mace_force",
                virial_key="mace_virial", r0=2.5, batch=4, rungs=("map",), predict_train=False, e0="lsq",
                opt="lbfgs", map_steps=40, arm="linear", uq="ard", ard_mode="joint").validate()
d = load_fit_data(cfg, train=f"{data}/train.xyz", test=f"{data}/test.xyz", ood=f"{data}/ood.xyz")
b = build_problem(cfg, d)
prob = b.prob
L = prob.cfg.len_basis
ard = json.load(open(f"{run}/ard.json"))
h = np.asarray(ard["h"], float)
sig = np.exp(h[:3])                                                                 # sigma_E, sigma_F, sigma_V
log("data", len(d.train), "train configs; L =", L, "; sigma_q =", sig)

with highest_precision():
    st = linear_statistics(prob.model, prob.cfg, d.ds_train)
    log("linear statistics")
    ev = ARDEvidence(joint_ard_stats(st), np.asarray(prob.gamma), body_order_columns(d.meta, prob.cfg))
    del st
    post = ard_posterior(ev, h, 1.0, d.meta)
    del ev
    Lc = jnp.asarray(post.chol, jnp.float64)                                        # S = D^-1 A D^-1 = Lc Lc^T
    dinv = jnp.asarray(post.dinv)
    c = jnp.asarray(post.mean)
    ref = np.load(f"{run}/posterior.npz")
    log("posterior rebuilt; |mean - saved| / |saved| =",
        float(np.linalg.norm(post.mean - ref["mean"]) / np.linalg.norm(ref["mean"])))

    rows_fn = jax.jit(lambda bt: linear_rows(prob.model, prob.cfg, bt)[0])

    @jax.jit
    def train_rows(bt):
        """Noise-weighted, prior-scaled rows psi~ (n, L), residuals rho (n,), leverage h (n,),
        atom-cluster and config-cluster ids for every E / F / V row of a batch."""
        r = rows_fn(bt)
        C, Ncap = r.E.shape[0], r.F.shape[0]
        P = jnp.concatenate([r.E, r.F.reshape(-1, L), r.V.reshape(-1, L)])
        y = jnp.concatenate([bt.y_E, bt.y_F.reshape(-1), bt.y_V.reshape(-1)])
        w = jnp.concatenate([bt.w_E / sig[0], jnp.repeat(bt.w_F, 3) / sig[1], jnp.repeat(bt.w_V, 6) / sig[2]])
        psi = P * w[:, None] * dinv[None, :]
        rho = (y - P @ c) * w
        lev = jnp.sum(solve_triangular(Lc, psi.T, lower=True) ** 2, axis=0)
        atom = jnp.concatenate([Ncap + jnp.arange(C), jnp.repeat(jnp.arange(Ncap), 3),
                                Ncap + C + jnp.arange(6 * C)])                       # force rows share their node
        cfgid = jnp.concatenate([jnp.arange(C), jnp.repeat(bt.node_cfg, 3), jnp.repeat(jnp.arange(C), 6)])
        return psi, rho, lev, atom, cfgid

    batches = [jax.tree.map(lambda a, i=i: a[i], d.ds_train) for i in range(d.ds_train.n_batches)]
    levs, rhos = [], []
    for bt in batches:
        _, rho, lev, _, _ = train_rows(bt)
        live = np.asarray(rho) != 0
        levs.append(np.asarray(lev)[live]); rhos.append(np.asarray(rho)[live])
    levs, rhos = np.concatenate(levs), np.concatenate(rhos)
    h_floor = float(np.percentile(levs, 5))
    log(f"pass 1: {len(levs)} live rows; leverage p5/p50/p95/max = "
        f"{np.percentile(levs, [5, 50, 95]).round(6).tolist()} / {levs.max():.4f}; mean rho^2 = {np.mean(rhos ** 2):.3f}; "
        f"h_floor {h_floor:.3g}")

    def accumulate(meats, bt):
        psi, rho, lev, atom, cfgid = train_rows(bt)
        n = psi.shape[0]
        out = {}
        for k, wgt in (("hc0", rho ** 2), ("hc3", rho ** 2 / (1 - jnp.minimum(lev, 0.99)) ** 2),
                       ("pops", rho ** 2 / jnp.maximum(lev, h_floor) ** 2)):
            q = psi * jnp.sqrt(wgt)[:, None]
            out[k] = meats[k] + q.T @ q
        for k, ids in (("atom", atom), ("cfg", cfgid)):
            g = jax.ops.segment_sum(psi * rho[:, None], ids, num_segments=n)
            out[k] = meats[k] + g.T @ g
        return out

    accumulate = jax.jit(accumulate, donate_argnums=0)
    meats = {k: jnp.zeros((L, L)) for k in VARIANTS}
    for bt in batches:
        meats = accumulate(meats, bt)
    jax.block_until_ready(meats)
    log("pass 2: meats", {k: float(jnp.trace(v)) for k, v in meats.items()})

    def atom_vars(Frows):
        """(ard, {variant}) per-atom sum over the 3 force components, from raw force rows (N, 3, Lt)."""
        N = Frows.shape[0]
        out_a, out_v = [], {k: [] for k in VARIANTS}
        for i in range(0, N, 1024):
            ph = Frows[i:i + 1024].reshape(-1, L) * dinv[None, :]
            U = cho_solve((Lc, True), ph.T)                                         # S^-1 phi~^T  (L, n)
            out_a.append(np.asarray(jnp.sum(ph.T * U, 0)).reshape(-1, 3).sum(1))
            for k in VARIANTS:
                out_v[k].append(np.asarray(jnp.sum(U * (meats[k] @ U), 0)).reshape(-1, 3).sum(1))
        return np.concatenate(out_a), {k: np.concatenate(v) for k, v in out_v.items()}

    res = {}
    crows = chunked_rows_fn(prob.model, prob.cfg)
    for split, ds in (("test", d.ds_test), ("ood", d.ds_ood)):
        a_, v_ = [], {k: [] for k in VARIANTS}
        for i in range(ds.n_batches):
            bt = jax.tree.map(lambda a, i=i: a[i], ds)
            sa, sv = atom_vars(crows(bt).F)
            nm = np.asarray(bt.node_mask)
            a_.append(sa[nm])
            for k in VARIANTS:
                v_[k].append(sv[k][nm])
        res[f"s2_ard_{split}"] = np.concatenate(a_)
        for k in VARIANTS:
            res[f"s2_{k}_{split}"] = np.concatenate(v_[k])
        pr = np.load(f"{run}/pred_{split}_map.npz")
        chk = np.max(np.abs(ard["kappa"] ** 2 * res[f"s2_ard_{split}"] - pr["F_var"].sum(1)) / pr["F_var"].sum(1))
        log(f"{split}: {len(res[f's2_ard_{split}'])} atoms; max rel |kappa^2 s2_ard - pred F_var| = {chk:.2e}")

    a_, v_ = [], {k: [] for k in VARIANTS}
    for at in read(f"{data}/big.xyz", ":"):
        cf = Config(at.get_positions(), at.get_atomic_numbers(), at.get_cell().array, at.get_pbc(),
                    None, None, None, 1.0, 1.0, 1.0)
        bt = jax.tree.map(lambda a: a[0], build_dataset([cf], d.meta, d.E0, 1))
        sa, sv = atom_vars(crows(bt).F)
        nm = np.asarray(bt.node_mask)
        a_.append(sa[nm])
        for k in VARIANTS:
            v_[k].append(sv[k][nm])
    res["s2_ard_big"] = np.concatenate(a_)
    for k in VARIANTS:
        res[f"s2_{k}_big"] = np.concatenate(v_[k])
    B = np.load(f"{run}/big_err.npz")
    log(f"big: {len(res['s2_ard_big'])} atoms; max rel |kappa s_ard - saved sd| = "
        f"{np.max(np.abs(ard['kappa'] * np.sqrt(res['s2_ard_big']) - B['sd']) / B['sd']):.2e}")

np.savez(f"{run}/sandwich.npz", lev_quantiles=np.percentile(levs, [1, 5, 25, 50, 75, 95, 99]),
         mean_rho2=np.mean(rhos ** 2), h_floor=h_floor, **res)
log("wrote", f"{run}/sandwich.npz")
