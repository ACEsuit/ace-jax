"""Validation errors in physical units for learned-radial candidates.

The held-out gate score is a sigma-normalised SSE; this reports what it means
physically. For each candidate radial set W it repeats the gate's procedure
(theta-MAP on the fit split warm-started from the init MAP, then the M = 0
posterior-mean readout) and predicts the validation split directly: energy
RMSE/MAE in meV/atom and force RMSE/MAE in meV/Å (and the training-split
errors, for the over-fitting check). Same model, data, split and seed flags as
bench/learn_radial/run.py.

    uv run python bench/learn_radial/rmse.py --model M.npz --data D.xyz --n-q 30 \
        --cand init --cand runs/sige/lam_0/rnl_Wnlq.npy ... --out rmse.json
"""
import argparse
import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ace_jax.construct.prior import prior_diagonal
from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import default_prior, from_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem, posterior
from ace_jax.fit.radial_learn import theta_map_linear
from ace_jax.fit.radial_model import normalise, radial_gram, row_active, to_analytic, with_radial
from ace_jax.fit.rows import linear_rows
from ace_jax.fit.stats import linear_statistics

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True); p.add_argument("--data", required=True)
p.add_argument("--energy-key", default="energy"); p.add_argument("--force-key", default="forces")
p.add_argument("--virial-key", default="virial")
p.add_argument("--ntrain", type=int, default=200); p.add_argument("--nval", type=int, default=200)
p.add_argument("--seed", type=int, default=0); p.add_argument("--batch", type=int, default=4)
p.add_argument("--r0", type=float, default=2.35)
p.add_argument("--n-q", type=int, default=12, help="must match the run that produced the candidates")
p.add_argument("--map-steps", type=int, default=300)
p.add_argument("--cand", action="append", required=True,
               help='"init" or a path to a learned rnl_Wnlq.npy (repeatable); label = its parent dir')
p.add_argument("--out", required=True)
p.add_argument("--save-models", default=None,
               help="directory: also write <label>.npz per candidate (its radials + the readout fitted for them)")
a = p.parse_args()

model, meta, z = load(a.model)
model, _ = to_analytic(model, a.n_q)
NZ = len(meta["elements"])
configs = load_configs(a.data, a.energy_key, a.force_key, a.virial_key)
perm = np.random.default_rng(a.seed).permutation(len(configs))
E0 = np.asarray(z["E0"]) if "E0" in z else np.zeros(NZ)
ds_fit = build_dataset([configs[i] for i in perm[:a.ntrain]], meta, E0, configs_per_batch=a.batch)
ds_val = build_dataset([configs[i] for i in perm[a.ntrain:a.ntrain + a.nval]], meta, E0,
                       configs_per_batch=a.batch)
cfg = GPConfig(r0=a.r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=NZ, C=a.batch)
X, S = site_features(model, cfg, ds_fit)
ind = select_inducing(X, S, ds_fit.node_z, ds_fit.node_mask, 0, descriptor_scale(X, ds_fit.node_mask))
prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
               jnp.asarray(prior_diagonal(z, meta, source=a.model)), default_prior(a.r0))

W0 = model.rnl_Wnlq
W_init = normalise(W0, radial_gram(model, ds_fit), row_active(W0))
a0 = theta_map_linear(prob, ds_fit, W_init, steps=a.map_steps)          # as fit_radial does


@jax.jit
def predict(m, c, b):
    """(E (C,), F (Ncap, 3)) predictions of the linear model with readout c for one batch."""
    r, _, _ = linear_rows(m, cfg, b)
    return r.E @ c, r.F @ c


def errors(m, c, ds):
    """Per-atom energy and per-component force errors (eV, eV/Å) over live rows."""
    eE, eF = [], []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda x: x[i], ds)
        pE, pF = predict(m, c, b)
        live_E = np.asarray(b.cfg_mask) & (np.asarray(b.w_E) > 0)
        dE = (np.asarray(pE) - np.asarray(b.y_E)) / np.maximum(np.asarray(b.n_atoms), 1)
        eE.append(dE[live_E])
        live_F = np.asarray(b.node_mask) & (np.asarray(b.w_F) > 0)
        eF.append((np.asarray(pF) - np.asarray(b.y_F))[live_F].ravel())
    return np.concatenate(eE), np.concatenate(eF)


rms = lambda e: float(1e3 * np.sqrt(np.mean(e ** 2)))
mae = lambda e: float(1e3 * np.mean(np.abs(e)))
res = {}
for cand in a.cand:
    if cand == "init":
        label, W = "init", W_init
    else:
        label, W = pathlib.Path(cand).parent.name, jnp.asarray(np.load(cand))
    a_fit = theta_map_linear(prob, ds_fit, W, steps=a.map_steps, init=a0)
    m = with_radial(model, W)
    c, _ = posterior(from_array(a_fit), linear_statistics(m, cfg, ds_fit), prob)
    if a.save_models:
        from ace_jax.construct.export import patch_radial_npz
        pathlib.Path(a.save_models).mkdir(parents=True, exist_ok=True)
        patch_radial_npz(a.model, pathlib.Path(a.save_models) / f"{label}.npz", m, readout=np.asarray(c))
    out = {}
    for split, ds in (("val", ds_val), ("train", ds_fit)):
        eE, eF = errors(m, c, ds)
        out[split] = {"E_rmse_meV_atom": rms(eE), "E_mae_meV_atom": mae(eE),
                      "F_rmse_meV_A": rms(eF), "F_mae_meV_A": mae(eF), "n_E": int(eE.size), "n_F": int(eF.size)}
    res[label] = out
    v, t = out["val"], out["train"]
    print(f"{label:22s} val E {v['E_rmse_meV_atom']:8.3f} meV/atom  F {v['F_rmse_meV_A']:8.2f} meV/Å"
          f"   | train E {t['E_rmse_meV_atom']:8.3f}  F {t['F_rmse_meV_A']:8.2f}", flush=True)
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
