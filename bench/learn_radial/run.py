"""Learn the tensor radials of an ACE model by VarPro, gate on a held-out split.

    uv run python bench/learn_radial/run.py --model M.npz --data D.xyz --out DIR \
        [--n-q 30] [--ntrain 200] [--steps 40] [--lam-grid 0,1e-2]

A splined (Julia-exported) model is converted to the analytic branch first
(to_analytic); an analytic one is widened to --n-q.  Writes DIR/model.npz (the
selected radials and the readout fitted for them patched into a copy of
--model), rnl_Wnlq.npy, readout.npy, radial_info.json and summary.json (gate
scores, selected label, to_analytic_relres_max).  Each lambda's radials are
checkpointed to DIR/lam_<lam>/ (rnl_Wnlq.npy, radial_info.json) as soon as its
run finishes.  The residual GP / UQ fit then runs on DIR/model.npz as usual.
"""
import argparse
import json
import pathlib
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ace_jax.construct.prior import prior_diagonal
from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import default_prior
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem
from ace_jax.fit.radial_learn import fit_radial, save_result
from ace_jax.fit.radial_model import rnl_degrees, to_analytic

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True); p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--energy-key", default="energy"); p.add_argument("--force-key", default="forces")
p.add_argument("--virial-key", default="virial")
p.add_argument("--ntrain", type=int, default=200); p.add_argument("--nval", type=int, default=200)
p.add_argument("--seed", type=int, default=0); p.add_argument("--batch", type=int, default=4)
p.add_argument("--r0", type=float, default=2.35, help="hyperprior length scale (default_prior)")
p.add_argument("--n-q", type=int, default=30, help="tensor-radial polynomial span after widening")
p.add_argument("--steps", type=int, default=40); p.add_argument("--reprofile-every", type=int, default=20)
p.add_argument("--lam-grid", default="0,1e-2", help="relative roughness weights")
p.add_argument("--map-steps", type=int, default=300)
a = p.parse_args()

t0 = time.time()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
model, meta, z = load(a.model)
model, relres = to_analytic(model, a.n_q)
relres_max = float(np.max(relres))
print(f"to_analytic: n_q={a.n_q} relres_max={relres_max:.3e}", flush=True)
NZ = len(meta["elements"])

configs = load_configs(a.data, a.energy_key, a.force_key, a.virial_key)
perm = np.random.default_rng(a.seed).permutation(len(configs))
if a.ntrain + a.nval > len(configs):
    raise SystemExit(f"--ntrain + --nval = {a.ntrain + a.nval} > {len(configs)} configs")
fit_c = [configs[i] for i in perm[:a.ntrain]]
val_c = [configs[i] for i in perm[a.ntrain:a.ntrain + a.nval]]
E0 = np.asarray(z["E0"]) if "E0" in z else np.zeros(NZ)
ds_fit = build_dataset(fit_c, meta, E0, configs_per_batch=a.batch)
ds_val = build_dataset(val_c, meta, E0, configs_per_batch=a.batch)
print(f"data loaded: {time.time() - t0:.1f}s", flush=True)

cfg = GPConfig(r0=a.r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=NZ, C=a.batch)
X, S = site_features(model, cfg, ds_fit)
ind = select_inducing(X, S, ds_fit.node_z, ds_fit.node_mask, 0, descriptor_scale(X, ds_fit.node_mask))
gamma = prior_diagonal(z, meta, source=a.model)
prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(gamma), default_prior(a.r0))
print(f"problem built: {time.time() - t0:.1f}s", flush=True)

wn = 1.0 / (1.0 + rnl_degrees(meta)) ** 2
lam_grid = tuple(float(x) for x in a.lam_grid.split(","))


def checkpoint(label, W_lam, run_info):
    """Persist each lambda's learned radials as soon as its run finishes, so an
    interrupted grid keeps them (no model.npz: the readout comes from the gate)."""
    save_result(out / f"lam_{label}", W_lam, run_info)
    print(f"checkpoint: {out / f'lam_{label}'}", flush=True)


W, info = fit_radial(prob, ds_fit, ds_val, model.rnl_Wnlq, lam_grid=lam_grid,
                     rough_weights=wn, steps=a.steps, reprofile_every=a.reprofile_every,
                     map_steps=a.map_steps, log=lambda s: print(s, flush=True),
                     checkpoint=checkpoint)
info["to_analytic_relres_max"] = relres_max
save_result(out, W, info, src_npz=a.model, model=model)
summary = {"selected": info["selected"], "scores": info["scores"], "n_q": a.n_q,
           "ntrain": a.ntrain, "nval": a.nval, "lam_grid": list(lam_grid),
           "steps": a.steps, "reprofile_every": a.reprofile_every,
           "to_analytic_relres_max": relres_max,
           "seconds": time.time() - t0}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
