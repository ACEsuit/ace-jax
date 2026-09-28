"""Learn the tensor radials of an ACE model by VarPro, gate on a held-out split.

    uv run python bench/learn_radial/run.py --model M.npz --data D.xyz --out DIR \
        [--n-q 12] [--ntrain 200] [--steps 40] [--lam-grid 0,1e-2]
    uv run python bench/learn_radial/run.py --model M.npz --data D.xyz --out DIR \
        --density full --P 1 --lam-eta-grid 0,1e-2

A splined (Julia-exported) model is converted to the analytic branch first
(to_analytic); an analytic one is widened to --n-q.  Writes DIR/model.npz (the
selected radials and the readout fitted for them patched into a copy of
--model), rnl_Wnlq.npy, readout.npy, radial_info.json and summary.json (gate
scores, selected label, to_analytic_relres_max).  Each lambda's radials are
checkpointed to DIR/lam_<lam>/ (rnl_Wnlq.npy, radial_info.json) as soon as its
run finishes.  With --density (a frozen sqrt-density term learned jointly with
the radials, gated against the plain radials-only fit; see
docs/specs/2026-09-28-radial-density-varpro-design.md), checkpoints go instead
to DIR/radials_only/ and DIR/density_lam_eta=<l>/, each with rnl_Wnlq.npy (and
eta.npy for the density runs).  The residual GP / UQ fit then runs on
DIR/model.npz as usual.
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
p.add_argument("--n-q", type=int, default=12, help="tensor-radial polynomial span after widening. 12 (a modest widening) optimises well and transfers to MD; 30 is ill-conditioned and learns only tiny high-frequency changes (docs/learn-radial-results.md)")
p.add_argument("--steps", type=int, default=40); p.add_argument("--reprofile-every", type=int, default=20)
p.add_argument("--lam-grid", default="0,1e-2", help="relative roughness weights")
p.add_argument("--spec-grid", default="0", help="relative spectral-prior weights on the radial change; useful range ~1e-6..1e-4 at --spec-p 4 (see relative_lambda_spec)")
p.add_argument("--spec-p", type=float, default=4.0, help="spectral prior degree power (1+q)^p")
p.add_argument("--gap-grid", default="0", help="relative data-gap-prior weights on the radial change, measured under a uniform-in-r Gram rather than the empirical pair-distance density; useful range ~1..30 (see relative_lambda_gap)")
p.add_argument("--map-steps", type=int, default=300)
p.add_argument("--learn-sigma-e-mult", type=float, default=1.0,
               help="scale sigma_E inside the radial objective only (>1 = force-heavier radial learning); "
                    "gate and final linear fit keep the MAP weights")
p.add_argument("--density", choices=["none", "pair", "full"], default="none",
               help="also learn sqrt-density features jointly with the radials over this span of the basis "
                    "(docs/specs/2026-09-28-radial-density-varpro-design.md); the gate keeps them only if "
                    "they win on the held-out split")
p.add_argument("--P", type=int, default=1, help="number of density features (with --density)")
p.add_argument("--density-mode", choices=["joint", "alternating"], default="joint")
p.add_argument("--lam-eta-grid", default="0", help="relative density shape-prior weights (with --density)")
a = p.parse_args()

t0 = time.time()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
model, meta, z = load(a.model)
if hasattr(model, "base"):              # an FSModel: learning starts from its linear part
    model = model.base
    print("note: the density term of the input model is dropped (it belongs to the old radials)", flush=True)
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
spec_grid = tuple(float(x) for x in a.spec_grid.split(","))
gap_grid = tuple(float(x) for x in a.gap_grid.split(","))


def checkpoint(label, W_lam, run_info):
    """Persist each lambda's learned radials as soon as its run finishes, so an
    interrupted grid keeps them (no model.npz: the readout comes from the gate)."""
    save_result(out / f"lam_{label}", W_lam, run_info)
    print(f"checkpoint: {out / f'lam_{label}'}", flush=True)


if a.density == "none":
    W, info = fit_radial(prob, ds_fit, ds_val, model.rnl_Wnlq, lam_grid=lam_grid, spec_grid=spec_grid,
                         gap_grid=gap_grid, rough_weights=wn, spec_p=a.spec_p, steps=a.steps,
                         reprofile_every=a.reprofile_every, map_steps=a.map_steps,
                         learn_sigma_e_mult=a.learn_sigma_e_mult,
                         log=lambda s: print(s, flush=True), checkpoint=checkpoint)
    eta = mask = None
else:
    from ace_jax.fit.density import density_mask
    from ace_jax.fit.radial_density import fit_radial_density
    if max(len(lam_grid), len(spec_grid), len(gap_grid)) > 1:
        raise SystemExit("--density needs single-valued --lam-grid, --spec-grid and --gap-grid "
                         "(sweep --lam-eta-grid instead)")
    if a.learn_sigma_e_mult != 1.0:
        raise SystemExit("--density does not support --learn-sigma-e-mult")
    mask = density_mask(cfg, a.density)

    def checkpoint_rd(label, W_c, eta_c, run_info):
        save_result(out / label, W_c, run_info, eta=eta_c, mask=mask)
        print(f"checkpoint: {out / label}", flush=True)

    W, eta, info = fit_radial_density(prob, ds_fit, ds_val, model.rnl_Wnlq, mask=mask, P=a.P,
                                      mode=a.density_mode,
                                      lam_eta_grid=tuple(float(x) for x in a.lam_eta_grid.split(",")),
                                      lam_rough=lam_grid[0], lam_spec=spec_grid[0], lam_gap=gap_grid[0],
                                      rough_weights=wn, spec_p=a.spec_p, steps=a.steps,
                                      reprofile_every=a.reprofile_every, map_steps=a.map_steps,
                                      log=lambda s: print(s, flush=True), checkpoint=checkpoint_rd)
info["to_analytic_relres_max"] = relres_max
save_result(out, W, info, src_npz=a.model, model=model, eta=eta, mask=mask)
summary = {"selected": info["selected"], "scores": info["scores"], "n_q": a.n_q,
           "ntrain": a.ntrain, "nval": a.nval, "lam_grid": list(lam_grid),
           "spec_grid": list(spec_grid), "spec_p": a.spec_p, "gap_grid": list(gap_grid),
           "steps": a.steps, "reprofile_every": a.reprofile_every,
           "learn_sigma_e_mult": a.learn_sigma_e_mult, "to_analytic_relres_max": relres_max,
           "density": a.density, "P_selected": int(info.get("P", 0)), "density_mode": a.density_mode,
           "lam_eta_grid": a.lam_eta_grid,
           "seconds": time.time() - t0}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
