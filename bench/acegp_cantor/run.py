"""Acceptance run: linear ACE (BLR, M = 0) vs ACEgp (hybrid) on the Cantor
CrMnFeCoNi set labelled by MACE-MH-1.  Same basis, same split, same code path;
the two arms differ only in the inducing set (M = 0 vs M = 5 x m_per_species).

    python run.py --model cantor_d6.npz --data cantor1k_b_mh1.xyz --out out/ \
        --arm gp --rungs map,laplace --ntrain 800

Writes per rung: pred_<split>_<rung>.npz (targets, means, variances, n_atoms),
draws_<rung>.npy, theta_map.json, metrics.json, timings.json.
E0 is the per-species least-squares offset fitted on the training energies
(the set has no isolated atoms), applied identically to both arms.
"""
import argparse, json

import jax
jax.config.update("jax_enable_x64", True)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--data", required=True)
p.add_argument("--out", required=True); p.add_argument("--arm", choices=["linear", "gp"], required=True)
p.add_argument("--ntrain", type=int, default=800); p.add_argument("--ntest", type=int, default=200)
p.add_argument("--test-start", type=int, default=None, help="absolute perm index for the test slice (default: ntrain); use a FIXED value to hold the test set constant across a --ntrain sweep")
p.add_argument("--seed", type=int, default=0); p.add_argument("--batch", type=int, default=4)
p.add_argument("--m-per-species", type=int, default=100)
p.add_argument("--kernel", default="cosine"); p.add_argument("--no-bump", action="store_true")
p.add_argument("--density", choices=["none", "pair", "pca"], default="none",
               help="residual feature map: none = isotropic B*scale (D dims); pair = the ACE pair-density "
                    "channels (n_pair dims); pca = the top --pca-d uncentred principal frame of the scaled "
                    "full descriptor (many-body included), host-cacheable")
p.add_argument("--pca-d", type=int, default=128, help="width of the --density pca feature map")
p.add_argument("--warp", choices=["none", "sqrt"], default="none",
               help="feature warp: sqrt gives the Finnis-Sinclair sqrt-density embedding")
p.add_argument("--no-deriv-dtc", action="store_true", help="force/virial variance SoR only (drop the derivative-DTC)")
p.add_argument("--uq", choices=["blr", "pops"], default="blr", help="linear-arm predictive UQ: blr (posterior variance, today's default) or pops (weight-space misspecification). pops requires --arm linear.")
p.add_argument("--pops-posterior", choices=["hypercube", "ensemble"], default="hypercube", help="POPS posterior form (uq=pops): hypercube (PCA/box misspecification covariance; DEFAULT, matches upstream popsregression) or ensemble (committee of weight samples; centred). ('samples' is reserved for a future draw-from-Sigma route.)")
p.add_argument("--pops-leverage-pct", type=float, default=0.0, help="POPS leverage percentile (uq=pops); 0 keeps every training point")
p.add_argument("--pops-ridge", default="auto",
               help="uq=pops uncertainty ridge (the mean is always the BLR mean): 'auto' (DEFAULT: per "
                    "quantity by validation CRPS), 'blr' (1/sigma_c^2), a relative ridge (lam / max eig of "
                    "the loss Gram), or per quantity as 'E=1e-11,F=1e-7,V=1e-6' "
                    "to pick it per quantity by CRPS on the last --pops-val-frac of train (one factorisation)")
p.add_argument("--pops-ridge-grid", default="1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14",
               help="comma-separated relative ridges searched by --pops-ridge auto")
p.add_argument("--pops-val-frac", type=float, default=0.2, help="held-out fraction of train for --pops-ridge auto")
p.add_argument("--pops-env-nf", type=int, default=2000,
               help="force components (random subsample of test) for the paper-mode envelope coverage")
p.add_argument("--delta-s-floor-q", type=float, default=None,
               help="floor the amplitude coordinate s at this quantile of the training sites' s: "
                    "delta(max(s, s_q)) cannot extrapolate to zero in compressed environments")
p.add_argument("--fix-rho", default=None,
               help="pin the bump lengthscale rho (L-BFGS only): a number, or 'auto' = the median "
                    "nearest-neighbour RMS distance within the inducing set; keeps the evidence from "
                    "flattening the GP's novelty term")
p.add_argument("--lml", choices=["device", "host-cache"], default="device",
               help="joint-LML engine. host-cache (needs --arm gp --density pair, --rungs map, --opt lbfgs): "
                    "cache the linear design rows in host RAM once and never re-evaluate the ACE basis "
                    "per evaluation (ace_jax.fit.hostcache; ~rows x L x 8 B of host RAM)")
p.add_argument("--lml-chunk", type=int, default=64, help="host-cache: batches per host->device transfer")
p.add_argument("--no-predict-train", action="store_true", help="skip train-set UQ prediction (a diagnostic; ~46%% of runtime at Cantor scale)")
p.add_argument("--rungs", default="map,laplace"); p.add_argument("--n-draws", type=int, default=64)
p.add_argument("--map-steps", type=int, default=150); p.add_argument("--map-lr", type=float, default=0.02)
p.add_argument("--opt", choices=["lbfgs", "adam"], default="lbfgs")
p.add_argument("--map-restarts", type=int, default=1,
               help="L-BFGS MAP from this many starts (the --init/prior-mean point plus seeded "
                    "hyperprior draws, clipped to the box); the best log-posterior wins.  The joint "
                    "LML is multimodal: single starts ended 620 nats apart on Cantor-1k")
p.add_argument("--vi-steps", type=int, default=1000)
p.add_argument("--nuts-warmup", type=int, default=100); p.add_argument("--nuts-samples", type=int, default=100)
p.add_argument("--nuts-chains", type=int, default=1)
p.add_argument("--pf-samples", type=int, default=4, help="pathfinder ELBO samples/L-BFGS iterate; peak ~ linear in this (Cantor M=500-safe default)")
p.add_argument("--pf-maxiter", type=int, default=10, help="pathfinder L-BFGS iterations; peak ~ linear in (this+1)")
p.add_argument("--r0", type=float, default=2.5)
p.add_argument("--init", default=None, help="theta_map.json to start from (e.g. the linear arm's)")
p.add_argument("--energy-key", default="mace_energy"); p.add_argument("--force-key", default="mace_force")
p.add_argument("--virial-key", default="mace_virial")
p.add_argument("--weights", default=None,
               help='JSON list of weight factors (ace_jax.fit.weights), composed in order and '
                    'multiplied into w_E/w_F/w_V, e.g. \'[{"Structural":{}},'
                    '{"ConfigType":{"table":{"defect":{"E":10,"F":10,"V":1}},"default":{"E":1,"F":1,"V":1}}},'
                    '{"PerConfig":{"key":"w"}}]\'. Each entry is {"<FactorClass>": {<constructor kwargs>}}, '
                    "kwargs matching the class's __init__ (table/key/default for ConfigType, key for "
                    "PerConfig, w for Quantity, exp for Structural). Omitted = the classic default "
                    "(structural 1/sqrt(n) on E,V, 1 on F; no per-config-type dict).")
p.add_argument("--baseline", default=None, help="dimer_mean.npz: subtract the MH-1 pair mean mu_0 from labels, add back at prediction")
p.add_argument("--base-npz", default=None, help="precomputed per-config (E,F,V) mu_0 offsets in data-file order (make_density_base.py); train+test only")
p.add_argument("--ood", default=None, help="extra out-of-distribution test xyz (same keys); predicted from the fitted model")
p.add_argument("--embedding", default=None, help="MACE element-embedding table (JSON); coregionalize the GP species factor. Absent = categorical block-diagonal.")
p.add_argument("--sigma-type", action="store_true",
               help="fit a per-config-type noise block (Task 6): assign each config a type from its "
                    "config_type label, build a ParamSet carrying the sigma_type LML block and fit the "
                    "inner MAP via run_map_ps so the block is optimised. The learned per-type log-ratios "
                    "(rows=type, cols E,F,V) are DIAGNOSTIC ONLY (written to sigma_type_ratios.json); "
                    "predictions/draws/Laplace still use the single theta_map noise -- per-type PREDICTIVE "
                    "noise is not yet wired. Off (default) = the classic single-noise L-BFGS/run_map fit, "
                    "numerically unchanged.")
p.add_argument("--route", default=None,
               help='per-block route override for the ParamSet fit, as a JSON object, e.g. '
                    '\'{"sigma_type":"lml"}\'; each route is fixed or lml.')
a = p.parse_args()
from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, write_outputs
from ace_jax.fit.paramset import parse_route
from ace_jax.fit.weights import ConfigType, PerConfig, Quantity, Structural

_FACTOR_CLASSES = {"Structural": Structural, "Quantity": Quantity,
                   "ConfigType": ConfigType, "PerConfig": PerConfig}
factors = None
if a.weights:
    factors = [_FACTOR_CLASSES[name](**kw) for entry in json.loads(a.weights) for name, kw in entry.items()]
    print("weight factors:", [type(f).__name__ for f in factors], flush=True)
if a.pops_ridge in ("auto", "blr"):
    ridge = a.pops_ridge
elif "=" in a.pops_ridge:
    ridge = {k.strip(): (v.strip() if v.strip() == "blr" else float(v))
             for k, v in (kv.split("=") for kv in a.pops_ridge.split(","))}
    if set(ridge) != set("EFV"):
        p.error("--pops-ridge per-quantity form needs E=,F=,V=")
else:
    ridge = float(a.pops_ridge)
cfg = FitConfig(
    model=a.model, arm=a.arm, energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key,
    ntrain=a.ntrain, ntest=a.ntest, test_start=a.test_start, seed=a.seed, batch=a.batch,
    factors=factors, sigma_type=a.sigma_type, route=parse_route(a.route), baseline=a.baseline,
    base_npz=a.base_npz, e0="lsq", m_per_species=a.m_per_species, kernel=a.kernel, bump=not a.no_bump,
    density=a.density, pca_d=a.pca_d, warp=a.warp, embedding=a.embedding,
    delta_s_floor_q=a.delta_s_floor_q, fix_rho=a.fix_rho, r0=a.r0, lml=a.lml, lml_chunk=a.lml_chunk,
    opt=a.opt, map_steps=a.map_steps, map_lr=a.map_lr, map_restarts=a.map_restarts,
    init=json.load(open(a.init)) if a.init else None,
    rungs=tuple(dict.fromkeys(["map", *[r.strip() for r in a.rungs.split(",")]])),
    laplace="fd", n_draws=a.n_draws, vi_steps=a.vi_steps, nuts_warmup=a.nuts_warmup,
    nuts_samples=a.nuts_samples, nuts_chains=a.nuts_chains, pf_samples=a.pf_samples,
    pf_maxiter=a.pf_maxiter, uq=a.uq, deriv_dtc=not a.no_deriv_dtc, predict_train=not a.no_predict_train,
    pops_posterior=a.pops_posterior, pops_leverage_pct=a.pops_leverage_pct, pops_ridge=ridge,
    pops_ridge_grid=tuple(float(x) for x in a.pops_ridge_grid.split(",")),
    pops_val_frac=a.pops_val_frac, pops_env_nf=a.pops_env_nf)
try:
    cfg.validate()
except ValueError as e:
    p.error(str(e))
data = load_fit_data(cfg, data=a.data, ood=a.ood)
res = fit(cfg, data, log=lambda *s: print(*s, flush=True))
write_outputs(res, a.out, layout=("run",), argv=vars(a))
print("done", {k: round(v, 1) for k, v in res.timings.items()}, flush=True)
