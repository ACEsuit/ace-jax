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
import argparse, json, pathlib, time

import jax
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.ladder import run_laplace_fd, run_map, run_nuts, run_pathfinder, run_vi
from ace_jax.fit.metrics import summarise
from ace_jax.fit.objective import Problem, make_log_density
from ace_jax.fit.predict import predict_mixture

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]   # Voigt 6-vector index pairs

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--data", required=True)
p.add_argument("--out", required=True); p.add_argument("--arm", choices=["linear", "gp"], required=True)
p.add_argument("--ntrain", type=int, default=800); p.add_argument("--ntest", type=int, default=200)
p.add_argument("--test-start", type=int, default=None, help="absolute perm index for the test slice (default: ntrain); use a FIXED value to hold the test set constant across a --ntrain sweep")
p.add_argument("--seed", type=int, default=0); p.add_argument("--batch", type=int, default=4)
p.add_argument("--m-per-species", type=int, default=100)
p.add_argument("--kernel", default="cosine"); p.add_argument("--no-bump", action="store_true")
p.add_argument("--density", choices=["none", "pair"], default="none",
               help="residual feature map: none = isotropic B*scale (D dims); pair = the ACE pair-density channels (n_pair dims)")
p.add_argument("--warp", choices=["none", "sqrt"], default="none",
               help="feature warp: sqrt gives the Finnis-Sinclair sqrt-density embedding")
p.add_argument("--no-deriv-dtc", action="store_true", help="force/virial variance SoR only (drop the derivative-DTC)")
p.add_argument("--no-predict-train", action="store_true", help="skip train-set UQ prediction (a diagnostic; ~46%% of runtime at Cantor scale)")
p.add_argument("--rungs", default="map,laplace"); p.add_argument("--n-draws", type=int, default=64)
p.add_argument("--map-steps", type=int, default=150); p.add_argument("--map-lr", type=float, default=0.02)
p.add_argument("--opt", choices=["lbfgs", "adam"], default="lbfgs")
p.add_argument("--vi-steps", type=int, default=1000)
p.add_argument("--nuts-warmup", type=int, default=100); p.add_argument("--nuts-samples", type=int, default=100)
p.add_argument("--nuts-chains", type=int, default=1)
p.add_argument("--pf-samples", type=int, default=4, help="pathfinder ELBO samples/L-BFGS iterate; peak ~ linear in this (Cantor M=500-safe default)")
p.add_argument("--pf-maxiter", type=int, default=10, help="pathfinder L-BFGS iterations; peak ~ linear in (this+1)")
p.add_argument("--r0", type=float, default=2.5)
p.add_argument("--init", default=None, help="theta_map.json to start from (e.g. the linear arm's)")
p.add_argument("--energy-key", default="mace_energy"); p.add_argument("--force-key", default="mace_force")
p.add_argument("--virial-key", default="mace_virial")
p.add_argument("--baseline", default=None, help="dimer_mean.npz: subtract the MH-1 pair mean mu_0 from labels, add back at prediction")
p.add_argument("--base-npz", default=None, help="precomputed per-config (E,F,V) mu_0 offsets in data-file order (make_density_base.py); train+test only")
p.add_argument("--ood", default=None, help="extra out-of-distribution test xyz (same keys); predicted from the fitted model")
p.add_argument("--embedding", default=None, help="MACE element-embedding table (JSON); coregionalize the GP species factor. Absent = categorical block-diagonal.")
p.add_argument("--learn-embedding", action="store_true", help="learn the GP species embedding by outer VarOpt (SVD-init from --embedding, freeze for the ladder). Off = frozen.")
p.add_argument("--embed-de", type=int, default=8, help="rank of the learned embedding (SVD of the MACE table)")
p.add_argument("--embed-anchor", type=float, default=1.0, help="lambda for the shrink-to-block-diagonal anchor")
p.add_argument("--embed-steps", type=int, default=30, help="outer VarOpt steps (0 = frozen)")
p.add_argument("--embed-holdout", type=float, default=0.2, help="fraction of train held out for the acceptance gate")
a = p.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
T0 = time.time(); timings = {}

model, meta, z = load(a.model)
keys = dict(energy_key=a.energy_key, force_key=a.force_key, virial_key=a.virial_key)
configs = load_configs(a.data, **keys)
rng = np.random.default_rng(a.seed); perm = rng.permutation(len(configs))
ts = a.ntrain if a.test_start is None else a.test_start
train_o = [configs[i] for i in perm[:a.ntrain]]
test_o = [configs[i] for i in perm[ts:ts + a.ntest]]
ood_o = load_configs(a.ood, **keys) if a.ood else []
np.save(out / "split_perm.npy", perm)

# mu_0: subtract the MH-1 pair mean from the labels (fit the many-body residual),
# add it back at prediction.  base_* are the per-config (E, F, V) of mu_0.
if a.baseline:
    import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from baseline import load_mean, subtract_baseline
    mean = load_mean(a.baseline)
    train, base_train = subtract_baseline(train_o, mean)
    test, base_test = subtract_baseline(test_o, mean)
    ood, base_ood = subtract_baseline(ood_o, mean) if ood_o else ([], [])
    print(f"mu_0 subtracted: train E/atom {np.mean([b[0]/len(c.numbers) for b,c in zip(base_train,train_o)]):.3f} eV", flush=True)
elif a.base_npz:
    zb = np.load(a.base_npz)
    off = np.concatenate([[0], np.cumsum(zb["natoms"])])
    def _base(idxs):
        return [(float(zb["E"][i]), zb["F"][off[i]:off[i + 1]], zb["V"][i]) for i in idxs]
    def _sub(cfgs, base):
        out = []
        for c, (E0_, F0_, V0_) in zip(cfgs, base):
            out.append(c._replace(energy=None if c.energy is None else c.energy - E0_,
                                  forces=None if c.forces is None else c.forces - F0_,
                                  virial=None if c.virial is None else c.virial - V0_))
        return out
    base_train = _base(perm[:a.ntrain]); base_test = _base(perm[ts:ts + a.ntest])
    train, test = _sub(train_o, base_train), _sub(test_o, base_test)
    ood = ood_o; base_ood = [(0.0, np.zeros((len(c.numbers), 3)), np.zeros((3, 3))) for c in ood_o]  # ood from a separate file
    print(f"mu_0 (density base): train E/atom {np.mean([b[0]/len(c.numbers) for b,c in zip(base_train,train_o)]):.3f} eV", flush=True)
else:
    train, test, ood = train_o, test_o, ood_o
    zero = lambda c: (0.0, np.zeros((len(c.numbers), 3)), np.zeros((3, 3)))
    base_train = [zero(c) for c in train_o]; base_test = [zero(c) for c in test_o]; base_ood = [zero(c) for c in ood_o]

# per-species E0 by least squares on the (residual) training energies
els = [int(e) for e in meta["elements"]]
counts = np.array([[np.sum(c.numbers == e) for e in els] for c in train], float)
E0, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in train]), rcond=None)
model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
print("E0 (eV):", dict(zip(els, np.round(E0, 4))), flush=True)

ds_train = build_dataset(train, meta, E0, a.batch)
ds_test = build_dataset(test, meta, E0, a.batch)
ds_ood = build_dataset(ood, meta, E0, a.batch) if ood else None
print(f"train batches {ds_train.n_batches}  Ncap {ds_train.node_z.shape[1]}  K {ds_train.nbr.shape[2]}  ood {len(ood)}", flush=True)
cfg = GPConfig(r0=a.r0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
               NZ=len(els), C=a.batch)
timings["setup"] = time.time() - T0

with highest_precision():
    t = time.time()
    X, S = site_features(model, cfg, ds_train)
    scale = descriptor_scale(X, ds_train.node_mask)
    m = a.m_per_species if a.arm == "gp" else 0
    from ace_jax.fit.inducing import build_pmap
    Pmap = build_pmap(cfg, scale, density=None if a.density == "none" else a.density)
    from ace_jax.fit.embedding import load_mace_embedding
    raw = None if a.embedding is None else np.asarray(load_mace_embedding(a.embedding, els))
    # Low-rank SVD init ONLY when learning; frozen --embedding stays full-width (exact prior
    # behaviour). load_mace_embedding returns (NZ, D) with NZ usually < D, so svd gives U (NZ, NZ)
    # and effective rank <= NZ; pass de = the ACTUAL embed width, never a.embed_de (which can
    # exceed the available columns -> select_inducing ValueError).
    if raw is not None and a.learn_embedding and a.embed_de:
        U, sv, _ = np.linalg.svd(raw, full_matrices=False)
        k = min(int(a.embed_de), U.shape[1])
        embed = jnp.asarray(U[:, :k] * sv[:k])                            # low-rank SVD init for learning
    else:
        embed = None if raw is None else jnp.asarray(raw)                 # frozen: full-width (exact prior behaviour)
    ind = select_inducing(X, S, ds_train.node_z, ds_train.node_mask, m, scale,
                          Pmap=Pmap, warp=a.warp, embed=embed, nz=len(els),
                          de=(int(embed.shape[1]) if embed is not None else None))
    print(f"feature map: density={a.density} warp={a.warp} -> d={ind.XM.shape[1] if ind.XM.shape[0] else cfg.D}", flush=True)
    timings["inducing"] = time.time() - t
    print(f"M = {ind.XM.shape[0]}  len_basis = {cfg.len_basis}", flush=True)
    prob = Problem(KernelSpec(a.kernel, not a.no_bump, cfg.D), model, ind, cfg,
                   jnp.asarray(z["gamma"]), default_prior(a.r0))
    if a.learn_embedding and embed is not None:
        from ace_jax.fit.varopt_embed import learn_embedding, theta_map_at
        from ace_jax.fit.predict import predict_fixed
        from ace_jax.fit.metrics import rmse
        from ace_jax.fit.hypers import from_array
        # GENUINELY held-out gate: split train into DISJOINT fit/val; learn E on train_fit,
        # score on train_val (NOT in train_fit). Conditioning val_score on ds_fit (not ds_train)
        # is what makes the gate protective.
        nval = max(1, int(a.embed_holdout * len(train)))
        train_fit, train_val = train[:-nval], train[-nval:]
        ds_fit = build_dataset(train_fit, meta, E0, a.batch)
        ds_val = build_dataset(train_val, meta, E0, a.batch)
        Fval = np.concatenate([c.forces for c in train_val]).reshape(-1)
        def val_score(E):
            ind_E = ind._replace(embed=E)
            a_star = theta_map_at(prob._replace(ind=ind_E), ds_fit, E, steps=a.map_steps)
            pr = predict_fixed(from_array(a_star), prob._replace(ind=ind_E), ds_fit, ds_val,
                               dtc=False, deriv_dtc=False)
            return rmse(Fval, np.asarray(pr.F_mean).reshape(-1))
        E_star, einfo = learn_embedding(prob, ds_fit, embed, lam=a.embed_anchor,
                                        steps=a.embed_steps, inner_steps=a.map_steps,
                                        val_score=val_score, seed=a.seed)
        print(f"learn-embedding: selected {einfo['selected']}, trace {np.round(einfo['trace'][-3:], 2)}", flush=True)
        ind = ind._replace(embed=E_star)
        prob = prob._replace(ind=ind)
    # theta-split cached likelihood: the theta-independent linear Gram G_BB is
    # streamed once; only the M residual columns move per evaluation (both arms).
    from ace_jax.fit.objective import make_lml
    from ace_jax.fit.stats import assemble_statistics, linear_statistics, residual_statistics
    t = time.time(); lik = make_lml(prob, ds_train, cache_linear=True)
    jax.block_until_ready(lik(to_array(prob.prior.mu))); timings["stats_once"] = time.time() - t
    # cache the theta-independent linear stats for PREDICTION too (predict_fixed
    # always conditions on ds_train), so each draw recomputes only the M columns
    import ace_jax.fit.predict as _P
    _lin = jax.jit(lambda: linear_statistics(prob.model, prob.cfg, ds_train))()
    _P.sufficient_statistics = (lambda th, spec, model, ind, cfg, ds:
        assemble_statistics(_lin, residual_statistics(th, spec, model, ind, cfg, ds)))

    init = None
    if a.init:
        init = Hypers(**json.load(open(a.init)))
    # no standalone lik/grad probes: each is a separate compiled executable
    # whose buffers stay resident and starved the L-BFGS value_and_grad of GPU
    # memory at the Cantor basis size; the first L-BFGS evaluation is timed instead

    rungs = [r.strip() for r in a.rungs.split(",")]
    t = time.time()
    if a.opt == "lbfgs":
        # 10-d smooth objective with an exact gradient: L-BFGS converges in a
        # few tens of evaluations where Adam needs hundreds of (expensive) steps
        from scipy.optimize import minimize
        from ace_jax.fit.hypers import from_array, log_prior
        logpost = jax.jit(lambda arr: lik(arr) + log_prior(from_array(arr), prob.prior))
        vg = jax.jit(jax.value_and_grad(logpost))
        hist = []
        def fg(x):
            t1 = time.time(); v, g = vg(jnp.asarray(x)); g.block_until_ready(); v = float(v); hist.append(v)
            print(f"  lbfgs eval {len(hist)}  logpost = {v:.6g}  ({time.time() - t1:.0f} s)", flush=True)
            if not np.isfinite(v) or not np.all(np.isfinite(np.asarray(g))):
                # an ill-conditioned trial point (e.g. sigma_E -> 0): a large
                # finite penalty makes the line search back off instead of aborting
                return 1e12, np.zeros_like(x)
            return -v, -np.asarray(g, float)
        x0 = np.asarray(to_array(init or prob.prior.mu), float)
        # log-space boxes: generous, but keep the Cholesky away from sigma -> 0
        lo = np.log([0.05, 1e-3, 0.1, 1.5, 1e-3, 0.1, 1e-2, 1e-4, 1e-4, 1e-4])
        hi = np.log([50.0, 10.0, 50.0, 4.0, 50.0, 100.0, 1e4, 10.0, 10.0, 10.0])
        x0 = np.clip(x0, lo, hi)
        res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                       options={"maxiter": a.map_steps, "maxfun": 4 * a.map_steps})
        theta_map = Hypers(*[float(v) for v in res.x])
        print("L-BFGS:", res.message, "nfev", res.nfev, flush=True)
    else:
        theta_map = run_map(lik, prob.prior, steps=a.map_steps, lr=a.map_lr, seed=a.seed, init=init)
    timings["map"] = time.time() - t
    json.dump(theta_map._asdict(), open(out / "theta_map.json", "w"), indent=1)
    print("MAP:", {k: round(float(np.exp(v)), 5) for k, v in theta_map._asdict().items()}, flush=True)

    draws = {"map": np.asarray(to_array(theta_map))[None]}
    if "laplace" in rungs:
        t = time.time()
        draws["laplace"], lap_info = run_laplace_fd(lik, prob.prior, theta_map, n_draws=a.n_draws, seed=a.seed)
        timings["laplace"] = time.time() - t
        json.dump(lap_info, open(out / "laplace_info.json", "w"), indent=1)
        print("Laplace std (log space):", dict(zip(lap_info["fields"], np.round(lap_info["std"], 4))),
              "floored:", lap_info["n_floored"], flush=True)
    if "pathfinder" in rungs:
        t = time.time()
        draws["pathfinder"], pf_info = run_pathfinder(lik, prob.prior, theta_map, n_draws=a.n_draws, seed=a.seed,
                                                num_samples=a.pf_samples, maxiter=a.pf_maxiter)
        timings["pathfinder"] = time.time() - t
        print("Pathfinder std (log space):", dict(zip(pf_info["fields"], np.round(pf_info["std"], 4))), flush=True)
    if "vi" in rungs:
        t = time.time()
        draws["vi"], _ = run_vi(lik, prob.prior, n_draws=a.n_draws, steps=a.vi_steps, seed=a.seed, init=theta_map)
        timings["vi"] = time.time() - t
    if "nuts" in rungs:
        t = time.time()
        draws["nuts"], summ = run_nuts(lik, prob.prior, num_warmup=a.nuts_warmup, num_samples=a.nuts_samples,
                                       num_chains=a.nuts_chains, seed=a.seed, init=theta_map)
        timings["nuts"] = time.time() - t
        json.dump(summ, open(out / "nuts_summary.json", "w"), indent=1)

    metrics = {}
    for rung, d in draws.items():
        np.save(out / f"draws_{rung}.npy", d)
        sub = d if len(d) <= a.n_draws else d[np.linspace(0, len(d) - 1, a.n_draws).astype(int)]
        splits = [("test", test_o, ds_test, base_test)]
        if not a.no_predict_train:
            splits.append(("train", train_o, ds_train, base_train))
        if ds_ood is not None:
            splits.append(("ood", ood_o, ds_ood, base_ood))
        for split, cfgs, ds, base in splits:
            t = time.time()
            pred = predict_mixture(sub, prob, ds_train, ds, deriv_dtc=not a.no_deriv_dtc)
            timings[f"predict_{split}_{rung}"] = time.time() - t
            nat = np.array([len(c.numbers) for c in cfgs])
            # references are the ORIGINAL labels; add mu_0 back to the predictions
            bE = np.array([b[0] for b in base]); bF = np.concatenate([b[1] for b in base]) if base else np.zeros((0,3))
            bV6 = np.array([[b[2][i, j] for i, j in VOIGT] for b in base])
            E = np.array([c.energy for c in cfgs]); F = np.concatenate([c.forces for c in cfgs])
            V = np.array([[c.virial[i, j] for i, j in VOIGT] for c in cfgs])
            pred = pred._replace(E_mean=np.asarray(pred.E_mean) + bE,
                                 F_mean=np.asarray(pred.F_mean) + bF,
                                 V_mean=np.asarray(pred.V_mean) + bV6)
            # inferred noise variances (mixture mean over draws) so the plots can show
            # the predictive sigma for a label: latent variance + noise; the noise
            # convention is tau = (w/sigma_t)^2 with w = 1/sqrt(n) on E and V rows
            s2 = {k: float(np.mean(np.exp(2 * sub[:, i]))) for k, i in (("E", 7), ("F", 8), ("V", 9))}
            np.savez(out / f"pred_{split}_{rung}.npz", nat=nat, E=E, E_mean=pred.E_mean, E_var=pred.E_var,
                     F=F, F_mean=pred.F_mean, F_var=pred.F_var, V=V, V_mean=pred.V_mean, V_var=pred.V_var,
                     noise_E=s2["E"] * nat, noise_F=np.full(len(F), s2["F"]), noise_V=s2["V"] * nat)      # one virial per config; all configs here carry one
            metrics[f"{split}/{rung}"] = {
                "E": summarise(1e3 * E / nat, 1e3 * pred.E_mean / nat, 1e3 * np.sqrt(pred.E_var) / nat),
                "F": summarise(F.reshape(-1), pred.F_mean.reshape(-1), np.sqrt(pred.F_var).reshape(-1)),
                "V": summarise(V.reshape(-1), pred.V_mean.reshape(-1), np.sqrt(pred.V_var).reshape(-1))}
            print(split, rung, {q: {k: round(v, 4) for k, v in m_.items() if k in ("rmse", "crps", "coverage", "rho", "rms_z")}
                                for q, m_ in metrics[f"{split}/{rung}"].items()}, flush=True)

timings["total"] = time.time() - T0
json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
json.dump(timings, open(out / "timings.json", "w"), indent=1)
json.dump({**vars(a), "M": int(ind.XM.shape[0]), "len_basis": cfg.len_basis, "E0": dict(zip(map(str, els), map(float, E0)))},
          open(out / "config.json", "w"), indent=1)
print("done", {k: round(v, 1) for k, v in timings.items()}, flush=True)
