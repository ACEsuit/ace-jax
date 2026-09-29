"""Bayesian replacements for the extrapolation grade, and body-order model averaging, on bench365.

Linear ACE (production basis, L = 15035), noise scales sigma_E/F/V and sigma_c fixed at the
bench365 POPS/BLR MAP (theta_map.json).  With M = sum_q G_q / sigma_q^2, b = sum_q b_q / sigma_q^2
(one sufficient-statistics pass) and prior precision Lambda, the posterior is N(A^-1 b, A^-1),
A = M + Lambda, and (up to Lambda-independent constants)

    log p(D | Lambda) = 1/2 b^T A^-1 b - 1/2 log|A| + 1/2 log|Lambda|.

Priors (per-species blocks share the scales):
  blr     Lambda = Gamma^2 / sigma_c^2                 (the fitted model)
  iso     Lambda = lam I                               (lam by evidence)
  ardG    Lambda = Gamma^2 * exp(a_k), k = body order  (3 scales by evidence)
  ardI    Lambda = exp(a_k) I per body order           (3 scales by evidence)
Nested truncations K = 2, 3, 4-body (columns of body order <= K), prior blr restricted: evidences,
per-atom forces F_K, and the BMA predictive  sum_K w_K [var_K + (F_K - Fbar)^2],  w_K ~ p(D|K).

Per-atom outputs (live sites, config-major, same order as pred_*.npz / sites.npz):
  sdF_<m>  sqrt(sum_xyz posterior force variance)  [test, ood]      (needs design rows)
  sdE_<m>  site-energy posterior std  x_i A^-1 x_i^T                 [test, ood, big]
  F_K<k>   forces of the K-body truncation; bma_sdF                 [test, ood]
    python bayes_bodyorder.py <data_dir> <theta_map.json> <out.npz>
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular      # noqa: E402

from ace_jax.eval import highest_precision                                # noqa: E402
from ace_jax.fit.data import build_dataset, load_configs                  # noqa: E402
from ace_jax.fit.hypers import Hypers                                     # noqa: E402
from ace_jax.fit.inducing import site_features                            # noqa: E402
from ace_jax.fit.pipeline import FitConfig, load_fit_data                 # noqa: E402
from ace_jax.fit.pipeline.problem import build_problem                    # noqa: E402
from ace_jax.fit.rows import linear_rows                                  # noqa: E402
from ace_jax.fit.stats import sufficient_statistics                       # noqa: E402

data, theta_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
T0 = time.time()
log = lambda *s: print(f"[{time.time() - T0:6.0f}s]", *s, flush=True)

cfg = FitConfig(model=f"{data}/cantor_embed_d16_deg10.npz", arm="linear", r0=2.5, batch=4, e0="lsq",
                energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial").validate()
d = load_fit_data(cfg, train=f"{data}/train.xyz", test=f"{data}/test.xyz", ood=f"{data}/ood.xyz")
b_ = build_problem(cfg, d)
prob, gcfg = b_.prob, b_.gpcfg
theta = Hypers(**json.load(open(theta_path)))
L, nB, nP, NZ = gcfg.len_basis, gcfg.n_B, gcfg.n_pair, gcfg.NZ
meta = d.meta
order_B = np.array([len(x) for x in meta["nnll"]])              # correlation order 1..3 -> body order +1
body = np.concatenate([np.tile(order_B + 1, 1), np.full(nP, 2)])   # per compact column: 2, 3, 4
body_col = np.concatenate([np.tile(body[:nB], NZ), np.tile(body[nB:], NZ)])   # layout: B blocks, then pair blocks
assert len(body_col) == L
groups = [np.flatnonzero(body_col == k) for k in (2, 3, 4)]
log("L", L, "columns per body order", [len(g) for g in groups])

with highest_precision():
    st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, d.ds_train)
    s2 = {q: float(jnp.exp(2 * getattr(theta, f"log_sigma_{q}"))) for q in "EFV"}
    M = sum(getattr(st, f"G_{q}") / s2[q] for q in "EFV")
    b = sum(getattr(st, f"b_{q}") / s2[q] for q in "EFV")
log("stats done")
gam2 = jnp.asarray(prob.gamma) ** 2
lam_blr = float(jnp.exp(-2 * theta.log_sigma_c))
G_idx = jnp.asarray(body_col - 2)                               # 0, 1, 2 per column


def logev(prior_diag, cols=None):
    Ms, bs, p = (M, b, prior_diag) if cols is None else (M[jnp.ix_(cols, cols)], b[cols], prior_diag[cols])
    A = Ms + jnp.diag(p)
    c, low = cho_factor(A, lower=True)
    x = cho_solve((c, low), bs)
    return 0.5 * bs @ x - jnp.sum(jnp.log(jnp.diag(c))) + 0.5 * jnp.sum(jnp.log(p)), (c, x)


def fit_scales(make, a0):
    f = jax.jit(jax.value_and_grad(lambda a: -logev(make(a))[0]))
    from scipy.optimize import minimize
    r = minimize(lambda a: tuple(np.asarray(v, float) for v in f(jnp.asarray(a))), np.asarray(a0, float),
                 jac=True, method="L-BFGS-B", options={"maxiter": 60})
    return r.x, -r.fun


priors = {"blr": gam2 * lam_blr}
a_iso, ev_iso = fit_scales(lambda a: jnp.full(L, jnp.exp(a[0])), [np.log(lam_blr * float(jnp.mean(gam2)))])
priors["iso"] = jnp.full(L, jnp.exp(a_iso[0]))
a_G, ev_G = fit_scales(lambda a: gam2 * jnp.exp(a)[G_idx], [np.log(lam_blr)] * 3)
priors["ardG"] = gam2 * jnp.exp(jnp.asarray(a_G))[G_idx]
a_I, ev_I = fit_scales(lambda a: jnp.exp(a)[G_idx], [a_iso[0]] * 3)
priors["ardI"] = jnp.exp(jnp.asarray(a_I))[G_idx]
summary = {"lam_blr": lam_blr, "log_scales": {"iso": a_iso.tolist(), "ardG": a_G.tolist(), "ardI": a_I.tolist()}}
post = {}
for m, p in priors.items():
    ev, (c, x) = logev(p)
    post[m] = (c, x, None); summary.setdefault("logev", {})[m] = float(ev)
nested = {}
for K in (2, 3, 4):
    cols = np.flatnonzero(body_col <= K)
    ev, (c, x) = logev(priors["blr"], jnp.asarray(cols))
    nested[K] = (c, x, cols); summary.setdefault("logev_nested", {})[f"K{K}"] = float(ev)
lw = np.array([summary["logev_nested"][f"K{K}"] for K in (2, 3, 4)])
w = np.exp(lw - lw.max()); w /= w.sum()
summary["bma_weights"] = w.tolist()
log("evidences", json.dumps(summary))


def place_sites(X, z):
    """Site-energy design rows (n, L) from compact descriptors X (n, D) and species z."""
    R = np.zeros((len(X), L))
    for s in range(NZ):
        m = z == s
        R[np.ix_(m, np.arange(s * nB, (s + 1) * nB))] = X[m, :nB]
        R[np.ix_(m, NZ * nB + np.arange(s * nP, (s + 1) * nP))] = X[m, nB:]
    return R


def var_rows(c, R, cols=None):
    Rr = R if cols is None else R[:, cols]
    v = solve_triangular(c, jnp.asarray(Rr).T, lower=True)
    return np.asarray(jnp.sum(v * v, 0))


out = {}
with highest_precision():
    for name, ds in (("test", d.ds_test), ("ood", d.ds_ood)):
        acc = {}
        X_all, _ = site_features(prob.model, gcfg, ds)
        for i in range(ds.n_batches):
            bt = jax.tree.map(lambda a, i=i: a[i], ds)
            live = np.asarray(bt.node_mask)
            lin = linear_rows(prob.model, gcfg, bt)[0]
            Fr = np.asarray(lin.F)[live].reshape(-1, L)                    # (3 n_live, L)
            Xs = np.asarray(X_all[i])[live]; zs = np.asarray(bt.node_z)[live]
            Er = place_sites(Xs, zs)
            for m, (c, _x, _) in post.items():
                acc.setdefault(f"sdF_{m}", []).append(np.sqrt(var_rows(c, Fr).reshape(-1, 3).sum(1)))
                acc.setdefault(f"sdE_{m}", []).append(np.sqrt(var_rows(c, Er)))
            for K, (c, x, cols) in nested.items():
                FK = (Fr[:, cols] @ np.asarray(x)).reshape(-1, 3)
                acc.setdefault(f"F_K{K}", []).append(FK)
                acc.setdefault(f"varF_K{K}", []).append(var_rows(c, Fr, cols).reshape(-1, 3))
        for k, v in acc.items():
            out[f"{name}/{k}"] = np.concatenate(v)
        FK = np.stack([out[f"{name}/F_K{K}"] for K in (2, 3, 4)])            # (3, n, 3)
        VK = np.stack([out[f"{name}/varF_K{K}"] for K in (2, 3, 4)])
        Fbar = np.tensordot(w, FK, 1)
        out[f"{name}/bma_sdF"] = np.sqrt(np.tensordot(w, VK + (FK - Fbar) ** 2, 1).sum(1))
        out[f"{name}/dF43"] = np.linalg.norm(FK[2] - FK[1], axis=1)
        out[f"{name}/dF42"] = np.linalg.norm(FK[2] - FK[0], axis=1)
        log(name, "done", len(out[f"{name}/dF43"]), "atoms")
    big = build_dataset(load_configs(f"{data}/big.xyz", "mace_energy", "mace_force", "mace_virial"),
                        meta, d.E0, 1)
    X_all, _ = site_features(prob.model, gcfg, big)
    acc = {}
    for i in range(big.n_batches):
        live = np.asarray(big.node_mask[i])
        Er = place_sites(np.asarray(X_all[i])[live], np.asarray(big.node_z[i])[live])
        for m, (c, _x, _) in post.items():
            acc.setdefault(f"sdE_{m}", []).append(np.sqrt(var_rows(c, Er)))
    for k, v in acc.items():
        out[f"big/{k}"] = np.concatenate(v)
    log("big done", len(out["big/sdE_blr"]), "atoms")
np.savez(out_path, **out)
json.dump(summary, open(out_path.replace(".npz", ".json"), "w"), indent=1)
log("wrote", out_path)
