"""Joint type-II ML over (log sigma_E, log sigma_F, log sigma_V, a_2, a_3, a_4) vs ARD-after-MAP, and a
Laplace diagnostic of hyperparameter uncertainty, on bench365 (linear ACE, L = 15035).

For the linear arm the weighted design rows do not depend on the hyperparameters, so one statistics
pass gives G_q, b_q, y^T y_q, n_q and the evidence is exact for any h:

    log p(D|h) = -1/2 sum_q yy_q / s_q^2 + 1/2 b^T A^-1 b - 1/2 log|A| + 1/2 log|Lambda| - sum_q n_q log s_q
    M = sum_q G_q / s_q^2,  b = sum_q b_q / s_q^2,  Lambda = Gamma^2 exp(a_k(j)),  A = M + Lambda

Models (each with its OWN posterior mean, so errors are per model):
  blr     sigma_q = linear MAP, Lambda = Gamma^2 / sigma_c^2
  ardG    sigma_q = linear MAP, a_k by evidence                (ARD after MAP)
  ardJ    all six by joint evidence                            (type-II ML)
  ardJm   ardJ with hyperparameters marginalised over the Laplace approximation (n_draws draws)
Outputs per atom (test, ood): F_<m> (predicted forces), sdF_<m>; summary JSON with evidences,
fitted scales, the Laplace std of h and the ardJm / ardJ variance ratio.

    python bayes_joint.py <data_dir> <theta_map.json> <out.npz> [n_draws]
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
from ace_jax.fit.hypers import Hypers                                     # noqa: E402
from ace_jax.fit.pipeline import FitConfig, load_fit_data                 # noqa: E402
from ace_jax.fit.pipeline.problem import build_problem                    # noqa: E402
from ace_jax.fit.rows import linear_rows                                  # noqa: E402
from ace_jax.fit.stats import sufficient_statistics                       # noqa: E402

data, theta_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
n_draws = int(sys.argv[4]) if len(sys.argv) > 4 else 6
T0 = time.time()
log = lambda *s: print(f"[{time.time() - T0:6.0f}s]", *s, flush=True)

cfg = FitConfig(model=f"{data}/cantor_embed_d16_deg10.npz", arm="linear", r0=2.5, batch=4, e0="lsq",
                energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial").validate()
d = load_fit_data(cfg, train=f"{data}/train.xyz", test=f"{data}/test.xyz", ood=f"{data}/ood.xyz")
bp = build_problem(cfg, d)
prob, gcfg = bp.prob, bp.gpcfg
theta = Hypers(**json.load(open(theta_path)))
L, nB, nP, NZ = gcfg.len_basis, gcfg.n_B, gcfg.n_pair, gcfg.NZ
order_B = np.array([len(x) for x in d.meta["nnll"]])
body_col = np.concatenate([np.tile(order_B + 1, NZ), np.tile(np.full(nP, 2), NZ)])
assert len(body_col) == L
G_idx = jnp.asarray(body_col - 2)

with highest_precision():
    st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, d.ds_train)
log("stats done")
Gq = [getattr(st, f"G_{q}") for q in "EFV"]
bq = [getattr(st, f"b_{q}") for q in "EFV"]
yy = jnp.asarray([float(getattr(st, f"yy_{q}")) for q in "EFV"])
nq = jnp.asarray([float(getattr(st, f"n_{q}")) for q in "EFV"])
gam2 = jnp.asarray(prob.gamma) ** 2
ls_map = np.array([float(getattr(theta, f"log_sigma_{q}")) for q in "EFV"])
a_blr = float(-2 * theta.log_sigma_c)                            # log(1/sigma_c^2)


def parts(h):
    ls, a = h[:3], h[3:]
    w = jnp.exp(-2 * ls)
    M = w[0] * Gq[0] + w[1] * Gq[1] + w[2] * Gq[2]
    b = w[0] * bq[0] + w[1] * bq[1] + w[2] * bq[2]
    lam = gam2 * jnp.exp(a)[G_idx]
    return M, b, lam, ls, w


def logev(h):
    M, b, lam, ls, w = parts(h)
    c, low = cho_factor(M + jnp.diag(lam), lower=True)
    x = cho_solve((c, low), b)
    return (-0.5 * jnp.sum(yy * w) + 0.5 * b @ x - jnp.sum(jnp.log(jnp.diag(c)))
            + 0.5 * jnp.sum(jnp.log(lam)) - jnp.sum(nq * ls))


vg = jax.jit(jax.value_and_grad(logev))


def maximise(h0, free):
    from scipy.optimize import minimize
    h0 = np.asarray(h0, float); free = np.asarray(free)
    def f(z):
        h = h0.copy(); h[free] = z
        v, g = vg(jnp.asarray(h))
        return -float(v), -np.asarray(g)[free]
    r = minimize(f, h0[free], jac=True, method="L-BFGS-B", options={"maxiter": 200})
    h = h0.copy(); h[free] = r.x
    return h, -r.fun, r


h_blr = np.concatenate([ls_map, [a_blr] * 3])
h_ardG, ev_ardG, rG = maximise(h_blr, [3, 4, 5])
h_ardJ, ev_ardJ, rJ = maximise(h_ardG, [0, 1, 2, 3, 4, 5])
ev_blr = float(vg(jnp.asarray(h_blr))[0])
log("evidence blr", ev_blr, "ardG", ev_ardG, "ardJ", ev_ardJ, "| ardJ h", np.round(h_ardJ, 3), rJ.message)

# Laplace: Hessian of log p at h_ardJ by central differences of the exact gradient
eps = 1e-3
H = np.zeros((6, 6))
for i in range(6):
    e = np.zeros(6); e[i] = eps
    H[:, i] = (np.asarray(vg(jnp.asarray(h_ardJ + e))[1]) - np.asarray(vg(jnp.asarray(h_ardJ - e))[1])) / (2 * eps)
H = 0.5 * (H + H.T)
cov_h = np.linalg.inv(-H)
std_h = np.sqrt(np.clip(np.diag(cov_h), 0, None))
log("Laplace std of h", np.round(std_h, 5), "eigs(-H)", np.round(np.linalg.eigvalsh(-H), 2))


def posterior(h):
    M, b, lam, _, _ = parts(jnp.asarray(h))
    c, low = cho_factor(M + jnp.diag(lam), lower=True)
    return c, cho_solve((c, low), b)


rng = np.random.default_rng(0)
draws = rng.multivariate_normal(h_ardJ, cov_h, size=n_draws)
models = {"blr": posterior(h_blr), "ardG": posterior(h_ardG), "ardJ": posterior(h_ardJ)}
dposts = [posterior(hd) for hd in draws]
out = {}
with highest_precision():
    for name, ds in (("test", d.ds_test), ("ood", d.ds_ood)):
        acc = {}
        for i in range(ds.n_batches):
            bt = jax.tree.map(lambda a: a[i], ds)
            live = np.asarray(bt.node_mask)
            Fr = jnp.asarray(np.asarray(linear_rows(prob.model, gcfg, bt)[0].F)[live].reshape(-1, L))
            for m, (c, x) in models.items():
                acc.setdefault(f"F_{m}", []).append(np.asarray(Fr @ x).reshape(-1, 3))
                v = solve_triangular(c, Fr.T, lower=True)
                acc.setdefault(f"varF_{m}", []).append(np.asarray(jnp.sum(v * v, 0)).reshape(-1, 3))
            Fd, Vd = [], []
            for c, x in dposts:
                Fd.append(np.asarray(Fr @ x).reshape(-1, 3))
                v = solve_triangular(c, Fr.T, lower=True)
                Vd.append(np.asarray(jnp.sum(v * v, 0)).reshape(-1, 3))
            Fd, Vd = np.stack(Fd), np.stack(Vd)
            acc.setdefault("F_ardJm", []).append(Fd.mean(0))
            acc.setdefault("varF_ardJm", []).append(Vd.mean(0) + Fd.var(0))
        for k, v in acc.items():
            out[f"{name}/{k}"] = np.concatenate(v)
        for m in ("blr", "ardG", "ardJ", "ardJm"):
            out[f"{name}/sdF_{m}"] = np.sqrt(out[f"{name}/varF_{m}"].sum(1))
        log(name, "done")
summary = {"logev": {"blr": ev_blr, "ardG": float(ev_ardG), "ardJ": float(ev_ardJ)},
           "h": {"blr": h_blr.tolist(), "ardG": h_ardG.tolist(), "ardJ": h_ardJ.tolist()},
           "h_names": ["log_sigma_E", "log_sigma_F", "log_sigma_V", "a_2body", "a_3body", "a_4body"],
           "laplace_std_h": std_h.tolist(),
           "ardJm_over_ardJ_var_median": {k: float(np.median(out[f"{k}/sdF_ardJm"] ** 2 / out[f"{k}/sdF_ardJ"] ** 2))
                                          for k in ("test", "ood")}}
np.savez(out_path, **out)
json.dump(summary, open(out_path.replace(".npz", ".json"), "w"), indent=1)
log("summary", json.dumps(summary))
