"""Plan step 3, item 1 analysis (float64, CPU): the LML, theta_MAP and posterior mean from the
streamed factors stream_qr.py saved, per precision, against the float64 reference.

Two LML forms:
  library  objective.log_marginal_likelihood_qr's: y'y - v'v, with y'y accumulated separately
  rho      the residual carried in the factor: rho_q^2 / sigma_q^2 summed, plus the residual of
           the reduced system from a QR with its target as an extra column (no difference of
           large numbers anywhere)

theta = (log sigma_c, log sigma_E, log sigma_F, log sigma_V) is optimised (L-BFGS-B, no
hyperprior) from the fit's theta_map.json, for each precision and LML form.  Predictions are
compared through the reference factors: |R_q,ref (mu - mu_ref)| / sigma_q / sqrt(n_q) is the RMS
change of the q predictions in units of their noise, against the posterior std in the same units.

    uv run --no-sync python bench/fp32/stream_analyse.py o3d12 o4d12 o4d16 > out.json
"""
import json
import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.optimize as so  # noqa: E402
from jax.scipy.linalg import solve_triangular  # noqa: E402

RUNS = Path("/storage/eng/essswb/projects/ace-jax/runs")
FIT = Path(os.environ.get("FP32_FIT", RUNS / "accuracy_cost/ace_per_quantity_noise"))   # <FIT>/<m>/{model.npz,theta_map.json}
DUMP = Path(os.environ.get("FP32_DUMP", RUNS / "fp32_step3"))                            # <DUMP>/<m>_<prec>.npz
KEYS = ("log_sigma_c", "log_sigma_E", "log_sigma_F", "log_sigma_V")


def factors(m, prec):
    z = np.load(DUMP / f"{m}_{prec}.npz")
    out = {}
    for q in "EFV":
        Rt = z[f"Rt_{q}"]
        s = np.where(np.diag(Rt) < 0, -1.0, 1.0)                    # positive diagonal (rows' sign)
        Rt = Rt * s[:, None]
        out[q] = dict(R=jnp.asarray(Rt[:-1, :-1]), c=jnp.asarray(Rt[:-1, -1]), rho=float(abs(Rt[-1, -1])),
                      yy=float(z[f"yy_{q}"]), n=float(z[f"n_{q}"]), logw=float(z[f"logw_{q}"]))
    return out, json.loads(str(z["info"]))


def reduced(x, F, gamma):
    """QR of [diag(sqrt(lin)); R_q / sigma_q | 0; c_q / sigma_q]: (R_S, v, r_tilde, lin)."""
    lin = gamma ** 2 * jnp.exp(-2.0 * x[0])
    s = {q: jnp.exp(x[1 + i]) for i, q in enumerate("EFV")}
    A = jnp.concatenate([jnp.diag(jnp.sqrt(lin))] + [F[q]["R"] / s[q] for q in "EFV"], 0)
    y = jnp.concatenate([jnp.zeros_like(lin)] + [F[q]["c"] / s[q] for q in "EFV"])
    Rt = jnp.linalg.qr(jnp.concatenate([A, y[:, None]], 1), mode="r")
    L = lin.shape[0]
    return Rt[:L, :L], Rt[:L, L], Rt[L, L], lin


def lml(x, F, gamma, form):
    R, v, rt, lin = reduced(x, F, gamma)
    s2 = {q: jnp.exp(2.0 * x[1 + i]) for i, q in enumerate("EFV")}
    if form == "rho":
        resid = rt ** 2 + sum(F[q]["rho"] ** 2 / s2[q] for q in "EFV")
    else:                                                          # the library's y'y - v'v
        resid = sum(F[q]["yy"] / s2[q] for q in "EFV") - v @ v
    logtau = sum(F[q]["logw"] - F[q]["n"] * jnp.log(s2[q]) for q in "EFV")
    N = sum(F[q]["n"] for q in "EFV")
    logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(R))))
    return -0.5 * resid - 0.5 * (logdet - jnp.sum(jnp.log(lin)) - logtau) - 0.5 * N * jnp.log(2 * jnp.pi)


def mean(x, F, gamma):
    R, v, _, _ = reduced(x, F, gamma)
    return solve_triangular(R, v, lower=False), R


def analyse(m):
    th = json.load(open(FIT / m / "theta_map.json"))
    gamma = jnp.asarray(np.load(FIT / m / "model.npz")["gamma"])
    x0 = np.array([th[k] for k in KEYS])
    ref, _ = factors(m, "f64")
    res = {"model": m, "L": int(gamma.shape[0])}
    fits = {}
    for prec in ("f64", "f32rows", "f32"):
        try:
            F, info = factors(m, prec)
        except FileNotFoundError:
            continue
        res.setdefault("info", {})[prec] = {k: info[k] for k in ("seconds", "n_batches", "n_configs")}
        for form in ("rho", "library"):
            f = jax.jit(jax.value_and_grad(lambda x: -lml(x, F, gamma, form)))  # noqa: B023 (used this iteration)
            v0 = float(-f(jnp.asarray(x0))[0])
            opt = so.minimize(lambda x: tuple(np.asarray(t, np.float64) for t in f(jnp.asarray(x))), x0,  # noqa: B023
                              jac=True, method="L-BFGS-B", options=dict(maxiter=200))
            fits[(prec, form)] = (opt.x, F)
            res[f"{prec}/{form}"] = dict(lml_at_theta_map=v0, lml_max=float(-opt.fun), theta=dict(zip(KEYS, opt.x.tolist())),
                                         converged=bool(opt.success), finite=bool(np.isfinite(v0)))
    xr, Fr = fits[("f64", "rho")]
    mu_r, Rs_r = mean(jnp.asarray(xr), Fr, gamma)
    for (prec, form), (x, F) in fits.items():
        mu, _ = mean(jnp.asarray(x), F, gamma)
        d = {}
        for i, q in enumerate("EFV"):
            sq = np.exp(xr[1 + i])
            d[q] = float(jnp.linalg.norm(ref[q]["R"] @ (mu - mu_r)) / sq / np.sqrt(ref[q]["n"]))
        res[f"{prec}/{form}"]["pred_change_in_sigma"] = d
        res[f"{prec}/{form}"]["dtheta"] = (np.asarray(x) - xr).tolist()
    # posterior std of the predictions, same units: |R_S^-T (R_q phi)| averaged -- trace form
    for i, q in enumerate("EFV"):
        sq = np.exp(xr[1 + i])
        Z = solve_triangular(Rs_r, ref[q]["R"].T, trans="T", lower=False)          # R_S^-T R_q^T
        res.setdefault("posterior_std_in_sigma", {})[q] = float(jnp.linalg.norm(Z) / sq / np.sqrt(ref[q]["n"]))
    return res


if __name__ == "__main__":
    print(json.dumps([analyse(m) for m in sys.argv[1:]], indent=1))
