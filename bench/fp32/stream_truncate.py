"""Plan step 3, item 1: numerical-null-space truncation of the streamed factors.

float32 rounding lifts the exactly singular directions of the design (basis combinations no
training data constrains) to ~eps32 x the largest singular value; where the prior is weak the
evidence reads that as data and sigma_c moves (o4d12: log sigma_c by -0.55 from float32 rows
alone).  Here each quantity's factor R_q is truncated after streaming, in float64: its SVD
U s V^T keeps s > rel * s_max, R_q -> diag(s_kept) V_kept^T, c_q -> (U^T c_q)_kept, and the
dropped target components join the residual rho_q.  The same truncation is applied to every
precision, so float32 and float64 should then agree; the float64 reference is also compared
untruncated, which is what truncation itself costs.

    uv run --no-sync python bench/fp32/stream_truncate.py o3d12 o4d12 o4d16 > out.json
"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as so

from stream_analyse import FIT, KEYS, factors, lml, mean


def truncate(F, rel):
    out = {}
    for q, f in F.items():
        U, s, Vt = np.linalg.svd(np.asarray(f["R"]))
        keep = s > rel * s[0]
        uc = U.T @ np.asarray(f["c"])
        out[q] = dict(f, R=jnp.asarray(s[keep, None] * Vt[keep]), c=jnp.asarray(uc[keep]),
                      rho=float(np.sqrt(f["rho"] ** 2 + np.sum(uc[~keep] ** 2))), dropped=int((~keep).sum()))
    return out


def fit(F, gamma, x0):
    f = jax.jit(jax.value_and_grad(lambda x: -lml(x, F, gamma, "rho")))
    opt = so.minimize(lambda x: tuple(np.asarray(t, np.float64) for t in f(jnp.asarray(x))), x0, jac=True,
                      method="L-BFGS-B", options=dict(maxiter=200))
    return opt.x, float(-opt.fun), bool(opt.success)


def analyse(m, rels=(1e-7, 1e-6, 1e-5)):
    th = json.load(open(FIT / m / "theta_map.json"))
    gamma = jnp.asarray(np.load(FIT / m / "model.npz")["gamma"])
    x0 = np.array([th[k] for k in KEYS])
    raw = {p: factors(m, p)[0] for p in ("f64", "f32rows", "f32")}
    xr, lr, _ = fit(raw["f64"], gamma, x0)                     # untruncated float64: the reference
    mu_r, _ = mean(jnp.asarray(xr), raw["f64"], gamma)
    ref = raw["f64"]

    def pred(x, F):
        mu, _ = mean(jnp.asarray(x), F, gamma)
        return {q: float(jnp.linalg.norm(ref[q]["R"] @ (mu - mu_r)) / np.exp(xr[1 + i]) / np.sqrt(ref[q]["n"]))
                for i, q in enumerate("EFV")}
    res = {"model": m, "theta_ref": xr.tolist(), "lml_ref": lr}
    for rel in rels:
        for p in ("f64", "f32rows", "f32"):
            F = truncate(raw[p], rel)
            x, lv, ok = fit(F, gamma, x0)
            res[f"{rel:g}/{p}"] = dict(dtheta=(x - xr).tolist(), lml=lv, converged=ok, pred_change_in_sigma=pred(x, F),
                                        dropped={q: F[q]["dropped"] for q in "EFV"})
    return res


if __name__ == "__main__":
    print(json.dumps([analyse(m) for m in sys.argv[1:]], indent=1))
