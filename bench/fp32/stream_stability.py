"""Plan step 3, item 1: a stability test for a float32 fit, from its own statistics.

Rows rounded with relative error eps act like extra random rows of column norms eps |Phi_j|:
they lift the design's null and weak directions.  Whether that moves theta depends on the model
(prior strength there, LML flatness in sigma_c), not on a spectrum threshold.  So: append such
noise rows to a factor (R_q -> qr([R_q; D_q]), c_q -> with zero targets), refit theta, and
look at the spread over seeds.  Calibrated on the float64 factors (does injected noise
reproduce the measured float32 shift?) and applied to the float32 factors (would the test flag
the unstable fits from float32 statistics alone?).

    uv run --no-sync python bench/fp32/stream_stability.py o3d12 o4d12 o4d16 > out.json
"""
import json
import sys

import jax.numpy as jnp
import numpy as np

from stream_analyse import FIT, KEYS, factors, mean
from stream_truncate import fit

EPS = 2e-6            # float32 row error measured in step 1 (column-norm relative, F rows)


def inject(F, eps, rng, m_rows=None):
    out = {}
    for q, f in F.items():
        R = np.asarray(f["R"])
        L = R.shape[1]
        m = m_rows or L
        D = rng.standard_normal((m, L)) * (eps * np.linalg.norm(R, axis=0) / np.sqrt(m))
        Rt = np.linalg.qr(np.vstack([np.column_stack([R, np.asarray(f["c"])]),
                                     np.column_stack([D, np.zeros(m)])]), mode="r")
        out[q] = dict(f, R=jnp.asarray(Rt[:L, :L]), c=jnp.asarray(Rt[:L, L]),
                      rho=float(np.hypot(f["rho"], Rt[L, L])))
    return out


def analyse(m, seeds=3):
    th = json.load(open(FIT / m / "theta_map.json"))
    gamma = jnp.asarray(np.load(FIT / m / "model.npz")["gamma"])
    x0 = np.array([th[k] for k in KEYS])
    raw = {p: factors(m, p)[0] for p in ("f64", "f32rows", "f32")}
    ref = raw["f64"]
    xr, _, _ = fit(ref, gamma, x0)
    mu_r, R_S = mean(jnp.asarray(xr), ref, gamma)
    from jax.scipy.linalg import solve_triangular
    pstd = {q: float(jnp.linalg.norm(solve_triangular(R_S, ref[q]["R"].T, trans="T", lower=False))
                     / np.exp(xr[1 + i]) / np.sqrt(ref[q]["n"])) for i, q in enumerate("EFV")}

    def score(F, x, mu_base):
        mu, _ = mean(jnp.asarray(x), F, gamma)
        return max(float(jnp.linalg.norm(ref[q]["R"] @ (mu - mu_base)) / np.exp(xr[1 + i]) / np.sqrt(ref[q]["n"]))
                   / pstd[q] for i, q in enumerate("EFV"))
    res = {"model": m}
    for p in ("f64", "f32rows", "f32"):
        xb, _, _ = fit(raw[p], gamma, x0)
        mub, _ = mean(jnp.asarray(xb), raw[p], gamma)
        actual = dict(dlogsc=float(xb[0] - xr[0]), pred_over_std=score(raw[p], xb, mu_r))
        trials = []
        for s in range(seeds):
            F = inject(raw[p], EPS, np.random.default_rng(s))
            x, _, _ = fit(F, gamma, xb)
            trials.append(dict(dlogsc=float(x[0] - xb[0]), pred_over_std=score(F, x, mub)))
        res[p] = dict(actual_vs_f64=actual, injected_vs_self=trials)
    return res


if __name__ == "__main__":
    print(json.dumps([analyse(m) for m in sys.argv[1:]], indent=1))
