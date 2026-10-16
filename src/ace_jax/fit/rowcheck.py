"""The stability check of a fit from float32 design rows (FitConfig.row_precision "float32").

float32 rounding of the rows acts like extra random rows of column norms ~ROW_EPS |Phi_j|: it
lifts the design's exactly singular and weak directions (the radial-only ACE functions and the
pair basis are both sums of f(r_ij), so the basis has exact redundancies) to ~1e-8 of the
largest, where float64 holds them down to ~1e-17.  Where the prior is weak there the evidence
reads that as data and sigma_c moves.  Whether it does is a property of the fit (basis, data,
prior), not of a spectrum threshold, so the check measures it: append such noise rows to the
fit's own QR statistics, refit theta from the MAP with the same optimiser, and compare the
posterior means.  It costs one MAP refit on the (L x L) statistics, no pass over the data.

Injected noise is a proxy for the rounding, not a replica, so the scale has a grey zone.  On
GAP-18 Si and MACE-labelled Si (docs/dev/fp32-results.md, step 3) fits whose float32 predictions
moved by <= 0.25 posterior stds read <= 0.57; those that moved by 1.2-3.5 read 0.67-2.8.  Hence
three verdicts: "stable" (<= STABLE), "marginal" (logged: compare with a float64 fit before
relying on it) and "unstable" (> UNSTABLE, a warning; such fits are fragile in float64 too)."""
import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

ROW_EPS = 2e-6       # float32 rows' column-norm relative error, measured (F rows, step 1)
STABLE = 0.5         # max over E/F/V of (prediction change / posterior std): at or below, stable
UNSTABLE = 1.0       # above, unstable (a warning); between, marginal


def perturb(qs, eps=ROW_EPS, seed=0, n_cols=None):
    """QRStats with eps-sized random rows (zero targets) appended to each quantity's factor, in
    its first n_cols columns (default all): the joint-E0 columns after the readout are atom
    counts, exact in float32, and are left alone."""
    rng = np.random.default_rng(seed)
    out = {}
    for q in "EFV":
        R, c = np.asarray(getattr(qs, f"R_{q}"), np.float64), np.asarray(getattr(qs, f"c_{q}"), np.float64)
        L = R.shape[1]
        scale = eps * np.linalg.norm(R, axis=0) / np.sqrt(L)
        scale[L if n_cols is None else n_cols:] = 0.0
        D = rng.standard_normal((L, L)) * scale
        Rt = np.linalg.qr(np.vstack([np.column_stack([R, c]), np.column_stack([D, np.zeros(L)])]), mode="r")
        out[f"R_{q}"], out[f"c_{q}"] = jnp.asarray(Rt[:L, :L]), jnp.asarray(Rt[:L, L])
        rho = getattr(qs, f"rho_{q}")
        if rho is not None:                    # the appended rows' share of the residual joins rho
            out[f"rho_{q}"] = jnp.hypot(jnp.asarray(rho, jnp.float64), Rt[L, L])
    return qs._replace(**out)


def _prediction_change(theta, mu, mu_p, R_S, qs):
    """Per quantity: RMS change of the predictions, and the RMS posterior std, both in units of
    sigma_q, over the training rows (through their factor R_q)."""
    out = {}
    for q in "EFV":
        R, s, n = getattr(qs, f"R_{q}"), jnp.exp(getattr(theta, f"log_sigma_{q}")), getattr(qs, f"n_{q}")
        if float(n) == 0:
            continue
        change = float(jnp.linalg.norm(R @ (mu_p - mu)) / s / jnp.sqrt(n))
        std = float(jnp.linalg.norm(solve_triangular(R_S, R.T, trans="T", lower=False)) / s / jnp.sqrt(n))
        out[q] = {"change": change, "posterior_std": std, "ratio": change / std if std > 0 else float("inf")}
    return out


def row_rounding_check(cfg, d, b, obj, mf, eps=ROW_EPS, seed=0, log=print):
    """Refit the MAP on the fit's QR statistics perturbed at float32 row rounding and report
    how far the posterior mean moves.  Returns None when the objective has no QR statistics,
    else {"eps", "verdict", "max_ratio", "dtheta", "quantities": {q: change, posterior_std, ratio}};
    warns when unstable."""
    from .hypers import from_array, log_prior, to_array
    from .objective import log_marginal_likelihood_qr, posterior_from_qr
    from .paramset import tied_likelihood
    from .pipeline.mapfit import fit_map
    from .pipeline.objective import Objective
    qs = getattr(obj.lik, "qr_stats", None)
    if qs is None:
        return None
    prob = b.prob
    qs_p = perturb(qs, eps, seed, n_cols=prob.cfg.len_readout)
    tied = obj.tied
    free = None if tied is None else ~tied
    lml = jax.jit(lambda a: log_marginal_likelihood_qr(from_array(a), qs_p, prob))
    lik = tied_likelihood(lml) if tied is not None else lml
    vg = jax.jit(jax.value_and_grad(lambda a: lik(a) + log_prior(from_array(a), prob.prior, free)))
    obj_p = Objective(lik, vg, None, None, obj.prior_mu, {}, None, tied)
    cfg_p = dataclasses.replace(cfg, init=dict(mf.theta._asdict()), map_restarts=1, strict=False, map_polish="off")
    mf_p = fit_map(cfg_p, d, b, obj_p, log=lambda *a, **k: None)
    mu, L_S = posterior_from_qr(mf.theta, qs, prob)
    mu_p, _ = posterior_from_qr(mf_p.theta, qs_p, prob)
    quant = _prediction_change(mf.theta, mu, mu_p, L_S.T, qs)
    worst = max(v["ratio"] for v in quant.values())
    verdict = "stable" if worst <= STABLE else "marginal" if worst <= UNSTABLE else "unstable"
    res = {"eps": eps, "verdict": verdict, "max_ratio": worst, "stable_max": STABLE, "unstable_min": UNSTABLE,
           "dtheta": dict(zip(type(mf.theta)._fields,
                              (np.asarray(to_array(mf_p.theta)) - np.asarray(to_array(mf.theta))).tolist())),
           "quantities": quant}
    msg = (f"float32 rows: perturbing the statistics at float32 rounding moves the predictions by "
           f"{worst:.2f} posterior std (stable <= {STABLE}, unstable > {UNSTABLE}); "
           f"log_sigma_c {res['dtheta']['log_sigma_c']:+.3f}")
    if verdict == "stable":
        log(msg + ": stable")
    elif verdict == "marginal":
        log(msg + ": marginal; compare with a --row-precision float64 fit before relying on it")
    else:
        log("WARNING: " + msg + ": unstable; refit with --row-precision float64 (the fit is sensitive to "
            "its weakest directions, a stronger prior may help)")
        warnings.warn(msg + ". Unstable: refit with row_precision float64", UserWarning, stacklevel=2)
    return res
