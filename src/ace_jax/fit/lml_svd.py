"""The linear arm's evidence from one SVD (`aj fit --lml-solver svd`): O(L) per LML evaluation instead of a QR.

With shared noise (sigma_E = sigma_F = sigma_V = sigma; the loss weights already scale the rows) and the
smoothness prior Lambda = D^2 / sigma_c^2 (D = diag(gamma), no joint-E0 columns), merge the per-quantity QR
factors into one (R, c) and take the SVD W = R D^-1 = U diag(s) V^T.  Then

    G + Lambda = D V diag(s^2 / sigma^2 + 1 / sigma_c^2) V^T D,        b = D V diag(s) U^T c / sigma^2,

so with chat = U^T c the evidence of objective.log_marginal_likelihood_qr is

    -1/2 [yy / sigma^2 - sum s^2 chat^2 / (sigma^4 (s^2/sigma^2 + 1/sigma_c^2))]
    -1/2 [sum log(s^2/sigma^2 + 1/sigma_c^2) + 2 L log sigma_c - sum log w + N log sigma^2] - N/2 log 2 pi

(the sum log gamma terms cancel).  One SVD of an L x L matrix (~5x one QR of the stacked system) replaces
the QR per evaluation; the gradient and the Newton polish's Hessian are O(L) too.  The SVD acts on R D^-1,
not on the Gram, so it keeps the QR form's kappa(Phi) (not kappa(Phi)^2) accuracy.

Only the MAP uses this: the posterior at the MAP is still the QR form's (objective.posterior_from_qr), so the
saved model is computed as before.  Not representable here, and left to the QR form (`svd_applicable`):
independent noises (three Gram terms with no common basis), joint-E0 columns (a second, fixed prior scale)
and QRStats from float32 rows (their residual must come from the factors, rho_q)."""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class SVDStats(NamedTuple):
    s2: jnp.ndarray       # (L,) squared singular values of R D^-1
    chat2: jnp.ndarray    # (L,) (U^T c)^2
    yy: jnp.ndarray       # sum of the weighted targets' squares (all quantities)
    n: jnp.ndarray        # live rows (all quantities)
    logw: jnp.ndarray     # sum of the log row weights


def svd_applicable(prob, qs, tied):
    """(ok, reason): whether the SVD form represents this problem's evidence exactly."""
    if not tied:
        return False, "the noises are not shared (--noise per-quantity)"
    if getattr(prob, "e0_prec", None) is not None:
        return False, "joint-E0 columns carry a second, fixed prior scale"
    if getattr(qs, "rho_E", None) is not None:
        return False, "float32-row statistics need the residual from the factors"
    return True, ""


def svd_statistics(qs, prob):
    """SVDStats of QRStats `qs`: the three quantities' factors merged by one stacked QR (host LAPACK), then
    the SVD of R D^-1 (gesdd).  Peak memory ~3 L x L doubles beyond qs."""
    import scipy.linalg as sl

    from .stats import _host_merge
    R, c = np.asarray(qs.R_E, np.float64), np.asarray(qs.c_E, np.float64)
    for q in "FV":
        R, c = _host_merge(R, c, np.asarray(getattr(qs, f"R_{q}"), np.float64),
                           np.asarray(getattr(qs, f"c_{q}"), np.float64))
    W = R / np.asarray(prob.gamma, np.float64)[None, :]
    del R
    U, s, _ = sl.svd(W, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver="gesdd")
    chat = U.T @ c
    del U
    tot = lambda k: sum(getattr(qs, f"{k}_{q}") for q in "EFV")                    # noqa: E731
    return SVDStats(jnp.asarray(s ** 2), jnp.asarray(chat ** 2), jnp.asarray(tot("yy")), jnp.asarray(tot("n")),
                    jnp.asarray(tot("logw")))


def log_marginal_likelihood_svd(theta, sv):
    """log_marginal_likelihood_qr at shared noise sigma = exp(theta.log_sigma_F), from SVDStats: O(L)."""
    ls, lc = theta.log_sigma_F, theta.log_sigma_c
    s2n, pc = jnp.exp(-2.0 * ls), jnp.exp(-2.0 * lc)            # 1 / sigma^2, 1 / sigma_c^2
    d = sv.s2 * s2n + pc
    vv = jnp.sum(sv.s2 * sv.chat2 / d) * s2n ** 2
    L = sv.s2.shape[0]
    return (-0.5 * (sv.yy * s2n - vv)
            - 0.5 * (jnp.sum(jnp.log(d)) + 2.0 * L * lc - sv.logw + sv.n * 2.0 * ls)
            - 0.5 * sv.n * jnp.log(2.0 * jnp.pi))
