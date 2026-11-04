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

Joint-E0 columns (k per species, fixed prior precision p0 independent of sigma_c) are eliminated exactly by a
k x k Schur complement in the SVD basis (`log_marginal_likelihood_svd`): O(L k^2) per evaluation.

Only the MAP uses this: the posterior at the MAP is still the QR form's (objective.posterior_from_qr), so the
saved model is computed as before.  Not representable here, and left to the QR form (`svd_applicable`):
independent noises (three Gram terms with no common basis) and QRStats from float32 rows (their residual must
come from the factors, rho_q)."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


SVD_DRIVER = "gesdd"       # divide and conquer: 1446 s against gesvd's 6403 s at L = 19,636 (48 cores), same MAP
#                            to 9 digits.  On spectra decaying past 1e-16, or exactly rank deficient, it loses up to
#                            ~2e-4 of y'y / sigma^2 (gesvd: roundoff; test_lml_svd_stability), far above
#                            mapfit.SVD_CHECK_TOL, so the QR check at the MAP catches it and the MAP finishes on QR.


class SVDStats(NamedTuple):
    s2: jnp.ndarray       # (L,) squared singular values of R_1 D^-1 (R_1: the smoothness-prior columns)
    chat: jnp.ndarray     # (L + k,) U^T c, U the full left singular basis
    P: jnp.ndarray        # (L + k, k) U^T R_2 (R_2: the k joint-E0 columns); (L + k, 0) without them
    p0: jnp.ndarray       # (k,) the joint-E0 prior precisions (fixed, independent of sigma_c)
    yy: jnp.ndarray       # sum of the weighted targets' squares (all quantities)
    n: jnp.ndarray        # live rows (all quantities)
    logw: jnp.ndarray     # sum of the log row weights


def svd_applicable(prob, qs, tied):
    """(ok, reason): whether the SVD form represents this problem's evidence exactly."""
    if not tied:
        return False, "the noises are not shared (--noise per-quantity)"
    if getattr(qs, "rho_E", None) is not None:
        return False, "float32-row statistics need the residual from the factors"
    return True, ""


def svd_statistics(qs, prob, driver=None):
    """SVDStats of QRStats `qs`: the three quantities' factors merged by one stacked QR (host LAPACK), then
    the SVD of R_1 D^-1 by LAPACK `driver` ("gesdd", divide and conquer, or "gesvd", QR iteration; default
    SVD_DRIVER), its full left basis U projecting c and the joint-E0 columns R_2.  Peak memory ~3 (L+k)^2
    doubles beyond qs."""
    import scipy.linalg as sl

    from .stats import _host_merge
    R, c = np.asarray(qs.R_E, np.float64), np.asarray(qs.c_E, np.float64)
    for q in "FV":
        R, c = _host_merge(R, c, np.asarray(getattr(qs, f"R_{q}"), np.float64),
                           np.asarray(getattr(qs, f"c_{q}"), np.float64))
    gamma = np.asarray(prob.gamma, np.float64)
    L = gamma.shape[0]
    p0 = np.zeros(0) if getattr(prob, "e0_prec", None) is None else np.asarray(prob.e0_prec, np.float64)
    R2 = R[:, L:].copy()
    W = R[:, :L] / gamma[None, :]
    del R
    U, s, _ = sl.svd(W, full_matrices=True, overwrite_a=True, check_finite=False,
                     lapack_driver=SVD_DRIVER if driver is None else driver)
    chat, P = U.T @ c, U.T @ R2
    del U
    tot = lambda k: sum(getattr(qs, f"{k}_{q}") for q in "EFV")                    # noqa: E731
    return SVDStats(jnp.asarray(s ** 2), jnp.asarray(chat), jnp.asarray(P), jnp.asarray(p0),
                    jnp.asarray(tot("yy")), jnp.asarray(tot("n")), jnp.asarray(tot("logw")))


def log_marginal_likelihood_svd(theta, sv):
    """log_marginal_likelihood_qr at shared noise sigma = exp(theta.log_sigma_F), from SVDStats: O(L k^2).

    With d_i = s_i^2/sigma^2 + 1/sigma_c^2 (s_i = 0 past the first L, where U spans R_1's complement) and
    w_i = (1/sigma_c^2) / (sigma^2 d_i), the k joint-E0 columns are eliminated exactly by the Schur complement
    S = diag(p0) + P^T diag(w) P with r = P^T diag(w) chat:
        log det(G + Lambda) = 2 sum log gamma + sum_{i<L} log d_i + log det S,
        v^T v = sum_{i<L} s_i^2 chat_i^2 / (sigma^4 d_i) + r^T S^-1 r."""
    ls, lc = theta.log_sigma_F, theta.log_sigma_c
    s2n, pc = jnp.exp(-2.0 * ls), jnp.exp(-2.0 * lc)            # 1 / sigma^2, 1 / sigma_c^2
    L, Lk, k = sv.s2.shape[0], sv.chat.shape[0], sv.p0.shape[0]
    s2 = jnp.concatenate([sv.s2, jnp.zeros(Lk - L)])
    d = s2 * s2n + pc
    vv = jnp.sum((s2 * sv.chat ** 2 / d)[:L]) * s2n ** 2
    logdet = jnp.sum(jnp.log(d[:L]))
    if k:
        w = pc * s2n / d
        S = jnp.diag(sv.p0) + (sv.P * w[:, None]).T @ sv.P
        r = sv.P.T @ (w * sv.chat)
        Lc = jnp.linalg.cholesky(S)
        z = jax.scipy.linalg.solve_triangular(Lc, r, lower=True)
        vv = vv + z @ z
        logdet = logdet + 2.0 * jnp.sum(jnp.log(jnp.diag(Lc))) - jnp.sum(jnp.log(sv.p0))
    return (-0.5 * (sv.yy * s2n - vv)
            - 0.5 * (logdet + 2.0 * L * lc - sv.logw + sv.n * 2.0 * ls)
            - 0.5 * sv.n * jnp.log(2.0 * jnp.pi))
