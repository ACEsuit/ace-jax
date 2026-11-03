"""Accuracy of the SVD evidence (fit/lml_svd.py) against the QR form and an exact (60-digit, mpmath) reference
on adversarial synthetic problems: R with prescribed singular spectra down to 1e-17 and exact rank deficiency,
prior scales gamma over 8 decades (the smoothness prior's range), and hyperparameters from near interpolation
(sigma at its floor, a nearly flat prior) to prior-dominated.  Errors are measured relative to the terms the
LML cancels (y'y / sigma^2; test_lml_qr._scale); the SVD form must stay within 10x the QR form's error."""
import jax

jax.config.update("jax_enable_x64", True)
from types import SimpleNamespace  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ace_jax.fit.hypers import Hypers  # noqa: E402
from ace_jax.fit.lml_svd import log_marginal_likelihood_svd, svd_statistics  # noqa: E402
from ace_jax.fit.objective import log_marginal_likelihood_qr  # noqa: E402
from ace_jax.fit.stats import QRStats  # noqa: E402

mp = pytest.importorskip("mpmath")
L = 36


def _problem(spectrum, gamma_decades, seed=0, e0_prec=None):
    """e0_prec (k,): k extra joint-E0 columns (one per species) with these fixed prior precisions."""
    g = np.random.default_rng(seed)
    k = 0 if e0_prec is None else len(e0_prec)
    U, _ = np.linalg.qr(g.normal(size=(L, L)))
    V, _ = np.linalg.qr(g.normal(size=(L, L)))
    gamma = np.logspace(0, gamma_decades, L)[g.permutation(L)]
    W = (U * spectrum) @ V.T                       # the prescribed spectrum of R D^-1
    A = W * gamma[None, :]
    if k:                                          # E0 columns: per-species atom counts, large and correlated
        A = np.concatenate([A, np.abs(g.normal(size=(L, k))) * np.linalg.norm(A) / np.sqrt(L)], 1)
    R = np.linalg.qr(np.concatenate([A, np.zeros((k, L + k))]) if k else A, mode="r")
    R *= np.sign(np.diag(R))[:, None]
    c = g.normal(size=L + k) * np.linalg.norm(R) / np.sqrt(L + k)
    yy = float(c @ c) * (1.0 + 1e-3)               # a small residual outside the range
    z = jnp.zeros((L + k, L + k)), jnp.zeros(L + k)
    qs = QRStats(jnp.asarray(R), *(z[0],) * 2, jnp.asarray(c), *(z[1],) * 2, jnp.asarray(yy), jnp.zeros(()),
                 jnp.zeros(()), jnp.asarray(900.0), jnp.zeros(()), jnp.zeros(()), jnp.asarray(-3.0), jnp.zeros(()),
                 jnp.zeros(()), None, None, None)
    prob = SimpleNamespace(gamma=jnp.asarray(gamma), e0_prec=None if k == 0 else jnp.asarray(e0_prec))
    return qs, prob, R, c, gamma, yy


def _exact(R, c, gamma, yy, n, logw, ls, lc, e0_prec=None):
    mp.mp.dps = 60
    s2, pc = mp.e ** (-2 * mp.mpf(ls)), mp.e ** (-2 * mp.mpf(lc))
    Rm, cm = mp.matrix(R.tolist()), mp.matrix(c.tolist())
    P = len(c)
    prior = [mp.mpf(gamma[i]) ** 2 * pc for i in range(L)] + [mp.mpf(p) for p in (e0_prec if e0_prec is not None else [])]
    A = Rm.T * Rm * s2
    for i in range(P):
        A[i, i] += prior[i]
    b = Rm.T * cm * s2
    Lc = mp.cholesky(A)
    v = mp.lu_solve(Lc, b)
    logdet = 2 * mp.fsum(mp.log(Lc[i, i]) for i in range(P))
    loglin = mp.fsum(mp.log(p) for p in prior)
    vv = mp.fsum(v[i] ** 2 for i in range(P))
    val = (-mp.mpf(0.5) * (mp.mpf(yy) * s2 - vv) - mp.mpf(0.5) * (logdet - loglin - (logw + n * mp.log(s2)))
           - mp.mpf(0.5) * n * mp.log(2 * mp.pi))
    return float(val)


def _theta(ls, lc):
    return Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=0.0, log_eps=0.0, log_rho=0.0, log_sigma_c=lc,
                  log_sigma_E=ls, log_sigma_F=ls, log_sigma_V=ls)


SPECTRA = {"decay-1e-14": np.logspace(0, -14, L), "decay-1e-17": np.logspace(0, -17, L),
           "rank-deficient": np.concatenate([np.logspace(0, -6, L - 6), np.zeros(6)])}
HYPERS = [(np.log(1e-4), np.log(1e6)), (np.log(1e-2), np.log(1e2)), (np.log(1.0), np.log(1e-3)),
          (np.log(1e-6), np.log(1e8))]


@pytest.mark.parametrize("driver", ["gesvd"])        # gesdd fails decay-1e-17 / rank-deficient by ~1e3x (SVD_DRIVER)
@pytest.mark.parametrize("spec", sorted(SPECTRA))
@pytest.mark.parametrize("decades", [2, 8])
def test_svd_evidence_as_accurate_as_qr(driver, spec, decades):
    qs, prob, R, c, gamma, yy = _problem(SPECTRA[spec], decades)
    sv = svd_statistics(qs, prob, driver=driver)
    for ls, lc in HYPERS:
        th = _theta(ls, lc)
        ref = _exact(R, c, gamma, yy, 900.0, -3.0, ls, lc)
        scale = yy * np.exp(-2.0 * ls)
        e_qr = abs(float(log_marginal_likelihood_qr(th, qs, prob)) - ref) / scale
        e_svd = abs(float(log_marginal_likelihood_svd(th, sv)) - ref) / scale
        assert e_svd <= max(10.0 * e_qr, 1e-14), (spec, decades, ls, lc, e_svd, e_qr)


@pytest.mark.parametrize("spec", sorted(SPECTRA))
def test_svd_gradient_as_accurate_as_qr(spec):
    """d LML / d(log sigma, log sigma_c), against a 60-digit central difference of the exact LML."""
    qs, prob, R, c, gamma, yy = _problem(SPECTRA[spec], 8)
    sv = svd_statistics(qs, prob)
    for ls, lc in HYPERS[:3]:
        h = 1e-12
        fd = [(_exact(R, c, gamma, yy, 900.0, -3.0, ls + h * (k == 0), lc + h * (k == 1))
               - _exact(R, c, gamma, yy, 900.0, -3.0, ls - h * (k == 0), lc - h * (k == 1))) / (2 * h) for k in (0, 1)]
        x = jnp.array([ls, lc])
        g_qr = np.asarray(jax.grad(lambda y: log_marginal_likelihood_qr(_theta(y[0], y[1]), qs, prob))(x))
        g_svd = np.asarray(jax.grad(lambda y: log_marginal_likelihood_svd(_theta(y[0], y[1]), sv))(x))
        scale = yy * np.exp(-2.0 * ls)
        e_qr, e_svd = np.abs(g_qr - fd).max() / scale, np.abs(g_svd - fd).max() / scale
        assert e_svd <= max(10.0 * e_qr, 1e-12), (spec, ls, lc, e_svd, e_qr)


@pytest.mark.parametrize("p0", [1e16, 1e-6, 1.0])           # pinned E0, nearly flat E0 prior, unit
@pytest.mark.parametrize("spec", ["decay-1e-14", "rank-deficient"])
def test_svd_evidence_with_joint_e0_columns(p0, spec):
    """Joint-E0 columns (k per species, fixed prior precision, independent of sigma_c) are eliminated exactly by
    a k x k Schur complement in the SVD basis: still as accurate as the QR form."""
    e0 = np.full(3, p0)
    qs, prob, R, c, gamma, yy = _problem(SPECTRA[spec], 8, e0_prec=e0)
    sv = svd_statistics(qs, prob)
    for ls, lc in HYPERS:
        th = _theta(ls, lc)
        ref = _exact(R, c, gamma, yy, 900.0, -3.0, ls, lc, e0_prec=e0)
        scale = yy * np.exp(-2.0 * ls)
        e_qr = abs(float(log_marginal_likelihood_qr(th, qs, prob)) - ref) / scale
        e_svd = abs(float(log_marginal_likelihood_svd(th, sv)) - ref) / scale
        assert e_svd <= max(10.0 * e_qr, 1e-14), (p0, spec, ls, lc, e_svd, e_qr)
