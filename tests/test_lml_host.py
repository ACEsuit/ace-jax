"""The host evaluation of the linear arm's QR evidence (fit/lml_host.py): triangle-triangle tpqrt merges of
[diag(sqrt(lin)); R_q / sigma_q], one triangular inverse, and the gradient and exact Hessian in the four scales
(sigma_c, sigma_E, sigma_F, sigma_V) written out, instead of a dense QR of the stacked (4L x L) matrix and
autodiff through it.  It must be objective.log_marginal_likelihood_qr -- value, gradient, Hessian -- to
roundoff relative to the cancelled y'y / sigma^2, with and without joint-E0 columns and float32-row residuals."""
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ace_jax.fit.hypers import Hypers, to_array  # noqa: E402
from ace_jax.fit.objective import Problem, log_marginal_likelihood_qr  # noqa: E402
from ace_jax.fit.stats import QRStats  # noqa: E402


def _theta(**kw):
    d = dict(log_ell=0.1, log_A=-3.0, log_alpha=0.0, log_r0=0.85, log_eps=-1.2, log_rho=1.4,
             log_sigma_c=np.log(0.3), log_sigma_E=np.log(1e-3), log_sigma_F=np.log(2e-2), log_sigma_V=np.log(2e-2))
    d.update(kw)
    return Hypers(**d)


def _problem(L=40, n_e0=0, rho=False, seed=0):
    """Synthetic QR statistics: per quantity, the QR of weighted rows of very different scales and counts."""
    g = np.random.default_rng(seed)
    Lt = L + n_e0
    w = g.normal(size=Lt) * 0.3
    f = {}
    for q, m, s in (("E", 3 * Lt, 3.0), ("F", 8 * Lt, 0.4), ("V", 2 * Lt, 1.5)):
        P = g.normal(size=(m, Lt)) * s * np.exp(-0.15 * np.arange(Lt))         # decaying spectrum
        y = P @ w + 1e-2 * s * g.normal(size=m)
        Q, R = np.linalg.qr(P)
        c = Q.T @ y
        res = y - Q @ c
        f[q] = (R, c, float(y @ y), float(m), float(np.log(1.0 + 0.1 * m)), float(np.sqrt(res @ res)))
    qs = QRStats(*(jnp.asarray(f[q][0]) for q in "EFV"), *(jnp.asarray(f[q][1]) for q in "EFV"),
                 *(jnp.asarray(f[q][k]) for k in (2, 3, 4) for q in "EFV"),
                 *((jnp.asarray(f[q][5]) for q in "EFV") if rho else ()))
    gamma = jnp.asarray(0.5 + g.random(L))
    prob = Problem(None, None, SimpleNamespace(XM=np.zeros((0, 1))), None, gamma, None, e0_prec=jnp.full(n_e0, 1e16) if n_e0 else None,
                   lml_solver="qr")
    return qs, prob


def _scale(theta, qs):
    return float(sum(getattr(qs, f"yy_{q}") * np.exp(-2.0 * getattr(theta, f"log_sigma_{q}")) for q in "EFV"))


CASES = [dict(), dict(n_e0=3), dict(rho=True), dict(n_e0=2, rho=True)]
THETAS = [_theta(), _theta(log_sigma_c=np.log(1e3), log_sigma_E=np.log(1e-4), log_sigma_F=np.log(1e-3),
                           log_sigma_V=np.log(3e-4))]


@pytest.mark.parametrize("case", CASES, ids=["plain", "e0", "rho", "e0-rho"])
@pytest.mark.parametrize("ti", range(len(THETAS)))
def test_host_evidence_matches_dense_qr(case, ti):
    from ace_jax.fit.lml_host import HostQREvidence
    qs, prob = _problem(**case)
    theta = THETAS[ti]
    a = to_array(theta)
    dense = lambda x: log_marginal_likelihood_qr(Hypers(*x), qs, prob)        # noqa: E731
    ev = HostQREvidence(qs, prob)
    host = lambda x: ev.lml(Hypers(*x))                                      # noqa: E731
    scale = _scale(theta, qs)
    v0, v1 = float(dense(a)), float(jax.jit(host)(a))
    assert abs(v1 - v0) < 1e-11 * scale, (v0, v1, scale)
    g0, g1 = np.asarray(jax.grad(dense)(a)), np.asarray(jax.jit(jax.grad(host))(a))
    assert np.allclose(g1[:6], 0.0) and np.allclose(g0[:6], 0.0)            # kernel hypers: not read at M = 0
    np.testing.assert_allclose(g1, g0, rtol=1e-10, atol=1e-12 * scale)
    H0, H1 = np.asarray(jax.hessian(dense)(a)), np.asarray(jax.jit(jax.hessian(host))(a))
    np.testing.assert_allclose(H1, H0, rtol=1e-10, atol=1e-11 * np.abs(H0).max())
    hvp = jax.jit(lambda x, v: jax.jvp(jax.grad(host), (x,), (v,))[1])    # the Newton polish's HVP
    e = jnp.zeros(10).at[8].set(1.0)
    np.testing.assert_allclose(np.asarray(hvp(a, e)), H0[:, 8], rtol=1e-10, atol=1e-11 * np.abs(H0).max())


def test_host_evidence_vmaps_and_caches():
    from ace_jax.fit.lml_host import HostQREvidence
    qs, prob = _problem()
    ev = HostQREvidence(qs, prob)
    A = jnp.stack([to_array(t) for t in THETAS])
    v = jax.jit(jax.vmap(lambda x: ev.lml(Hypers(*x))))(A)
    for t, vi in zip(THETAS, np.asarray(v)):
        assert abs(vi - float(log_marginal_likelihood_qr(t, qs, prob))) < 1e-11 * _scale(t, qs)
    n = ev.n_factor
    jax.jit(jax.value_and_grad(lambda x: ev.lml(Hypers(*x))))(to_array(THETAS[0]))
    assert ev.n_factor - n == 1                     # value and gradient share one factorisation


def test_make_lml_takes_the_host_form_on_cpu(monkeypatch):
    """make_lml's QR branch uses the host evaluation on the CPU backend (objective.QR_EVAL 'auto'), and the
    dense form with QR_EVAL 'dense'."""
    from ace_jax.fit import objective
    qs, prob = _problem()
    monkeypatch.setattr("ace_jax.fit.stats.linear_qr_statistics", lambda *a, **k: qs)
    f = objective.make_lml(prob, None)
    assert f.qr_stats is qs and getattr(f, "host_evidence", None) is not None
    a = to_array(THETAS[0])
    assert abs(float(f(a)) - float(log_marginal_likelihood_qr(THETAS[0], qs, prob))) < 1e-11 * _scale(THETAS[0], qs)
    monkeypatch.setattr(objective, "QR_EVAL", "dense")
    assert getattr(objective.make_lml(prob, None), "host_evidence", None) is None


def test_host_evidence_on_si_fixture_at_the_stiff_point():
    """Real QR statistics (Si fixture, 12 configs) where kappa(G + Lambda) ~ 1e16 (noise at its floor, prior nearly
    flat: the Cholesky form fails there): value, gradient and Hessian still match the dense QR form."""
    import test_lml_qr as T
    from conftest import FIXTURE_DIR  # noqa: F401
    from ace_jax.eval import highest_precision, load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.lml_host import HostQREvidence
    from ace_jax.fit.stats import linear_qr_statistics
    if not T.XYZ.exists():
        pytest.skip("missing GP fixtures")
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    ds = build_dataset(load_configs(T.XYZ, "dft_energy", "dft_force", "dft_virial")[:12], meta, np.asarray(z["E0"]), 4)
    from ace_jax.fit.inducing import GPConfig
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=4)
    with highest_precision():
        qs = linear_qr_statistics(model, cfg, ds)
    prob = Problem(None, model, SimpleNamespace(XM=np.zeros((0, 1))), cfg, jnp.asarray(z["gamma"]), None,
                   lml_solver="qr")
    ev = HostQREvidence(qs, prob)
    for theta in (T._theta(), T._theta(**T.STIFF)):
        a = to_array(theta)
        dense = lambda x: log_marginal_likelihood_qr(Hypers(*x), qs, prob)    # noqa: E731
        host = lambda x: ev.lml(Hypers(*x))                                  # noqa: E731
        scale = _scale(theta, qs)
        with highest_precision():
            v0, g0, H0 = float(dense(a)), np.asarray(jax.grad(dense)(a)), np.asarray(jax.hessian(dense)(a))
        v1, g1, H1 = float(host(a)), np.asarray(jax.grad(host)(a)), np.asarray(jax.hessian(host)(a))
        assert abs(v1 - v0) < 1e-10 * scale, (v0, v1, scale)
        np.testing.assert_allclose(g1[6:], g0[6:], rtol=1e-6, atol=1e-9 * scale)
        np.testing.assert_allclose(H1[6:, 6:], H0[6:, 6:], rtol=1e-5, atol=1e-8 * scale)
