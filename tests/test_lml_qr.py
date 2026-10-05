"""The QR form of the linear-arm LML (objective.log_marginal_likelihood_qr): compressed
per-quantity QR statistics (stats.linear_qr_statistics) instead of the Gram, so the evidence,
its gradient and the posterior are computed at kappa(Phi), not kappa(Phi)^2.

Where the Gram is well conditioned it is the Cholesky LML to roundoff; where it is not (an
evidence optimum that nearly interpolates: noise at its floor, prior near flat) the Cholesky
LML is NaN or silently wrong, and the QR form still matches a dense QR of the full design."""
import numpy as np
import pytest

from conftest import FIXTURE_DIR

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import (Problem, combine, log_marginal_likelihood, log_marginal_likelihood_qr,
                                   make_lml, posterior, posterior_from_qr, prior_precision)
from ace_jax.fit.solve import posterior_qr
from ace_jax.fit.stats import linear_qr_statistics, linear_statistics

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing GP fixtures")


def _theta(**kw):
    d = dict(log_ell=np.log(0.8), log_A=np.log(0.05), log_alpha=0.0, log_r0=np.log(2.35),
             log_eps=np.log(0.3), log_rho=np.log(4.0), log_sigma_c=np.log(0.3),
             log_sigma_E=np.log(1e-3), log_sigma_F=np.log(2e-2), log_sigma_V=np.log(2e-2))
    d.update(kw)
    return Hypers(**d)


# noise at its floor, prior almost flat: kappa(G + Lambda) ~ 1e16, its Cholesky fails
STIFF = dict(log_sigma_c=np.log(1e4), log_sigma_E=np.log(1e-4), log_sigma_F=np.log(1e-4),
             log_sigma_V=np.log(1e-4))


@pytest.fixture(scope="module")
def m0():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:12]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), 4)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=4)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))  # M = 0
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    with highest_precision():
        lin = linear_statistics(model, cfg, ds)
        qs = linear_qr_statistics(model, cfg, ds)
    return prob, ds, lin, qs


def test_qr_statistics_reproduce_the_gram(m0):
    """R_q^T R_q and R_q^T c_q are the Gram statistics; yy, n and logw are copied exactly."""
    _, _, lin, qs = m0
    for q in "EFV":
        G, b = np.asarray(getattr(lin, f"G_{q}")), np.asarray(getattr(lin, f"b_{q}"))
        assert np.abs(np.asarray(getattr(qs, f"G_{q}")) - G).max() < 1e-12 * np.abs(G).max(), q
        assert np.abs(np.asarray(getattr(qs, f"b_{q}")) - b).max() < 1e-12 * np.abs(b).max(), q
        R = np.asarray(getattr(qs, f"R_{q}"))
        assert np.allclose(np.tril(R, -1), 0.0), q                        # upper triangular
        for k in ("yy", "n", "logw"):
            a, r = float(getattr(qs, f"{k}_{q}")), float(getattr(lin, f"{k}_{q}"))
            assert np.isclose(a, r, rtol=1e-13, atol=0), (q, k)


def _scale(theta, lin):
    """The size of the terms the LML cancels (y^T T y, ~1e8-1e11 here, against an LML of ~10):
    roundoff in the LML is relative to this, not to the LML itself."""
    return float(combine(theta, lin)[2])


def test_qr_lml_matches_cholesky_where_well_conditioned(m0):
    prob, ds, lin, qs = m0
    theta = _theta()
    with highest_precision():
        v_c, g_c = jax.value_and_grad(lambda t: log_marginal_likelihood(t, lin, prob))(theta)
        v_q, g_q = jax.value_and_grad(lambda t: log_marginal_likelihood_qr(t, qs, prob))(theta)
        ref = _dense_reference(prob, ds, theta)
    tol = 1e-13 * _scale(theta, lin)
    assert abs(float(v_q) - ref) < tol and abs(float(v_c) - ref) < tol
    for a, b in zip(to_array(g_q), to_array(g_c)):
        assert np.isclose(float(a), float(b), rtol=1e-6, atol=tol)


def _dense_reference(prob, ds, theta):
    """The LML from a dense QR of the full weighted design stacked with the prior square root:
    an independent route (solve.posterior_qr's factor) to the same quantity."""
    from ace_jax.fit.solve import _qr_factor
    R, d = _qr_factor(prob, ds, theta)
    st = linear_statistics(prob.model, prob.cfg, ds)
    _, _, yy, logtau, N = combine(theta, st)
    _, logdet_Lam = prior_precision(theta, prob)
    return float(-0.5 * (yy - d @ d) - 0.5 * (2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(R)))) - logdet_Lam - logtau)
                 - 0.5 * N * jnp.log(2 * jnp.pi))


def test_qr_lml_stays_exact_where_cholesky_fails(m0):
    prob, ds, lin, qs = m0
    theta = _theta(**STIFF)
    with highest_precision():
        chol = float(log_marginal_likelihood(theta, lin, prob))
        v, g = jax.value_and_grad(lambda t: log_marginal_likelihood_qr(t, qs, prob))(theta)
        ref = _dense_reference(prob, ds, theta)
    assert not np.isfinite(chol)                                    # the failure, reproduced
    assert np.isfinite(float(v)) and np.isfinite(np.asarray(to_array(g))).all()
    assert abs(float(v) - ref) < 1e-13 * _scale(theta, lin)


@pytest.mark.parametrize("stiff", [False, True])
def test_qr_posterior_is_the_lml_factorisation(m0, stiff):
    """posterior_from_qr: the same mean and factor as the streaming QR of the full design (and,
    where it works, the Cholesky posterior); L L^T = G + Lambda."""
    prob, ds, lin, qs = m0
    theta = _theta(**STIFF) if stiff else _theta()
    with highest_precision():
        mu, L = posterior_from_qr(theta, qs, prob)
        mu_s, _ = posterior_qr(prob, ds, theta)
        G, *_ = combine(theta, lin)
        Lam, _ = prior_precision(theta, prob)
        if not stiff:
            mu_c, _ = posterior(theta, lin, prob)
    mu, L, mu_s, A = map(np.asarray, (mu, L, mu_s, G + Lam))
    assert np.allclose(np.triu(L, 1), 0.0) and (np.diag(L) > 0).all()
    assert np.abs(L @ L.T - A).max() < 1e-10 * np.abs(A).max()
    # both are stable QR solves; they agree to about kappa(A) eps (A the stacked design, ~1e9 stiff)
    assert np.abs(mu - mu_s).max() < (1e-5 if stiff else 1e-8) * np.abs(mu_s).max()
    if not stiff:
        assert np.abs(mu - np.asarray(mu_c)).max() < 1e-8 * np.abs(mu).max()


def test_make_lml_follows_the_problem_solver(m0):
    """Problem.lml_solver picks the factorisation: 'cholesky' (the Problem default, so direct
    constructions stay bit-exact) or 'qr' (the pipeline default for the linear arm)."""
    prob, ds, lin, qs = m0
    a, tol = to_array(_theta()), 1e-13 * _scale(_theta(), lin)
    with highest_precision():
        v_c = float(make_lml(prob, ds)(a))
        v_q = float(make_lml(prob._replace(lml_solver="qr"), ds)(a))
        assert abs(v_c - float(log_marginal_likelihood(_theta(), lin, prob))) < tol
        assert abs(v_q - float(log_marginal_likelihood_qr(_theta(), qs, prob))) < tol
        stiff = to_array(_theta(**STIFF))
        assert not np.isfinite(float(make_lml(prob, ds)(stiff)))            # Cholesky: NaN
        assert np.isfinite(float(make_lml(prob._replace(lml_solver="qr"), ds)(stiff)))


@pytest.fixture(scope="module")
def pipeline_fits(tmp_path_factory):
    """The linear evidence fit on the Si fixture (its isolated atom included, so e0='lsq' pins
    E0 with the 1e16 prior row), by QR (the default) and by Cholesky."""
    from ase.io import read, write
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    work = tmp_path_factory.mktemp("lml_qr")
    fr = read(XYZ, ":")
    write(work / "train.xyz", fr[:16]); write(work / "test.xyz", fr[16:22])
    keys = dict(energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
    out = {}
    for solver in ("qr", "cholesky"):
        cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), arm="linear", m_per_species=0, e0="lsq",
                        opt="lbfgs", r0=2.35, rungs=("map",), map_steps=40, predict_stats="recompute",
                        predict_train=False, lml_solver=solver, **keys).validate()
        d = load_fit_data(cfg, train=str(work / "train.xyz"), test=str(work / "test.xyz"), log=lambda *a: None)
        out[solver] = fit(cfg, d, log=lambda *a: None)
    return out


def test_pipeline_defaults_to_qr_and_is_consistent(pipeline_fits):
    """The default fit takes the QR form throughout: the reported evidence is the QR LML at
    the MAP, and the saved readout is the QR posterior mean of the same statistics."""
    from ace_jax.fit.pipeline import FitConfig
    from ace_jax.fit.pipeline.export import linear_model_arrays
    assert FitConfig.lml_solver == "qr"
    res = pipeline_fits["qr"]
    prob = res.built.prob
    assert prob.lml_solver == "qr"
    with highest_precision():
        qs = linear_qr_statistics(prob.model, prob.cfg, res.data.ds_train)
        lml = float(log_marginal_likelihood_qr(res.theta, qs, prob))
        mu, _ = posterior_from_qr(res.theta, qs, prob)
    scale = sum(float(getattr(qs, f"yy_{q}")) * np.exp(-2.0 * float(getattr(res.theta, f"log_sigma_{q}")))
                for q in "EFV")                                   # the cancelled terms (see _scale)
    assert abs(res.map.log_evidence - lml) < 1e-13 * scale
    nz, nB = prob.cfg.NZ, prob.cfg.n_B
    WB = linear_model_arrays(res)["WB"]
    assert np.allclose(WB, np.asarray(mu)[:nz * nB].reshape(nz, nB).T, rtol=1e-12, atol=0)
    assert np.isfinite(res.preds.metrics["test/map"]["F"]["rmse"])


def test_pipeline_fixture_map_already_needs_qr(pipeline_fits):
    """The default fit of this fixture already sits where the Cholesky is inaccurate without
    failing: its MAP noise is near the floor (sigma_E ~ 1.2e-4), and there the Cholesky LML is
    ~250x further from a dense QR of the full design than the QR form (~1e-13 against ~1e-15 of
    the cancelled terms) and its posterior mean -- the saved readout -- is off by ~1e-3, while
    the QR mean matches the full-design QR.  A fall back to QR only when the Cholesky fails
    would never trigger here.  The two MAP runs still reach the same evidence to L-BFGS
    stopping tolerance."""
    q, c = pipeline_fits["qr"], pipeline_fits["cholesky"]
    assert c.built.prob.lml_solver == "cholesky"
    assert np.isclose(q.map.log_evidence, c.map.log_evidence, rtol=1e-5)
    prob, ds, th = q.built.prob, q.data.ds_train, q.theta
    assert float(th.log_sigma_E) < np.log(2e-4)
    with highest_precision():
        qs, lin = linear_qr_statistics(prob.model, prob.cfg, ds), linear_statistics(prob.model, prob.cfg, ds)
        v_q, v_c = float(log_marginal_likelihood_qr(th, qs, prob)), float(log_marginal_likelihood(th, lin, prob))
        mu_q, mu_c = (np.asarray(m) for m in (posterior_from_qr(th, qs, prob)[0], posterior(th, lin, prob)[0]))
        mu_s = np.asarray(posterior_qr(prob, ds, th)[0])
        ref = _dense_reference(prob, ds, th)
    sc = _scale(th, lin)
    assert abs(v_q - ref) < 1e-14 * sc                                  # QR: at roundoff
    assert 1e-14 * sc < abs(v_c - ref) < 1e-11 * sc                     # Cholesky: finite, worse
    assert np.abs(mu_q - mu_s).max() < 1e-6 * np.abs(mu_s).max()        # QR mean: the full-design QR's
    assert np.abs(mu_c - mu_s).max() > 1e-4 * np.abs(mu_s).max()        # Cholesky mean: off


def test_qr_merge_leaves_no_near_underflow_residue(m0):
    """A rank-deficient updating QR (the energy block: a few rows per batch against L columns)
    must not leave rounding residue near underflow in the rows it has not filled: XLA's
    flush-to-zero LAPACK returned NaN on such a factor on AVX-512 runners.  Every entry of each
    merged factor is zero or above QR_DUST of its largest, and the Gram is unchanged."""
    from ace_jax.fit.rows import linear_rows_bounded
    from ace_jax.fit.stats import QR_DUST, _qr_merge
    prob, ds, lin, qs = m0
    L = prob.cfg.len_basis
    R, c = jnp.zeros((L, L)), jnp.zeros(L)
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a: a[i], ds)
        r = linear_rows_bounded(prob.model, prob.cfg, bt)
        R, c = _qr_merge(R, c, r.E * bt.w_E[:, None], bt.y_E * bt.w_E)
        a = np.abs(np.asarray(R))
        assert ((a == 0) | (a >= QR_DUST * a.max())).all(), i
    G = np.asarray(lin.G_E)
    assert np.abs(np.asarray(R).T @ np.asarray(R) - G).max() < 1e-12 * np.abs(G).max()
