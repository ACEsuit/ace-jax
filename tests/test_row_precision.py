# tests/test_row_precision.py
"""Fits from float32 design rows (FitConfig.row_precision "float32"; docs/dev/fp32-results.md,
step 3): QR statistics from float32 rows carry the residual (QRStats.rho_q) and the LML reads it
instead of y'y - v'v; the stability check (fit/rowcheck.py) runs after the MAP and is recorded."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


def _cfg(**kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=30, ntest=8, batch=4, r0=2.35, arm="linear", uq="blr",
                rungs=("map",), map_steps=30)
    return FitConfig(**{**base, **kw})


@pytest.fixture(scope="module")
def problem():
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = _cfg().validate()
    d = load_fit_data(cfg, data=str(XYZ))
    return cfg, d, build_problem(cfg, d)


@pytest.fixture(scope="module")
def stats(problem):
    """float64 (host merges) and float32 (device path on the CPU backend: small L) statistics."""
    from ace_jax.fit.stats import linear_qr_statistics
    _, d, b = problem
    p = b.prob
    return (linear_qr_statistics(p.model, p.cfg, d.ds_train),
            linear_qr_statistics(p.model, p.cfg, d.ds_train, host=False, row_dtype="float32"))


def test_float32_statistics_carry_the_residual(stats):
    q64, q32 = stats
    assert q64.rho_E is None and q32.rho_E is not None
    for q in "EFV":
        # the carried residual is the factor's own: rho^2 = y'y - |c|^2 to float32 accuracy
        yy, c, rho = float(getattr(q32, f"yy_{q}")), np.asarray(getattr(q32, f"c_{q}")), float(getattr(q32, f"rho_{q}"))
        if yy > 0:
            assert abs(rho ** 2 - (yy - c @ c)) <= 1e-5 * yy
        assert getattr(q32, f"R_{q}").dtype == np.float64            # returned in float64 for the LML


def test_rho_lml_equals_the_library_form_in_float64(problem, stats):
    """With rho made exact from float64 statistics the two LML forms agree."""
    from ace_jax.fit.objective import log_marginal_likelihood_qr
    _, _, b = problem
    q64, _ = stats
    rho = {f"rho_{q}": np.sqrt(max(float(getattr(q64, f"yy_{q}")) - float(getattr(q64, f"c_{q}") @ getattr(q64, f"c_{q}")), 0.0))
           for q in "EFV"}
    th = b.prob.prior.mu
    lib, rh = log_marginal_likelihood_qr(th, q64, b.prob), log_marginal_likelihood_qr(th, q64._replace(**rho), b.prob)
    assert np.isclose(float(lib), float(rh), rtol=1e-9, atol=1e-6)


def test_float32_rows_change_the_evidence_and_mean_little(problem, stats):
    from ace_jax.fit.objective import log_marginal_likelihood_qr, posterior_from_qr
    _, _, b = problem
    q64, q32 = stats
    th = b.prob.prior.mu
    l64, l32 = float(log_marginal_likelihood_qr(th, q64, b.prob)), float(log_marginal_likelihood_qr(th, q32, b.prob))
    assert abs(l32 - l64) <= 0.5                     # nats (measured 0.07; full GAP-18 0.2-8 of ~4e5)
    m64, m32 = posterior_from_qr(th, q64, b.prob)[0], posterior_from_qr(th, q32, b.prob)[0]
    # compared through the float64 design (the coefficients themselves are ill-determined)
    for q in "EFV":
        R = np.asarray(getattr(q64, f"R_{q}"))
        assert np.linalg.norm(R @ (m32 - m64)) <= 1e-3 * max(np.linalg.norm(R @ m64), 1e-12)


def test_perturb_at_zero_is_the_identity(problem, stats):
    from ace_jax.fit.objective import log_marginal_likelihood_qr
    from ace_jax.fit.rowcheck import perturb
    _, _, b = problem
    _, q32 = stats
    th = b.prob.prior.mu
    p = perturb(q32, eps=0.0)
    assert np.isclose(float(log_marginal_likelihood_qr(th, p, b.prob)),
                      float(log_marginal_likelihood_qr(th, q32, b.prob)), rtol=1e-12)


def test_row_check_needs_float32_rows():
    with pytest.raises(ValueError, match="row_check"):
        _cfg(row_check=True).validate()


@pytest.mark.parametrize("bad", [dict(arm="gp", m_per_species=4), dict(lml_solver="cholesky"),
                                 dict(learn_radial=True), dict(objective="loo"), dict(devices=2)])
def test_float32_rows_reject_unsupported_settings(bad):
    with pytest.raises(ValueError, match="row_precision float32"):
        _cfg(row_precision="float32", **bad).validate()


def test_pipeline_records_the_row_rounding_check():
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.run import fit
    cfg = _cfg(row_precision="float32", row_check=True).validate()
    res = fit(cfg, load_fit_data(cfg, data=str(XYZ)), log=lambda *a, **k: None)
    chk = res.map.convergence["row_rounding_check"]
    assert chk["verdict"] in ("stable", "marginal", "unstable")
    assert set(chk["quantities"]) <= set("EFV") and np.isfinite(chk["max_rmse_change"])
    for v in chk["quantities"].values():
        assert set(v) == {"rmse_change", "change_over_rmse", "change_over_posterior_std"}


def test_float32_row_predictions_match_float64(problem, stats):
    """predict_fixed with prob.row_dtype "float32" evaluates the test rows in float32 (the
    posterior stays float64): means and variances agree with float64 to float32 rounding."""
    from ace_jax.fit.predict import predict_fixed
    _, d, b = problem
    q64, _ = stats
    th = b.prob.prior.mu
    p64 = predict_fixed(th, b.prob, d.ds_train, d.ds_test, stats=lambda t: q64)
    p32 = predict_fixed(th, b.prob._replace(row_dtype="float32"), d.ds_train, d.ds_test, stats=lambda t: q64)
    for k in ("E_mean", "F_mean", "V_mean"):
        a, c = np.asarray(getattr(p64, k)), np.asarray(getattr(p32, k))
        # float32 rows times a readout whose terms cancel (step 1, section 4): ~4e-5 of the scale here
        assert np.max(np.abs(a - c)) <= 2e-4 * max(np.max(np.abs(a)), 1.0), k
    for k in ("E_var", "F_var"):
        a, c = np.asarray(getattr(p64, k)), np.asarray(getattr(p32, k))
        assert np.allclose(a, c, rtol=1e-3, atol=1e-6 * np.max(a)), k


def test_float32_rows_through_the_chunked_scan(problem):
    """The node-chunked rows (linear_rows_chunked / _rows_scan, which predictions take when a
    batch exceeds ROWS_EDGE_BUDGET) accept float32 rows: their accumulators keep their dtype."""
    from ace_jax.fit.rows import linear_rows, linear_rows_chunked
    from ace_jax.fit.stats import float32_rows
    _, d, b = problem
    p = b.prob
    m32, ds32 = float32_rows(p.model, d.ds_train)
    batch64 = jax.tree.map(lambda a: a[0], d.ds_train)
    batch32 = jax.tree.map(lambda a: a[0], ds32)
    ref = linear_rows(p.model, p.cfg, batch64)[0]
    got = linear_rows_chunked(m32, p.cfg, batch32, node_chunk=7)
    for k in ("E", "F", "V"):
        a, c = np.asarray(getattr(ref, k)), np.asarray(getattr(got, k))
        assert np.max(np.abs(a - c)) <= 1e-5 * np.max(np.abs(a)), k
