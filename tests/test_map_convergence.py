"""The default hyperparameter MAP reaches a stationary point of the log-posterior.

The CLI's old default (Adam, 500 steps, lr 0.02 decaying) moves each log-hyperparameter at most
~5 units from the prior mean; on GAP-18 Si the optimum is ~8 units away in log sigma_E, and the
shipped MAP sat about 1e7 nats below it (-1.04e7 against +1.92e5) with no warning.  These tests pin the default to a converged MAP.
"""
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from conftest import FIXTURE_DIR  # noqa: E402

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")
QUIET = dict(log=lambda *a, **k: None)


def _setup(*extra):
    """A small linear fit through the CLI's own defaults (no --opt): config, data, problem, objective."""
    from ace_jax.cli import _fit_config, _parse
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    a = _parse(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--ntrain", "40",
                "--ntest", "8", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key",
                "dft_virial", "--m-per-species", "0", "--rungs", "map", "--configs-per-batch", "4",
                "--r0", "2.35", "--out", "unused", *extra])
    cfg = _fit_config(a)
    d = load_fit_data(cfg, data=str(XYZ), **QUIET)
    b = build_problem(cfg, d)
    return cfg, d, b, make_objective(cfg, d, b)


def _reference_optimum(obj, prior):
    """An independent, tightly converged bounded optimum: scipy L-BFGS-B at ftol 1e-15 / gtol 1e-10."""
    from scipy.optimize import minimize
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.pipeline.mapfit import LBFGS_HI, LBFGS_LO

    def f(z):
        v, g = obj.vg(jnp.asarray(z))
        return -float(v), -np.asarray(g, float)
    r = minimize(f, np.asarray(to_array(prior.mu), float), jac=True, method="L-BFGS-B",
                 bounds=list(zip(LBFGS_LO, LBFGS_HI)), options={"maxiter": 5000, "maxfun": 20000,
                                                                "ftol": 1e-15, "gtol": 1e-10})
    return r.x, -float(r.fun)


def test_default_map_is_stationary_and_matches_the_reference_optimum():
    from ace_jax.eval import highest_precision
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.newton import _projected_gradient
    from ace_jax.fit.pipeline.mapfit import LBFGS_HI, LBFGS_LO, MAP_GAIN_TOL, fit_map
    with highest_precision():
        cfg, d, b, obj = _setup()
        mf = fit_map(cfg, d, b, obj, **QUIET)
        x = np.asarray(to_array(mf.theta), float)
        v, g = obj.vg(jnp.asarray(x))
        pg = _projected_gradient(x, -np.asarray(g, float), LBFGS_LO, LBFGS_HI)
        x_ref, v_ref = _reference_optimum(obj, b.prob.prior)
    c = mf.convergence
    print(f"default MAP: logpost {float(v):.10g}  |pg|_inf {np.abs(pg).max():.2e}  gain {c['gain']:.2e}; "
          f"reference {v_ref:.10g}")
    assert c["converged"] and c["gain"] <= MAP_GAIN_TOL and not c["resolution_limited"]
    assert c["polish"]["hessian"] == "fd"                            # the default polish: gradient-level memory
    assert np.abs(pg).max() <= 1e-2                                  # stationary (bounded: projected)
    assert float(v) >= v_ref - 1e-6 * max(1.0, abs(v_ref))           # at least as good as the reference
    # the data-identified noise and prior scales agree with the reference (log space)
    assert np.allclose(x[6:], x_ref[6:], atol=1e-3)


def test_fd_polish_reaches_the_exact_hessian_optimum():
    """The default finite-difference polish and the opt-in exact-Hessian polish end at the same MAP."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.pipeline.mapfit import fit_map
    out = {}
    with highest_precision():
        for mode in ("on", "exact"):
            cfg, d, b, obj = _setup("--map-polish", mode)
            mf = fit_map(cfg, d, b, obj, **QUIET)
            out[mode] = (np.asarray(to_array(mf.theta), float), mf.convergence)
    (xf, cf), (xe, ce) = out["on"], out["exact"]
    assert cf["polish"]["hessian"] == "fd" and ce["polish"]["hessian"] == "exact"
    assert cf["converged"] and ce["converged"]
    assert np.allclose(xf, xe, rtol=0, atol=1e-6), np.abs(xf - xe).max()
    assert abs(cf["logpost"] - ce["logpost"]) <= 10 * max(cf["noise"], ce["noise"], 1e-12)


def test_polish_auto_is_the_cheap_linear_path_only():
    from ace_jax.fit.pipeline import FitConfig
    from ace_jax.fit.pipeline.mapfit import polish_mode
    m = lambda **k: polish_mode(FitConfig(model="m", **k))       # noqa: E731
    assert m(arm="linear") == "fd"
    assert m(arm="gp") is None and m(arm="linear", objective="loo") is None and m(arm="linear", devices=2) is None
    assert m(arm="gp", map_polish="on") == "fd" and m(arm="linear", map_polish="exact") == "exact"
    assert m(arm="linear", map_polish="off") is None


def test_a_line_search_stop_restarts_lbfgs_once(monkeypatch):
    """L-BFGS-B's 'ABNORMAL' (line-search) stop is not convergence: one bounded restart from there."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.pipeline import mapfit
    real, calls = mapfit.multistart_map, []

    def abnormal(*a, **k):
        best, runs = real(*a, **k)
        best = {**best, "message": "ABNORMAL: "}
        return best, runs
    monkeypatch.setattr(mapfit, "multistart_map", abnormal)
    real_l = mapfit.lbfgs_map
    monkeypatch.setattr(mapfit, "lbfgs_map", lambda *a, **k: calls.append(a[4]) or real_l(*a, **k))
    with highest_precision():
        cfg, d, b, obj = _setup("--map-polish", "off")
        mf = mapfit.fit_map(cfg, d, b, obj, **QUIET)
    assert calls == [min(cfg.map_steps, mapfit.RESTART_ITERS)]
    r = mf.convergence["restart"]
    assert r is not None and r["logpost"] >= r["logpost_before"] - 1e-9


def test_an_unconverged_map_warns_and_strict_raises():
    """Adam at its old 500-step default stops far from the optimum: a loud warning, and an error
    under --strict, instead of a silent under-fitted MAP."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.pipeline.mapfit import MapNotConverged, fit_map
    with highest_precision():
        cfg, d, b, obj = _setup("--opt", "adam", "--map-steps", "500")
        with pytest.warns(UserWarning, match="MAP did not converge"):
            mf = fit_map(cfg, d, b, obj, **QUIET)
        assert not mf.convergence["converged"] and mf.convergence["gain"] > 1.0
        cfg, d, b, obj = _setup("--opt", "adam", "--map-steps", "500", "--strict")
        with pytest.raises(MapNotConverged, match="MAP did not converge"):
            fit_map(cfg, d, b, obj, **QUIET)


def test_converged_default_map_does_not_warn():
    from ace_jax.eval import highest_precision
    from ace_jax.fit.pipeline.mapfit import fit_map
    with highest_precision(), warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        cfg, d, b, obj = _setup("--strict")
        fit_map(cfg, d, b, obj, **QUIET)
