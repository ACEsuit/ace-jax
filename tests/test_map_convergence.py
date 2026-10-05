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


def _setup_files(train, *extra):
    """As _setup, on a given training file (scored on itself)."""
    from ace_jax.cli import _fit_config, _parse
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    a = _parse(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--train", str(train), "--energy-key",
                "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial", "--m-per-species", "0",
                "--rungs", "map", "--configs-per-batch", "4", "--r0", "2.35", "--out", "unused", *extra])
    cfg = _fit_config(a)
    d = load_fit_data(cfg, train=str(train), **QUIET)
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
    assert c["polish"]["hvps"] > 0                                   # the default polish: exact, by HVP columns
    assert np.abs(pg).max() <= 1e-2                                  # stationary (bounded: projected)
    assert float(v) >= v_ref - 1e-6 * max(1.0, abs(v_ref))           # at least as good as the reference
    # the data-identified noise and prior scales agree with the reference (log space)
    assert np.allclose(x[6:], x_ref[6:], atol=1e-3)


def _exact_reference(obj, prior, x0):
    """An independent exact-Hessian optimum: scipy trust-constr (bounded) with jax.hessian, from x0."""
    from scipy.optimize import Bounds, minimize
    from ace_jax.fit.hypers import from_array, log_prior
    from ace_jax.fit.pipeline.mapfit import LBFGS_HI, LBFGS_LO
    hess = jax.jit(jax.hessian(lambda a: obj.lik(a) + log_prior(from_array(a), prior)))

    def f(z):
        v, g = obj.vg(jnp.asarray(z))
        return -float(v), -np.asarray(g, float)
    r = minimize(f, x0, jac=True, hess=lambda z: -np.asarray(hess(jnp.asarray(z))), method="trust-constr",
                 bounds=Bounds(LBFGS_LO, LBFGS_HI), options={"gtol": 1e-10, "xtol": 1e-14, "maxiter": 2000})
    return r.x, -float(r.fun), f


def test_default_polish_reaches_the_exact_optimum():
    """40 configs (well conditioned): the default polish is within 1e-3 nats of an independent
    exact-Hessian trust-region optimum, converged without the roundoff floor."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.pipeline.mapfit import fit_map
    with highest_precision():
        cfg, d, b, obj = _setup()
        mf = fit_map(cfg, d, b, obj, **QUIET)
        _, v_ref, _ = _exact_reference(obj, b.prob.prior, np.asarray(to_array(b.prob.prior.mu), float))
    c = mf.convergence
    print(f"40 configs: default {c['logpost']:.8f}, exact trust-constr {v_ref:.8f}")
    assert abs(c["logpost"] - v_ref) <= 1e-3 and c["converged"] and not c["resolution_limited"]


def test_default_polish_on_12_configs_is_not_stopped_short(tmp_path):
    """The 12-config linear fit of test_gp_cli: sigma_F on its bound (the basis interpolates the
    forces), weakly curved in log sigma_c, and cond(S) so large that the log-posterior itself is
    noisy: a 1e-9 shift of theta moves it by up to 0.6 nats (polish-measured roundoff ~0.3).  A
    finite-difference-Hessian polish stopped at 482.52 (|g| 14.8 at log sigma_c, ~100x its roundoff)
    and called it converged; the exact polish reached ~485.4.  So: the default must reach the exact
    reference to within 3x the measured roundoff (1e-3 nats is below what this objective resolves),
    and a 'converged' verdict must be honest -- every free gradient component within 10x its
    roundoff."""
    from conftest import small_si_xyz
    from ace_jax.eval import highest_precision
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.pipeline.mapfit import LBFGS_HI, LBFGS_LO, _free, fit_map
    tr = small_si_xyz(tmp_path / "tr.xyz", 12)
    with highest_precision():
        cfg, d, b, obj = _setup_files(tr)
        mf = fit_map(cfg, d, b, obj, **QUIET)
        x = np.asarray(to_array(mf.theta), float)
        x_ref, v_ref, f = _exact_reference(obj, b.prob.prior, x)
        g = -f(x)[1]
    c, noise = mf.convergence, mf.convergence["polish"]["noise"]
    print(f"12 configs: default {c['logpost']:.4f} (gain {c['gain']:.2e}, |pg| {c['pgrad_inf']:.2e}, "
          f"gnoise {c['gnoise_max']:.2e}, F roundoff {noise:.2f}), exact trust-constr {v_ref:.4f}")
    assert c["logpost"] >= v_ref - max(1e-3, 3 * noise)
    assert c["logpost"] > 484.0                                       # the FD shortfall was 482.52
    assert abs(x[8] - LBFGS_LO[8]) < 1e-12                            # sigma_F on its bound
    if c["converged"]:
        free = _free(x, g, LBFGS_LO, LBFGS_HI)
        assert np.all(np.abs(g[free]) <= 10 * c["gnoise_max"])


def test_polish_auto_is_the_cheap_linear_path_only():
    from ace_jax.fit.pipeline import FitConfig
    from ace_jax.fit.pipeline.mapfit import polish_wanted
    m = lambda **k: polish_wanted(FitConfig(model="m", **k))     # noqa: E731
    assert m(arm="linear")
    assert not m(arm="gp") and not m(arm="linear", objective="loo") and not m(arm="linear", devices=2)
    assert m(arm="gp", map_polish="on") and not m(arm="linear", map_polish="off")


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
    monkeypatch.setattr(mapfit, "lbfgs_map", lambda *a, **k: calls.append((a[4], k["maxfun"])) or real_l(*a, **k))
    with highest_precision():
        cfg, d, b, obj = _setup("--map-polish", "off")
        mf = mapfit.fit_map(cfg, d, b, obj, **QUIET)
    n = min(cfg.map_steps, mapfit.RESTART_EVALS)
    assert calls == [(n, n)]                                         # bounded in evaluations, not just iterations
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
