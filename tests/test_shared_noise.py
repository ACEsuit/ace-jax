"""--noise shared: one noise scale drives the E, F and V rows (ACEfit's BLR), so the user's E:F:V
weights set the balance.  Per-quantity noise (the default) learns sigma_E, sigma_F and sigma_V
separately, and at a converged MAP each cancels its quantity's weight."""
import json
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from conftest import FIXTURE_DIR, small_si_xyz  # noqa: E402

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
MODEL = FIXTURE_DIR / "si_fitted.npz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")
KEYS = ["--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial"]
QUIET = dict(log=lambda *a, **k: None)
SIG = ("log_sigma_E", "log_sigma_F", "log_sigma_V")
ACEFIT_WEIGHTS = '{"default": {"E": 30.0, "F": 1.0, "V": 1.0}}'      # ACEpotentials' default_weights()


def _setup(*extra, train=None):
    """A linear fit through the CLI's own defaults: config, data, problem, objective."""
    from ace_jax.cli import _fit_config, _parse
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    src = ["--train", str(train)] if train else ["--data", str(XYZ), "--ntrain", "40", "--ntest", "8"]
    a = _parse(["fit", "--model", str(MODEL), *src, *KEYS, "--m-per-species", "0", "--rungs", "map",
                "--configs-per-batch", "4", "--r0", "2.35", "--out", "unused", *extra])
    cfg = _fit_config(a)
    d = load_fit_data(cfg, **({"train": str(train)} if train else {"data": str(XYZ)}), **QUIET)
    b = build_problem(cfg, d)
    return cfg, d, b, make_objective(cfg, d, b)


def _sigmas(theta):
    return [float(getattr(theta, s)) for s in SIG]


# ---------------------------------------------------------------- plumbing

def test_noise_flag_default_and_choices():
    from ace_jax.cli import _fit_config, _parse
    base = ["fit", "--model", str(MODEL), "--train", str(XYZ), "--r0", "2.35", "--m-per-species", "0",
            "--out", "o"]
    assert _fit_config(_parse(base)).noise == "per-quantity"
    assert _fit_config(_parse(base + ["--noise", "shared"])).noise == "shared"
    with pytest.raises(SystemExit):
        _parse(base + ["--noise", "tied"])


def test_noise_from_fit_yaml(tmp_path, capsys):
    from ace_jax.cli import _parse
    f = tmp_path / "fit.yaml"
    f.write_text(yaml.safe_dump({"train": str(XYZ), "out": "o", "model": str(MODEL), "r0": 2.35, "noise": "shared"}))
    assert _parse(["fit", "--config", str(f)]).noise == "shared"
    assert _parse(["fit", "--config", str(f), "--noise", "per-quantity"]).noise == "per-quantity"
    f.write_text(yaml.safe_dump({"train": str(XYZ), "out": "o", "model": str(MODEL), "r0": 2.35, "noise": "tied"}))
    with pytest.raises(SystemExit):
        _parse(["fit", "--config", str(f)])
    assert "'noise' must be one of" in capsys.readouterr().err


def test_fitconfig_validates_noise():
    from ace_jax.fit.pipeline import FitConfig
    assert FitConfig(model="m").noise == "per-quantity"
    FitConfig(model="m", noise="shared").validate()
    FitConfig(model="m", arm="linear", uq="ard", ard_mode="sequential", noise="shared").validate()
    for bad in (dict(noise="tied"), dict(noise="shared", sigma_type=True),
                dict(noise="shared", arm="linear", uq="ard"),                 # joint ARD refits sigma_q itself
                dict(noise="shared", learn_radial=True),
                dict(noise="shared", arm="linear", solver="lstsq")):
        with pytest.raises(ValueError, match="noise"):
            FitConfig(model="m", **bad).validate()


def test_cli_records_shared_noise(tmp_path):
    from ace_jax.cli import main
    xyz = small_si_xyz(tmp_path / "si12.xyz")
    out = tmp_path / "o"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(["fit", "--model", str(MODEL), "--train", str(xyz), "--r0", "2.35", *KEYS, "--m-per-species", "0",
              "--rungs", "map", "--configs-per-batch", "4", "--noise", "shared", "--out", str(out)])
    assert yaml.safe_load((out / "fit.yaml").read_text())["noise"] == "shared"
    conv = json.loads((out / "map_convergence.json").read_text())
    assert conv["noise"] == "shared"
    assert conv["tied"] == {"log_sigma_E": "log_sigma_F", "log_sigma_V": "log_sigma_F"}
    th = json.loads((out / "theta_map.json").read_text())
    assert th["log_sigma_E"] == th["log_sigma_F"] == th["log_sigma_V"]


def test_per_quantity_records_nothing_new(tmp_path):
    """The default writes no noise key into map_convergence.json (bit-exact pipeline goldens)."""
    from ace_jax.cli import main
    xyz = small_si_xyz(tmp_path / "si12.xyz")
    out = tmp_path / "o"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(["fit", "--model", str(MODEL), "--train", str(xyz), "--r0", "2.35", *KEYS, "--m-per-species", "0",
              "--rungs", "map", "--configs-per-batch", "4", "--out", str(out)])
    conv = json.loads((out / "map_convergence.json").read_text())
    assert "noise" not in conv and "tied" not in conv
    assert yaml.safe_load((out / "fit.yaml").read_text())["noise"] == "per-quantity"


# ---------------------------------------------------------------- the tie

def test_tie_routing_and_its_gradient():
    """tie_noise copies the shared coordinate into the tied ones; untie_grad is its vector-Jacobian
    product, so a host-side value_and_grad can be chained through the tie."""
    from ace_jax.fit.hypers import Hypers
    from ace_jax.fit.paramset import noise_tie, tie_noise, untie_grad
    m = noise_tie("shared")
    assert [Hypers._fields[i] for i in np.flatnonzero(m)] == ["log_sigma_E", "log_sigma_V"]
    assert not noise_tie("per-quantity").any()
    a = jnp.arange(10.0) / 7
    t = np.asarray(tie_noise(a))
    assert t[7] == t[8] == t[9] == float(a[8]) and np.array_equal(t[:7], np.asarray(a[:7]))
    f = lambda v: jnp.sum(jnp.sin(v) * jnp.arange(1.0, 11.0))         # noqa: E731
    g_chain = jax.grad(lambda v: f(tie_noise(v)))(a)
    assert np.allclose(np.asarray(untie_grad(jax.grad(f)(tie_noise(a)))), np.asarray(g_chain), rtol=1e-14)
    draws = np.random.default_rng(0).normal(size=(5, 10))
    td = tie_noise(draws)
    assert isinstance(td, np.ndarray) and np.array_equal(td[:, 7], draws[:, 8]) and np.array_equal(td[:, 9], draws[:, 8])


def test_shared_objective_ignores_the_tied_coordinates():
    """The shared log-posterior depends on log_sigma_F only: its gradient in the tied coordinates
    is exactly zero (likelihood and hyperprior alike), and it equals the per-quantity
    log-likelihood with all three sigmas set to sigma_F."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.hypers import to_array
    with highest_precision():
        *_, b, obj = _setup("--noise", "shared")
        *_, _, obj_pq = _setup()
        x = to_array(b.prob.prior.mu).at[8].set(np.log(0.03))
        v, g = obj.vg(x)
        assert float(g[7]) == 0.0 and float(g[9]) == 0.0 and float(g[8]) != 0.0
        xt = x.at[7].set(x[8]).at[9].set(x[8])
        assert np.isclose(float(obj.lik(x)), float(obj_pq.lik(xt)), rtol=1e-13)


@pytest.fixture(scope="module")
def shared40():
    """The shared-noise MAP on 40 configs, shared by the tests that only read it."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.pipeline.mapfit import fit_map
    with highest_precision():
        cfg, d, b, obj = _setup("--noise", "shared")
        return cfg, d, b, obj, fit_map(cfg, d, b, obj, **QUIET)


def test_shared_map_ties_sigmas_and_is_converged(shared40):
    from ace_jax.eval import highest_precision
    from ace_jax.fit.pipeline.mapfit import fit_map
    *_, mf = shared40
    with highest_precision():
        cfg2, d2, b2, obj2 = _setup()
        mf2 = fit_map(cfg2, d2, b2, obj2, **QUIET)
    sE, sF, sV = _sigmas(mf.theta)
    assert sE == sF == sV
    c = mf.convergence
    assert c["noise"] == "shared" and c["converged"] and c["optimiser"] == "lbfgs"
    assert c["polish"] is not None and c["polish"]["hvps"] > 0
    # the tie counts once: the polish differentiates the free coordinates only, never the tied two
    assert c["polish"]["hvps"] <= 8 * c["polish"]["hessian_evals"]
    assert mf.fixed is not None and not mf.fixed[7] and not mf.fixed[9]
    pE, pF, pV = _sigmas(mf2.theta)
    assert not np.isclose(pE, pF)                                # per-quantity noise is not tied


def test_shared_posterior_mean_is_the_blr_ridge_solve(shared40):
    """At the shared MAP the posterior mean is the ridge solution on the weighted, prior-scaled design
    with lambda = sigma^2 / sigma_c^2 -- the BLR identity (ACEfit: ridge on W A / P)."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.predict import fit_posterior
    from ace_jax.fit.solve import stacked_design
    cfg, d, b, obj, mf = shared40
    assert getattr(b.prob, "e0_prec", None) is None               # e0 'model': no fixed-precision E0 columns
    with highest_precision():
        th = mf.theta
        mu, _ = fit_posterior(th, obj.stats(th), b.prob, d.ds_train)
        unit = th._replace(log_sigma_E=0.0, log_sigma_F=0.0, log_sigma_V=0.0, log_sigma_c=0.0)
        Phi, y = stacked_design(b.prob, d.ds_train, unit)        # rows w phi; prior rows diag(gamma)
    L = Phi.shape[1]
    A, yA = np.asarray(Phi[:-L]), np.asarray(y[:-L])
    gam = np.asarray(b.prob.gamma)
    lam = float(np.exp(2 * th.log_sigma_F - 2 * th.log_sigma_c))
    c, *_ = np.linalg.lstsq(np.vstack([A, np.sqrt(lam) * np.diag(gam)]), np.concatenate([yA, np.zeros(L)]),
                            rcond=None)
    mu = np.asarray(mu)
    assert np.linalg.norm(mu - c) <= 1e-8 * np.linalg.norm(c)


def test_matches_the_acefit_blr_reference():
    """fixtures/si_fitted.npz was fitted by acefit!(Si_tiny, BLR) (julia/export_model.jl) with
    ACEpotentials' defaults: weights E 30 / F 1 / V 1, smoothness prior p = 4, the model's E0, and
    ONE noise variance, at the evidence optimum (no hyperprior).  With the same basis, data and
    weights, the shared-noise evidence optimum is that posterior mean to ACEfit's tolerance; the
    shared MAP differs only by the weak hyperprior; per-quantity noise is a different fit."""
    from scipy.optimize import minimize
    from ace_jax.eval import highest_precision
    from ace_jax.fit.hypers import from_array, to_array
    from ace_jax.fit.pipeline import fit
    from ace_jax.fit.pipeline.export import linear_arrays_from_mean, linear_model_arrays
    from ace_jax.fit.predict import fit_posterior
    ref = np.load(MODEL)

    def rel(got):
        return max(np.linalg.norm(got[k] - ref[k]) / np.linalg.norm(ref[k]) for k in ("WB", "Wpair"))
    with highest_precision():
        cfg, d, b, obj = _setup("--noise", "shared", "--weights", ACEFIT_WEIGHTS, train=XYZ)
        res = fit(cfg, d, **QUIET)
        x0 = np.asarray(to_array(res.theta), float)
        lik = jax.jit(jax.value_and_grad(lambda z: obj.lik(jnp.asarray(x0).at[6].set(z[0]).at[8].set(z[1]))))

        def f(z):                                                  # -evidence over (log sigma_c, log sigma)
            v, g = lik(jnp.asarray(z))
            return -float(v), -np.asarray(g, float)
        r = minimize(f, x0[[6, 8]], jac=True, method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-10})
        x = x0.copy(); x[6], x[7:10] = r.x[0], r.x[1]
        th = from_array(jnp.asarray(x))
        mu, _ = fit_posterior(th, obj.stats(th), b.prob, d.ds_train)
        evidence = linear_arrays_from_mean(d.z, d.E0, b.prob.cfg, np.asarray(mu))
        cfg2, d2, *_ = _setup("--weights", ACEFIT_WEIGHTS, train=XYZ)
        per_quantity = linear_model_arrays(fit(cfg2, d2, **QUIET))
    r_ev, r_map, r_pq = rel(evidence), rel(linear_model_arrays(res)), rel(per_quantity)
    print(f"|ace-jax - ACEfit BLR| / |ACEfit BLR|: shared evidence optimum {r_ev:.1e}, shared MAP {r_map:.1e}, "
          f"per-quantity MAP {r_pq:.1e}")
    assert r_ev <= 1e-5                    # measured 4e-7
    assert r_map <= 1e-2                   # the hyperprior on log sigma, log sigma_c: measured 5.6e-3
    assert r_pq >= 0.1                     # measured 0.55: per-quantity noise undoes the weights


# ---------------------------------------------------------------- rungs and the GP arm

@pytest.mark.parametrize("laplace", ["fd", "svi"])
def test_laplace_counts_the_shared_noise_once(shared40, laplace):
    import dataclasses
    from ace_jax.eval import highest_precision
    from ace_jax.fit.pipeline.rungs import run_rungs
    cfg, d, b, obj, mf = shared40
    cfg = dataclasses.replace(cfg, rungs=("map", "laplace"), laplace=laplace, n_draws=16, map_steps=200)
    with highest_precision():
        rg = run_rungs(cfg, b, obj, mf.theta, fixed=mf.fixed, **QUIET)
    for dr in rg.draws.values():
        assert np.array_equal(dr[:, 7], dr[:, 8]) and np.array_equal(dr[:, 9], dr[:, 8])
    assert np.std(rg.draws["laplace"][:, 8]) > 0
    if laplace == "fd":
        info = rg.info["laplace"]
        assert {"log_sigma_E", "log_sigma_V"} <= set(info["fixed"])
        assert len(info["eigenvalues"]) == 10 - len(info["fixed"])


def test_gp_arm_shared_smoke(tmp_path):
    from ace_jax.eval import highest_precision
    from ace_jax.fit.pipeline.mapfit import fit_map
    xyz = small_si_xyz(tmp_path / "si12.xyz")
    with highest_precision(), warnings.catch_warnings():
        warnings.simplefilter("ignore")                             # 10 L-BFGS steps: not converged
        cfg, d, b, obj = _setup("--noise", "shared", "--m-per-species", "4", "--map-steps", "10", train=xyz)
        assert cfg.arm == "gp"
        mf = fit_map(cfg, d, b, obj, **QUIET)
        _, g = obj.vg(jnp.asarray([float(v) for v in mf.theta]))
    sE, sF, sV = _sigmas(mf.theta)
    assert sE == sF == sV and np.isfinite(sF) and np.isfinite(mf.log_evidence)
    assert float(g[7]) == 0.0 and float(g[9]) == 0.0
    assert mf.convergence["noise"] == "shared"
