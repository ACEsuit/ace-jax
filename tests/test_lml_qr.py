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
    """At STIFF the Cholesky LML is NaN on lestrade (on other machines it may instead be finite
    and wrong, so it is not asserted on); the QR LML and its gradient are finite and at
    roundoff of the dense full-design QR."""
    prob, ds, lin, qs = m0
    theta = _theta(**STIFF)
    with highest_precision():
        v, g = jax.value_and_grad(lambda t: log_marginal_likelihood_qr(t, qs, prob))(theta)
        ref = _dense_reference(prob, ds, theta)
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
        assert np.isfinite(float(make_lml(prob._replace(lml_solver="qr"), ds)(to_array(_theta(**STIFF)))))


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
    """The default fit of this fixture already sits where only the QR form is reliable: its MAP
    noise is near the floor (sigma_E ~ 1.2e-4).  There the QR LML is at roundoff of a dense QR of
    the full design and its posterior mean matches the full-design QR's, on every machine.

    The Cholesky form is not asserted on, because its error there depends on the machine: its
    evidence at this point is ~250x further from the reference than the QR's on lestrade, its
    posterior mean off by ~1e-3, and the Cholesky MAP run ends at a machine-dependent, too-high
    evidence (573.3014 on lestrade, 573.3091 on a GitHub runner, against 573.2992 from QR on both)."""
    q, c = pipeline_fits["qr"], pipeline_fits["cholesky"]
    assert c.built.prob.lml_solver == "cholesky" and np.isfinite(c.map.log_evidence)
    prob, ds, th = q.built.prob, q.data.ds_train, q.theta
    assert float(th.log_sigma_E) < np.log(2e-4)
    with highest_precision():
        qs, lin = linear_qr_statistics(prob.model, prob.cfg, ds), linear_statistics(prob.model, prob.cfg, ds)
        v_q = float(log_marginal_likelihood_qr(th, qs, prob))
        mu_q = np.asarray(posterior_from_qr(th, qs, prob)[0])
        mu_s = np.asarray(posterior_qr(prob, ds, th)[0])
        ref = _dense_reference(prob, ds, th)
    assert abs(v_q - ref) < 1e-14 * _scale(th, lin)                     # QR: at roundoff
    assert np.abs(mu_q - mu_s).max() < 1e-6 * np.abs(mu_s).max()        # QR mean: the full-design QR's


_THREADED_QR = r"""
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, pathlib, sys
from ase.io import read, write
from ace_jax.basis.model import BasisSpec, build_basis
from ace_jax.fit.pipeline import FitConfig, load_fit_data
from ace_jax.fit.pipeline.problem import build_problem
from ace_jax.fit.stats import linear_qr_statistics
data, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
write(work / "train.xyz", read(data / "cantor_train.xyz", ":"))
b = build_basis(BasisSpec(order=2, max_degree=4, elements=("Cr", "Mn", "Fe", "Co", "Ni"), rcut=5.5))
cfg = FitConfig(model=b, arm="linear", m_per_species=0, e0="lsq", opt="lbfgs", r0=None, rungs=("map",),
                energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial").validate()
d = load_fit_data(cfg, train=str(work / "train.xyz"), log=lambda *a: None)
prob = build_problem(cfg, d).prob
qs = linear_qr_statistics(prob.model, prob.cfg, d.ds_train)
print("NONFINITE", sum(int((~np.isfinite(np.asarray(v))).sum()) for v in qs._asdict().values()))
"""


@pytest.mark.parametrize("threads", ["4", "8"])
def test_qr_statistics_finite_with_threaded_openblas(tmp_path, threads):
    """Regression (#61): built as a lax.scan of jnp.linalg.qr on the CPU, the energy factor of
    tutorial 3's 655-function categorical basis came out NaN whenever OpenBLAS ran 4 or 8
    threads -- a GitHub runner's default -- because XLA calls LAPACK from its worker threads.
    The CPU path merges on the host (stats.host_qr_stream).  OPENBLAS_NUM_THREADS must be set
    before OpenBLAS loads, so this runs in a fresh process."""
    import os, pathlib, subprocess, sys
    data = pathlib.Path(__file__).resolve().parents[1] / "docs" / "user" / "tutorials" / "data" / "cantor"
    env = dict(os.environ, OPENBLAS_NUM_THREADS=threads, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    r = subprocess.run([sys.executable, "-c", _THREADED_QR, str(data), str(tmp_path)], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "NONFINITE 0" in r.stdout, r.stdout[-500:]
