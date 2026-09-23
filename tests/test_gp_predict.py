import jax
import numpy as np
import pytest
from scipy.stats import norm, spearmanr

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.metrics import coverage, crps_gaussian, sigma_ratio, spearman, summarise
from ace_jax.fit.objective import Problem
from ace_jax.fit.predict import predict_fixed, predict_mixture

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")


def test_crps_gaussian_against_quadrature():
    y, mu, s = 0.3, 0.0, 0.7
    x = np.linspace(-10, 10, 200001)
    F = norm.cdf(x, mu, s)
    ref = np.trapezoid((F - (x >= y)) ** 2, x)
    # trapezoid error is O(h) at the kink x = y, ~2e-5 on this grid
    assert abs(float(crps_gaussian(np.array([y]), np.array([mu]), np.array([s]))[0]) - ref) < 1e-4


def test_metrics_basic():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.1, 1.0, 1000); y = rng.normal(0, s); mu = np.zeros(1000)
    assert abs(coverage(y, mu, s) - 0.68) < 0.05
    assert abs(spearman(s, np.abs(y)) - spearmanr(s, np.abs(y)).correlation) < 1e-12
    assert sigma_ratio(s) > 5.0
    d = summarise(y, mu, s)
    assert set(d) >= {"rmse", "mae", "crps", "coverage", "rho", "sigma_ratio", "rms_z"}
    d0 = summarise(np.r_[y, 0.0], np.r_[mu, 0.0], np.r_[s, 0.0])       # a zero-sigma exact prediction
    assert d0["n_dropped"] == 1 and np.isfinite(d0["crps"]) and abs(d0["rms_z"] - d["rms_z"]) < 1e-12


@pytest.fixture(scope="module")
def fitted():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    E0 = np.asarray(z["E0"])
    train, test = build_dataset(configs[:8], meta, E0, 4), build_dataset(configs[8:12], meta, E0, 4)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=4)
    X, S = site_features(model, cfg, train)
    ind = select_inducing(X, S, train.node_z, train.node_mask, 10, descriptor_scale(X, train.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    theta = Hypers(log_ell=np.log(0.8), log_A=np.log(0.05), log_alpha=np.log(1.0), log_r0=np.log(2.35),
                   log_eps=np.log(0.3), log_rho=np.log(4.0), log_sigma_c=np.log(0.3),
                   log_sigma_E=np.log(1e-3), log_sigma_F=np.log(0.02), log_sigma_V=np.log(0.02))
    return prob, train, test, theta, configs, E0


def test_predict_fixed_reproduces_training_data(fitted):
    prob, train, test, theta, configs, E0 = fitted
    with highest_precision():
        p = predict_fixed(theta, prob, train, train)
    E_true = np.array([c.energy for c in configs[:8]])
    nat = np.array([len(c.numbers) for c in configs[:8]])
    assert np.sqrt(np.mean(((np.asarray(p.E_mean) - E_true) / nat) ** 2)) < 2e-2   # eV/atom
    assert bool(jnp.all(p.E_var >= 0)) and bool(jnp.all(p.F_var >= 0)) and bool(jnp.all(p.V_var >= 0))
    assert p.F_mean.shape == (sum(len(c.numbers) for c in configs[:8]), 3)


def test_mixture_variance_exceeds_mean_variance(fitted):
    prob, train, test, theta, configs, E0 = fitted
    a = np.asarray(to_array(theta))
    draws = np.stack([a, a + np.array([0.3] + [0.0] * 9), a - np.array([0.3] + [0.0] * 9)])
    with highest_precision():
        pm = predict_mixture(draws, prob, train, test)
        p0 = predict_fixed(theta, prob, train, test)
    assert pm.E_mean.shape == p0.E_mean.shape
    assert float(pm.E_var.mean()) >= 0.99 * float(p0.E_var.mean())


def test_energy_variance_includes_dtc_prior_residual(fitted):
    """Ruling R29: E_var = SoR + sum_{ij in a} [k(x_i,x_j) - k_iM K_MM^-1 k_Mj].  With one
    inducing point and an O(0.1) residual amplitude, a configuration far from it in
    descriptor space gets a strictly larger energy variance; the term is never negative."""
    prob, train, test, theta, configs, E0 = fitted
    X, S = site_features(prob.model, prob.cfg, train)
    scale = descriptor_scale(X, train.node_mask)
    ind1 = select_inducing(X, S, train.node_z, train.node_mask, 1, scale)
    assert ind1.XM.shape[0] == 1
    prob1 = prob._replace(ind=ind1)
    theta1 = theta._replace(log_A=np.log(0.1))
    with highest_precision():
        p_dtc = predict_fixed(theta1, prob1, train, test)
        p_sor = predict_fixed(theta1, prob1, train, test, dtc=False)
        p_nod = predict_fixed(theta1, prob1, train, test, deriv_dtc=False)   # E-DTC on, F/V-DTC off
    term = np.asarray(p_dtc.E_var) - np.asarray(p_sor.E_var)
    assert term.shape == p_sor.E_var.shape and np.all(term >= 0.0)
    # the test config whose live sites are (RMS) farthest from the inducing point
    Xt, _ = site_features(prob.model, prob.cfg, test)
    xt = np.asarray(Xt * scale).reshape(-1, Xt.shape[-1])
    live, cfg_id = np.asarray(test.node_mask).reshape(-1), np.asarray(test.node_cfg).reshape(-1)
    d2 = np.sum((xt - np.asarray(ind1.XM)[0]) ** 2, axis=1)
    cfgs = np.unique(cfg_id[live])
    rms = [np.sqrt(d2[live & (cfg_id == c)].mean()) for c in cfgs]
    far = int(np.argmax(rms))
    assert term[far] > 0.0 and p_dtc.E_var[far] > p_sor.E_var[far]
    assert np.allclose(p_dtc.E_mean, p_sor.E_mean)
    # forces/virials now carry the derivative-DTC: >= the SoR variance; deriv_dtc=False
    # leaves them at SoR (dtc=False turns off both energy and derivative residuals).
    assert np.allclose(np.asarray(p_nod.F_var), np.asarray(p_sor.F_var))
    assert np.all(np.asarray(p_dtc.F_var) >= np.asarray(p_nod.F_var) - 1e-12)
    assert np.all(np.asarray(p_dtc.V_var) >= np.asarray(p_nod.V_var) - 1e-12)


def test_linear_limit_has_no_residual_variance(fitted):
    """M = 0 is the BLR limit: E_var must be the linear posterior variance only,
    independent of the (unidentified) kernel hyperparameters."""
    prob, train, test, theta, configs, E0 = fitted
    ind0 = select_inducing(jnp.zeros((1, 1, prob.cfg.D)), jnp.ones((1, 1)), jnp.zeros((1, 1), jnp.int32),
                           jnp.ones((1, 1), bool), 0, prob.ind.scale)
    prob0 = prob._replace(ind=ind0)
    with highest_precision():
        p1 = predict_fixed(theta, prob0, train, test)
        p2 = predict_fixed(theta._replace(log_A=theta.log_A + 2.0), prob0, train, test)
        p_sor = predict_fixed(theta, prob0, train, test, dtc=False)
    assert np.abs(p1.E_var - p2.E_var).max() < 1e-12 * np.abs(p1.E_var).max()
    assert np.abs(p1.E_var - p_sor.E_var).max() < 1e-12 * np.abs(p1.E_var).max()


def test_force_virial_dtc_is_derivative_of_energy(fitted):
    """Ruling R30: the force/virial DTC prior residuals are the matched mixed
    second derivative of the energy DTC residual _dtc_D, so F_var, V_var are the
    position/strain derivative of E_var (self-consistent predictive).  Checked
    against finite differences of _dtc_D on the cosine-SE kernel."""
    from ace_jax.fit.data import flat_edges
    from ace_jax.fit.predict import _dtc_D, _dtc_deriv_residual, _dtc_energy_residual
    from ace_jax.fit.rows import linear_rows
    prob, train, test, theta, configs, E0 = fitted
    batch = jax.tree.map(lambda a: a[0], train)
    rij, send, recv, m = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    ncfg = np.asarray(batch.node_cfg); nmask = np.asarray(batch.node_mask)
    with highest_precision():
        X = linear_rows(prob.model, prob.cfg, batch)[1]
        assert np.abs(np.asarray(_dtc_D(theta, prob, batch, jnp.zeros_like(rij), jnp.zeros_like(rij))
                                 - _dtc_energy_residual(theta, prob, batch, X))).max() < 1e-10
        Fv, Vv = _dtc_deriv_residual(theta, prob, batch)
    assert bool((Fv >= 0).all()) and bool((Vv >= 0).all())
    inc = lambda d: jnp.where(m[:, None], d[recv] - d[send], 0.0)
    E = rij.shape[0]
    def Dm(n, a, s, tt):
        dL = jnp.zeros((batch.nbr.shape[0], 3)).at[n, a].set(s)
        dR = jnp.zeros((batch.nbr.shape[0], 3)).at[n, a].set(tt)
        return float(_dtc_D(theta, prob, batch, inc(dL), inc(dR))[ncfg[n]])
    h = 1e-4
    for n in np.flatnonzero(nmask)[:4]:
        for a in range(3):
            fd = (Dm(n, a, h, h) - Dm(n, a, h, -h) - Dm(n, a, -h, h) + Dm(n, a, -h, -h)) / (4 * h * h)
            assert abs(fd - float(Fv[n, a])) < 1e-4 + 1e-3 * abs(float(Fv[n, a]))


def test_pops_predict_finite_and_aleatoric_inflates(tiny_linear_problem):
    """Task 10: uq='pops' on the linear arm returns FINITE per-config E sigma and
    per-component F sigma with the right shapes; --aleatoric strictly inflates it."""
    prob, ds = tiny_linear_problem
    theta = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
                   log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
                   log_sigma_F=np.log(0.01), log_sigma_V=np.log(0.01))
    with highest_precision():
        p = predict_fixed(theta, prob, ds, ds, uq="pops", aleatoric=False)  # misspec-only baseline
        pa = predict_fixed(theta, prob, ds, ds, uq="pops", aleatoric=True)
        pblr = predict_fixed(theta, prob, ds, ds)                       # uq='blr' default
    assert np.all(np.isfinite(p.E_var)) and np.all(p.E_var >= 0.0)
    assert np.all(np.isfinite(p.F_var)) and np.all(p.F_var >= 0.0)
    assert p.E_var.shape == p.E_mean.shape                             # per config
    assert p.F_var.shape == p.F_mean.shape == (p.F_mean.shape[0], 3)   # per component
    # mean is the BLR linear posterior mean -- untouched by the POPS variance path
    assert np.allclose(np.asarray(p.E_mean), np.asarray(pblr.E_mean))
    assert np.allclose(np.asarray(p.F_mean), np.asarray(pblr.F_mean))
    # aleatoric ADDS (sigma_q / w)^2 > 0 to every observed row -> strictly larger
    assert np.all(pa.E_var > p.E_var) and np.all(pa.F_var > p.F_var)


def test_pops_default_is_hypercube_aleatoric(tiny_linear_problem):
    """Revised decision: the POPS default is hypercube + aleatoric (matches the
    upstream popsregression default; the calibrated form).  'ensemble' centred with
    no aleatoric is the overconfident form and must NOT be the default."""
    prob, ds = tiny_linear_problem
    theta = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
                   log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
                   log_sigma_F=np.log(0.01), log_sigma_V=np.log(0.01))
    with highest_precision():
        d  = predict_fixed(theta, prob, ds, ds, uq="pops")                                    # defaults
        hc = predict_fixed(theta, prob, ds, ds, uq="pops", pops_form="hypercube", aleatoric=True)
        so = predict_fixed(theta, prob, ds, ds, uq="pops", pops_form="ensemble", aleatoric=False)
    # the default binds to hypercube + aleatoric ...
    assert np.allclose(np.asarray(d.E_var), np.asarray(hc.E_var))
    assert np.allclose(np.asarray(d.F_var), np.asarray(hc.F_var))
    # ... and is NOT the old (ensemble, misspec-only) form
    assert not np.allclose(np.asarray(d.E_var), np.asarray(so.E_var))


def test_pops_hypercube_form_and_leverage(tiny_linear_problem):
    """The hypercube posterior form and a >0 leverage percentile both stay finite."""
    prob, ds = tiny_linear_problem
    theta = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
                   log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
                   log_sigma_F=np.log(0.01), log_sigma_V=np.log(0.01))
    with highest_precision():
        p = predict_fixed(theta, prob, ds, ds, uq="pops", pops_form="hypercube",
                          leverage_pct=50.0)
    assert np.all(np.isfinite(p.E_var)) and np.all(p.E_var >= 0.0)
    assert np.all(np.isfinite(p.F_var)) and np.all(p.F_var >= 0.0)


def test_pops_requires_linear_arm(fitted):
    """uq='pops' is linear-arm only: the GP arm (M>0) must error clearly."""
    prob, train, test, theta, configs, E0 = fitted
    assert prob.ind.XM.shape[0] > 0
    with pytest.raises(ValueError, match="linear"):
        predict_fixed(theta, prob, train, test, uq="pops")


def test_deriv_dtc_requires_cosine(fitted):
    """matern32 is not twice differentiable at coincidence (Ruling R30); the
    derivative-DTC must refuse it rather than emit untrustworthy force UQ."""
    from ace_jax.fit.predict import _dtc_deriv_residual
    prob, train, test, theta, configs, E0 = fitted
    prob_m = prob._replace(spec=KernelSpec("matern32", True, prob.cfg.D))
    batch = jax.tree.map(lambda a: a[0], train)
    with pytest.raises(ValueError, match="cosine"):
        _dtc_deriv_residual(theta, prob_m, batch)
