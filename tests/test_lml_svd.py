"""The SVD form of the linear arm's evidence (`aj fit --lml-solver svd`, fit/lml_svd.py): with shared noise and the
smoothness prior gamma^2 / sigma_c^2, one SVD of R D^-1 (R the merged QR factor, D = diag(gamma)) makes every LML
evaluation and its gradient O(L).  It must equal the QR form (objective.log_marginal_likelihood_qr) to roundoff."""
import jax

jax.config.update("jax_enable_x64", True)
import json  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from conftest import FIXTURE_DIR  # noqa: E402

from ace_jax.eval import highest_precision, load  # noqa: E402
from ace_jax.fit.data import build_dataset, load_configs  # noqa: E402
from ace_jax.fit.hypers import Hypers, default_prior  # noqa: E402
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features  # noqa: E402
from ace_jax.fit.kernels import KernelSpec  # noqa: E402
from ace_jax.fit.objective import Problem, log_marginal_likelihood_qr  # noqa: E402
from ace_jax.fit.stats import linear_qr_statistics  # noqa: E402

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"


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
                   default_prior(2.35), lml_solver="qr")
    with highest_precision():
        qs = linear_qr_statistics(model, cfg, ds)
    return prob, qs


def _theta(log_sigma, log_sigma_c):
    """Shared noise: sigma_E = sigma_F = sigma_V (the tie mapfit applies)."""
    return Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0, log_rho=0.0,
                  log_sigma_c=log_sigma_c, log_sigma_E=log_sigma, log_sigma_F=log_sigma, log_sigma_V=log_sigma)


POINTS = [(np.log(0.05), np.log(1.0)), (np.log(0.3), np.log(10.0)), (np.log(1e-4), np.log(1e4)),   # last: stiff
          (np.log(2.0), np.log(0.01))]


@pytest.mark.parametrize("ls,lc", POINTS)
def test_svd_lml_matches_qr(m0, ls, lc):
    from ace_jax.fit.lml_svd import log_marginal_likelihood_svd, svd_statistics
    prob, qs = m0
    sv = svd_statistics(qs, prob)
    th = _theta(ls, lc)
    with highest_precision():
        ref = float(log_marginal_likelihood_qr(th, qs, prob))
    val = float(log_marginal_likelihood_svd(th, sv))
    # roundoff is relative to the terms the LML cancels (y^T y / sigma^2: ~5e10 at the stiff point, against an
    # LML of ~190), as in test_lml_qr._scale -- not to the LML itself
    scale = sum(float(getattr(qs, f"yy_{q}")) for q in "EFV") * float(np.exp(-2.0 * ls))
    assert abs(val - ref) <= 1e-13 * max(scale, abs(ref), 1.0), (val, ref, scale)


@pytest.mark.parametrize("ls,lc", POINTS[:2])
def test_svd_lml_gradient_matches_qr(m0, ls, lc):
    """d/d(log_sigma, log_sigma_c) under the tie: the only two coordinates the linear arm's evidence reads."""
    from ace_jax.fit.lml_svd import log_marginal_likelihood_svd, svd_statistics
    prob, qs = m0
    sv = svd_statistics(qs, prob)

    def f(lml, x):
        return lml(_theta(x[0], x[1]))
    x = jnp.array([ls, lc])
    with highest_precision():
        g_ref = jax.grad(lambda x: f(lambda t: log_marginal_likelihood_qr(t, qs, prob), x))(x)
    g = jax.grad(lambda x: f(lambda t: log_marginal_likelihood_svd(t, sv), x))(x)
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), rtol=1e-8, atol=1e-8 * float(jnp.abs(g_ref).max()))


def test_svd_form_refuses_what_it_cannot_represent(m0):
    from ace_jax.fit.lml_svd import svd_applicable
    prob, qs = m0
    assert svd_applicable(prob, qs, tied=True)[0]
    assert not svd_applicable(prob, qs, tied=False)[0]                               # independent noises
    assert not svd_applicable(prob._replace(e0_prec=np.full(1, 1e16)), qs, tied=True)[0]   # joint-E0 columns
    assert not svd_applicable(prob, qs._replace(rho_E=jnp.zeros(()), rho_F=jnp.zeros(()), rho_V=jnp.zeros(())),
                              tied=True)[0]                                          # float32 rows


def test_fit_with_svd_solver_matches_qr(tmp_path, capsys):
    from ace_jax.cli import main
    args = ["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--ntrain", "20",
            "--ntest", "6", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial",
            "--m-per-species", "0", "--opt", "lbfgs", "--noise", "shared", "--r0", "2.35"]
    assert main(args + ["--out", str(tmp_path / "qr")]) == 0
    capsys.readouterr()
    assert main(args + ["--lml-solver", "svd", "--out", str(tmp_path / "svd")]) == 0
    assert "lml-solver svd: one SVD" in capsys.readouterr().out       # the SVD form ran (no silent QR fallback)
    a = json.loads((tmp_path / "qr" / "theta_map.json").read_text())
    b = json.loads((tmp_path / "svd" / "theta_map.json").read_text())
    for k in ("log_sigma_c", "log_sigma_F"):
        assert b[k] == pytest.approx(a[k], rel=1e-6, abs=1e-7), k
    ma = np.load(tmp_path / "qr" / "model.npz"); mb = np.load(tmp_path / "svd" / "model.npz")
    for k in ma.files:
        if np.issubdtype(ma[k].dtype, np.floating):
            # two L-BFGS runs stop within the MAP's tolerance of each other (theta to ~1e-7 above), which moves
            # coefficients by a few 1e-6 relative; the two evidence forms themselves agree to roundoff
            np.testing.assert_allclose(mb[k], ma[k], rtol=2e-5, atol=1e-8 * max(1.0, float(np.abs(ma[k]).max())))
