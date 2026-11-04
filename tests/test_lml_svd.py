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
    assert svd_applicable(prob._replace(e0_prec=np.full(1, 1e16)), qs, tied=True)[0]       # joint-E0: handled
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
    out = capsys.readouterr().out
    assert "lml-solver svd: one SVD" in out                           # the SVD form ran (no silent QR fallback)
    assert "QR check at the MAP" in out and "agrees" in out           # and the QR check passed
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


def test_svd_map_is_checked_against_qr_and_falls_back(tmp_path, capsys, monkeypatch):
    """At the SVD-found MAP the QR evidence is evaluated once; when the two disagree (here: corrupted SVD
    statistics), the fit warns and finishes the MAP on the QR form, ending where a QR fit ends."""
    import warnings

    from ace_jax.cli import main
    from ace_jax.fit import lml_svd
    real = lml_svd.svd_statistics
    monkeypatch.setattr(lml_svd, "svd_statistics", lambda qs, prob, **kw: real(qs, prob, **kw)._replace(
        s2=real(qs, prob, **kw).s2 * 1.5))
    args = ["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(XYZ), "--ntrain", "20",
            "--ntest", "6", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial",
            "--m-per-species", "0", "--opt", "lbfgs", "--noise", "shared", "--r0", "2.35"]
    assert main(args + ["--out", str(tmp_path / "qr")]) == 0
    capsys.readouterr()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert main(args + ["--lml-solver", "svd", "--out", str(tmp_path / "svd")]) == 0
    out = capsys.readouterr().out
    assert "QR check at the MAP" in out
    assert any("continuing the MAP on the QR form" in str(x.message) for x in w)
    a = json.loads((tmp_path / "qr" / "theta_map.json").read_text())
    b = json.loads((tmp_path / "svd" / "theta_map.json").read_text())
    for k in ("log_sigma_c", "log_sigma_F"):
        assert b[k] == pytest.approx(a[k], rel=1e-5, abs=1e-6), k
    ea = json.loads((tmp_path / "qr" / "map_convergence.json").read_text())["log_evidence"]
    eb = json.loads((tmp_path / "svd" / "map_convergence.json").read_text())["log_evidence"]
    assert eb == pytest.approx(ea, rel=1e-8)                          # reported by the QR form, not the bad SVD


def test_svd_map_on_near_interpolating_mace_labels(tmp_path):
    """The case the QR form exists for: Cantor with MACE labels (tutorial 3's data), which the model nearly
    interpolates.  The SVD-form MAP passes its QR check without falling back and matches the QR fit."""
    import pathlib

    from ase.io import read, write

    from ace_jax.basis.model import BasisSpec, build_basis
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    data = pathlib.Path(__file__).resolve().parents[1] / "docs" / "user" / "tutorials" / "data" / "cantor"
    write(tmp_path / "train.xyz", read(data / "cantor_train.xyz", ":"))
    b = build_basis(BasisSpec(order=2, max_degree=4, elements=("Cr", "Mn", "Fe", "Co", "Ni"), rcut=5.5))
    res, logs = {}, {}
    for solver in ("qr", "svd"):
        cfg = FitConfig(model=b, arm="linear", m_per_species=0, e0="lsq", opt="lbfgs", r0=None, rungs=("map",),
                        noise="shared", lml_solver=solver, energy_key="mace_energy", force_key="mace_force",
                        virial_key="mace_virial").validate()
        d = load_fit_data(cfg, train=str(tmp_path / "train.xyz"), log=lambda *a: None)
        logs[solver] = []
        res[solver] = fit(cfg, d, log=logs[solver].append)
    assert any("QR check at the MAP agrees" in line for line in logs["svd"]), logs["svd"][-5:]
    q, s = res["qr"].theta, res["svd"].theta
    for k in ("log_sigma_c", "log_sigma_F"):
        assert float(getattr(s, k)) == pytest.approx(float(getattr(q, k)), rel=1e-5, abs=1e-6), k
    assert res["svd"].map.log_evidence == pytest.approx(res["qr"].map.log_evidence, rel=1e-8)


def test_svd_past_scipy_lp64_limit_uses_numpy_ilp64(m0, monkeypatch):
    """Past SciPy's 32-bit LAPACK indexing (L > 46,340: scipy.linalg.svd refuses a full U) the SVD goes through
    NumPy's ILP64 LAPACK; forced here by a small limit, the statistics and the evidence are the SciPy path's."""
    from ace_jax.fit import lml_svd
    prob, qs = m0
    assert lml_svd._numpy_ilp64()
    ref = lml_svd.svd_statistics(qs, prob, driver="gesdd")
    monkeypatch.setattr(lml_svd, "LP64_MAX", 10)
    sv = lml_svd.svd_statistics(qs, prob, driver="gesdd")
    np.testing.assert_allclose(np.asarray(sv.s2), np.asarray(ref.s2), rtol=1e-10, atol=1e-12 * float(ref.s2.max()))
    for ls, lc in POINTS:
        th = _theta(ls, lc)
        a, b = float(lml_svd.log_marginal_likelihood_svd(th, sv)), float(lml_svd.log_marginal_likelihood_svd(th, ref))
        assert abs(a - b) < 1e-9 * float(ref.yy) * np.exp(-2 * ls) + 1e-9 * abs(b), (ls, lc, a, b)
    with pytest.warns(UserWarning, match="ILP64"):
        lml_svd.svd_statistics(qs, prob, driver="gesvd")
