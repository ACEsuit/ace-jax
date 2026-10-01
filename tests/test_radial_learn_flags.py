"""Learned-radial helpers and bench-driver options: prior sized from gamma, theta-MAP /
holdout on precomputed statistics, run.py --extra-train / --tol / --init-radials, and
rmse_npz.py (validation errors of the written model.npz)."""
import json
import subprocess
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from conftest import ROOT
from test_gp_learn_radial import MODEL, THETA, XYZ, make_problem

pytestmark = pytest.mark.skipif(not (XYZ.exists() and MODEL.exists()), reason="missing fixtures")


@pytest.fixture(scope="module")
def small():
    return make_problem()


def test_prior_precision_sizes_from_gamma(small):
    from ace_jax.fit.objective import prior_precision
    prob, _, _ = small
    L = prob.cfg.len_basis
    Lam, _ = prior_precision(THETA, prob._replace(gamma=jnp.full(L + 3, 2.0)))
    assert Lam.shape == (L + 3, L + 3)
    np.testing.assert_allclose(np.diag(Lam), 4.0 / 0.3 ** 2)
    Lam0, _ = prior_precision(THETA, prob)
    assert Lam0.shape == (L, L)


def test_theta_map_and_holdout_take_precomputed_stats(small):
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.radial_learn import holdout_score, theta_map_linear
    from ace_jax.fit.stats import linear_statistics
    prob, ds, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    W = prob.model.rnl_Wnlq
    lin = linear_statistics(prob.model, prob.cfg, ds)
    a1 = theta_map_linear(prob, ds, W, steps=20)
    a2 = theta_map_linear(prob, ds, None, steps=20, lin=lin)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    a = to_array(THETA)
    lv = linear_statistics(prob.model, prob.cfg, ds_val)
    s1 = holdout_score(W, a, a, prob, ds, ds_val)
    s2 = holdout_score(None, a, a, prob, ds, ds_val, lin_fit=lin, lin_val=lv)
    np.testing.assert_allclose(s1, s2, rtol=1e-12)


def _driver(tmp_path, *extra):
    args = [sys.executable, str(ROOT / "bench/learn_radial/run.py"), "--model", str(MODEL),
            "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
            "--virial-key", "dft_virial", "--ntrain", "8", "--nval", "8", "--batch", "4",
            "--n-q", "20", "--steps", "3", "--map-steps", "20", "--lam-grid", "0", "--out", str(tmp_path), *extra]
    return subprocess.run(args, capture_output=True, text=True)


def test_bench_driver_extra_train_tol_and_rmse_npz(tmp_path):
    """--extra-train appends configs to the training split only; --tol reaches the learners;
    rmse_npz.py scores the written model.npz."""
    r = _driver(tmp_path, "--extra-train", str(XYZ), "--tol", "0")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "extra training configs:" in r.stdout
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["extra_train"] == [str(XYZ)] and s["tol"] == 0.0 and s["nval"] == 8
    r2 = subprocess.run([sys.executable, str(ROOT / "bench/learn_radial/rmse_npz.py"), "--model", str(MODEL),
                         "--data", str(XYZ), "--energy-key", "dft_energy", "--force-key", "dft_force",
                         "--virial-key", "dft_virial", "--ntrain", "8", "--nval", "8",
                         "--model-npz", str(tmp_path / "model.npz"), "--out", str(tmp_path / "rmse.json")],
                        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr[-3000:]
    (row,) = json.loads((tmp_path / "rmse.json").read_text()).values()
    assert np.isfinite(row["E_rmse_meV_atom"]) and np.isfinite(row["F_rmse_meV_A"])


def test_bench_driver_init_radials(tmp_path):
    """--init-radials replaces the start (with --steps 0 the written radials are that start,
    normalised, not the model's own); a wrong shape fails fast."""
    from ace_jax.eval import load
    from ace_jax.fit.radial_model import to_analytic
    m, _, _ = load(MODEL)
    W = np.asarray(to_analytic(m, 20)[0].rnl_Wnlq)
    act = np.abs(W).sum(-1) > 0
    Wp = W + 0.3 * np.abs(W).mean() * np.random.default_rng(0).standard_normal(W.shape) * act[..., None]
    np.save(tmp_path / "W.npy", Wp)
    r0 = _driver(tmp_path / "base", "--steps", "0")
    r = _driver(tmp_path / "init", "--init-radials", str(tmp_path / "W.npy"), "--steps", "0")
    assert r0.returncode == 0 and r.returncode == 0, (r0.stderr + r.stderr)[-3000:]
    assert "starting from radials" in r.stdout
    assert json.loads((tmp_path / "init" / "summary.json").read_text())["init_radials"] == str(tmp_path / "W.npy")
    Wb, Wi = np.load(tmp_path / "base" / "rnl_Wnlq.npy"), np.load(tmp_path / "init" / "rnl_Wnlq.npy")
    assert not np.allclose(Wb, Wi)
    # same span per radial row up to the gauge scale: Wi is a positive rescaling of Wp row by row
    cos = np.sum(Wi * Wp, -1)[act] / (np.linalg.norm(Wi, axis=-1)[act] * np.linalg.norm(Wp, axis=-1)[act])
    np.testing.assert_allclose(cos, 1.0, atol=1e-10)
    np.save(tmp_path / "bad.npy", W[..., :5])
    r = _driver(tmp_path / "bad", "--init-radials", str(tmp_path / "bad.npy"))
    assert r.returncode != 0 and "--init-radials shape" in (r.stderr + r.stdout)
