"""Batched updating-QR merges (stats.linear_qr_statistics, stats.host_qr_stream,
stats.device_qr_stream): buffering the live weighted rows of consecutive batches and merging
once per ~L rows gives the same QRStats as merging every batch, up to the QR's row-sign
ambiguity.  Consumers read R only through R^T R and R^T c (QRStats.G/b, and objective._qr_S,
which re-QRs and fixes the signs), so those, and the LML, its gradient, the posterior mean and
log-det downstream, are what is compared -- at ~1e-12 relative (the mean: at its conditioning),
on the host and the scan path, with the default buffer, a ragged one, and per-batch merging as
the reference."""
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
from ace_jax.fit.objective import Problem, log_marginal_likelihood_qr, posterior_from_qr
from ace_jax.fit.stats import QRStats, linear_qr_statistics, linear_statistics, merge_target

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing GP fixtures")


def _theta():
    return Hypers(log_ell=np.log(0.8), log_A=np.log(0.05), log_alpha=0.0, log_r0=np.log(2.35),
                  log_eps=np.log(0.3), log_rho=np.log(4.0), log_sigma_c=np.log(0.3),
                  log_sigma_E=np.log(1e-3), log_sigma_F=np.log(2e-2), log_sigma_V=np.log(2e-2))


def _problem(name, n_configs, per_batch):
    model, meta, z = load(FIXTURE_DIR / name)
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:n_configs]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), per_batch, log=lambda *a: None)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=per_batch)
    L = cfg.len_basis
    gamma = jnp.asarray(z["gamma"]) if "gamma" in z else jnp.ones(L)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))  # M = 0
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, gamma, default_prior(2.35),
                   lml_solver="qr")
    return prob, ds


@pytest.fixture(scope="module", params=[("si_fitted.npz", 24, 2), ("si_1429.npz", 30, 2)],
                ids=["si_tiny", "si_1429"])
def case(request):
    prob, ds = _problem(*request.param)
    with highest_precision():
        ref = linear_qr_statistics(prob.model, prob.cfg, ds, merge_rows=0, host=True)   # every batch
        lin = linear_statistics(prob.model, prob.cfg, ds)
    return prob, ds, ref, lin


def _downstream(theta, qs, prob):
    with highest_precision():
        v, g = jax.value_and_grad(lambda t: log_marginal_likelihood_qr(t, qs, prob))(theta)
        mu, Lc = posterior_from_qr(theta, qs, prob)
    return float(v), np.asarray(to_array(g)), np.asarray(mu), 2.0 * float(jnp.sum(jnp.log(jnp.diag(Lc))))


def _assert_equivalent(qs, ref, lin, prob):
    for q in "EFV":
        G, b = np.asarray(getattr(ref, f"G_{q}")), np.asarray(getattr(ref, f"b_{q}"))
        assert np.abs(np.asarray(getattr(qs, f"G_{q}")) - G).max() < 1e-12 * np.abs(G).max(), q
        assert np.abs(np.asarray(getattr(qs, f"b_{q}")) - b).max() < 1e-12 * np.abs(b).max(), q
        Gl = np.asarray(getattr(lin, f"G_{q}"))                           # and the Gram stream's
        assert np.abs(np.asarray(getattr(qs, f"G_{q}")) - Gl).max() < 1e-12 * np.abs(Gl).max(), q
        R = np.asarray(getattr(qs, f"R_{q}"))
        assert R.shape == np.asarray(getattr(ref, f"R_{q}")).shape and np.allclose(np.tril(R, -1), 0.0), q
        for k in ("yy", "n", "logw"):
            assert np.isclose(float(getattr(qs, f"{k}_{q}")), float(getattr(ref, f"{k}_{q}")), rtol=1e-13, atol=0)
    th = _theta()
    v, g, mu, ld = _downstream(th, qs, prob)
    v0, g0, mu0, ld0 = _downstream(th, ref, prob)
    scale = sum(float(getattr(ref, f"yy_{q}")) * np.exp(-2.0 * float(getattr(th, f"log_sigma_{q}"))) for q in "EFV")
    assert abs(v - v0) < 1e-12 * scale                                    # LML: relative to its cancelled terms
    assert np.abs(g - g0).max() < 1e-12 * scale
    # the mean is a solve at kappa(G + Lambda) (~2e13 on si_1429): two per-batch streams that differ
    # only in LAPACK (host vs scan) already disagree by ~1e-10 there; batching moves it no further
    assert np.abs(mu - mu0).max() < 1e-8 * np.abs(mu0).max()
    assert abs(ld - ld0) < 1e-12 * abs(ld0)


@pytest.mark.parametrize("host", [True, False], ids=["host", "scan"])
@pytest.mark.parametrize("merge_rows", [None, 37], ids=["default", "ragged"])
def test_batched_merges_equal_per_batch_merges(case, host, merge_rows):
    """The default buffer (~L rows per quantity) and a small ragged one (37 rows: merges land
    mid-stream, the tail is flushed) give the per-batch QRStats, on the host loop and the scan."""
    prob, ds, ref, lin = case
    with highest_precision():
        qs = linear_qr_statistics(prob.model, prob.cfg, ds, merge_rows=merge_rows, host=host)
    assert isinstance(qs, QRStats)
    _assert_equivalent(qs, ref, lin, prob)


def test_scan_per_batch_merge_matches_host(case):
    """merge_rows=0 on the scan is the old per-batch scan: the same statistics as the host loop."""
    prob, ds, ref, lin = case
    with highest_precision():
        qs = linear_qr_statistics(prob.model, prob.cfg, ds, merge_rows=0, host=False)
    _assert_equivalent(qs, ref, lin, prob)


def test_merges_count_live_rows_only(case, monkeypatch):
    """The host merges once per >= target live rows per quantity: zero rows (padded configs and
    nodes, absent virials, an isolated atom's zero-target energy row) neither cost a merge nor fill
    the buffer."""
    from ace_jax.fit import stats
    prob, ds, ref, lin = case
    L, calls = prob.cfg.len_basis, []
    merge = stats._host_merge
    monkeypatch.setattr(stats, "_host_merge", lambda R, c, P, y: calls.append(len(y)) or merge(R, c, P, y))
    assert merge_target(L) == L and merge_target(L, 0) == 0
    rows = jax.jit(lambda b: stats._qr_rows(prob.model, prob.cfg, b))
    per = [[int((np.any(np.asarray(P) != 0, 1) | (np.asarray(y) != 0)).sum()) for P, y, *_ in
            rows(jax.tree.map(lambda a: a[i], ds))] for i in range(ds.n_batches)]  # noqa: B023  live rows per batch
    w_live = sum(int((np.asarray(w) > 0).sum()) * k for w, k in ((ds.w_E, 1), (ds.w_F, 3), (ds.w_V, 6)))
    assert sum(map(sum, per)) <= w_live < sum(np.asarray(y).size for y in (ds.y_E, ds.y_F, ds.y_V))   # padding dropped
    for mr in (None, 37, 0):
        calls.clear()
        linear_qr_statistics(prob.model, prob.cfg, ds, merge_rows=mr, host=True)
        t, want = max(merge_target(L, mr), 1), []
        for q in range(3):
            fill = 0
            for p in per:
                fill += p[q]
                if fill >= t:
                    want.append(fill); fill = 0
            if fill:
                want.append(fill)
        assert sorted(calls) == sorted(want), (mr, calls, want)


def test_solve_qr_factor_scan_matches_host():
    """solve._qr_factor (the GP / fallback posterior's streaming QR, prior-seeded) gives the same
    R^T R and R^T d on the scan as on the host."""
    from ace_jax.fit.solve import _qr_factor
    prob, ds = _problem("si_fitted.npz", 24, 2)
    th = _theta()
    with highest_precision():
        Rh, dh = _qr_factor(prob, ds, th, host=True)
        Rs, ds_ = _qr_factor(prob, ds, th, host=False)
    Rh, dh, Rs, ds_ = map(np.asarray, (Rh, dh, Rs, ds_))
    G = Rh.T @ Rh
    assert np.abs(Rs.T @ Rs - G).max() < 1e-12 * np.abs(G).max()
    assert np.abs(Rs.T @ ds_ - Rh.T @ dh).max() < 1e-12 * np.abs(Rh.T @ dh).max()
