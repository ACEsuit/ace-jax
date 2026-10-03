"""POPS (Swinburne & Perez, arXiv:2402.01810) on the BLR's own loss: rows weighted
by w/sigma_q (the fit's relative E/F/V loss weights), regulariser ridge * Gamma^2
with the default ridge 'blr' = 1/sigma_c^2, so c* IS the BLR mean and A the BLR
posterior covariance -- the corrections sit around the optimum of the loss that
defines A.  sigma_q never enters the PREDICTIVE (no aleatoric/noise term)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision
from ace_jax.fit.hypers import Hypers
from ace_jax.fit.objective import posterior
from ace_jax.fit.pops import _fit_hypercube_cov
from ace_jax.fit.predict import PopsRidgePath, predict_fixed, select_pops_ridge
from ace_jax.fit.rows import linear_rows
from ace_jax.fit.stats import sufficient_statistics

# sigma_q deliberately far from 1 and unequal: the loss weights must be 1/sigma_q
THETA = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
               log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
               log_sigma_F=np.log(0.05), log_sigma_V=np.log(0.3))



@pytest.fixture(autouse=True)
def _clear_jax_caches_per_test():
    """This module only: clear JAX's in-process caches after every test (conftest clears
    per module). Keeping the host-row tests' compiled state across tests here aborted
    XLA intermittently inside backend_compile (jaxlib 0.11.2, CPU, local runs; CI passed)."""
    yield
    import jax
    jax.clear_caches()

def _dense_structural(prob, ds, c, theta=THETA):
    """Loss-weighted member rows (w/sigma_q * phi) and residuals (w/sigma_q * r) of
    every non-padded observation, plus the raw energy rows -- a dense reference."""
    sg = {q: float(np.exp(getattr(theta, f"log_sigma_{q}"))) for q in "EFV"}
    Pw, rw, PE = [], [], []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        r, _, _ = linear_rows(prob.model, prob.cfg, b)
        L = r.E.shape[-1]
        for q, phi, y, w in (("E", r.E, b.y_E, b.w_E),
                             ("F", r.F.reshape(-1, L), b.y_F.reshape(-1), jnp.repeat(b.w_F, 3)),
                             ("V", r.V.reshape(-1, L), b.y_V.reshape(-1), jnp.repeat(b.w_V, 6))):
            phi, y, w = map(np.asarray, (phi, y, w))
            k = w > 0
            w = w / sg[q]
            Pw.append(w[k, None] * phi[k]); rw.append(w[k] * (y[k] - phi[k] @ c))
        PE.append(np.asarray(r.E)[np.asarray(b.w_E) > 0])
    return np.concatenate(Pw), np.concatenate(rw), np.concatenate(PE)


def _lam(ridge, M, g2, theta=THETA):
    """Absolute ridge: 'blr' = 1/sigma_c^2 (the BLR prior); a number is relative to
    the largest eigenvalue of the Gamma-scaled loss Gram."""
    if ridge == "blr":
        return float(np.exp(-2.0 * theta.log_sigma_c))
    D = np.sqrt(g2)
    return ridge * np.linalg.eigvalsh(M / D[:, None] / D[None, :]).max()


def _dense_ridge_mean(prob, ds, ridge, theta=THETA):
    """c*(ridge) = (M + lam Gamma^2)^-1 b from the dense loss-weighted rows."""
    Pw, yw, _ = _dense_structural(prob, ds, np.zeros(prob.cfg.len_basis), theta)   # yw = w/s * y at c = 0
    g2 = np.asarray(prob.gamma) ** 2
    M = Pw.T @ Pw
    return np.linalg.solve(M + _lam(ridge, M, g2, theta) * np.diag(g2), Pw.T @ yw)


def _reference_var(prob, ds, ridge):
    c = _dense_ridge_mean(prob, ds, "blr")                # the mean is ALWAYS the BLR mean
    Pw, rw, PE = _dense_structural(prob, ds, c)
    g2 = np.asarray(prob.gamma) ** 2
    M = Pw.T @ Pw
    A = np.linalg.inv(M + _lam(ridge, M, g2) * np.diag(g2))
    h = np.einsum("ij,jk,ik->i", Pw, A, Pw)
    k = h > 0                                       # zero-leverage rows (phi = 0) are not members
    deltas = (Pw[k] @ A) * (rw[k] / h[k])[:, None]
    cov = np.asarray(_fit_hypercube_cov(jnp.asarray(deltas)))
    return np.sum((PE @ cov) * PE, axis=1), M, g2


@pytest.mark.parametrize("ridge", ["blr", 1e-3])
def test_pops_matches_dense_blr_loss_reference(tiny_linear_problem, ridge):
    """Rows weighted w/sigma_q, regulariser ridge*Gamma^2 for A; the mean is the
    BLR mean whatever the uncertainty ridge."""
    prob, ds = tiny_linear_problem
    with highest_precision():
        p = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge=ridge)
        ref, _, _ = _reference_var(prob, ds, ridge)
    assert np.allclose(np.asarray(p.E_var), ref, rtol=1e-6, atol=1e-14)


def test_blr_ridge_is_the_blr_posterior(tiny_linear_problem):
    """Default ridge 'blr': c* = the BLR posterior mean, A = the BLR posterior
    covariance (objective.posterior) -- POPS around the fit that is reported."""
    prob, ds = tiny_linear_problem
    with highest_precision():
        st = sufficient_statistics(THETA, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        mu, L = posterior(THETA, st, prob)
        path = PopsRidgePath(THETA, prob, ds)
        cov = np.linalg.inv(np.asarray(L) @ np.asarray(L).T)
        assert np.allclose(np.asarray(path.c_star_at("blr")), np.asarray(mu), rtol=1e-6, atol=1e-10)
        # eigh- vs Cholesky-based inverse of a precision with cond ~1e7: they agree
        # to ~1e-9 of the largest entry (cond * eps), so atol is relative to max|cov|
        assert np.allclose(np.asarray(path.A("blr")), cov, rtol=1e-6, atol=1e-8 * np.abs(cov).max())
        p = predict_fixed(THETA, prob, ds, ds, uq="pops")                  # default ridge
        pb = predict_fixed(THETA, prob, ds, ds)
    for f in ("E_mean", "F_mean", "V_mean"):
        assert np.allclose(np.asarray(getattr(p, f)), np.asarray(getattr(pb, f)), rtol=1e-8, atol=1e-12), f


def test_pops_paper_per_quantity_ridge(tiny_linear_problem):
    """A per-quantity ridge dict: ONE mean -- the BLR mean, pinned whatever the
    ridges -- and each quantity's own A(ridge_q) for its uncertainty."""
    prob, ds = tiny_linear_problem
    with highest_precision():
        pd = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge={"E": 1e-2, "F": 1e-4, "V": 1e-4})
        pF = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge=1e-4)
        pb = predict_fixed(THETA, prob, ds, ds)                             # BLR
        path = PopsRidgePath(THETA, prob, ds)
        path.use_mean("blr")
        vE = pops_var(_test_rows(prob, ds), path.posterior(1e-2))
    for f in ("E_mean", "F_mean", "V_mean"):
        assert np.allclose(np.asarray(getattr(pd, f)), np.asarray(getattr(pb, f)), rtol=1e-8, atol=1e-12), f
        assert np.allclose(np.asarray(getattr(pF, f)), np.asarray(getattr(pb, f)), rtol=1e-8, atol=1e-12), f
    for f in ("F_var", "V_var"):
        assert np.allclose(np.asarray(getattr(pd, f)), np.asarray(getattr(pF, f)), rtol=1e-10, atol=1e-14), f
    assert np.allclose(np.asarray(pd.E_var), np.asarray(vE), rtol=1e-8, atol=1e-14)
    assert not np.allclose(np.asarray(pd.E_var), np.asarray(pF.E_var))


def test_pops_mean_is_the_ridge_solution_from_the_shared_factorisation(tiny_linear_problem):
    """c*(ridge) = (M + lam Gamma^2)^-1 b on the loss-weighted rows, read off the
    one eigendecomposition, for 'blr' and numeric ridges."""
    prob, ds = tiny_linear_problem
    with highest_precision():
        path = PopsRidgePath(THETA, prob, ds)
        for r in ("blr", 1e-2, 1e-6):
            ref = _dense_ridge_mean(prob, ds, r)
            assert np.allclose(np.asarray(path.c_star_at(r)), ref, rtol=1e-6, atol=1e-10), r


def test_ridge_path_reuses_one_factorisation(tiny_linear_problem):
    """A(ridge) from the single eigendecomposition equals the direct inverse."""
    prob, ds = tiny_linear_problem
    with highest_precision():
        path = PopsRidgePath(THETA, prob, ds)
        _, M, g2 = _reference_var(prob, ds, 1e-3)
        for r in (1e-1, 1e-5):
            direct = np.linalg.inv(M + path.ridge_abs(r) * np.diag(g2))
            assert np.allclose(np.asarray(path.A(r)), direct, rtol=1e-7, atol=1e-12)


def test_select_pops_ridge_picks_argmin_of_its_own_score(tiny_linear_problem):
    prob, ds = tiny_linear_problem
    grid = (1e-1, 1e-3, 1e-5)
    with highest_precision():
        ridge, scores = select_pops_ridge(THETA, prob, ds, ds, grid)
    for q in "EF":
        assert ridge[q] in grid
        assert ridge[q] == grid[int(np.argmin(scores[q]))]


# ---------------------------------------------------------------------------
# Streaming POPS: same predictive, O(L^2) memory -- the K x L corrections are
# never materialised.  The in-memory members() path is the oracle.
# ---------------------------------------------------------------------------
from ace_jax.fit.pops import pops_envelope, pops_posterior, pops_var


def _test_rows(prob, ds):
    return jnp.asarray(_dense_structural(prob, ds, np.zeros(prob.cfg.len_basis))[2])   # raw E rows


@pytest.mark.parametrize("lev", [0.0, 50.0])
def test_streamed_hypercube_matches_in_memory(tiny_linear_problem, lev):
    prob, ds = tiny_linear_problem
    with highest_precision():
        path = PopsRidgePath(THETA, prob, ds)
        PE = _test_rows(prob, ds)
        ref = pops_var(PE, pops_posterior(path.members(1e-3, lev), path.c_star_at(1e-3), form="hypercube"))
        got = pops_var(PE, path.posterior(1e-3, form="hypercube", leverage_pct=lev))
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=1e-14)


@pytest.mark.parametrize("lev", [0.0, 50.0])
def test_streamed_ensemble_matches_in_memory(tiny_linear_problem, lev):
    prob, ds = tiny_linear_problem
    with highest_precision():
        path = PopsRidgePath(THETA, prob, ds)
        PE = _test_rows(prob, ds)
        ref = pops_var(PE, pops_posterior(path.members(1e-3, lev), path.c_star_at(1e-3), form="ensemble"))
        got = pops_var(PE, path.posterior(1e-3, form="ensemble", leverage_pct=lev))
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=1e-14)


def test_streamed_envelope_matches_in_memory(tiny_linear_problem):
    prob, ds = tiny_linear_problem
    with highest_precision():
        path = PopsRidgePath(THETA, prob, ds)
        PE = _test_rows(prob, ds)
        lo_ref, hi_ref = pops_envelope(PE, path.members(1e-3))
        lo, hi = path.envelope(PE, 1e-3)
    assert np.allclose(np.asarray(lo), np.asarray(lo_ref), rtol=1e-8, atol=1e-12)
    assert np.allclose(np.asarray(hi), np.asarray(hi_ref), rtol=1e-8, atol=1e-12)


def test_predict_and_ridge_selection_never_materialise_members(tiny_linear_problem, monkeypatch):
    """The production paths must stream: calling the (K x L) members() oracle is a bug."""
    prob, ds = tiny_linear_problem
    def boom(*a, **k):
        raise AssertionError("members() materialises every correction -- must not be used here")
    monkeypatch.setattr(PopsRidgePath, "members", boom)
    with highest_precision():
        predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge=1e-3)
        predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge=1e-3, pops_form="ensemble")
        select_pops_ridge(THETA, prob, ds, ds, (1e-2, 1e-4))


def test_predict_reuses_a_prebuilt_path_and_its_posteriors(tiny_linear_problem, monkeypatch):
    """At production size (L ~ 2.8e4) a second factorisation does not fit next to the
    first, and every posterior is several streamed passes: predict_fixed must reuse a
    caller's path, and the path must not rebuild a posterior it already has."""
    prob, ds = tiny_linear_problem
    with highest_precision():
        ref = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge={"E": 1e-2, "F": 1e-4, "V": 1e-4})
        path = PopsRidgePath(THETA, prob, ds)
        calls = []                                            # one streamed moment pass per posterior built
        import ace_jax.fit.stats as S                         # DeviceRows calls it there
        real = S.pops_moment_sums
        monkeypatch.setattr(S, "pops_moment_sums", lambda *a, **k: calls.append(1) or real(*a, **k))
        monkeypatch.setattr(PopsRidgePath, "__init__",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("second factorisation")))
        for _ in range(2):                                    # e.g. the test and ood splits
            got = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge={"E": 1e-2, "F": 1e-4, "V": 1e-4},
                                pops_path=path)
    for f in ref._fields:
        assert np.allclose(np.asarray(getattr(got, f)), np.asarray(getattr(ref, f)), rtol=1e-10, atol=1e-14), f
    assert len(calls) == 2, calls                             # one per distinct ridge, across both calls


def test_ridge_selection_compiles_the_predictor_once(tiny_linear_problem, monkeypatch):
    """The batch predictor takes the posteriors as ARGUMENTS: jitting a closure over
    them re-traced (and re-compiled, with the L x L posteriors embedded as constants
    -- 6 GB each at the production Cantor basis) once per grid ridge."""
    import ace_jax.fit.predict as P
    prob, ds = tiny_linear_problem
    traces = []
    real = P._pops_paper_batch
    monkeypatch.setattr(P, "_pops_paper_batch", lambda *a: traces.append(1) or real(*a))
    with highest_precision():
        select_pops_ridge(THETA, prob, ds, ds, (1e-2, 1e-4, 1e-6))
    assert len(traces) == 1


# --- host-cached rows: one ACE pass, every POPS pass streams the cache -------------

def _host(prob, ds, block=7):
    from ace_jax.fit.stats import HostRows
    qs = PopsRidgePath(THETA, prob, ds).qs
    return HostRows(prob.model, prob.cfg, ds, qs, block=block)     # 7: many blocks + a padded tail


def test_host_rows_path_equals_device_path(tiny_linear_problem):
    prob, ds = tiny_linear_problem
    with highest_precision():
        dev = PopsRidgePath(THETA, prob, ds)
        host = PopsRidgePath(THETA, prob, ds, rows=_host(prob, ds))
        for p in (dev, host):
            p.use_mean("blr")
        # the Gram is summed row-wise vs per quantity: equal to the solve's conditioning
        assert np.allclose(host.c_star, dev.c_star, rtol=1e-7, atol=1e-12)
        # compared on what is predicted -- the variance / envelope at real rows
        from ace_jax.fit.pops import pops_var
        from ace_jax.fit.rows import linear_rows
        lin = linear_rows(prob.model, prob.cfg, jax.tree.map(lambda a: a[0], ds))[0]
        for phi in (lin.E, lin.F.reshape(-1, lin.F.shape[-1])):
            for form in ("hypercube", "ensemble"):
                a = pops_var(phi, dev.posterior(1e-4, form=form))
                b = pops_var(phi, host.posterior(1e-4, form=form))
                assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-14), form
            for x, y in zip(dev.envelope(phi, 1e-4), host.envelope(phi, 1e-4)):
                assert np.allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=1e-12)


def test_host_rows_ridge_selection_equals_device(tiny_linear_problem):
    prob, ds = tiny_linear_problem
    grid = (1e-2, 1e-4, 1e-6)
    with highest_precision():
        r_dev, s_dev = select_pops_ridge(THETA, prob, ds, ds, grid)
        r_host, s_host = select_pops_ridge(THETA, prob, ds, ds, grid, rows="host")
    assert r_dev == r_host
    for q in "EFV":
        assert np.allclose(s_dev[q], s_host[q], rtol=1e-6)


def test_host_rows_hold_only_live_rows(tiny_linear_problem):
    """Padded rows (w = 0) are dropped: the cache is the live rows plus one block's tail."""
    prob, ds = tiny_linear_problem
    h = _host(prob, ds)
    live = int(sum(np.sum(np.asarray(w) > 0) * k for w, k in
                   ((ds.w_E, 1), (ds.w_F, 3), (ds.w_V, 6))))
    assert h.n_rows == live and h.blocks[0].shape[0] == 7
    assert sum(b.shape[0] for b in h.blocks) - h.n_rows < 7


def test_blr_mean_is_accurate_not_the_eigh_formula():
    """c*('blr') must be the BLR posterior mean to solver accuracy.  Reading it off
    the eigendecomposition of the Gram (kappa^2, with round-off negatives clipped)
    was 1e-3 off a QR reference on the Si fixture at its MAP hyperparameters, and
    4e-4 eV/A in forces at the production Cantor basis; the Cholesky BLR posterior
    agrees with QR to 1e-7 there."""
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.solve import solve_qr
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                    virial_key="dft_virial", ntrain=16, ntest=6, batch=4, r0=2.35, arm="linear",
                    opt="lbfgs", rungs=("map",), map_steps=5, e0="prefit").validate()   # POPS: E0 fixed
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
        ref = np.asarray(solve_qr(b.prob, d.ds_train, theta))
        for rows in ("device", "host"):
            c = np.asarray(PopsRidgePath(theta, b.prob, d.ds_train, rows=rows).c_star_at("blr"))
            assert np.linalg.norm(c - ref) <= 1e-5 * np.linalg.norm(ref), (rows, np.linalg.norm(c - ref) / np.linalg.norm(ref))
