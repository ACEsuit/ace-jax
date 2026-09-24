"""Paper-faithful POPS (Swinburne & Perez, arXiv:2402.01810): the intrinsic noise
Sigma_Y is FIXED (structural weights only, never the evidence-fit sigma_q), the
regulariser is the paper's Sigma_Y Sigma_0^-1 term = ridge * Gamma^2 (smoothness
prior shape), and there is no separate aleatoric/epistemic term."""
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

# sigma_q deliberately far from 1: a sigma-whitened implementation would differ
THETA = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
               log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
               log_sigma_F=np.log(0.05), log_sigma_V=np.log(0.3))


def _dense_structural(prob, ds, c):
    """Structurally-weighted member rows (w*phi) and residuals (w*r) of every
    non-padded observation, plus the raw energy rows -- a dense reference."""
    Pw, rw, PE = [], [], []
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        r, _, _ = linear_rows(prob.model, prob.cfg, b)
        L = r.E.shape[-1]
        for phi, y, w in ((r.E, b.y_E, b.w_E),
                          (r.F.reshape(-1, L), b.y_F.reshape(-1), jnp.repeat(b.w_F, 3)),
                          (r.V.reshape(-1, L), b.y_V.reshape(-1), jnp.repeat(b.w_V, 6))):
            phi, y, w = map(np.asarray, (phi, y, w))
            k = w > 0
            Pw.append(w[k, None] * phi[k]); rw.append(w[k] * (y[k] - phi[k] @ c))
        PE.append(np.asarray(r.E)[np.asarray(b.w_E) > 0])
    return np.concatenate(Pw), np.concatenate(rw), np.concatenate(PE)


def _reference_var(prob, ds, ridge):
    st = sufficient_statistics(THETA, prob.spec, prob.model, prob.ind, prob.cfg, ds)
    c = np.asarray(posterior(THETA, st, prob)[0])
    Pw, rw, PE = _dense_structural(prob, ds, c)
    g2 = np.asarray(prob.gamma) ** 2; D = np.sqrt(g2)
    M = Pw.T @ Pw
    lam = ridge * np.linalg.eigvalsh(M / D[:, None] / D[None, :]).max()
    A = np.linalg.inv(M + lam * np.diag(g2))
    h = np.einsum("ij,jk,ik->i", Pw, A, Pw)
    k = h > 0                                       # zero-leverage rows (phi = 0) are not members
    deltas = (Pw[k] @ A) * (rw[k] / h[k])[:, None]
    cov = np.asarray(_fit_hypercube_cov(jnp.asarray(deltas)))
    return np.sum((PE @ cov) * PE, axis=1), M, g2


def test_pops_paper_matches_dense_structural_reference(tiny_linear_problem):
    """Paper mode = structural weights (sigma_q never enters) + ridge*Gamma^2."""
    prob, ds = tiny_linear_problem
    with highest_precision():
        p = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge=1e-3)
        ref, _, _ = _reference_var(prob, ds, 1e-3)
    assert np.allclose(np.asarray(p.E_var), ref, rtol=1e-6, atol=1e-14)


def test_pops_paper_per_quantity_ridge(tiny_linear_problem):
    """A per-quantity ridge dict uses each quantity's own ridge."""
    prob, ds = tiny_linear_problem
    with highest_precision():
        pd = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge={"E": 1e-2, "F": 1e-4, "V": 1e-4})
        pE = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge=1e-2)
        pF = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge=1e-4)
    assert np.allclose(np.asarray(pd.E_var), np.asarray(pE.E_var))
    assert np.allclose(np.asarray(pd.F_var), np.asarray(pF.F_var))
    assert not np.allclose(np.asarray(pE.E_var), np.asarray(pF.E_var))


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
    st = sufficient_statistics(THETA, prob.spec, prob.model, prob.ind, prob.cfg, ds)
    c = np.asarray(posterior(THETA, st, prob)[0])
    return jnp.asarray(_dense_structural(prob, ds, c)[2])


@pytest.mark.parametrize("lev", [0.0, 50.0])
def test_streamed_hypercube_matches_in_memory(tiny_linear_problem, lev):
    prob, ds = tiny_linear_problem
    with highest_precision():
        path = PopsRidgePath(THETA, prob, ds)
        PE = _test_rows(prob, ds)
        ref = pops_var(PE, pops_posterior(path.members(1e-3, lev), path.c_star, form="hypercube"))
        got = pops_var(PE, path.posterior(1e-3, form="hypercube", leverage_pct=lev))
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=1e-14)


@pytest.mark.parametrize("lev", [0.0, 50.0])
def test_streamed_ensemble_matches_in_memory(tiny_linear_problem, lev):
    prob, ds = tiny_linear_problem
    with highest_precision():
        path = PopsRidgePath(THETA, prob, ds)
        PE = _test_rows(prob, ds)
        ref = pops_var(PE, pops_posterior(path.members(1e-3, lev), path.c_star, form="ensemble"))
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
        import ace_jax.fit.predict as P
        calls = []                                            # one streamed moment pass per posterior built
        real = P.pops_moment_sums
        monkeypatch.setattr(P, "pops_moment_sums", lambda *a, **k: calls.append(1) or real(*a, **k))
        monkeypatch.setattr(PopsRidgePath, "__init__",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("second factorisation")))
        for _ in range(2):                                    # e.g. the test and ood splits
            got = predict_fixed(THETA, prob, ds, ds, uq="pops", pops_ridge={"E": 1e-2, "F": 1e-4, "V": 1e-4},
                                pops_path=path)
    for f in ref._fields:
        assert np.allclose(np.asarray(getattr(got, f)), np.asarray(getattr(ref, f)), rtol=1e-10, atol=1e-14), f
    assert len(calls) == 2, calls                             # one per distinct ridge, across both calls
