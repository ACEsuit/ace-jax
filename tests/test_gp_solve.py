"""Stable direct solves for the linear (M = 0) fit (acegp.solve): QR and
matrix-free LSQR against the normal-equations Cholesky (objective.posterior),
and against ACEfit's own exported design matrix."""
import numpy as np
import pytest

from conftest import FIXTURE_DIR

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem, posterior, combine, prior_precision
from ace_jax.fit.stats import sufficient_statistics
from ace_jax.fit.solve import solve_qr, solve_qr_streaming, solve_lsqr, lsqr, stacked_design, streamed_operators

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
DESIGN = FIXTURE_DIR / "si_tiny_design.npz"
ACEFIT_QR = FIXTURE_DIR / "si_tiny_acefit_qr.npz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing GP fixtures")


def _theta(**kw):
    d = dict(log_ell=np.log(0.8), log_A=np.log(0.05), log_alpha=0.0, log_r0=np.log(2.35),
             log_eps=np.log(0.3), log_rho=np.log(4.0), log_sigma_c=np.log(0.3),
             log_sigma_E=np.log(1e-3), log_sigma_F=np.log(2e-2), log_sigma_V=np.log(2e-2))
    d.update(kw)
    return Hypers(**d)


@pytest.fixture(scope="module")
def m0():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), 4)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=4)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))  # M = 0
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    return prob, ds


@pytest.fixture(scope="module")
def gp():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), 4)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=4)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 6, descriptor_scale(X, ds.node_mask))  # M = 6
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    return prob, ds


def test_qr_matches_cholesky_gp(gp):
    """QR on the joint [linear | residual] design, with the block prior square root
    blockdiag(diag(gamma/sigma_c), chol(K_MM)^T), reproduces the GP posterior mean
    [c ; w] -- stably.  The joint Gram is far stiffer than the linear one, so this
    is where kappa vs kappa^2 matters most."""
    prob, ds = gp
    theta = _theta()
    with highest_precision():
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        c_chol, _ = posterior(theta, st, prob)
        c_qr = solve_qr(prob, ds, theta)
        c_str = solve_qr_streaming(prob, ds, theta)
    c_chol = np.asarray(c_chol)
    assert c_chol.shape[0] == prob.cfg.len_basis + prob.ind.XM.shape[0]     # [c ; w]
    assert np.abs(np.asarray(c_qr) - c_chol).max() < 1e-6 * max(1.0, np.abs(c_chol).max())
    assert np.abs(np.asarray(c_str) - c_chol).max() < 1e-6 * max(1.0, np.abs(c_chol).max())


def test_streamed_operators_equal_joint_design_gp(gp):
    """The matrix-free operators equal the materialised joint design's action at
    M > 0 too (the residual block and the chol(K_MM)^T prior included)."""
    prob, ds = gp
    theta = _theta()
    rng = np.random.default_rng(0)
    with highest_precision():
        Phi, _ = stacked_design(prob, ds, theta)
        matvec, rmatvec, _ = streamed_operators(prob, ds, theta)
        Dt = prob.cfg.len_basis + prob.ind.XM.shape[0]
        x = jnp.asarray(rng.standard_normal(Dt))
        u = jnp.asarray(rng.standard_normal(Phi.shape[0]))
        Av, Atu = matvec(x), rmatvec(u)
    Phi = np.asarray(Phi)
    assert np.abs(np.asarray(Av) - Phi @ np.asarray(x)).max() < 1e-9 * max(1.0, np.abs(Phi @ np.asarray(x)).max())
    assert np.abs(np.asarray(Atu) - Phi.T @ np.asarray(u)).max() < 1e-9 * max(1.0, np.abs(Phi.T @ np.asarray(u)).max())


def test_qr_matches_cholesky(m0):
    """lineax QR least-squares on the weighted, prior-augmented design equals the
    normal-equations posterior mean -- the ACEfit-style stable route to the same fit."""
    prob, ds = m0
    theta = _theta()
    with highest_precision():
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        c_chol, _ = posterior(theta, st, prob)
        c_qr = solve_qr(prob, ds, theta)
    c_chol, c_qr = np.asarray(c_chol), np.asarray(c_qr)
    assert np.abs(c_qr - c_chol).max() < 1e-6 * max(1.0, np.abs(c_chol).max())


def test_qr_streaming_matches_cholesky(m0):
    """Updating (tall-skinny) QR streams the rows, keeps only the L x L factor,
    and reaches the same fit as the Cholesky posterior -- stable AND O(L^2)
    memory in one pass (basis-limited, not observation-limited)."""
    prob, ds = m0
    theta = _theta()
    with highest_precision():
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        c_chol, _ = posterior(theta, st, prob)
        c_str = solve_qr_streaming(prob, ds, theta)
    c_chol, c_str = np.asarray(c_chol), np.asarray(c_str)
    assert np.abs(c_str - c_chol).max() < 1e-6 * max(1.0, np.abs(c_chol).max())


def test_qr_conditioning_is_sqrt_of_gram(m0):
    """The design QR sees kappa(Phi); the Gram Cholesky sees kappa(Phi)^2.  This is
    the reason to prefer QR for an ill-conditioned basis -- quantified here."""
    prob, ds = m0
    theta = _theta()
    with highest_precision():
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        Phi, _ = stacked_design(prob, ds, theta)
        G, _, *_ = combine(theta, st)
        Lam, _ = prior_precision(theta, prob)
    kA = np.linalg.cond(np.asarray(Phi))
    kG = np.linalg.cond(np.asarray(G + Lam))
    assert kG > 1e3 * kA                              # Gram is dramatically worse conditioned
    assert 0.3 < (kA ** 2) / kG < 3.0                 # and it is exactly the square


def test_lsqr_algorithm_matches_scipy():
    """The JAX LSQR recurrence solves least-squares correctly (vs scipy) on a
    well-conditioned operator -- the algorithm is right; only its iteration count
    scales with conditioning."""
    from scipy.sparse.linalg import lsqr as sp_lsqr
    rng = np.random.default_rng(0)
    m, n = 300, 60
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    Vt, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (U * np.logspace(0, -2, n)) @ Vt.T            # cond = 1e2
    b = rng.standard_normal(m)
    x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
    x_sp = sp_lsqr(A, b, atol=0, btol=0, iter_lim=1000)[0]
    x_jax = np.asarray(lsqr(lambda v: jnp.asarray(A) @ v, lambda u: jnp.asarray(A).T @ u,
                            jnp.asarray(b), n, maxiter=4 * n))
    assert np.abs(x_sp - x_ref).max() < 1e-8          # scipy sanity
    assert np.abs(x_jax - x_ref).max() < 1e-6 * max(1.0, np.abs(x_ref).max())


def test_streamed_operators_equal_materialised_design(m0):
    """The matrix-free matvec / rmatvec are exactly the action of the (weighted,
    prior-augmented) design matrix stacked_design materialises -- so LSQR on them
    solves the same least-squares problem, without ever forming n_obs x L.  This
    is the correctness of the streaming assembly (LSQR's own convergence rate is
    conditioning-bound and tested separately)."""
    prob, ds = m0
    theta = _theta()
    rng = np.random.default_rng(0)
    with highest_precision():
        Phi, y = stacked_design(prob, ds, theta)
        matvec, rmatvec, y_op = streamed_operators(prob, ds, theta)
        x = jnp.asarray(rng.standard_normal(prob.cfg.len_basis))
        u = jnp.asarray(rng.standard_normal(Phi.shape[0]))
        Av, Atu = matvec(x), rmatvec(u)
    Phi = np.asarray(Phi)
    assert np.abs(np.asarray(Av) - Phi @ np.asarray(x)).max() < 1e-9 * max(1.0, np.abs(Phi @ np.asarray(x)).max())
    assert np.abs(np.asarray(Atu) - Phi.T @ np.asarray(u)).max() < 1e-9 * max(1.0, np.abs(Phi.T @ np.asarray(u)).max())
    assert np.abs(np.asarray(y_op) - np.asarray(y)).max() < 1e-12


@pytest.mark.skipif(not DESIGN.exists(), reason="missing si_tiny_design.npz")
def test_qr_equivalent_on_acefit_exported_design():
    """On ACEfit's own exported design matrix A (Y targets, W row weights, gamma
    smoothness prior), the QR least-squares of the weighted, prior-augmented
    stack equals the regularised normal-equations solution -- i.e. acegp's linear
    estimator is ACEfit's, solved stably.  (Design rows already match ACEfit to
    1e-8 in test_gp_rows.)"""
    d = np.load(DESIGN)
    A, Y, W, gamma = d["A"], d["Y"], d["W"], d["gamma"]
    sc = 0.5
    sw = np.sqrt(W)
    Aw, yw = sw[:, None] * A, sw * Y
    Astk = np.concatenate([Aw, np.diag(gamma / sc)], 0)
    ystk = np.concatenate([yw, np.zeros(A.shape[1])])
    c_qr = np.linalg.lstsq(Astk, ystk, rcond=None)[0]
    c_ne = np.linalg.solve(Aw.T @ Aw + np.diag((gamma / sc) ** 2), Aw.T @ yw)
    assert np.abs(c_qr - c_ne).max() < 1e-6 * np.abs(c_ne).max()
    assert np.linalg.cond(Astk) ** 2 == pytest.approx(np.linalg.cond(Aw.T @ Aw + np.diag((gamma/sc)**2)), rel=0.3)


@pytest.mark.skipif(not (DESIGN.exists() and ACEFIT_QR.exists()), reason="missing design / ACEfit reference")
def test_matches_acefit_qr_reference():
    """The same weighted, prior-augmented least squares, solved by ACEfit's OWN
    solve(QR) in Julia (fixtures/si_tiny_acefit_qr.npz, regenerated by
    julia/acefit_qr_reference.jl), equals the QR/normal-equations solution here to
    machine precision -- so acegp's linear estimator reproduces ACEfit's linear
    ACE fit through ACEfit's actual solver code (design rows already match to 1e-8)."""
    d = np.load(DESIGN)
    A, Y, W, gamma = d["A"], d["Y"], d["W"], d["gamma"]
    ref = np.load(ACEFIT_QR)
    C_acefit, sc = ref["C"], float(ref["sigma_c"][0])
    sw = np.sqrt(W)
    Aw, yw = sw[:, None] * A, sw * Y
    Astk = np.concatenate([Aw, np.diag(gamma / sc)], 0)
    ystk = np.concatenate([yw, np.zeros(A.shape[1])])
    c_py = np.linalg.lstsq(Astk, ystk, rcond=None)[0]
    assert np.abs(c_py - C_acefit).max() < 1e-9 * max(1.0, np.abs(C_acefit).max())
