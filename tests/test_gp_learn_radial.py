"""fit.radial_learn: VarPro objective, L-BFGS loop, held-out gate."""
import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
MODEL = FIXTURE_DIR / "si_ace_model.npz"
pytestmark = pytest.mark.skipif(not (XYZ.exists() and MODEL.exists()), reason="missing fixtures")

THETA = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
               log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
               log_sigma_F=np.log(0.01), log_sigma_V=np.log(0.01))


def make_problem(ncfg=6, per_batch=3, start=0, force_key="dft_force"):
    """Linear (M = 0) problem on si_tiny with the analytic Si model; gamma = 1."""
    model, meta, z = load(MODEL)
    configs = load_configs(XYZ, "dft_energy", force_key, "dft_virial")[start:start + ncfg]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=per_batch)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=per_batch)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   jnp.ones(cfg.len_basis), default_prior(2.35))
    return prob, ds, meta


def relabel(prob, ds, W, c):
    """ds with targets replaced by the noiseless linear ACE with radials W and readout c."""
    from ace_jax.fit.radial_model import with_radial
    from ace_jax.fit.rows import linear_rows
    m = with_radial(prob.model, W)
    yE, yF, yV = [], [], []
    for i in range(ds.n_batches):
        r, _, _ = linear_rows(m, prob.cfg, jax.tree.map(lambda a: a[i], ds))
        yE.append(r.E @ c); yF.append(r.F @ c); yV.append(r.V @ c)
    return ds._replace(y_E=jnp.stack(yE), y_F=jnp.stack(yF), y_V=jnp.stack(yV))


@pytest.fixture(scope="module")
def small():
    return make_problem()


def _in_memory_residual(W, prob, ds, theta):
    """min_c ||Phi~ c - y~||^2 on the materialised prior-augmented design."""
    from ace_jax.fit.radial_model import with_radial
    from ace_jax.fit.solve import stacked_design
    Phi, y = stacked_design(prob._replace(model=with_radial(prob.model, W)), ds, theta)
    # jnp.linalg.lstsq has no custom VJP: its gradient is plain autodiff through
    # svd, whose singular-VECTOR gradient is NaN on this degenerate ACE design
    # (repeated singular values). Reduced QR is differentiable for a
    # full-column-rank design, and the prior rows make Phi~ full rank.
    Qm, _ = jnp.linalg.qr(Phi)          # reduced (mode="reduced" is the default)
    v = Qm.T @ y
    return y @ y - v @ v


def test_projected_residual_matches_in_memory_lstsq(small):
    from ace_jax.fit.radial_learn import projected_residual
    prob, ds, _ = small
    W = prob.model.rnl_Wnlq
    got = float(projected_residual(W, THETA, prob, ds))
    ref = float(_in_memory_residual(W, prob, ds, THETA))
    assert abs(got - ref) < 1e-7 * abs(ref)


def test_projected_residual_gradient(small):
    from ace_jax.fit.radial_learn import projected_residual
    prob, ds, _ = small
    W = prob.model.rnl_Wnlq
    f = lambda X: projected_residual(X, THETA, prob, ds)
    g = jax.grad(f)(W)
    D = jnp.asarray(np.random.default_rng(0).standard_normal(W.shape))
    h = 1e-5
    fd = (float(f(W + h * D)) - float(f(W - h * D))) / (2 * h)
    assert abs(float(jnp.vdot(g, D)) - fd) < 1e-5 * abs(fd)
    g_mem = jax.grad(lambda X: _in_memory_residual(X, prob, ds, THETA))(W)
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_mem), rtol=1e-6,
                               atol=1e-8 * float(jnp.abs(g_mem).max()))
