"""Host-cached joint LML: per-evaluation cost without the ACE many-body basis.

With --density pair the residual GP only sees the pair channels, so its inputs
come from the pair radials alone; the theta-independent linear rows are cached in
host RAM and streamed back for the cross block.  Everything must agree with the
existing device path (make_lml) to f64 roundoff."""
import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, default_prior, to_array
from ace_jax.fit.inducing import (GPConfig, build_pmap, descriptor_scale, select_inducing,
                                   site_features)
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.objective import Problem, make_lml

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")

THETA = Hypers(log_ell=np.log(0.8), log_A=np.log(0.05), log_alpha=np.log(1.0), log_r0=np.log(2.35),
               log_eps=np.log(0.3), log_rho=np.log(4.0), log_sigma_c=np.log(0.3),
               log_sigma_E=np.log(1e-3), log_sigma_F=np.log(0.02), log_sigma_V=np.log(0.02))


def _problem(density):
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:9]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), 3)          # 3 batches
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    scale = descriptor_scale(X, ds.node_mask)
    Pmap = build_pmap(cfg, scale, density=density)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 4, scale, Pmap=Pmap)
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    return prob, ds


@pytest.fixture(scope="module")
def pair_problem():
    return _problem("pair")


def test_pair_features_match_full_jacobian(pair_problem):
    """model.pair_features_dense == the pair slice of edge_jacobian_dense."""
    prob, ds = pair_problem
    b = jax.tree.map(lambda a: a[0], ds)
    Ncap, K = b.nbr.shape
    zi = jnp.broadcast_to(b.node_z[:, None], (Ncap, K)); zj = b.node_z[b.nbr]
    nB = prob.cfg.n_B
    with highest_precision():
        X, J = prob.model.edge_jacobian_dense(b.rij, zi, zj, b.nbr_mask)
        Xp, dRp = prob.model.pair_features_dense(b.rij, zi, zj, b.nbr_mask)
    assert np.allclose(np.asarray(Xp), np.asarray(X[:, nB:]), rtol=1e-12, atol=1e-14)
    assert np.allclose(np.asarray(dRp), np.asarray(J[:, nB:, :]), rtol=1e-12, atol=1e-14)


def test_pair_residual_inputs_match_projection(pair_problem):
    """(U0, JU0) from the pair radials equal X @ Pmap and Pmap^T J."""
    from ace_jax.fit.rows import linear_rows, pair_feature_inputs
    prob, ds = pair_problem
    b = jax.tree.map(lambda a: a[0], ds)
    Ncap, K = b.nbr.shape
    with highest_precision():
        _, X, J = linear_rows(prob.model, prob.cfg, b)
        U0, JU0 = pair_feature_inputs(prob.model, prob.ind, prob.cfg, b)
    P = np.asarray(prob.ind.Pmap)
    assert np.allclose(np.asarray(U0), np.asarray(X) @ P, rtol=1e-12, atol=1e-14)
    JUref = np.einsum("nkDa,Dq->nkqa", np.asarray(J).reshape(Ncap, K, -1, 3), P)
    assert np.allclose(np.asarray(JU0), JUref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("chunk", [1, 2])
def test_hostcache_statistics_match_device(pair_problem, chunk):
    """The correctness check proper: cached linear statistics and streamed residual
    statistics equal the device path's to f64 roundoff, for any chunking."""
    from ace_jax.fit.hostcache import HostCachedLML
    from ace_jax.fit.hypers import from_array
    from ace_jax.fit.stats import linear_statistics, residual_statistics
    prob, ds = pair_problem
    a = jnp.asarray(to_array(THETA))
    with highest_precision():
        lik = HostCachedLML(prob, ds, chunk=chunk)
        lin = linear_statistics(prob.model, prob.cfg, ds)
        res = residual_statistics(from_array(a), prob.spec, prob.model, prob.ind, prob.cfg, ds)
        got = lik._residual_stats(a)
    rel = lambda x, y: float(jnp.max(jnp.abs(x - y)) / (jnp.max(jnp.abs(y)) + 1e-300))
    for f in lin._fields:
        assert rel(getattr(lik.lin, f), getattr(lin, f)) < 1e-12, f
    for f in res._fields:
        assert rel(getattr(got, f), getattr(res, f)) < 1e-12, f


@pytest.mark.parametrize("chunk", [1, 2])
def test_hostcached_lml_matches_device_lml(pair_problem, chunk):
    """Value and gradient of the host-cached LML equal make_lml's.  The statistics
    agree to ~1e-15 (above), but the posterior precision here has cond ~1e19, so
    summation-order roundoff reaches the LML at ~1e-9 relative: tolerances say so."""
    from ace_jax.fit.hostcache import HostCachedLML
    prob, ds = pair_problem
    a = jnp.asarray(to_array(THETA))
    with highest_precision():
        ref_v, ref_g = jax.value_and_grad(make_lml(prob, ds, cache_linear=True))(a)
        lik = HostCachedLML(prob, ds, chunk=chunk)
        v, g = lik.value_and_grad(a)
        v_only = lik(a)
    assert np.isclose(float(v), float(ref_v), rtol=1e-7)
    assert np.isclose(float(v_only), float(v), rtol=1e-12)
    assert float(jnp.max(jnp.abs(g - ref_g))) < 1e-6 * float(jnp.max(jnp.abs(ref_g)))


def test_hostcache_requires_pair_features():
    """With the isotropic (full-descriptor) feature map the residual needs the ACE
    Jacobian every evaluation, so the host cache would save nothing: refuse."""
    from ace_jax.fit.hostcache import HostCachedLML
    prob, ds = _problem(None)
    with pytest.raises(ValueError, match="density"):
        HostCachedLML(prob, ds)
