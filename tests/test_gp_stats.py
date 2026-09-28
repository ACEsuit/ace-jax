"""Streamed statistics equal the dense single-pass statistics, and so does
their gradient in theta (the two-pass VJP through scan + checkpoint)."""
import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import highest_precision, load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, from_array, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.rows import batch_rows
from ace_jax.fit.stats import Stats, batch_stats, sufficient_statistics

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")

THETA = Hypers(log_ell=np.log(0.8), log_A=np.log(0.3), log_alpha=np.log(1.2), log_r0=np.log(2.35),
               log_eps=np.log(0.4), log_rho=np.log(4.0), log_sigma_c=0.0,
               log_sigma_E=0.0, log_sigma_F=0.0, log_sigma_V=0.0)


@pytest.fixture(scope="module")
def setup():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:6]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=2)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=2)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 5, descriptor_scale(X, ds.node_mask))
    spec = KernelSpec(kind="cosine", bump=True, D=cfg.D)
    return model, ds, cfg, ind, spec


def _dense(theta, model, ds, cfg, ind, spec):
    """Concatenate every batch's rows and form the statistics in one go."""
    PE, yE, PF, yF, PV, yV = [], [], [], [], [], []
    for b in range(ds.n_batches):
        batch = jax.tree.map(lambda a: a[b], ds)
        r = batch_rows(theta, spec, model, ind, cfg, batch)
        PE.append(r.E * batch.w_E[:, None]); yE.append(batch.y_E * batch.w_E)
        wF = jnp.repeat(batch.w_F, 3)
        PF.append(r.F.reshape(-1, r.F.shape[-1]) * wF[:, None]); yF.append(batch.y_F.reshape(-1) * wF)
        wV = jnp.repeat(batch.w_V, 6)
        PV.append(r.V.reshape(-1, r.V.shape[-1]) * wV[:, None]); yV.append(batch.y_V.reshape(-1) * wV)
    cat = jnp.concatenate
    PE, yE, PF, yF, PV, yV = map(cat, (PE, yE, PF, yF, PV, yV))
    return PE.T @ PE, PF.T @ PF, PV.T @ PV, PE.T @ yE, PF.T @ yF, PV.T @ yV


def test_streamed_equals_dense(setup):
    model, ds, cfg, ind, spec = setup
    with highest_precision():
        st = jax.jit(lambda th: sufficient_statistics(th, spec, model, ind, cfg, ds))(THETA)
        GE, GF, GV, bE, bF, bV = _dense(THETA, model, ds, cfg, ind, spec)
    for a, b in ((st.G_E, GE), (st.G_F, GF), (st.G_V, GV), (st.b_E, bE), (st.b_F, bF), (st.b_V, bV)):
        assert float(jnp.abs(a - b).max()) < 1e-9 * max(1.0, float(jnp.abs(b).max()))
    assert int(st.n_E) == 6
    assert int(st.n_V) == 6 * int(jnp.sum(ds.w_V > 0))      # config 0 (isolated atom) has no virial
    assert int(st.n_F) == 3 * int(ds.n_atoms.sum())


def test_streamed_gradient_equals_dense_gradient(setup):
    model, ds, cfg, ind, spec = setup
    probe = jax.random.normal(jax.random.PRNGKey(0), (cfg.len_basis + ind.XM.shape[0],))

    def f_stream(a):
        st = sufficient_statistics(from_array(a), spec, model, ind, cfg, ds)
        return probe @ st.G_F @ probe + probe @ st.b_E

    def f_dense(a):
        GE, GF, GV, bE, bF, bV = _dense(from_array(a), model, ds, cfg, ind, spec)
        return probe @ GF @ probe + probe @ bE

    with highest_precision():
        a0 = to_array(THETA)
        g1, g2 = jax.grad(f_stream)(a0), jax.grad(f_dense)(a0)
        # kernel hypers only; sigma_* do not enter the statistics
        assert float(jnp.abs(g1[6:]).max()) == 0.0
        assert float(jnp.abs(g1 - g2).max()) < 1e-8 * max(1.0, float(jnp.abs(g2).max()))
        h = 1e-5
        e = jnp.zeros(10).at[0].set(h)
        fd = (f_stream(a0 + e) - f_stream(a0 - e)) / (2 * h)
        assert abs(float(g1[0]) - float(fd)) < 1e-5 * max(1.0, abs(float(fd)))


def test_theta_split_equals_full(setup):
    """assemble(linear, residual(theta)) == sufficient_statistics(theta), value
    and theta-gradient, to f64 roundoff."""
    from ace_jax.fit.stats import (assemble_statistics, linear_statistics,
                             residual_statistics)
    model, ds, cfg, ind, spec = setup
    with highest_precision():
        lin = jax.jit(lambda: linear_statistics(model, cfg, ds))()
        st_full = jax.jit(lambda th: sufficient_statistics(th, spec, model, ind, cfg, ds))(THETA)
        st_split = jax.jit(lambda th: assemble_statistics(
            lin, residual_statistics(th, spec, model, ind, cfg, ds)))(THETA)
    for f in st_full._fields:
        a, b = getattr(st_full, f), getattr(st_split, f)
        assert float(jnp.abs(a - b).max()) < 1e-8 * max(1.0, float(jnp.abs(a).max())), f

    probe = jax.random.normal(jax.random.PRNGKey(1), (cfg.len_basis + ind.XM.shape[0],))
    def obj(fn):
        def f(a):
            st = fn(from_array(a))
            return probe @ st.G_F @ probe + probe @ st.b_E
        return f
    full = obj(lambda th: sufficient_statistics(th, spec, model, ind, cfg, ds))
    split = obj(lambda th: assemble_statistics(lin, residual_statistics(th, spec, model, ind, cfg, ds)))
    with highest_precision():
        g_full = jax.grad(full)(to_array(THETA))
        g_split = jax.grad(split)(to_array(THETA))
    assert float(jnp.abs(g_full - g_split).max()) < 1e-7 * max(1.0, float(jnp.abs(g_full).max()))


def test_streamed_pops_equals_monolithic(tiny_linear_problem):
    """Task 9: streamed pops_statistics over a 2-batch dataset gives the SAME
    deltas as Task-8's pops_corrections on the whole whitened design assembled
    in one go (streaming == monolithic), for leverage_pct 0 and a top-fraction."""
    from jax.scipy.linalg import cho_solve

    from ace_jax.fit.objective import posterior
    from ace_jax.fit.pops import pops_corrections, whiten
    from ace_jax.fit.rows import linear_rows
    from ace_jax.fit.stats import pops_statistics

    prob, ds = tiny_linear_problem
    assert ds.n_batches == 2                        # 6 configs / 3 per batch
    theta = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
                   log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
                   log_sigma_F=np.log(0.01), log_sigma_V=np.log(0.01))
    with highest_precision():
        st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
        c_star, cholA = posterior(theta, st, prob)          # A^{-1} b, chol(A)
        Ldim = c_star.shape[0]
        Sigma0 = cho_solve((cholA, True), jnp.eye(Ldim))    # A^{-1}
        sigma = {q: float(jnp.exp(getattr(theta, f"log_sigma_{q}"))) for q in "EFV"}

        # Monolithic reference: whiten every batch's linear rows (in the E,F,V
        # per-batch order the stream visits), concatenate, one pops_corrections.
        def whitened(batch):
            r, _, _ = linear_rows(prob.model, prob.cfg, batch)
            Lc = r.E.shape[-1]
            phis, resids = [], []
            for phi, y, w, sq in (
                    (r.E, batch.y_E, batch.w_E, sigma["E"]),
                    (r.F.reshape(-1, Lc), batch.y_F.reshape(-1), jnp.repeat(batch.w_F, 3), sigma["F"]),
                    (r.V.reshape(-1, Lc), batch.y_V.reshape(-1), jnp.repeat(batch.w_V, 6), sigma["V"])):
                pt, rt = whiten(phi, y - phi @ c_star, w, sq)
                phis.append(pt); resids.append(rt)
            return jnp.concatenate(phis, 0), jnp.concatenate(resids, 0)

        blocks = [whitened(jax.tree.map(lambda a: a[b], ds)) for b in range(ds.n_batches)]
        phi_all = jnp.concatenate([p for p, _ in blocks], 0)
        r_all = jnp.concatenate([rr for _, rr in blocks], 0)

        for lev in (0.0, 50.0):
            mono = pops_corrections(Sigma0, phi_all, r_all, lev)
            stream = pops_statistics(c_star, Sigma0, prob, ds, sigma, leverage_pct=lev)
            assert stream.shape == mono.shape, lev
            tol = 1e-8 * max(1.0, float(jnp.abs(mono).max()))
            assert float(jnp.abs(stream - mono).max()) < tol, lev
