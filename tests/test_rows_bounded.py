"""Node-chunked training-side rows (rows.ROWS_EDGE_BUDGET): a batch whose whole-batch edge
Jacobian J (Ncap*K, D, 3) exceeds the budget is evaluated a node chunk at a time, everywhere the
fit builds rows -- linear statistics, the hybrid rows + residual block, predictive rows, site
features.  The chunked results equal the unchunked ones up to summation order, and the traced
program holds no intermediate of size ~Ncap*K*D."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")



def _close(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(1.0, float(np.abs(b).max())) if b.size else 1.0
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * scale, err_msg=what)


@pytest.fixture(scope="module")
def packed():
    """si_tiny configs plus one rattled 800-atom diamond cell, size-aware packed (n_cap = 800):
    a batch holding the big cell next to batches of small cells."""
    from ase.build import bulk
    from ace_jax.eval import load
    from ace_jax.fit.data import Config, build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((5, 5, 4))
    at.rattle(0.05, seed=1)
    rng = np.random.default_rng(0)
    big = Config(at.positions, at.numbers, at.cell.array, at.pbc, -4.0 * len(at),
                 rng.normal(size=(len(at), 3)), rng.normal(size=(3, 3)), 1.0, 1.0, 1.0)
    cfgs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    cfgs = cfgs[:5] + [big] + cfgs[5:9]
    ds = build_dataset(cfgs, meta, np.asarray(z["E0"]), 4, pack=True, log=lambda *a: None)
    g = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                 NZ=len(meta["elements"]), C=ds.y_E.shape[1])
    return model, g, ds


def _force_chunk(monkeypatch, model, g, ds, nodes=64):
    """Lower the budget so a batch chunks at `nodes` centre nodes."""
    from ace_jax.fit import rows
    K = ds.nbr.shape[2]
    monkeypatch.setattr(rows, "ROWS_EDGE_BUDGET", nodes * rows._node_elems(model, g, K))
    nc = rows.rows_node_chunk(model, g, ds.nbr.shape[1], K)
    assert nc == nodes
    return nc


def test_linear_rows_bounded_equals_unchunked(packed, monkeypatch):
    from ace_jax.fit.rows import linear_rows, linear_rows_bounded
    model, g, ds = packed
    assert ds.nbr.shape[1] >= 800
    refs = [linear_rows(model, g, jax.tree.map(lambda a, i=i: a[i], ds))[0] for i in range(ds.n_batches)]
    _force_chunk(monkeypatch, model, g, ds)
    f = jax.jit(lambda b: linear_rows_bounded(model, g, b))
    for i, ref in enumerate(refs):
        got = f(jax.tree.map(lambda a, i=i: a[i], ds))
        for k in "EFV":
            _close(getattr(got, k), getattr(ref, k), f"{k} batch {i}")


def test_linear_statistics_chunked_equal(packed, monkeypatch):
    from ace_jax.fit.stats import linear_statistics
    model, g, ds = packed
    ref = jax.jit(lambda: linear_statistics(model, g, ds))()
    _force_chunk(monkeypatch, model, g, ds, nodes=96)
    got = jax.jit(lambda: linear_statistics(model, g, ds))()
    for k in ref._fields:
        _close(getattr(got, k), getattr(ref, k), k)


def test_site_features_chunked_equal(packed, monkeypatch):
    from ace_jax.fit.inducing import site_features
    model, g, ds = packed
    X0, S0 = site_features(model, g, ds)
    _force_chunk(monkeypatch, model, g, ds)
    X1, S1 = site_features(model, g, ds)
    _close(X1, X0, "X"); _close(S1, S0, "S")


@pytest.fixture(scope="module")
def hybrid():
    """M > 0 hybrid problem on small si_tiny batches (Ncap a few node_chunks)."""
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.hypers import Hypers, default_prior
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    cfgs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")
    cfgs = [c for c in cfgs if len(c.numbers) == 64][:1] + cfgs[:5]
    ds = build_dataset(cfgs, meta, np.asarray(z["E0"]), configs_per_batch=3)
    g = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                 NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, g, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 5, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, g.D), model, ind, g, jnp.asarray(z["gamma"]),
                   default_prior(2.35))
    theta = Hypers(log_ell=np.log(0.8), log_A=np.log(0.3), log_alpha=np.log(1.2), log_r0=np.log(2.35),
                   log_eps=np.log(0.4), log_rho=np.log(4.0), log_sigma_c=0.0,
                   log_sigma_E=0.0, log_sigma_F=0.0, log_sigma_V=0.0)
    assert ds.nbr.shape[1] >= 2 * g.node_chunk
    return prob, ds, theta


def test_hybrid_statistics_and_gradient_chunked_equal(hybrid, monkeypatch):
    """sufficient_statistics (linear + residual block, fused per chunk) and its theta-gradient."""
    from ace_jax.fit.hypers import from_array, to_array
    from ace_jax.fit.stats import sufficient_statistics
    prob, ds, theta = hybrid
    probe = jax.random.normal(jax.random.PRNGKey(0), (prob.cfg.len_basis + prob.ind.XM.shape[0],))

    def run():
        def f(a):
            st = sufficient_statistics(from_array(a), prob.spec, prob.model, prob.ind, prob.cfg, ds)
            return probe @ st.G_F @ probe + probe @ st.b_E + probe @ st.G_V @ probe, st
        (v, st), gr = jax.jit(jax.value_and_grad(f, has_aux=True))(to_array(theta))
        return st, gr

    ref_st, ref_g = run()
    _force_chunk(monkeypatch, prob.model, prob.cfg, ds, nodes=prob.cfg.node_chunk)
    st, gr = run()
    for k in ref_st._fields:
        _close(getattr(st, k), getattr(ref_st, k), k)
    assert np.all(np.isfinite(np.asarray(gr)))
    np.testing.assert_allclose(np.asarray(gr), np.asarray(ref_g), rtol=1e-10, atol=1e-10)


def test_predict_batch_chunked_equal(hybrid, monkeypatch):
    """The predictive rows path (rows + residual + energy and derivative DTC) chunked."""
    from ace_jax.fit.objective import posterior
    from ace_jax.fit.predict import _predict_batch
    from ace_jax.fit.stats import sufficient_statistics
    prob, ds, theta = hybrid
    st = sufficient_statistics(theta, prob.spec, prob.model, prob.ind, prob.cfg, ds)
    mu, L = posterior(theta, st, prob)
    b = jax.tree.map(lambda a: a[0], ds)
    f = lambda: jax.jit(lambda bb: _predict_batch(theta, prob, mu, L, bb))(b)
    ref = f()
    _force_chunk(monkeypatch, prob.model, prob.cfg, ds, nodes=prob.cfg.node_chunk)
    got = f()
    for a, r in zip(got, ref):
        _close(a, r)


def _abstract_batch(ds, Ncap, K):
    """A one-batch abstract Dataset of ds's layout with n_cap = Ncap, k_cap = K."""
    S = lambda shape, a: jax.ShapeDtypeStruct(shape, a.dtype)
    big = jax.tree.map(lambda a: S((1,) + a.shape[1:], a), ds)
    return big._replace(rij=S((1, Ncap, K, 3), ds.rij), nbr=S((1, Ncap, K), ds.nbr),
                        nbr_mask=S((1, Ncap, K), ds.nbr_mask), node_z=S((1, Ncap), ds.node_z),
                        node_cfg=S((1, Ncap), ds.node_cfg), node_mask=S((1, Ncap), ds.node_mask),
                        node_type=S((1, Ncap), ds.node_type), y_F=S((1, Ncap, 3), ds.y_F),
                        w_F=S((1, Ncap), ds.w_F))


def test_statistics_memory_shape_bounded(packed, monkeypatch):
    """The linear statistics of one n_cap = 3296, k_cap = 928 batch (the bench365 + crack packing),
    traced abstractly: the largest intermediate stays within ROWS_EDGE_BUDGET and does not grow
    when n_cap doubles (only the (Ncap, 3, L) rows do, negligible at this L), while the unchunked
    trace holds the whole-batch edge Jacobian Ncap*K*D -- the check is live."""
    from ace_jax.fit import rows
    from ace_jax.fit.stats import linear_statistics
    model, g, ds = packed
    Ncap, K = 3296, 928
    peak = lambda n: rows._max_elems(
        jax.make_jaxpr(lambda d: linear_statistics(model, g, d))(_abstract_batch(ds, n, K)).jaxpr)
    nc = rows.rows_node_chunk(model, g, Ncap, K)
    assert nc is not None and nc % g.node_chunk == 0
    assert nc * rows._node_elems(model, g, K) <= rows.ROWS_EDGE_BUDGET
    p1, p2 = peak(Ncap), peak(2 * Ncap)
    assert p1 <= rows.ROWS_EDGE_BUDGET, p1
    assert p2 <= p1 + 2 * Ncap * 3 * g.len_basis * 6, (p1, p2)
    monkeypatch.setattr(rows, "ROWS_EDGE_BUDGET", 1 << 62)
    assert peak(Ncap) >= Ncap * K * g.D
