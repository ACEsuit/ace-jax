import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh

from ace_jax.eval import highest_precision, load
from ace_jax.cli import _pad_to_multiple
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.hypers import Hypers, from_array, to_array
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.sharding import sufficient_statistics_sharded
from ace_jax.fit.stats import sufficient_statistics

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
# Module-wide requirement is just the fixture; the padding test below is pure
# single-device logic and must not be skipped for want of a second device
# (fix round 1 -- the multi-device skip is applied only to the sharded test).
pytestmark = pytest.mark.skipif(not XYZ.exists(), reason="needs si_tiny_train.xyz")


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs 2 devices")
def test_sharded_equals_single_device():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:8]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), 2)           # 4 batches
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=2)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 4, descriptor_scale(X, ds.node_mask))
    spec = KernelSpec("cosine", True, cfg.D)
    theta = Hypers(np.log(0.8), np.log(0.3), 0.0, np.log(2.35), np.log(0.4), np.log(4.0), 0.0, 0.0, 0.0, 0.0)
    mesh = Mesh(np.array(jax.devices()[:2]), ("data",))
    probe = jax.random.normal(jax.random.PRNGKey(0), (cfg.len_basis + 4,))
    f1 = lambda a: probe @ sufficient_statistics(from_array(a), spec, model, ind, cfg, ds).G_F @ probe
    f2 = lambda a: probe @ sufficient_statistics_sharded(from_array(a), spec, model, ind, cfg, ds, mesh).G_F @ probe
    with highest_precision():
        a = to_array(theta)
        v1, g1 = jax.value_and_grad(f1)(a); v2, g2 = jax.value_and_grad(f2)(a)
    assert abs(float(v1 - v2)) < 1e-9 * abs(float(v1))
    assert float(jnp.abs(g1 - g2).max()) < 1e-9 * max(1.0, float(jnp.abs(g1).max()))


def test_pad_to_multiple():
    """Pure single-device padding logic (fix round 1): the appended batch
    must be inert to every consumer, not just the weighted statistics, and
    sufficient_statistics on the padded dataset must equal the unpadded one."""
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[1:7]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), 2)              # 3 batches
    assert ds.n_batches == 3
    ds_pad = _pad_to_multiple(ds, 2)
    assert ds_pad.n_batches == 4
    assert bool(jnp.all(ds_pad.w_E[-1] == 0.0))
    assert bool(jnp.all(ds_pad.w_F[-1] == 0.0))
    assert bool(jnp.all(ds_pad.w_V[-1] == 0.0))
    assert not bool(jnp.any(ds_pad.cfg_mask[-1]))
    assert not bool(jnp.any(ds_pad.node_mask[-1]))
    assert not bool(jnp.any(ds_pad.nbr_mask[-1]))

    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=2)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 4, descriptor_scale(X, ds.node_mask))
    spec = KernelSpec("cosine", True, cfg.D)
    theta = Hypers(np.log(0.8), np.log(0.3), 0.0, np.log(2.35), np.log(0.4), np.log(4.0), 0.0, 0.0, 0.0, 0.0)
    with highest_precision():
        st = sufficient_statistics(theta, spec, model, ind, cfg, ds)
        st_pad = sufficient_statistics(theta, spec, model, ind, cfg, ds_pad)
    assert float(jnp.abs(st_pad.G_F - st.G_F).max()) < 1e-12 * max(1.0, float(jnp.abs(st.G_F).max()))
    assert float(jnp.abs(st_pad.b_E - st.b_E).max()) < 1e-12 * max(1.0, float(jnp.abs(st.b_E).max()))
