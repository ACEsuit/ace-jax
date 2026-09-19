import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.inducing import (GPConfig, descriptor_scale, farthest_point, select_inducing,
                            site_features)


def test_farthest_point_picks_square_corners():
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [0.4, 0.6]], float)
    idx = farthest_point(pts, 4, start=0)
    assert sorted(idx.tolist()) == [0, 1, 2, 3]


def test_farthest_point_never_duplicates():
    """Ruling R30: stop once the remaining maximum distance is zero."""
    idx = farthest_point(np.array([[0, 0], [1, 0], [0, 0], [1, 0]], float), 4)
    assert len(idx) == 2 and len(set(idx.tolist())) == 2
    assert sorted(idx.tolist()) == [0, 1]


def test_descriptor_scale_inverts_std():
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.normal(size=(50, 3)) * np.array([1.0, 10.0, 0.1]))
    sc = descriptor_scale(X, jnp.ones(50, bool))
    assert np.allclose(np.asarray(sc) * np.std(np.asarray(X), 0), 1.0, rtol=1e-3)


def test_select_inducing_per_species_counts():
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.normal(size=(30, 4))); S = jnp.ones(30); Z = jnp.asarray([0] * 20 + [1] * 10, jnp.int32)
    mask = jnp.ones(30, bool).at[25:].set(False)          # species 1 has 5 live nodes
    ind = select_inducing(X, S, Z, mask, {0: 6, 1: 8}, jnp.ones(4))
    assert ind.XM.shape == (11, 4) and int((ind.ZM == 0).sum()) == 6 and int((ind.ZM == 1).sum()) == 5


XYZ = FIXTURE_DIR / "si_tiny_train.xyz"


@pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")
def test_site_features_shapes_and_summary_range():
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    # configs[0] is the isolated-atom reference (no neighbours): its s sits at the floor by design
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[1:5]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=2)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=2)
    X, S = site_features(model, cfg, ds)
    assert X.shape[:2] == ds.node_z.shape and X.shape[2] == meta["n_B"] + meta["n_pair"]
    live = np.asarray(ds.node_mask)
    assert 1.5 < float(np.asarray(S)[live].min()) and float(np.asarray(S)[live].max()) < 4.0
    assert float(np.asarray(S)[~live].min()) > 100 * 2.35


@pytest.mark.skipif(not XYZ.exists(), reason="missing si_tiny_train.xyz")
def test_select_inducing_skips_zero_descriptor_nodes():
    """Ruling R30: configs[0] is the isolated atom (all-zero descriptor); with start=0 it
    used to be the first pick of every species pool."""
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:4]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=2)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=2)
    X, S = site_features(model, cfg, ds)
    Xl = np.asarray(X).reshape(-1, X.shape[-1]); live = np.asarray(ds.node_mask).reshape(-1)
    assert np.any(live & ~np.any(Xl != 0, axis=1))          # the isolated atom is live and zero
    for m in (1, 4, 100):
        ind = select_inducing(X, S, ds.node_z, ds.node_mask, m, descriptor_scale(X, ds.node_mask))
        assert ind.XM.shape[0] >= 1 and np.all(np.any(np.asarray(ind.XM) != 0, axis=1))
        assert float(np.asarray(ind.SM).max()) < 100 * 2.35   # never the padding/isolated summary
