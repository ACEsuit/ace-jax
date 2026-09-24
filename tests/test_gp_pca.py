"""Low-rank principal-frame projection of the site descriptors for the residual
GP (--density pca): the same uncentred principal-frame reduction as the species
embedding (ACEpotentials _pca_reduce; run.py's --learn-embedding SVD init)."""
import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)

from ace_jax.fit.inducing import build_pmap, principal_frame


def test_principal_frame_is_the_truncated_uncentred_svd():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(40, 6)) @ rng.normal(size=(6, 15))           # rank 6
    C, V = principal_frame(R, 4)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    assert C.shape == (40, 4) and V.shape == (15, 4)
    assert np.allclose(V.T @ V, np.eye(4), atol=1e-12)                # orthonormal frame
    assert np.allclose(C, R @ V, atol=1e-10)                           # coordinates = projection
    assert np.allclose(np.abs(C), np.abs(U[:, :4] * s[:4]), atol=1e-10)  # the embedding-init form
    C6, _ = principal_frame(R, 6)                                      # d >= rank: exact Gram
    assert np.allclose(C6 @ C6.T, R @ R.T, atol=1e-9)


def test_principal_frame_caps_d_at_the_rank():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(30, 3)) @ rng.normal(size=(3, 10))           # rank 3
    C, V = principal_frame(R, 8)
    assert C.shape[1] == V.shape[1] == 3


def _si():
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, site_features
    XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
    if not XYZ.exists():
        pytest.skip("missing si_tiny_train.xyz")
    model, meta, z = load(FIXTURE_DIR / "si_fitted.npz")
    ds = build_dataset(load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:9], meta,
                       np.asarray(z["E0"]), 3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, _ = site_features(model, cfg, ds)
    scale = descriptor_scale(X, ds.node_mask)
    return cfg, np.asarray(X), np.asarray(ds.node_mask), np.asarray(scale)


def test_pca_pmap_is_the_principal_frame_of_the_scaled_live_sites():
    cfg, X, mask, scale = _si()
    Xs = X[mask] * scale                                               # live sites, scaled
    P = build_pmap(cfg, scale, density="pca", d=5, X=X, mask=mask)
    assert P.shape == (cfg.D, 5)
    V = P / scale[:, None]                                             # P = diag(scale) V
    assert np.allclose(V.T @ V, np.eye(5), atol=1e-10)
    _, Vref = principal_frame(Xs, 5)
    assert np.allclose(np.abs(V.T @ Vref), np.eye(5), atol=1e-6)       # same frame up to signs
    r = np.linalg.matrix_rank(Xs)
    Pf = build_pmap(cfg, scale, density="pca", d=r, X=X, mask=mask)   # full rank: Gram kept
    assert np.allclose((X[mask] @ Pf) @ (X[mask] @ Pf).T, Xs @ Xs.T, rtol=1e-8, atol=1e-8)


def test_pca_pmap_needs_the_descriptors():
    cfg, X, mask, scale = _si()
    with pytest.raises(ValueError, match="pca"):
        build_pmap(cfg, scale, density="pca", d=5)
