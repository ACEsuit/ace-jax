import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.fit.hypers import Hypers, default_prior
from ace_jax.fit.kernels import K_MM, KernelSpec, delta, grad_k_rows, k_rows, kernel

THETA = Hypers(log_ell=np.log(0.7), log_A=np.log(0.2), log_alpha=np.log(1.0),
               log_r0=np.log(2.3), log_eps=np.log(0.3), log_rho=np.log(3.0),
               log_sigma_c=0.0, log_sigma_E=0.0, log_sigma_F=0.0, log_sigma_V=0.0)
D = 12


def _data(n, seed):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.normal(size=(n, D)))
    S = jnp.asarray(rng.uniform(1.8, 3.5, size=n))
    Z = jnp.asarray(rng.integers(0, 2, size=n), jnp.int32)
    return X, S, Z


@pytest.mark.parametrize("kind", ["cosine", "matern32"])
@pytest.mark.parametrize("bump", [True, False])
def test_kmm_symmetric_positive_definite(kind, bump):
    spec = KernelSpec(kind=kind, bump=bump, D=D)
    XM, SM, ZM = _data(40, 0)
    K = K_MM(THETA, spec, XM, SM, ZM, jnp.eye(2), jitter=0.0)
    assert float(jnp.abs(K - K.T).max()) < 1e-12
    ev = jnp.linalg.eigvalsh(K)
    assert float(ev.min()) > -1e-10 * float(ev.max())


def test_species_delta_and_diagonal():
    spec = KernelSpec(kind="cosine", bump=True, D=D)
    X, S, Z = _data(5, 1)
    K = k_rows(THETA, spec, X, S, Z, X, S, Z, jnp.eye(2))
    for i in range(5):
        assert abs(float(K[i, i]) - float(delta(S[i], THETA)) ** 2) < 1e-12
        for j in range(5):
            if int(Z[i]) != int(Z[j]):
                assert float(K[i, j]) == 0.0


def test_delta_vanishes_for_isolated_atom_with_finite_gradient():
    g = jax.grad(lambda s: delta(s, THETA))
    assert float(delta(jnp.asarray(1e30), THETA)) == 0.0
    assert np.isfinite(float(g(jnp.asarray(1e30))))
    assert float(delta(jnp.asarray(2.3), THETA)) > 0.0


@pytest.mark.parametrize("kind", ["cosine", "matern32"])
def test_grad_k_rows_matches_finite_difference(kind):
    spec = KernelSpec(kind=kind, bump=True, D=D)
    X, S, Z = _data(3, 2)
    XM, SM, ZM = _data(4, 3)
    Z = jnp.zeros(3, jnp.int32); ZM = jnp.zeros(4, jnp.int32)
    embed = jnp.eye(2)
    dKx, dKs = grad_k_rows(THETA, spec, X, S, Z, XM, SM, ZM, embed)
    h = 1e-6
    for i in range(3):
        for m in range(4):
            for d in range(D):
                fd = (kernel(THETA, spec, X[i].at[d].add(h), S[i], Z[i], XM[m], SM[m], ZM[m], embed)
                      - kernel(THETA, spec, X[i].at[d].add(-h), S[i], Z[i], XM[m], SM[m], ZM[m], embed)) / (2 * h)
                assert abs(float(dKx[i, m, d]) - float(fd)) < 1e-6
            fd = (kernel(THETA, spec, X[i], S[i] + h, Z[i], XM[m], SM[m], ZM[m], embed)
                  - kernel(THETA, spec, X[i], S[i] - h, Z[i], XM[m], SM[m], ZM[m], embed)) / (2 * h)
            assert abs(float(dKs[i, m]) - float(fd)) < 1e-6


def test_gradient_defined_at_coincident_points():
    """Inducing points are drawn from the training environments, so x == xm occurs."""
    spec = KernelSpec(kind="matern32", bump=True, D=D)
    X, S, Z = _data(2, 4)
    dKx, dKs = grad_k_rows(THETA, spec, X, S, Z, X, S, Z, jnp.eye(2))
    assert bool(jnp.all(jnp.isfinite(dKx))) and bool(jnp.all(jnp.isfinite(dKs)))


def test_unknown_kernel_kind_raises():
    X, S, Z = _data(2, 3)
    with pytest.raises(ValueError, match="unknown kernel kind"):
        kernel(THETA, KernelSpec(kind="rbf", D=D), X[0], S[0], Z[0], X[1], S[1], Z[1], jnp.eye(2))


def _toy(NZ=3, N=5, M=4, D=6, seed=0):
    r = np.random.RandomState(seed)
    theta = default_prior(2.35).mu
    spec = KernelSpec("cosine", True, D)
    X = jnp.asarray(r.randn(N, D)); S = jnp.asarray(2.5 + r.rand(N))
    Z = jnp.asarray(r.randint(0, NZ, N), jnp.int32)
    XM = jnp.asarray(r.randn(M, D)); SM = jnp.asarray(2.5 + r.rand(M))
    ZM = jnp.asarray(r.randint(0, NZ, M), jnp.int32)
    return theta, spec, X, S, Z, XM, SM, ZM, NZ


def test_onehot_embed_equals_block_diagonal():
    theta, spec, X, S, Z, XM, SM, ZM, NZ = _toy()
    eye = jnp.eye(NZ)
    K_embed = k_rows(theta, spec, X, S, Z, XM, SM, ZM, eye)
    # reference: same call, then zero the cross-species entries by hand
    K_ref = k_rows(theta, spec, X, S, Z, XM, SM, ZM, eye)
    mask = (np.asarray(Z)[:, None] == np.asarray(ZM)[None, :]).astype(float)
    assert np.allclose(np.asarray(K_embed), np.asarray(K_embed) * mask, atol=1e-12)
    assert np.allclose(np.asarray(K_embed), np.asarray(K_ref) * mask, atol=1e-12)


def test_offdiagonal_embed_couples_species():
    theta, spec, X, S, Z, XM, SM, ZM, NZ = _toy()
    # _toy()'s fixed seed happens to draw no species-0 atoms; force one 0/1
    # pair so the cross-species coupling this test checks is actually exercised.
    Z = Z.at[0].set(0); ZM = ZM.at[0].set(1)
    # two species share a direction -> nonzero cross-covariance
    E = np.eye(NZ); E[1] = E[0]                      # species 0 and 1 identical
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    K = np.asarray(k_rows(theta, spec, X, S, Z, XM, SM, ZM, jnp.asarray(E)))
    cross = [(i, j) for i in range(len(Z)) for j in range(len(ZM))
             if {int(Z[i]), int(ZM[j])} == {0, 1}]
    assert any(abs(K[i, j]) > 0 for i, j in cross)   # 0-1 pairs now couple


def test_default_inducing_embed_is_onehot():
    import numpy as np
    from conftest import FIXTURE_DIR
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
    model, meta, z = load(FIXTURE_DIR / "sige_nofit.npz")
    cfgs = load_configs(FIXTURE_DIR / "si_tiny_train.xyz", "dft_energy", "dft_force", "dft_virial")[:3]
    ds = build_dataset(cfgs, meta, np.asarray(z["E0"]), 3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 4, descriptor_scale(X, ds.node_mask))
    E = np.asarray(ind.embed)
    assert E.ndim == 2 and E.shape[0] == E.shape[1]                 # square identity of its own width
    assert np.allclose(E, np.eye(E.shape[0]))
    ZM = np.asarray(ind.ZM)                                          # the gate: embed[z].embed[zm]==(z==zm)
    G = E[ZM] @ E[ZM].T
    assert np.allclose(G, (ZM[:, None] == ZM[None, :]).astype(float))


def test_inducing_embed_nz_gives_full_width_onehot():
    # I1 regression: with nz set, the default one-hot spans ALL model species even when the
    # data presents only some, so a predict-time gather of a test-only species cannot clamp.
    import numpy as np, jax.numpy as jnp
    from ace_jax.fit.inducing import descriptor_scale, select_inducing
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.normal(size=(8, 4))); S = jnp.asarray(2.5 + rng.random(8))
    Z = jnp.asarray([0] * 8, jnp.int32); mask = jnp.ones(8, bool)     # only species 0 present
    ind = select_inducing(X, S, Z, mask, 3, descriptor_scale(X, mask), nz=3)
    E = np.asarray(ind.embed)
    assert E.shape == (3, 3) and np.allclose(E, np.eye(3))            # full width despite 1 species
    ind0 = select_inducing(X, S, Z, mask, 3, descriptor_scale(X, mask))
    assert np.asarray(ind0.embed).shape == (1, 1)                     # default: present-species width
