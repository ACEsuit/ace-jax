import json, numpy as np, jax.numpy as jnp, pytest
from ace_jax.fit.embedding import species_onehot, normalize_rows, load_mace_embedding

def test_gram_is_correlation_matrix():
    import numpy as np, jax.numpy as jnp
    from ace_jax.fit.embedding import gram
    E = jnp.asarray([[3.0, 0.0], [0.6, 0.8]])       # row 1 already unit
    B = np.asarray(gram(E))
    assert B.shape == (2, 2)
    assert np.allclose(np.diag(B), 1.0)             # unit rows -> diag 1
    assert np.allclose(B, B.T) and abs(B[0, 1] - 0.6) < 1e-9

def test_anchor_penalty_zero_at_orthonormal_and_grad_finite():
    import numpy as np, jax, jax.numpy as jnp
    from ace_jax.fit.embedding import anchor_penalty
    eye = jnp.eye(3)
    assert float(anchor_penalty(eye, 2.0)) < 1e-18   # B = I -> penalty 0
    E = jnp.asarray([[1.0, 0.0], [0.7, 0.7]])
    assert float(anchor_penalty(E, 1.0)) > 0         # coupled -> positive
    g = jax.grad(lambda X: anchor_penalty(X, 1.0))(E)
    assert np.all(np.isfinite(np.asarray(g)))

def test_onehot_is_identity():
    E = np.asarray(species_onehot(4))
    assert E.shape == (4, 4) and np.allclose(E, np.eye(4))

def test_normalize_rows_unit_and_zero_safe():
    E = normalize_rows(jnp.asarray([[3.0, 4.0], [0.0, 0.0]]))
    E = np.asarray(E)
    assert np.allclose(np.linalg.norm(E, axis=1), [1.0, 0.0], atol=1e-9)  # zero row stays zero

def test_load_mace_reindexes_and_normalizes(tmp_path):
    # real MACE artifact format: parallel "emb" (rows) and "Z" (atomic numbers)
    table = {"emb": [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], "Z": [14, 32]}
    p = tmp_path / "emb.json"; p.write_text(json.dumps(table))
    E = np.asarray(load_mace_embedding(str(p), elements=[32, 14]))   # note reordered
    assert E.shape == (2, 3)
    assert np.allclose(np.linalg.norm(E, axis=1), 1.0)               # unit rows
    assert np.allclose(E[1], [1.0, 0.0, 0.0])                        # row for Z=14

def test_load_mace_missing_element_errors(tmp_path):
    p = tmp_path / "emb.json"; p.write_text(json.dumps({"emb": [[1.0, 0.0]], "Z": [14]}))
    with pytest.raises(KeyError):
        load_mace_embedding(str(p), elements=[14, 32])              # 32 absent


def test_sige_coregionalization_end_to_end():
    import numpy as np, jax.numpy as jnp
    from ace_jax.eval import highest_precision
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.kernels import KernelSpec, K_MM
    from ace_jax.fit.inducing import descriptor_scale, select_inducing
    rng = np.random.default_rng(0)
    D, k = 6, 5
    X = jnp.asarray(rng.normal(size=(2 * k, D)))         # synthetic site features, 2 species
    S = jnp.asarray(2.5 + rng.random(2 * k))
    Z = jnp.asarray([0] * k + [1] * k, jnp.int32)
    mask = jnp.ones(2 * k, bool)
    scale = descriptor_scale(X, mask)
    theta = default_prior(2.35).mu
    spec = KernelSpec("cosine", True, D)
    # correlated 2-species embedding (cos-sim 0.6) vs the one-hot default
    E = np.array([[1.0, 0.0], [0.6, np.sqrt(1 - 0.36)]]); E /= np.linalg.norm(E, axis=1, keepdims=True)
    ind_e = select_inducing(X, S, Z, mask, 3, scale, embed=jnp.asarray(E))
    ind_0 = select_inducing(X, S, Z, mask, 3, scale)                    # one-hot default
    with highest_precision():
        Ke = np.asarray(K_MM(theta, spec, ind_e.XM, ind_e.SM, ind_e.ZM, ind_e.embed))
        K0 = np.asarray(K_MM(theta, spec, ind_0.XM, ind_0.SM, ind_0.ZM, ind_0.embed))
    ZM = np.asarray(ind_0.ZM)
    assert (ZM == 0).any() and (ZM == 1).any()          # both species were induced
    cross = (ZM[:, None] != ZM[None, :])
    assert np.abs(K0[cross]).max() < 1e-12               # block-diagonal: no cross coupling
    assert np.abs(Ke[cross]).max() > 1e-8               # coregionalized: cross coupling present
