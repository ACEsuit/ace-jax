import json, numpy as np, jax.numpy as jnp, pytest
from ace_jax.fit.embedding import species_onehot, normalize_rows, load_mace_embedding

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
