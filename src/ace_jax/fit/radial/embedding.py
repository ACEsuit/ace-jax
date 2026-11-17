"""Species embeddings for the coregionalized GP kernel (frozen).

species_onehot -> the block-diagonal (categorical) default; load_mace_embedding
imports a MACE element table (per-element rows), reindexes to the model's
element order, and unit-normalizes rows so B[z,zm] = e_hat(z).e_hat(zm) is a
correlation matrix (diag 1)."""
import json
import numpy as np
import jax.numpy as jnp

_TINY = 1e-30


def species_onehot(NZ):
    return jnp.eye(int(NZ), dtype=jnp.float64)


def normalize_rows(E):
    E = jnp.asarray(E, jnp.float64)
    return E / jnp.sqrt(jnp.sum(E * E, axis=1, keepdims=True) + _TINY)


def gram(E):
    """Species correlation matrix B = Ê Êᵀ (Ê = unit-normalized rows), diag 1."""
    Eh = normalize_rows(E)
    return Eh @ Eh.T


def load_mace_embedding(path, elements):
    """path: MACE embedding artifact (JSON with parallel arrays "emb" (list of raw
    float rows) and "Z" (list of atomic numbers), row i belonging to Z[i]).
    elements: list of atomic numbers in the model's species order. Returns
    (NZ, de) unit-normalized, row z = the artifact row whose Z is elements[z].
    KeyError if the artifact lacks emb/Z or any element is missing."""
    with open(path) as fh:
        d = json.load(fh)
    if "emb" not in d or "Z" not in d:
        raise KeyError("emb")                       # not an embedding artifact
    by_z = {int(zz): row for zz, row in zip(d["Z"], d["emb"])}
    rows = [by_z[int(z)] for z in elements]         # KeyError on a missing element
    return normalize_rows(jnp.asarray(np.asarray(rows, float)))
