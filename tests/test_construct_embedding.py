"""Element-embedding reduction ported from ACEpotentials (src/models/embeddings.jl):
embedding_rows (:pca / :truncate, normalised or not), _pca_reduce, _generic_frame,
embedding_widths -- checked against the Julia oracle (julia/embedding_reference.jl).
Pure numpy; no Julia at test time."""
import pathlib

import numpy as np
import pytest

from ace_jax.construct.embedding import (_fix_signs, _generic_frame, embedding_rows,
                                         embedding_widths, read_embedding)

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
CASES = ["identity5", "mh1_CrMnFeCoNi", "mh1_SiGe"]


def _ref(name):
    p = FIX / f"embedding_ref_{name}.npz"
    if not p.exists():
        pytest.skip(f"missing {p.name}; see julia/embedding_reference.jl")
    return np.load(p)


@pytest.mark.parametrize("r,d", [(3, 5), (5, 15), (5, 35), (2, 7)])
def test_generic_frame_is_bit_identical_to_julia(r, d):
    z = _ref("identity5")
    Q = _generic_frame(r, d)
    assert Q.shape == (r, d)
    assert np.allclose(Q @ Q.T, np.eye(r), atol=1e-13)            # orthonormal rows
    assert np.allclose(Q, z[f"frame_r{r}_d{d}"], rtol=0, atol=1e-13)


@pytest.mark.parametrize("case", CASES)
def test_embedding_widths_match_julia(case):
    z = _ref(case)
    S = len(z["zlist"])
    for order in (2, 3, 4):
        for dmax in (0, 4, 16):
            got = embedding_widths(S, order, d_max=None if dmax == 0 else dmax)
            assert got == [int(x) for x in z[f"widths_o{order}_dmax{dmax}"]], (order, dmax)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("normalise", [True, False])
def test_pca_rows_match_julia(case, normalise):
    z = _ref(case)
    table, zlist = z["table"], [int(x) for x in z["zlist"]]
    for d in (int(x) for x in z["ds"]):
        ref = z[f"pca_{'norm' if normalise else 'raw'}_d{d}"]
        got = embedding_rows(table, zlist, zlist, d=d, reduction="pca", normalise=normalise)
        assert got.shape == ref.shape
        # the contract: the element Gram (similarity) is preserved exactly
        assert np.allclose(got @ got.T, ref @ ref.T, atol=1e-12), d
        # the rows themselves: Julia's principal coordinates with the port's
        # deterministic sign convention (Julia leaves the sign to LAPACK)
        P = _fix_signs(z["pca_P"])                                  # P = U S, s > 0: same flips as U
        want = P[:, :d] if d <= P.shape[1] else P @ _generic_frame(P.shape[1], d)
        if normalise:
            want = want / np.linalg.norm(want, axis=1, keepdims=True)
        assert np.allclose(got, want, atol=1e-12), d
        if np.allclose(P, z["pca_P"]):
            assert np.allclose(got, ref, atol=1e-12), d            # Julia's LAPACK happened to agree


@pytest.mark.parametrize("case", CASES)
def test_truncate_rows_match_julia(case):
    z = _ref(case)
    table, zlist = z["table"], [int(x) for x in z["zlist"]]
    for d in (int(x) for x in z["ds"]):
        k = f"trunc_norm_d{d}"
        if k in z.files:
            got = embedding_rows(table, zlist, zlist, d=d, reduction="truncate", normalise=True)
            assert np.allclose(got, z[k], atol=1e-14), d


def test_rows_follow_zlist_order_and_reject_unknown_elements():
    rng = np.random.default_rng(0)
    table = rng.normal(size=(4, 9)); Z = [26, 24, 28, 25]
    a = embedding_rows(table, Z, [24, 26], d=3)
    b = embedding_rows(table, Z, [26, 24], d=3)
    assert np.allclose(a @ a.T, (b @ b.T)[::-1, ::-1], atol=1e-12)
    with pytest.raises(ValueError, match="27"):
        embedding_rows(table, Z, [27], d=2)


def test_zero_row_after_reduction_raises_like_julia():
    table = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])            # element 3 has a zero row
    with pytest.raises(ValueError, match="zero embedding row"):
        embedding_rows(table, [1, 2, 3], [1, 2, 3], d=2, reduction="truncate")


def test_sign_convention_is_stable_under_exact_ties():
    """Two elements: the normalised U columns tie exactly in |.|; tiny
    perturbations (as different LAPACKs produce) must not flip the result."""
    h = 1 / np.sqrt(2)
    for eps in (0.0, 1e-15, -1e-15):
        U = np.array([[h + eps, h], [h, -h - eps]])
        for flips in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            V = _fix_signs(U * np.array(flips))
            assert np.allclose(V, [[h, h], [h, -h]], atol=1e-12)


def test_read_embedding_json(tmp_path):
    import json
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"Z": [24, 26], "emb": [[1.0, 2.0], [3.0, 4.0]], "checkpoint": "x"}))
    Z, table, meta = read_embedding(p)
    assert Z == [24, 26] and table.shape == (2, 2) and meta["checkpoint"] == "x"
