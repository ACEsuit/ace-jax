"""The sparse A2B coupling on the fitting path: B, dB/dA and the rows' Jacobian parts
from the (row-sorted) triplets must equal the dense contraction to roundoff, built
bases must come out sparse (no dense A2B held), and the rows' node-chunk policy must
not mistake the model-sized A2B for per-node work."""
import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from conftest import FIXTURE_DIR, require_coupling_lib  # noqa: E402
from ace_jax.eval import highest_precision, load  # noqa: E402
from ace_jax.eval.model import a2b_sparse_auto, fold_readout, with_a2b_sparse  # noqa: E402

TOL = 1e-13
SIZES = [(3, 8), (4, 12)]


@pytest.fixture(scope="module", params=SIZES, ids=lambda s: f"o{s[0]}d{s[1]}")
def built(request):
    require_coupling_lib()
    from ace_jax.basis.model import build_model
    o, d = request.param
    return build_model([14], o, d, rcut=5.0)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _nodes(model, n=5, seed=0):
    """A realistic node-major A (n, n_A): the pooled A basis of a Si config."""
    n_A = int(model.aspec_r.shape[0])
    rng = np.random.default_rng(seed)
    K = 12
    rij = jnp.asarray(rng.normal(size=(n, K, 3)) * 0.6 + np.array([2.3, 0.0, 0.0]) * rng.choice([-1, 1], (n, K, 1)))
    zi = jnp.zeros((n, K), jnp.int32)
    mask = jnp.asarray(rng.random((n, K)) < 0.9)
    A, _, _, _, _ = model._dense_jacobian_parts(rij, zi, zi, mask)
    assert A.shape == (n, n_A)
    return A, (rij, zi, mask)


def test_built_basis_is_sparse(built):
    m = built.model
    nB, nAA = built.meta["n_B"], built.meta["n_AA"]
    assert m.a2b_sparse and m.A2B is None
    assert m.a2b_shape == (nB, nAA)
    assert a2b_sparse_auto(int(m.a2b_vals.shape[0]), nB, nAA)
    D = np.asarray(m.a2b_matrix())
    assert D.shape == (nB, nAA) and np.count_nonzero(D) == m.a2b_vals.shape[0]


def test_node_B_and_jacobian_match_dense(built):
    """B, jacfwd(B) and the structured dB/dA the rows use, sparse vs dense."""
    ms = built.model
    md = with_a2b_sparse(ms, False)
    assert not md.a2b_sparse and md.A2B is not None
    A, _ = _nodes(ms)
    with highest_precision():
        Bd = jax.vmap(md._node_B)(A)
        Bs = jax.vmap(ms._node_B)(A)
        Jd = jax.vmap(jax.jacfwd(md._node_B))(A)
        Js = jax.vmap(jax.jacfwd(ms._node_B))(A)
        Jst = ms._b_jacobian(A)
        Bb = ms._b_from_a(A)
    assert np.abs(np.asarray(Bd)).max() > 0 and np.abs(np.asarray(Jd)).max() > 0
    assert _rel(Bs, Bd) < TOL
    assert _rel(Bb, Bd) < TOL
    assert _rel(Js, Jd) < TOL
    assert Jst.shape == Jd.shape and _rel(Jst, Jd) < TOL


REPEATED = {(3, 8): (39, 3), (4, 12): (446, 4)}     # AA columns with a repeated A factor, max multiplicity


def test_bases_have_repeated_factors(built):
    """The leave-one-out dB/dA sums one entry per factor position, so a repeated factor (A_m^k)
    must come out as k A_m^(k-1) -- these bases exercise that: 39 / 446 such columns, up to A_m^3 / A_m^4."""
    n, mx = 0, 1
    for g in built.model.aa_specs:
        for row in np.asarray(g):
            c = int(np.bincount(row).max())
            n, mx = n + (c > 1), max(mx, c)
    o, d = built.meta["order"], built.meta["totaldegree"]
    assert (n, mx) == REPEATED[(o, d)]


def test_jacobian_with_exact_zeros_in_A(built):
    """Exact zeros in A (a radial or Y_lm channel that vanishes, a node with no neighbour): the
    leave-one-out products need no division, so dB/dA stays exact -- including d(A_m^k)/dA_m at A_m = 0."""
    ms = built.model
    md = with_a2b_sparse(ms, False)
    A, _ = _nodes(ms)
    A = np.array(A)
    A[0] = 0.0                                    # a node with no neighbours
    A[1, ::3] = 0.0                               # every third channel of another
    for g in ms.aa_specs:                         # zero a factor that is repeated in some AA column
        rows = [r for r in np.asarray(g) if np.bincount(r).max() > 1]
        if rows:
            A[2, rows[0][np.argmax([list(rows[0]).count(x) for x in rows[0]])]] = 0.0
            break
    A = jnp.asarray(A)
    with highest_precision():
        Jd = jax.vmap(jax.jacfwd(md._node_B))(A)
        Js = ms._b_jacobian(A)
        Bd, Bs = jax.vmap(md._node_B)(A), ms._b_from_a(A)
    assert np.isfinite(np.asarray(Js)).all()
    assert _rel(Js, Jd) < TOL and _rel(Bs, Bd) < TOL
    np.testing.assert_array_equal(np.asarray(Js[0]), np.asarray(Jd[0]))   # all-zero node: exactly what jacfwd gives


def test_dense_jacobian_parts_match_dense(built):
    ms = built.model
    md = with_a2b_sparse(ms, False)
    _, (rij, zi, mask) = _nodes(ms)
    with highest_precision():
        Xs, Js = ms.edge_jacobian_dense(rij, zi, zi, mask)
        Xd, Jd = md.edge_jacobian_dense(rij, zi, zi, mask)
    assert _rel(Xs, Xd) < TOL and _rel(Js, Jd) < TOL


def test_sparse_a2b_roundtrip_dense():
    """Dense -> sparse -> dense is the identity, and fold_readout agrees."""
    md, _, _ = load(FIXTURE_DIR / "si_1429.npz", a2b_sparse=False, fold=False)
    ms = with_a2b_sparse(md, True)
    assert ms.A2B is None and ms.a2b_shape == md.A2B.shape
    np.testing.assert_array_equal(np.asarray(ms.a2b_matrix()), np.asarray(md.A2B))
    np.testing.assert_array_equal(np.asarray(with_a2b_sparse(ms, False).A2B), np.asarray(md.A2B))
    WB = jnp.asarray(np.random.default_rng(0).normal(size=md.WB.shape))
    fd = fold_readout(_with_wb(md, WB))
    fs = fold_readout(_with_wb(ms, WB))
    assert _rel(fs.ctilde, fd.ctilde) < 1e-15


def _with_wb(m, WB):
    import dataclasses
    return dataclasses.replace(m, WB=WB)


@pytest.mark.parametrize("name", ["si_1429.npz", "si_ace_model.npz", "si_fitted.npz"])
def test_load_auto_is_sparse(name):
    """`load` converts to sparse when A2B is sparse enough -- including npz files that store
    a dense A2B (si_ace_model.npz) -- and `a2b_sparse=False` keeps the dense form."""
    m, meta, _ = load(FIXTURE_DIR / name)
    assert m.a2b_sparse and m.A2B is None and m.a2b_shape == (meta["n_B"], meta["n_AA"])
    md, _, _ = load(FIXTURE_DIR / name, a2b_sparse=False)
    assert not md.a2b_sparse and md.A2B.shape == (meta["n_B"], meta["n_AA"])


def test_node_elems_excludes_model_sized_a2b():
    """`_node_elems` is a per-node figure: the dense A2B (n_B x n_AA, here 16.4M elements) is a
    model-sized intermediate of the dense Jacobian, not per-node work.  The sparse model's
    per-node figure is far below it and below the dense path's own per-node work (its
    (n_AA, n_A) jacfwd tangents), so the rows' node chunk no longer scales with A2B."""
    from ace_jax.fit import rows
    md, meta, _ = load(FIXTURE_DIR / "si_1429.npz", a2b_sparse=False)
    ms, _, _ = load(FIXTURE_DIR / "si_1429.npz")
    nB, nAA = md.A2B.shape
    K = 30
    cfg = SimpleNamespace(D=nB + meta["n_pair"])
    rows._NODE_ELEMS.clear()
    pd, ps = rows._node_elems(md, cfg, K), rows._node_elems(ms, cfg, K)
    assert ps >= K * cfg.D * 3
    assert ps < nB * nAA / 3 and ps < pd / 4, (ps, pd, nB * nAA)     # 4.8M, 43M, 16.4M
