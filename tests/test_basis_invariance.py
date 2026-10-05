"""A basis built in Python (`build_basis`) must be a basis of INVARIANTS.

The coupling parity tests (test_coupling_parity.py) compare the coupling
coefficients against ACEpotentials through each column's (n, l, m) signature,
so they cannot see a model that pairs those coefficients with the wrong AA
products.  These tests check the built model itself: site descriptors are
unchanged by a proper rotation and by relabelling the atoms (which permutes
every neighbour list), and the smoothness prior gamma names the (n, l)
channels each B column actually evaluates.  The shapes are not in the
committed coupling cache, so the tests need the compiled coupling library.
"""
import jax
import numpy as np
import pytest

from conftest import require_coupling_lib

jax.config.update("jax_enable_x64", True)

CASES = [(("Si",), 2, 6), (("Si",), 3, 8), (("Si",), 4, 10), (("Si", "Ge"), 3, 6)]
IDS = ["Si-o2d6", "Si-o3d8", "Si-o4d10", "SiGe-o3d6"]
TOL = 1e-10


def _build(elements, order, degree):
    require_coupling_lib()
    from ace_jax.basis.model import BasisSpec, build_basis
    return build_basis(BasisSpec(order=order, max_degree=degree, elements=elements, rcut=5.0))


def _cluster(elements, seed=3, n=12):
    """An open cluster (no pbc): uniform in a 6 A box, mixed species."""
    from ase.data import atomic_numbers
    rng = np.random.default_rng(seed)
    zs = [atomic_numbers[e] for e in elements]
    return rng.uniform(0, 6, size=(n, 3)), np.array([zs[i % len(zs)] for i in range(n)]), rng


def _rel(X0, X1):
    """Per-column change relative to the column's scale."""
    return np.abs(X1 - X0).max(0) / np.maximum(np.abs(X0).max(0), 1e-300)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_built_basis_rotation_and_permutation_invariant(case):
    from ace_jax.eval.api import site_descriptors
    b = _build(*case)
    model, meta = b.eval_pair()
    pos, num, rng = _cluster(case[0])
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))                                # proper rotation
    X0 = np.asarray(site_descriptors(model, pos, num, meta=meta))
    assert np.abs(X0).max() > 0
    Xr = np.asarray(site_descriptors(model, pos @ Q.T, num, meta=meta))
    d = _rel(X0, Xr)
    assert d.max() < TOL, f"{int((d > TOL).sum())} of {d.size} columns not rotation invariant, max {d.max():.3g}"
    p = rng.permutation(len(num))                                  # relabel atoms: permutes neighbour lists
    Xp = np.asarray(site_descriptors(model, pos[p], num[p], meta=meta))
    d = _rel(X0[p], Xp)
    assert d.max() < TOL, f"{int((d > TOL).sum())} of {d.size} columns not permutation invariant, max {d.max():.3g}"


def test_built_basis_invariant_after_npz_roundtrip():
    """The exported file evaluates the same invariants (save_npz -> load)."""
    import io
    from ace_jax.basis.export import save_npz
    from ace_jax.eval import load
    from ace_jax.eval.api import site_descriptors
    b = _build(("Si",), 2, 6)
    buf = io.BytesIO()
    save_npz(buf, b)
    buf.seek(0)
    model, meta, _ = load(buf)
    pos, num, rng = _cluster(("Si",))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    X0 = np.asarray(site_descriptors(model, pos, num, meta=meta))
    Xr = np.asarray(site_descriptors(model, pos @ Q.T, num, meta=meta))
    assert _rel(X0, Xr).max() < TOL


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_built_basis_gamma_matches_evaluated_columns(case):
    """gamma[r] is the prior of the body B row r EVALUATES: the (n, l) channels
    of the AA products its A2B row reads, through aa_specs -> aspec -> Rnl_spec
    (the evaluation path), not through the coupling's own signatures."""
    from ace_jax.basis.prior import smoothness_prior
    b = _build(*case)
    rows = [r for g in b.aa_specs for r in np.asarray(g)]          # eval AA column order
    body = lambda j: tuple(sorted(b.Rnl_spec[b.aspec[a][0]] for a in rows[j]))
    A2B = np.asarray(b.model.A2B)
    n_B = A2B.shape[0]
    for r in range(n_B):
        cols = np.flatnonzero(np.abs(A2B[r]) > 1e-12)
        bodies = {body(j) for j in cols}
        assert len(bodies) == 1, f"B row {r} mixes bodies {bodies}"
        (bb,) = bodies
        assert bb == tuple(sorted(b.nnll[r])), f"B row {r}: evaluates {bb}, nnll says {b.nnll[r]}"
    NZ = len(case[0])
    tensor = [[b.Rnl_spec[b.aspec[a][0]] for a in rows[np.flatnonzero(np.abs(A2B[r]) > 1e-12)[0]]]
              for r in range(n_B)]
    np.testing.assert_array_equal(b.gamma[:n_B * NZ], smoothness_prior(tensor * NZ))


def test_align_columns_repairs_an_unaligned_entry():
    """A cache entry written before the fix holds the columns in the library's
    𝔸spec order; `align_columns` (applied to every build) must restore the
    evaluation order exactly from any column permutation, and is idempotent."""
    require_coupling_lib()
    from ace_jax.basis.coupling import align_columns, couple
    from ace_jax.basis.spec import build_spec
    mb, Rnl, Ylm = build_spec(1, 3, 8)
    cpl = couple(mb, Rnl, Ylm)
    assert align_columns(cpl, Rnl, Ylm) is cpl                     # already aligned: unchanged
    p = np.random.default_rng(0).permutation(cpl.A2B.shape[1])
    stale = cpl._replace(A2B=cpl.A2B[:, p], aa_sig=tuple(cpl.aa_sig[j] for j in p))
    fixed = align_columns(stale, Rnl, Ylm)
    np.testing.assert_array_equal(fixed.A2B, cpl.A2B)
    assert fixed.aa_sig == cpl.aa_sig


def test_built_basis_descriptors_match_acepotentials():
    """End-to-end parity with ACEpotentials `ace_model` (fixtures/si_ace_model.npz,
    Si order 3, totaldegree 10): build_spec's mb order through the LIVE coupling
    (not the fixture's grouped nnll order, which hid the column bug), the
    fixture's radial coefficients injected, then each B descriptor column over
    the 64 test sites must be a nonzero multiple of the Julia column with the
    same nnll (every block has multiplicity 1 at this size; ET main rescales
    B rows against the fixture's ET 0.4.3)."""
    import dataclasses
    import json
    import jax.numpy as jnp
    from conftest import FIXTURE_DIR
    from ace_jax.eval.api import site_descriptors
    path = FIXTURE_DIR / "si_ace_model.npz"
    if not path.exists():
        pytest.skip("fixture missing")
    require_coupling_lib()
    from ace_jax.basis.model import build_model
    zf = np.load(path)
    fmeta = json.loads(bytes(zf["meta_json"]).decode())
    b = build_model([14], 3, 10, rcut=fmeta["rcut"], coupling_cache=False)
    model = dataclasses.replace(b.model, **{k: jnp.asarray(zf[k]) for k in (
        "rnl_Wnlq", "polys_A", "polys_B", "polys_C", "rnl_transform", "rnl_envelope")})
    model, meta = b._replace(model=model).eval_pair()
    n_B = meta["n_B"]
    assert n_B == fmeta["n_B"]
    D = np.asarray(site_descriptors(model, zf["test_pos"].T, zf["test_Z"], cell=zf["test_cell"].T,
                                    pbc=zf["test_pbc"].astype(bool), meta=meta))[:, :n_B]
    J = np.asarray(zf["test_desc"]).T[:, :n_B]                      # (sites, B) from ACEpotentials
    jcol = {tuple(sorted(map(tuple, bb))): j for j, bb in enumerate(fmeta["nnll"])}
    assert len(jcol) == n_B and sorted(jcol) == sorted(tuple(sorted(bb)) for bb in b.nnll)
    worst = 0.0
    for r, bb in enumerate(b.nnll):
        d, j = D[:, r], J[:, jcol[tuple(sorted(bb))]]
        s = float(d @ j / (j @ j))
        assert s != 0.0
        worst = max(worst, float(np.abs(d - s * j).max() / np.abs(d).max()))
    assert worst < 1e-10, worst
