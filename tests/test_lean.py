"""The lean evaluation form of an ACEModel (`ace_jax.eval.model.lean`) is exact.

Each load-time transform removes per-edge work the energy never reads, and must
leave E, forces and the virial unchanged to roundoff (docs/dev/ace-vs-pace-gap.md):

  prune       R_nl columns no A entry reads, and Y_lm above the largest l used
  pairfold    the pair readout Wpair[:, z_i] folded into the pair radial table
  block       the dense A pooled per l-block (species-compact where R_nl is
              block-sparse in z_j), feature-major for the product basis

Fitting never sees these: `load` returns the full model, and a pair-folded
model refuses the basis methods (its pair channel is the readout, not Apair).

`lean` also splines an analytic radial (`to_spline`), which is an
approximation to its tolerance, not exact: here `lean_exact` (spline_tol=None)
keeps it analytic, and tests/test_to_spline.py covers the splined form.
"""
import dataclasses
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk

from ace_jax.eval import load
from ace_jax.eval.model import block_dense, fold_pair, highest_precision, lean, prune_columns
from ace_jax.eval.nlist import dense_from_sparse, sparse_graph

ROOT = pathlib.Path(__file__).parent.parent
MODELS = {
    "si_fitted": ROOT / "fixtures" / "si_fitted.npz",            # 1 species, spline, spherical
    "si_ace_model": ROOT / "fixtures" / "si_ace_model.npz",      # 1 species, analytic, solid
    "sige_nofit": ROOT / "fixtures" / "sige_nofit.npz",          # 2 species, spline
    "emb_SiGe": ROOT / "fixtures" / "emb_ref_SiGe_o2d6.npz",     # 2 species, factorised radial
    "Cantor_small": ROOT / "fixtures" / "ace_cantor5_small.npz",  # 5 species (bench Cantor small ACE)
}
TOL = 1e-12


def lean_exact(m):
    """`lean` without `to_spline`: every transform exact to roundoff."""
    return lean(m, spline_tol=None)


def _structure(meta, seed=0):
    """A rattled cell with every species present, dense enough for a pair term
    and many-body terms at the model cutoff."""
    els = [int(z) for z in meta["elements"]]
    if len(els) > 2:
        at = bulk("Ni", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    else:
        at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    rng = np.random.default_rng(seed)
    at.numbers = np.asarray(els)[np.arange(len(at)) % len(els)]
    rng.shuffle(at.numbers)
    at.rattle(0.08, seed=seed)
    return at


# jitted, the model a pytree argument: each distinct model structure compiles once and
# new weights reuse it (run eagerly, every op dispatched one by one, 3-5x slower)
_efv_sparse = jax.jit(lambda m, *a: m.energy_forces_virial(*a), static_argnums=(6,))
_efv_dense = jax.jit(lambda m, *a: m.energy_forces_virial_dense(*a))


def _efv(model, meta, at, layout):
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    with highest_precision():
        if layout == "sparse":
            s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
            out = _efv_sparse(model, jnp.asarray(g.rij), nz[s], nz[r], s, r, len(at), nz)
        else:
            d = dense_from_sparse(g, meta["rcut"])
            idx = jnp.asarray(d.idx)
            out = _efv_dense(model, jnp.asarray(d.rij), jnp.broadcast_to(nz[:, None], idx.shape), nz[idx],
                             idx, jnp.asarray(d.mask), nz)
    return tuple(np.asarray(x) for x in out)


def _assert_same(a, b):
    (E0, F0, V0), (E1, F1, V1) = a, b
    print(f"\n  |dE| {abs(E0 - E1):.2e}  |dF| {np.abs(F0 - F1).max():.2e}"
          f"  |dV| {np.abs(V0 - V1).max():.2e}")
    # as tests/test_fold.py: the total energy is judged relative (a sum over the
    # cell, below one ulp at 1e-12 absolute), forces and virial absolute
    assert abs(E0 - E1) <= TOL * max(1.0, abs(E0))
    assert np.abs(F0 - F1).max() < TOL
    assert np.abs(V0 - V1).max() < TOL


@pytest.fixture(scope="module", params=list(MODELS))
def loaded(request):
    m, meta, _ = load(str(MODELS[request.param]))
    return request.param, m, meta, _structure(meta)


TRANSFORMS = {
    "prune": prune_columns,
    "pairfold": fold_pair,
    "prune+pairfold": lambda m: fold_pair(prune_columns(m)),
    "block": block_dense,
    "lean": lean_exact,
}


@pytest.mark.parametrize("layout", ["sparse", "dense"])
@pytest.mark.parametrize("transform", list(TRANSFORMS))
def test_transform_is_exact(loaded, transform, layout):
    name, m, meta, at = loaded
    m1 = TRANSFORMS[transform](m)
    _assert_same(_efv(m, meta, at, layout), _efv(m1, meta, at, layout))


def test_prune_drops_unread_columns_and_harmonics(loaded):
    name, m, meta, at = loaded
    m1 = prune_columns(m)
    n_r0, n_y0 = m.edge_a_widths()
    n_r1, n_y1 = m1.edge_a_widths()
    used = np.unique(np.asarray(m.aspec_r))
    assert n_r1 == len(used) <= n_r0
    l_used = int(np.floor(np.sqrt(np.asarray(m.aspec_y).max())))
    assert m1.lmax == l_used and n_y1 == (l_used + 1) ** 2 <= n_y0
    assert not m1.energy_only                              # the A basis is unchanged
    assert prune_columns(m1).edge_a_widths() == m1.edge_a_widths()


def test_prune_keeps_the_basis(loaded):
    """Pruned columns carry no A entry, so B and Apair are unchanged."""
    name, m, meta, at = loaded
    m1 = prune_columns(m)
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    with highest_precision():
        B0, P0 = m.site_basis(jnp.asarray(g.rij), nz[s], nz[r], s, len(at))
        B1, P1 = m1.site_basis(jnp.asarray(g.rij), nz[s], nz[r], s, len(at))
    np.testing.assert_allclose(np.asarray(B1), np.asarray(B0), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(np.asarray(P1), np.asarray(P0))


def test_pairfold_is_one_column_and_energy_only(loaded):
    name, m, meta, at = loaded
    m1 = fold_pair(m)
    assert m1.Wpair.shape == (1, m.Wpair.shape[1]) and m1.energy_only
    assert fold_pair(m1) is m1
    z = jnp.zeros((1,), jnp.int32)
    with pytest.raises(ValueError, match="energy-only"):
        m1.site_basis(jnp.ones((1, 3)), z, z, z, 1)


def test_load_is_not_lean():
    """Fitting loads the model with `load`: it must stay the full model (every
    pair column, every R_nl column, the exported lmax)."""
    for p in MODELS.values():
        m, meta, z = load(str(p))
        assert not m.energy_only
        assert m.Wpair.shape == tuple(np.asarray(z["Wpair"]).shape)
        assert m.lmax == int(meta["lmax"])


def test_lean_needs_a_folded_model():
    m, _, _ = load(str(MODELS["si_fitted"]), fold=False)
    assert lean(m) is m                                    # unfolded: used as given
    with pytest.raises(ValueError, match="folded"):
        fold_pair(m)


def test_lean_is_idempotent(loaded):
    name, m, meta, at = loaded
    m1 = lean(m)
    assert lean(m1) is m1


def test_lean_keeps_the_matmul_form(loaded):
    """A model in the one-hot matmul A form keeps it, with selectors sized for
    the pruned widths."""
    from ace_jax.eval.edge_model import with_edge_a_kind
    name, m, meta, at = loaded
    mm = with_edge_a_kind(m, "matmul")
    m1 = lean_exact(mm)
    assert m1.edge_a_kind == "matmul"
    assert m1.a_sel_r.shape[0] == m1.edge_a_widths()[0]
    _assert_same(_efv(m, meta, at, "sparse"), _efv(m1, meta, at, "sparse"))


@pytest.mark.parametrize("dense", [False, True])
def test_lean_readout_stays_trainable(dense):
    """dE/d(ctilde) through the lean paths is live and matches the full model's."""
    m, meta, _ = load(str(MODELS["sige_nofit"]))
    at = _structure(meta)
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    if dense:
        d = dense_from_sparse(g, meta["rcut"])
        idx, mask, rij = jnp.asarray(d.idx), jnp.asarray(d.mask), jnp.asarray(d.rij)
        energies = lambda mm: mm.site_energies_dense(rij, jnp.broadcast_to(nz[:, None], idx.shape),
                                                     nz[idx], mask, nz)
    else:
        s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
        energies = lambda mm: mm.site_energies(jnp.asarray(g.rij), nz[s], nz[r], s, len(at), nz)
    grads = []
    for mm in (m, lean(m)):
        E = lambda w, mm=mm: jnp.sum(energies(dataclasses.replace(mm, ctilde=w)))
        with highest_precision():
            grads.append(np.asarray(jax.grad(E)(mm.ctilde)))
    assert np.abs(grads[1]).max() > 0
    np.testing.assert_allclose(grads[1], grads[0], rtol=1e-10, atol=1e-10)


# species-compact where R_nl is block-sparse in z_j (splined, more than one species)
COMPACT = {"si_fitted": False, "si_ace_model": False, "sige_nofit": True, "emb_SiGe": False,
           "Cantor_small": True}


def test_block_layout(loaded):
    """One block per l the A basis uses; the product basis is remapped onto the
    blocks and the original aa_specs are kept for the sparse layout."""
    name, m, meta, at = loaded
    m1 = block_dense(m)
    ls = sorted({int(v) for v in np.floor(np.sqrt(np.asarray(m.aspec_y)))})
    assert [b[0] for b in m1.blk] == ls
    assert m1.blk_compact == COMPACT[name]
    assert (m1.blk_rnl_coefs is not None) == COMPACT[name]
    assert len(m1.blk_aa_specs) == len(m1.aa_specs)
    for g0, g1 in zip(m1.aa_specs, m1.blk_aa_specs):
        assert g0.shape == g1.shape
    assert not m1.energy_only                              # the basis is untouched
    assert block_dense(m1) is m1
    assert lean(m).blk == m1.blk


def test_block_needs_a_folded_model():
    m, _, _ = load(str(MODELS["sige_nofit"]), fold=False)
    with pytest.raises(ValueError, match="folded"):
        block_dense(m)


# ------------------------------------------------------------------ evaluation paths
@pytest.mark.parametrize("layout", ["sparse", "dense"])
@pytest.mark.parametrize("skin", [0.0, 1.0])
def test_calculator_evaluates_the_lean_form(layout, skin):
    from ace_jax.calc.point import ACECalculator
    from ace_jax.eval import site_descriptors
    path = str(MODELS["Cantor_small"])
    m, meta, _ = load(path)
    at = _structure(meta)
    res = {}
    for use in (True, False):
        calc = ACECalculator(path, layout=layout, skin=skin, lean=use)
        assert calc.model.energy_only is False                  # the model as loaded
        assert calc.eval_model.energy_only is use
        a = at.copy()
        a.calc = calc
        res[use] = (a.get_potential_energy(), a.get_forces(), a.get_stress())
        # descriptors come from the full model, lean or not
        np.testing.assert_array_equal(
            calc.get_site_descriptors(a),
            site_descriptors(m, a.positions, a.numbers, a.cell.array, a.pbc, meta=meta))
    E0, F0, S0 = res[False]
    E1, F1, S1 = res[True]
    assert abs(E1 - E0) <= TOL * max(1.0, abs(E0))
    np.testing.assert_allclose(F1, F0, rtol=0, atol=TOL)
    np.testing.assert_allclose(S1, S0, rtol=0, atol=TOL)


def test_calculator_relean_on_new_model():
    from ace_jax.calc.point import ACECalculator
    m, meta, _ = load(str(MODELS["sige_nofit"]))
    calc = ACECalculator(m, meta)
    first = calc.eval_model
    m2 = dataclasses.replace(m, ctilde=2 * m.ctilde)
    calc.model = m2
    assert calc.eval_model is not first and calc.eval_model.energy_only
    np.testing.assert_allclose(np.asarray(calc.eval_model.ctilde), np.asarray(m2.ctilde))


def test_calculator_pace_is_as_given():
    from ace_jax.calc.point import ACECalculator
    from conftest import pace_fixture
    y = str(pace_fixture(ROOT / "fixtures" / "pace" / "gesi_sbessel.yace"))
    calc = ACECalculator(y)
    assert calc.eval_model is calc.model


# ------------------------------------------------------------------ edits need the full model
def test_radial_edits_refuse_a_lean_model():
    """A lean model keeps the radial twice (rnl_coefs for the sparse layout,
    blk_rnl_coefs for the dense one) and its pair channel is the readout, so
    editing its radials would silently desynchronise the layouts."""
    from ace_jax.fit.radial_model import to_analytic, widen_radial, with_radial
    m, meta, _ = load(str(MODELS["sige_nofit"]))
    a, _ = to_analytic(m, 6)
    for mm in (lean(m), block_dense(m)):
        with pytest.raises(ValueError, match="lean"):
            to_analytic(mm, 6)
        with pytest.raises(ValueError, match="lean"):
            mm.require_full()
    for mm in (lean(a), block_dense(a)):
        with pytest.raises(ValueError, match="lean"):
            with_radial(mm, mm.rnl_Wnlq)
        with pytest.raises(ValueError, match="lean"):
            widen_radial(mm, 8)
    m.require_full()                                       # the full model passes
    prune_columns(m).require_full()                        # pruning keeps the full basis


# ------------------------------------------------------------------ synthetic variants
def _legendre(n_q):
    from ace_jax.basis.radial_init import legendre_3term
    return tuple(jnp.asarray(x) for x in legendre_3term(n_q))


def _analytic_pair(m, n_q=6, seed=0):
    NZ, n_pair = m.Wpair.shape[1], m.Wpair.shape[0]
    W = np.random.default_rng(seed).normal(size=(NZ, NZ, n_pair, n_q)) * 0.3
    A, B, C = _legendre(n_q)
    return dataclasses.replace(m, pair_radial_kind="analytic", pair_Wnlq=jnp.asarray(W),
                               pair_polys_A=A, pair_polys_B=B, pair_polys_C=C)


def _two_l_column(m):
    """One A entry re-pointed at a radial column another entry uses at a
    different l: a column used at two l, so block_dense must fall back."""
    ar, ay = np.array(m.aspec_r), np.asarray(m.aspec_y)
    la = np.floor(np.sqrt(ay)).astype(int)
    a0 = int(np.flatnonzero(la == 0)[0])
    b1 = int(np.flatnonzero(la == 1)[0])
    ar[a0] = ar[b1]
    return dataclasses.replace(m, aspec_r=jnp.asarray(ar, m.aspec_r.dtype))


def _no_pair(m):
    NZ, c = m.Wpair.shape[1], m.pair_coefs.shape[2]
    return dataclasses.replace(m, pair_coefs=jnp.zeros((NZ, NZ, c, 0)),
                               Wpair=jnp.zeros((0, NZ)))


def _analytic_rnl(m):
    from ace_jax.fit.radial_model import to_analytic
    return to_analytic(m, 8)[0]


VARIANTS = {"analytic_pair": _analytic_pair, "analytic_rnl": _analytic_rnl,
            "two_l_column": _two_l_column, "no_pair": _no_pair}


@pytest.mark.parametrize("layout", ["sparse", "dense"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_lean_is_exact_on_synthetic_variants(variant, layout):
    m, meta, _ = load(str(MODELS["sige_nofit"]))
    m = VARIANTS[variant](m)
    at = _structure(meta)
    m1 = lean_exact(m)
    assert m1.energy_only
    if variant == "two_l_column":
        assert m1.blk == ()                                # the unblocked fallback
    if variant == "analytic_rnl":
        assert m1.blk and not m1.blk_compact
    _assert_same(_efv(m, meta, at, layout), _efv(m1, meta, at, layout))


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_lean_float32(layout):
    """float32: the lean form agrees with the full float32 model at float32
    precision (the transforms themselves are done in float64)."""
    m, meta, _ = load(str(MODELS["Cantor_small"]), dtype=jnp.float32)
    at = _structure(meta)
    (E0, F0, V0), (E1, F1, V1) = _efv(m, meta, at, layout), _efv(lean(m), meta, at, layout)
    assert abs(E1 - E0) <= 5e-6 * abs(E0) + 1e-6 * len(at)
    assert np.abs(F1 - F0).max() <= 2e-5 * np.abs(F0).max() + 1e-4
    assert np.abs(V1 - V0).max() <= 5e-5 * np.abs(V0).max() + 1e-4


def test_authored_model_through_the_calculator(tmp_path, monkeypatch):
    """A Python-authored model (analytic R_nl and pair radial) with a nonzero
    readout: ACECalculator(lean=True) equals lean=False to roundoff.  Its
    radials are analytic but not learned, so the default lean keeps them
    analytic (no splining)."""
    from test_basis_build import _primed_cache

    from ace_jax.calc.point import ACECalculator
    from ace_jax.basis.model import build_model
    from ace_jax.eval.model import fold_readout
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    auth = build_model([14], 3, 10, coupling_cache_dir=_primed_cache(tmp_path))
    m, meta = auth.eval_pair()
    assert m.pair_radial_kind == "analytic" and m.radial_kind == "analytic"
    rng = np.random.default_rng(3)
    m = fold_readout(dataclasses.replace(
        m, WB=jnp.asarray(rng.normal(size=m.WB.shape) * 0.1),
        Wpair=jnp.asarray(rng.normal(size=m.Wpair.shape) * 0.1), folded=False, ctilde=None))
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    at.rattle(0.08, seed=4)
    res = []
    for use in (False, True):
        a = at.copy()
        a.calc = ACECalculator(m, meta, lean=use, layout="dense")
        assert a.calc.eval_model.energy_only is use
        assert a.calc.eval_model.radial_kind == "analytic" and a.calc.splined is None
        res.append((a.get_potential_energy(), a.get_forces(), a.get_stress()))
    (E0, F0, S0), (E1, F1, S1) = res
    assert abs(E1 - E0) <= TOL * max(1.0, abs(E0))
    np.testing.assert_allclose(F1, F0, rtol=0, atol=TOL)
    np.testing.assert_allclose(S1, S0, rtol=0, atol=TOL)


def test_blocked_a_forms_agree():
    """The species-compact blocked A has two forms: the one-hot expansion over z_j
    (GPU, and the CPU below `BLK_SCATTER_MIN_NZ` species) and a segment_sum per
    (node, z_j) (the CPU otherwise).  Same A and same reverse-mode gradients."""
    from ace_jax.eval import model as acemod
    m, meta, _ = load(str(MODELS["Cantor_small"]))
    m = lean_exact(m)
    assert m.blk_compact and m.E0.shape[0] >= acemod.BLK_SCATTER_MIN_NZ
    n, K = 7, 5
    rng = np.random.default_rng(0)
    w = sum(b[2] for b in m.blk)
    R = jnp.asarray(rng.standard_normal((n * K, w)))
    Y = jnp.asarray(rng.standard_normal((n * K, (max(b[0] for b in m.blk) + 1) ** 2)))
    zj = jnp.asarray(rng.integers(0, m.E0.shape[0], n * K), jnp.int32)
    one, sc = m._blocked_a_onehot(R, Y, zj, n, K), m._blocked_a_scatter(R, Y, zj, n, K)
    np.testing.assert_allclose(sc, one, rtol=1e-13, atol=1e-14)
    C = jnp.asarray(rng.standard_normal(one.shape))
    g = lambda f: jax.grad(lambda R, Y: jnp.sum(C * f(R, Y, zj, n, K)), argnums=(0, 1))(R, Y)
    for a, b in zip(g(m._blocked_a_scatter), g(m._blocked_a_onehot)):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-14)
