"""The lean evaluation form of an ACEModel (`ace_jax.eval.model.lean`) is exact.

Each load-time transform removes per-edge work the energy never reads, and must
leave E, forces and the virial unchanged to roundoff (docs/ace-vs-pace-gap.md):

  prune       R_nl columns no A entry reads, and Y_lm above the largest l used
  pairfold    the pair readout Wpair[:, z_i] folded into the pair radial table
  block       the dense A pooled per l-block (species-compact where R_nl is
              block-sparse in z_j), feature-major for the product basis

Fitting never sees these: `load` returns the full model, and a pair-folded
model refuses the basis methods (its pair channel is the readout, not Apair).
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
from ace_jax.eval.model import fold_pair, highest_precision, lean, prune_columns
from ace_jax.eval.nlist import dense_from_sparse, sparse_graph

ROOT = pathlib.Path(__file__).parent.parent
MODELS = {
    "si_fitted": ROOT / "fixtures" / "si_fitted.npz",            # 1 species, spline, spherical
    "si_ace_model": ROOT / "fixtures" / "si_ace_model.npz",      # 1 species, analytic, solid
    "sige_nofit": ROOT / "fixtures" / "sige_nofit.npz",          # 2 species, spline
    "emb_SiGe": ROOT / "fixtures" / "emb_ref_SiGe_o2d6.npz",     # 2 species, factorised radial
    "Cantor_small": ROOT / "bench" / "scaling" / "models" / "ace_Cantor_small.npz",  # 5 species
}
TOL = 1e-12


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


def _efv(model, meta, at, layout):
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    with highest_precision():
        if layout == "sparse":
            s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
            out = model.energy_forces_virial(jnp.asarray(g.rij), nz[s], nz[r], s, r,
                                             len(at), nz)
        else:
            d = dense_from_sparse(g, meta["rcut"])
            idx = jnp.asarray(d.idx)
            out = model.energy_forces_virial_dense(
                jnp.asarray(d.rij), jnp.broadcast_to(nz[:, None], idx.shape), nz[idx], idx,
                jnp.asarray(d.mask), nz)
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
    "lean": lean,
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
    m1 = lean(mm)
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
