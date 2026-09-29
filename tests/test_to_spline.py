"""Analytic (learned) radials to the spline branch: `to_spline`, and `lean` on
analytic models (docs/learned-radial-splining.md).

`to_spline` is an approximation, not a restructuring: it agrees with the
analytic model to its tolerance, not to roundoff.  So the parity bounds here
scale with `tol`, where tests/test_lean.py holds the exact transforms to 1e-12.
"""
import dataclasses
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.eval.edge_model import EdgeSiteModel
from ace_jax.eval.model import _rnl_owner, highest_precision, lean, lean_keep_basis, prune_columns
from ace_jax.eval.nlist import dense_from_sparse, sparse_graph
from ace_jax.eval.splinify import to_spline
from ace_jax.fit.radial_model import to_analytic
from test_lean import _analytic_pair, _efv, _structure

ROOT = pathlib.Path(__file__).parent.parent
SIGE = ROOT / "fixtures" / "sige_nofit.npz"              # 2 species, spline
CANTOR = ROOT / "fixtures" / "ace_cantor5_small.npz"     # 5 species, spline
MULTI = {"sige_nofit": SIGE, "Cantor_small": CANTOR}
N_Q = 12


def _perturbed(m, n_q=N_Q, rel=0.2, seed=0, fill=False):
    """to_analytic(m, n_q) with Wnlq perturbed on its active rows (the zero
    pattern kept, as learning keeps it); fill=True also fills the zero rows."""
    a, _ = to_analytic(m, n_q)
    W = np.asarray(a.rnl_Wnlq)
    act = np.abs(W).max(-1) > 0
    rms = np.sqrt((W ** 2).mean(-1, keepdims=True))
    noise = np.random.default_rng(seed).normal(size=W.shape)
    W2 = W + rel * rms * noise * act[..., None]
    if fill:
        W2 = np.where(act[..., None], W2, rel * rms[act].mean() * noise)
    return dataclasses.replace(a, rnl_Wnlq=jnp.asarray(W2))


def _close(ref, got, tol, scale=100.0):
    """E, F, V agree to `scale * tol` relative to the reference's own size."""
    (E0, F0, V0), (E1, F1, V1) = ref, got
    dE, dF, dV = abs(E1 - E0), np.abs(F1 - F0).max(), np.abs(V1 - V0).max()
    print(f"\n  |dE|/|E| {dE / abs(E0):.2e}  |dF|/|F| {dF / np.abs(F0).max():.2e}"
          f"  |dV|/|V| {dV / np.abs(V0).max():.2e}")
    assert dE <= scale * tol * abs(E0)
    assert dF <= scale * tol * np.abs(F0).max()
    assert dV <= scale * tol * np.abs(V0).max()


@pytest.fixture(scope="module", params=list(MULTI))
def multi(request):
    m, meta, _ = load(str(MULTI[request.param]))
    return request.param, m, meta, _structure(meta)


# ------------------------------------------------------------------ the conversion
@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_roundtrip_recovers_the_analytic_model(multi, layout):
    """to_spline(to_analytic(m)) evaluates as to_analytic(m) to within tol.
    (Not as m itself: to_analytic is a projection with its own residual.)"""
    name, m, meta, at = multi
    a, _ = to_analytic(m, N_Q)
    s, err = to_spline(a, tol=1e-9)
    assert s.radial_kind == "spline" and err <= 1e-9
    _close(_efv(a, meta, at, layout), _efv(s, meta, at, layout), 1e-9)


def test_grid_layout(multi):
    name, m, meta, at = multi
    a = _perturbed(m)
    s, _ = to_spline(a, n_intervals=64)
    assert s.rnl_grid == (-1.0, 2.0 / 64, 65)
    assert s.rnl_coefs.shape == a.rnl_Wnlq.shape[:2] + (67, a.rnl_Wnlq.shape[2])
    assert s.rnl_coefs.dtype == a.rnl_Wnlq.dtype
    assert s.edge_a_widths() == a.edge_a_widths()


def test_perturbed_keeps_the_species_pattern(multi):
    """A learned-like radial (zero rows kept) splines to a table _rnl_owner
    reads as block-sparse in z_j, as ACE1's own table; a filled one does not."""
    name, m, meta, at = multi
    s, _ = to_spline(_perturbed(m))
    own = _rnl_owner(s)
    assert own is not None
    np.testing.assert_array_equal(own, _rnl_owner(m))
    assert _rnl_owner(to_spline(_perturbed(m, fill=True))[0]) is None


def test_error_control():
    """Tighter tol, more intervals; the reported error is within tol and is the
    error actually measured on that grid."""
    m, meta, _ = load(str(SIGE))
    a = _perturbed(m)
    grids = {}
    for tol in (1e-5, 1e-8, 1e-11):
        s, err = to_spline(a, tol=tol)
        assert err <= tol
        grids[tol] = s.rnl_grid[2] - 1
        _, err_fixed = to_spline(a, n_intervals=grids[tol])
        assert err_fixed == err
    assert grids[1e-5] < grids[1e-8] < grids[1e-11]
    _, coarse = to_spline(a, n_intervals=8)
    assert coarse > 1e-5                                     # the error is measured, not assumed


def test_error_is_measured_against_the_model():
    """The reported error bounds R_nl as the model evaluates it: a dense check
    of ACEModel.radial on r, independent of to_spline's own check grid."""
    m, meta, _ = load(str(SIGE))
    a = _perturbed(m)
    s, err = to_spline(a, tol=1e-7)
    NZ = a.rnl_Wnlq.shape[0]
    r = jnp.linspace(0.5, float(meta["rcut"]), 5003)
    rij = jnp.stack([r, 0 * r, 0 * r], -1)
    for i in range(NZ):
        for j in range(NZ):
            zi, zj = jnp.full(r.shape, i), jnp.full(r.shape, j)
            R0, P0 = a.radial(rij, zi, zj)
            R1, P1 = s.radial(rij, zi, zj)
            R0, R1 = np.asarray(R0), np.asarray(R1)
            live = np.abs(R0).max(0) > 0
            rel = np.abs(R1 - R0).max(0)[live] / np.abs(R0).max(0)[live]
            assert rel.max() <= 1.01 * err
            np.testing.assert_array_equal(np.asarray(P1), np.asarray(P0))   # pair: spline, untouched


def test_pair_radial():
    """An analytic pair radial converts too; E/F/V within tol."""
    m, meta, _ = load(str(SIGE))
    p = _analytic_pair(m)
    s, err = to_spline(p, tol=1e-9)
    assert s.pair_radial_kind == "spline" and s.radial_kind == "spline" and err <= 1e-9
    at = _structure(meta)
    for layout in ("sparse", "dense"):
        _close(_efv(p, meta, at, layout), _efv(s, meta, at, layout), 1e-9)
        _close(_efv(p, meta, at, layout), _efv(lean(p, spline_tol=1e-9), meta, at, layout), 1e-9)


def test_nothing_analytic_is_a_no_op(multi):
    name, m, meta, at = multi
    s, err = to_spline(m)
    assert s is m and err == 0.0


def test_spline_factorised_rnl_is_kept():
    m, meta, _ = load(str(ROOT / "fixtures" / "emb_ref_SiGe_o2d6.npz"))
    assert m.radial_kind == "spline_factorised"
    s, err = to_spline(m)
    assert s is m and err == 0.0
    p = _analytic_pair(m)
    s, err = to_spline(p)
    assert s.radial_kind == "spline_factorised" and s.pair_radial_kind == "spline"


def test_result_is_a_full_trainable_spline_model():
    """The converted model is a normal full model: require_full passes, the
    basis methods work, and the radial table is a live leaf."""
    m, meta, _ = load(str(SIGE))
    s, _ = to_spline(_perturbed(m))
    s.require_full()
    assert not s.energy_only and not s.blk
    at = _structure(meta)
    assert all(np.isfinite(x).all() for x in _basis(s, meta, at)[0])
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    sd, rc = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    E = lambda c: jnp.sum(eqx.tree_at(lambda mm: mm.rnl_coefs, s, c).site_energies(   # noqa: E731
        jnp.asarray(g.rij), nz[sd], nz[rc], sd, len(at), nz))
    with highest_precision():
        assert np.abs(np.asarray(jax.grad(E)(s.rnl_coefs))).max() > 0


def test_real_learned_radial_keeps_the_pattern():
    """A radial actually learned by `learn_radial` (VarPro, a few L-BFGS steps on
    the si_tiny data) keeps ACE1's per-z_j column pattern: learning freezes the
    zero rows, so its splined table is species-compact under `lean`."""
    from test_gp_learn_radial import THETA, XYZ

    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem
    from ace_jax.fit.radial_learn import learn_radial
    from ace_jax.fit.radial_model import with_radial
    spline, meta, z = load(str(SIGE))
    model, _ = to_analytic(spline, N_Q)
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:6]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   jnp.ones(cfg.len_basis), default_prior(2.35))
    W, _ = learn_radial(prob, ds, model.rnl_Wnlq, theta0=THETA, steps=5, reprofile_every=5,
                        map_steps=20)
    assert np.abs(np.asarray(W) - np.asarray(model.rnl_Wnlq)).max() > 1e-3   # it learned
    learned = with_radial(model, W)
    s, err = to_spline(learned)
    assert err <= 1e-8
    np.testing.assert_array_equal(_rnl_owner(s), _rnl_owner(spline))
    ml = lean(learned)
    assert ml.blk_compact
    at = _structure(meta)
    for layout in ("sparse", "dense"):
        _close(_efv(learned, meta, at, layout), _efv(ml, meta, at, layout), 1e-8)


# ------------------------------------------------------------------ lean on analytic models
@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_lean_of_a_learned_radial_is_compact_and_within_tol(multi, layout):
    name, m, meta, at = multi
    a = _perturbed(m)
    tol = 1e-8
    ml = lean(a, spline_tol=tol)
    assert ml.energy_only and ml.blk and ml.blk_compact     # compaction applies again
    _close(_efv(a, meta, at, layout), _efv(ml, meta, at, layout), tol)
    exact = lean(a, spline_tol=None)                         # analytic kept: exact, not compact
    assert exact.blk and not exact.blk_compact
    (E0, F0, V0), (E1, F1, V1) = _efv(a, meta, at, layout), _efv(exact, meta, at, layout)
    assert abs(E1 - E0) <= 1e-12 * max(1.0, abs(E0)) and np.abs(F1 - F0).max() < 1e-12
    assert np.abs(V1 - V0).max() < 1e-12


def test_lean_of_a_filled_radial_is_not_compact():
    m, meta, _ = load(str(CANTOR))
    ml = lean(_perturbed(m, fill=True))
    assert ml.radial_kind == "spline" and ml.blk and not ml.blk_compact


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_lean_float32(layout):
    """float32: the conversion is done in float64 and cast; lean agrees with
    the full float32 analytic model at float32 precision."""
    m, meta, _ = load(str(CANTOR))
    a = _perturbed(m)
    a32 = jax.tree.map(lambda v: v.astype(jnp.float32)
                       if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating) else v, a)
    ml = lean(a32)
    assert ml.rnl_coefs.dtype == jnp.float32 and ml.blk_compact
    at = _structure(meta)
    (E0, F0, V0), (E1, F1, V1) = _efv(a32, meta, at, layout), _efv(ml, meta, at, layout)
    assert abs(E1 - E0) <= 5e-6 * abs(E0) + 1e-6 * len(at)
    assert np.abs(F1 - F0).max() <= 2e-5 * np.abs(F0).max() + 1e-4
    assert np.abs(V1 - V0).max() <= 5e-5 * np.abs(V0).max() + 1e-4


def test_require_full_guards():
    m, meta, _ = load(str(SIGE))
    a = _perturbed(m)
    for mm in (lean(a), lean(a, spline_tol=None)):
        with pytest.raises(ValueError, match="lean"):
            to_spline(mm)
        with pytest.raises(ValueError, match="lean"):
            lean_keep_basis(mm)
    prune_columns(a).require_full()
    to_spline(prune_columns(a))                              # pruning keeps the full model


def test_lean_is_idempotent_on_analytic():
    m, meta, _ = load(str(SIGE))
    ml = lean(_perturbed(m))
    assert lean(ml) is ml


# ------------------------------------------------------------------ wrapper models
class Wrapper(EdgeSiteModel):
    """A minimal stand-in for FSModel(base, ...): energies through the base's
    basis and its unfolded readout, plus a nonlinear term of B so the test
    depends on B itself, not just the linear energy.  Forces come from
    EdgeSiteModel.energy_forces_virial*, a value_and_grad of site_energies."""
    base: object
    eta: float = 0.01

    def with_base(self, new_base):
        return dataclasses.replace(self, base=new_base)

    def _e(self, B, Apair, node_z):
        return self.base._readout(B, Apair, node_z) - self.eta * jnp.sqrt(1.0 + jnp.sum(B * B, -1))

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        return self._e(*self.base.site_basis(rij, zi, zj, segment_ids, n_nodes, mask), node_z)

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        return self._e(*self.base.site_basis_dense(rij, zi, zj, mask), node_z)

    def pad_cutoff(self):
        return self.base.pad_cutoff()

    def __getattr__(self, k):                 # the rest delegates, as FSModel does
        if k == "base":
            raise AttributeError(k)
        return getattr(self.base, k)


def _basis(model, meta, at):
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    d = dense_from_sparse(g, meta["rcut"])
    idx = jnp.asarray(d.idx)
    with highest_precision():
        sp = model.site_basis(jnp.asarray(g.rij), nz[s], nz[r], s, len(at))
        de = model.site_basis_dense(jnp.asarray(d.rij), jnp.broadcast_to(nz[:, None], idx.shape),
                                    nz[idx], jnp.asarray(d.mask))
    return [np.asarray(x) for x in (*sp, *de)], nz


@pytest.mark.parametrize("kind", ["spline", "analytic"])
def test_lean_keep_basis_keeps_the_basis(multi, kind):
    """prune_columns leaves B and Apair unchanged (sparse and dense) and the
    unfolded readout working; to_spline changes them only to within tol.

    Pruning is exact in arithmetic but not always bitwise in B: the spline
    contraction is compiled for the pruned column count, and XLA may then
    vectorise its 4-term sum differently (measured: bitwise on Cantor_small,
    <= 6e-16 absolute on sige_nofit).  Apair is bitwise (no column moves)."""
    name, m, meta, at = multi
    m = m if kind == "spline" else _perturbed(m)
    k = lean_keep_basis(m)
    assert not k.energy_only and not k.blk and k.radial_kind == "spline"
    b0, nz = _basis(m, meta, at)
    b1, _ = _basis(k, meta, at)
    ref = _basis(to_spline(m)[0], meta, at)[0]                # the splined, unpruned basis
    for i, (x0, x1, xr) in enumerate(zip(b0, b1, ref)):
        if i % 2:                                              # Apair
            np.testing.assert_array_equal(x1, xr)
        else:                                                  # B: prune to roundoff
            np.testing.assert_allclose(x1, xr, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(x1, x0, rtol=0, atol=1e-7 * max(1.0, np.abs(x0).max()))
    e0 = np.asarray(m._readout(jnp.asarray(b0[0]), jnp.asarray(b0[1]), nz))
    e1 = np.asarray(k._readout(jnp.asarray(b1[0]), jnp.asarray(b1[1]), nz))
    np.testing.assert_allclose(e1, e0, rtol=1e-6)


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_wrapper_lean_hook(multi, layout):
    """lean(wrapper) = wrapper.with_base(lean_keep_basis(base)): the base is
    splined and pruned, never pair-folded or blocked, and E/F/V match the
    unwrapped-full wrapper within tol; with a spline base, to roundoff."""
    name, m, meta, at = multi
    for base, tol in ((m, 1e-12), (_perturbed(m), 1e-8)):
        w = Wrapper(base)
        wl = lean(w, spline_tol=tol if tol > 1e-12 else 1e-8)
        assert isinstance(wl, Wrapper)
        assert wl.base.radial_kind == "spline" and not wl.base.energy_only and not wl.base.blk
        assert wl.base.edge_a_widths()[0] == len(np.unique(np.asarray(base.aspec_r)))
        _close(_efv(w, meta, at, layout), _efv(wl, meta, at, layout), tol)


def test_wrapper_through_the_calculator():
    from ace_jax.calc.point import ACECalculator
    m, meta, _ = load(str(CANTOR))
    w = Wrapper(_perturbed(m))
    at = _structure(meta)
    res = {}
    for use in (False, True):
        calc = ACECalculator(w, meta, lean=use, layout="dense")
        assert isinstance(calc.eval_model, Wrapper)
        assert calc.eval_model.base.radial_kind == ("spline" if use else "analytic")
        a = at.copy()
        a.calc = calc
        res[use] = (a.get_potential_energy(), a.get_forces(), a.get_stress())
    _close(res[False], res[True], 1e-8)


# ------------------------------------------------------------------ LAMMPS export
@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_lammps_bundle_of_a_learned_radial(tmp_path, monkeypatch, layout):
    """export_lammps(analytic model) traces lean(model): splined, species-compact,
    and within tol of the full analytic model's energy function."""
    from conftest import require_optional
    from test_export_lammps import _cluster, _lammps_graph
    require_optional("lammps_jax")
    from ace_jax.export import lammps as lx
    m, meta, _ = load(str(SIGE))
    a = _perturbed(m)
    at = _cluster()
    at.numbers = np.where(at.numbers == 32, 32, 14)
    graph, g = _lammps_graph(at, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    species = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    pos = jnp.asarray(at.positions)
    seen = []
    real = lx.make_energy_fn

    def spy(mm, *args, **kw):
        f = real(mm, *args, **kw)
        seen.append((mm, f))
        return f

    monkeypatch.setattr(lx, "make_energy_fn", spy)
    b = lx.export_lammps(a, meta, tmp_path / "m.json", max_atoms=256, max_edges=256 * 64,
                         k_dense=64, layout=layout)
    assert b["ace_jax"]["lean"] is True
    ml, f = seen[-1]
    assert ml.radial_kind == "spline" and (layout == "sparse" or ml.blk_compact)
    full = real(a, len(meta["elements"]), layout, k_dense=64)
    vg = lambda fn: jax.jit(jax.value_and_grad(lambda p: jnp.sum(fn(p, species, graph))))(pos)  # noqa: E731
    (E0, G0), (E1, G1) = vg(full), vg(f)
    assert abs(float(E1) - float(E0)) <= 1e-6 * abs(float(E0))
    assert np.abs(np.asarray(G1 - G0)).max() <= 1e-6 * np.abs(np.asarray(G0)).max()
