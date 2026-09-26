"""fit.radial_model: analytic-radial helpers (swap / widen / convert, Gram, gauge, roughness)."""
import dataclasses

import jax
import numpy as np
import pytest

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, flat_edges, load_configs
from ace_jax.fit.inducing import GPConfig
from ace_jax.fit.rows import linear_rows

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
SI_ANALYTIC = FIXTURE_DIR / "si_ace_model.npz"
SI_SPLINE = FIXTURE_DIR / "si_fitted.npz"
SIGE_SPLINE = FIXTURE_DIR / "sige_nofit.npz"
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (XYZ, SI_ANALYTIC, SI_SPLINE, SIGE_SPLINE)), reason="missing fixtures")


def _setup(path, ncfg=6, per_batch=3):
    model, meta, z = load(path)
    configs = load_configs(XYZ, "dft_energy", "dft_force", "dft_virial")[:ncfg]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=per_batch)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=per_batch)
    return model, meta, ds, cfg


def _batch(ds, i=0):
    return jax.tree.map(lambda a: a[i], ds)


def _live_edges(ds):
    """(rij, zi, zj) over every live edge of every batch."""
    out = []
    for i in range(ds.n_batches):
        b = _batch(ds, i)
        rij, send, recv, mask = flat_edges(b.rij, b.nbr, b.nbr_mask)
        m = np.asarray(mask)
        out.append((np.asarray(rij)[m], np.asarray(b.node_z[send])[m], np.asarray(b.node_z[recv])[m]))
    return tuple(jnp.asarray(np.concatenate(k)) for k in zip(*out))


@pytest.fixture(scope="module")
def si():
    return _setup(SI_ANALYTIC)


def test_with_radial_identity_is_exact(si):
    from ace_jax.fit.radial_model import with_radial
    model, _, ds, cfg = si
    b = _batch(ds)
    r0, _, _ = linear_rows(model, cfg, b)
    r1, _, _ = linear_rows(with_radial(model, model.rnl_Wnlq), cfg, b)
    np.testing.assert_array_equal(np.asarray(r1.E), np.asarray(r0.E))
    with pytest.raises(ValueError, match="shape"):
        with_radial(model, model.rnl_Wnlq[..., :-1])


def test_poly_env_reproduces_model_radial(si):
    from ace_jax.fit.radial_model import poly_env
    model, _, ds, _ = si
    rij, zi, zj = _live_edges(ds)
    Rnl, _ = model.radial(rij, zi, zj)
    r = jnp.linalg.norm(rij, axis=-1)
    mine = jnp.einsum("eq,enq->en", poly_env(model, r, zi, zj), model.rnl_Wnlq[zi, zj])
    np.testing.assert_allclose(np.asarray(mine), np.asarray(Rnl), rtol=1e-12, atol=1e-13)


def test_widen_radial_preserves_descriptors(si):
    from ace_jax.fit.radial_model import widen_radial
    model, _, ds, cfg = si
    wide = widen_radial(model, 30)
    assert wide.rnl_Wnlq.shape[-1] == 30 and wide.polys_A.shape == (30,)
    b = _batch(ds)
    r0, _, _ = linear_rows(model, cfg, b)
    r1, _, _ = linear_rows(wide, cfg, b)
    np.testing.assert_allclose(np.asarray(r1.E), np.asarray(r0.E), rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match="n_q"):
        widen_radial(model, 5)
    with pytest.raises(ValueError, match="Legendre"):
        widen_radial(dataclasses.replace(model, polys_A=2.0 * model.polys_A), 30)


def test_non_analytic_model_raises():
    from ace_jax.fit.radial_model import with_radial
    model, *_ = _setup(SI_SPLINE)
    with pytest.raises(ValueError, match="to_analytic"):
        with_radial(model, model.rnl_Wnlq)


def test_to_analytic_matches_spline_radials(si):
    from ace_jax.fit.radial_model import to_analytic
    _, _, ds, _ = si
    spline, *_ = _setup(SI_SPLINE)
    ana, rel = to_analytic(spline, 30)
    assert ana.radial_kind == "analytic" and ana.rnl_Wnlq.shape[-1] == 30
    rij, zi, zj = _live_edges(ds)
    Rs, Ps = spline.radial(rij, zi, zj)
    Ra, Pa = ana.radial(rij, zi, zj)
    err = float(jnp.abs(Ra - Rs).max() / jnp.abs(Rs).max())
    print(f"to_analytic max rel radial error {err:.3e}, max relres {rel.max():.3e}")
    # ACE1 radials are not polynomials in the agnesi x of ace_model: measured
    # 9.2e-5 at n_q=30 (relres max 2.1e-4), converging with n_q; tolerance =
    # 10x measured, rounded. The pointwise error at these specific dataset
    # bond lengths is non-monotone in n_q (e.g. n_q=40/50 measure worse than
    # n_q=30 before n_q=60 recovers) -- it is a fixed finite set of evaluation
    # points off the fitting grid, and polynomial approximation error at such
    # points is classically non-monotonic in degree (Runge/Gibbs-type
    # behaviour). The internal L2 projection residual `rel` is the quantity
    # the fit actually minimises and is guaranteed non-increasing in n_q (each
    # n_q is a superset basis), so the convergence check below asserts on it
    # instead of on a second pointwise measurement.
    assert err < 1e-3
    _, rel60 = to_analytic(spline, 60)
    assert rel60.max() < 0.5 * rel.max()
    np.testing.assert_array_equal(np.asarray(Pa), np.asarray(Ps))   # pair basis untouched


def test_patch_radial_npz_roundtrip(tmp_path, si):
    from ace_jax.construct.export import patch_radial_npz
    from ace_jax.fit.radial_model import to_analytic
    spline, *_ = _setup(SI_SPLINE)
    ana, _ = to_analytic(spline, 30)
    patch_radial_npz(SI_SPLINE, tmp_path / "m.npz", ana)
    back, meta, _ = load(tmp_path / "m.npz")
    assert back.radial_kind == "analytic" and meta["radial_kind"] == "analytic"
    np.testing.assert_array_equal(np.asarray(back.rnl_Wnlq), np.asarray(ana.rnl_Wnlq))
    _, _, ds, _ = si
    rij, zi, zj = _live_edges(ds)
    np.testing.assert_allclose(np.asarray(back.radial(rij, zi, zj)[0]),
                               np.asarray(ana.radial(rij, zi, zj)[0]), rtol=0, atol=1e-14)
