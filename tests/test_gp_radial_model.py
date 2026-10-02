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
    from ace_jax.basis.export import patch_radial_npz
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


def test_radial_gram_gives_unit_empirical_norm(si):
    from ace_jax.fit.radial_model import normalise, poly_env, radial_gram, row_active
    model, _, ds, _ = si
    Q = radial_gram(model, ds, n_prior=0.0)
    W = normalise(model.rnl_Wnlq, Q, row_active(model.rnl_Wnlq))
    rij, zi, zj = _live_edges(ds)
    R = jnp.einsum("eq,enq->en", poly_env(model, jnp.linalg.norm(rij, axis=-1), zi, zj), W[zi, zj])
    np.testing.assert_allclose(np.asarray(jnp.mean(R ** 2, axis=0)), 1.0, rtol=1e-10)


def test_normalise_is_scale_invariant(si):
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    model, _, ds, _ = si
    Q = radial_gram(model, ds)
    V = model.rnl_Wnlq
    act = row_active(V)
    np.testing.assert_allclose(np.asarray(normalise(3.7 * V, Q, act)),
                               np.asarray(normalise(V, Q, act)), rtol=1e-12, atol=1e-14)


def test_normalise_zero_rows_stay_zero_and_finite(si):
    from ace_jax.fit.radial_model import normalise, radial_gram, row_active
    model, _, ds, _ = si
    Q = radial_gram(model, ds)
    V0 = model.rnl_Wnlq.at[0, 0, 3].set(0.0)            # a structurally-zero radial
    act = row_active(V0)
    assert not bool(act[0, 0, 3]) and bool(act[0, 0, 2])
    W = normalise(V0, Q, act)
    assert bool(jnp.all(W[0, 0, 3] == 0.0)) and bool(jnp.all(jnp.isfinite(W)))
    g = jax.grad(lambda V: jnp.sum(normalise(V, Q, act) ** 2))(V0)
    assert bool(jnp.all(jnp.isfinite(g))) and bool(jnp.all(g[0, 0, 3] == 0.0))


def test_radial_gram_absent_pair_is_positive_definite():
    from ace_jax.fit.radial_model import radial_gram, to_analytic
    spline, _, ds, _ = _setup(SIGE_SPLINE)              # elements Si, Ge; data is all-Si
    model, _ = to_analytic(spline, 12)
    Q = radial_gram(model, ds)
    ev = np.linalg.eigvalsh(np.asarray(Q))              # (NZ, NZ, n_q)
    assert ev.min() > 0.0


def test_roughness_matrix_properties(si):
    from ace_jax.fit.radial_model import roughness_matrix
    model, *_ = si
    D2 = np.asarray(roughness_matrix(model))
    np.testing.assert_allclose(D2, D2.T, atol=1e-10)
    assert np.abs(D2[:2]).max() < 1e-9                  # degrees 0, 1 have zero curvature
    assert np.linalg.eigvalsh(D2).min() > -1e-8 * np.abs(D2).max()
    from ace_jax.basis.radial_init import poly_eval
    # finite-difference check of one entry: int P_5'' P_7'' dx.  The double
    # np.gradient + trapezoid reference converges only at O(h) here (measured:
    # reldiff 1.29e-2 at n=2001, 1.28e-3 at n=20001, 1.28e-4 at n=200001,
    # 1.28e-5 at n=2e6 -- a clean 10x-per-10x rate, extrapolating to the exact
    # value the Gauss-Legendre+autodiff D2 already matches), so the brief's
    # n=20001 sits right at the 1e-3 tolerance boundary (measured reldiff
    # 1.28e-3, i.e. the assertion as originally written fails by ~2.8e-4).
    # Raised to n=200001 (reldiff 1.28e-4, 8x inside tolerance; ~0.03s) rather
    # than loosen the tolerance.
    x = np.linspace(-1, 1, 200_001)
    P = poly_eval(x, *(np.asarray(a) for a in (model.polys_A, model.polys_B, model.polys_C)))
    d2 = np.gradient(np.gradient(P, x, axis=0), x, axis=0)
    ref = np.trapezoid(d2[:, 5] * d2[:, 7], x)
    assert abs(D2[5, 7] - ref) < 1e-3 * abs(ref)


def test_roughness_is_weighted_quadratic_form(si):
    from ace_jax.fit.radial_model import roughness, roughness_matrix
    model, *_ = si
    W, D2 = model.rnl_Wnlq, roughness_matrix(model)
    wn = jnp.linspace(1.0, 0.1, W.shape[2])
    ref = sum(float(wn[n]) * float(W[0, 0, n] @ D2 @ W[0, 0, n]) for n in range(W.shape[2]))
    assert abs(float(roughness(W, D2, wn)) - ref) < 1e-10 * abs(ref)


def test_spectral_weights_p0_is_ones():
    from ace_jax.fit.radial_model import spectral_weights
    sw = spectral_weights(10, 0.0)
    np.testing.assert_allclose(np.asarray(sw), np.ones(10))


def test_spectral_penalty_zero_at_reference(si):
    from ace_jax.fit.radial_model import spectral_penalty, spectral_weights
    model, *_ = si
    W = model.rnl_Wnlq
    sw = spectral_weights(W.shape[-1], 4.0)
    assert float(spectral_penalty(W, W, sw)) == 0.0


def test_spectral_penalty_matches_explicit_loop(si):
    from ace_jax.fit.radial_model import spectral_penalty, spectral_weights
    model, *_ = si
    W = model.rnl_Wnlq
    W_ref = W + 0.1
    sw = spectral_weights(W.shape[-1], 3.0)
    NZ, _, n_rnl, n_q = W.shape
    ref = sum(float(sw[q]) * float((W[zi, zj, n, q] - W_ref[zi, zj, n, q]) ** 2)
             for zi in range(NZ) for zj in range(NZ) for n in range(n_rnl) for q in range(n_q))
    got = float(spectral_penalty(W, W_ref, sw))
    assert abs(got - ref) < 1e-10 * max(abs(ref), 1.0)


def test_gap_penalty_zero_at_reference(si):
    from ace_jax.fit.radial_model import gap_penalty, uniform_gram
    model, *_ = si
    U = uniform_gram(model, 2.0, 5.0, n=50)
    W = model.rnl_Wnlq
    assert float(gap_penalty(W, W, U)) == 0.0


def test_gap_penalty_matches_uniform_grid_mean(si):
    from ace_jax.fit.radial_model import gap_penalty, poly_env, uniform_gram
    model, *_ = si
    r_lo, r_hi, n = 2.0, 5.0, 400
    U = uniform_gram(model, r_lo, r_hi, n=n)
    W = model.rnl_Wnlq
    W_ref = W + 0.05 * jnp.abs(W).mean()
    got = float(gap_penalty(W, W_ref, U))
    # Independent reference on the SAME grid: for each species pair and each
    # radial n, R_n(r) = poly_env(r) @ W[zi,zj,n], so the mean squared change
    # over the grid is exactly dW^T (mean p p^T) dW = dW^T U dW for any grid
    # (not just a fine one) -- this checks uniform_gram/gap_penalty agree with
    # a direct pointwise computation, not merely converge as n grows.
    NZ, _, n_rnl, _ = W.shape
    r = jnp.linspace(r_lo, r_hi, n)
    ref = 0.0
    for zi in range(NZ):
        for zj in range(NZ):
            zi_arr = jnp.full(r.shape, zi, dtype=jnp.int32)
            zj_arr = jnp.full(r.shape, zj, dtype=jnp.int32)
            p = poly_env(model, r, zi_arr, zj_arr)                  # (n, n_q)
            dR = jnp.einsum("xq,mq->mx", p, (W - W_ref)[zi, zj])    # (n_rnl, n)
            ref += float(jnp.mean(dR ** 2, axis=-1).sum())
    assert abs(got - ref) < 1e-8 * max(abs(ref), 1.0)


def test_uniform_gram_symmetric_psd(si):
    from ace_jax.fit.radial_model import uniform_gram
    model, *_ = si
    U = np.asarray(uniform_gram(model, 2.0, 5.0, n=200))
    NZ = U.shape[0]
    for zi in range(NZ):
        for zj in range(NZ):
            m = U[zi, zj]
            np.testing.assert_allclose(m, m.T, atol=1e-10)
            ev = np.linalg.eigvalsh(m)
            assert ev.min() > -1e-8 * np.abs(m).max()


def test_rnl_degrees(si):
    from ace_jax.fit.radial_model import rnl_degrees
    _, meta, _, _ = si
    d = rnl_degrees(meta)
    assert d.shape == (meta["n_rnl"],) and d.min() == 0
    # Si fixture (NZ = 1, order 3, totaldegree 10): the spec starts
    # (1,0) .. (10,0), (1,1), (2,1), so (n - 1) // NZ = n - 1 there
    spec_n = list(range(1, 11)) + [1, 2]
    np.testing.assert_array_equal(d[:12], [(n - 1) // 1 for n in spec_n])
    np.testing.assert_array_equal(d[:12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    with pytest.raises(ValueError, match="n_rnl"):
        rnl_degrees({**meta, "n_rnl": meta["n_rnl"] + 1})
