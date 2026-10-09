"""ACEModel(rnl_basis="sbessel"): the analytic tensor radial on PACE's simplified spherical Bessel
basis g_k(r) (pacemaker's radials), learnable like the polynomial basis."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from test_basis_build import _primed_cache  # noqa: E402

import dataclasses  # noqa: E402

from ace_jax.eval.model import fold_readout  # noqa: E402


def _build(tmp_path, monkeypatch, rnl_basis="sbessel"):
    """Si order 3, degree 10 on the primed coupling cache, with a seeded nonzero readout."""
    from ace_jax.basis.model import build_model
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    b = build_model([14], 3, 10, coupling_cache_dir=_primed_cache(tmp_path), rnl_basis=rnl_basis)
    rng = np.random.default_rng(0)
    m = eqx.tree_at(lambda m: (m.WB, m.Wpair), b.model,
                    (jnp.asarray(0.05 * rng.standard_normal(b.model.WB.shape)),
                     jnp.asarray(0.05 * rng.standard_normal(b.model.Wpair.shape))))
    m = fold_readout(dataclasses.replace(m, folded=False))     # a stale ctilde would hide the many-body term
    return b._replace(model=m)


def _atoms(seed=0):
    from ase.build import bulk
    a = bulk("Si", "diamond", a=5.43, cubic=True)
    a.rattle(0.1, seed=seed)
    return a


def _efv(model, atoms, lean=False, meta=None):
    from ace_jax.calc.point import ACECalculator
    calc = ACECalculator(model, meta=meta, lean=lean, skin=0.0)
    a = atoms.copy(); a.calc = calc
    return a.get_potential_energy(), a.get_forces(), a.get_stress(voigt=False)


def test_onehot_start_is_the_sbessel_basis_and_zero_beyond_the_cutoff(tmp_path, monkeypatch):
    """With the onehot start R_n = g_n: the model's radials equal pace_radial._sbessel's columns
    on [0, rc), and vanish from rc on (the polynomial basis' envelope does not apply)."""
    from ace_jax.eval.pace_radial import _sbessel
    m = _build(tmp_path, monkeypatch).model
    rc = float(m.pair_envelope[0, 0, 0])
    r = jnp.linspace(0.5, rc + 1.0, 400)
    z = jnp.zeros(r.shape, jnp.int32)
    rij = jnp.stack([r, 0 * r, 0 * r], -1)
    Rnl, _ = m.radial(rij, z, z)
    n_q = m.rnl_Wnlq.shape[-1]
    g = np.asarray(_sbessel(r, rc, n_q))
    inside = np.asarray(r < rc)
    W = np.asarray(m.rnl_Wnlq[0, 0])
    np.testing.assert_allclose(np.asarray(Rnl)[inside], (g @ W.T)[inside], rtol=1e-12, atol=1e-14)
    assert np.all(np.asarray(Rnl)[~inside] == 0.0)
    assert np.all((W != 0).sum(1) == 1) and np.all(W[W != 0] == 1.0)   # onehot: each radial reads one g_k


def test_forces_are_the_energy_gradient(tmp_path, monkeypatch):
    b = _build(tmp_path, monkeypatch); m, meta = b.model, b.meta
    a = _atoms()
    E, F, _ = _efv(m, a, meta=meta)
    h, i = 1e-5, 3
    for k in range(3):
        ap, am = a.copy(), a.copy()
        ap.positions[i, k] += h; am.positions[i, k] -= h
        fd = -(_efv(m, ap, meta=meta)[0] - _efv(m, am, meta=meta)[0]) / (2 * h)
        assert abs(fd - F[i, k]) < 1e-6 * max(1.0, abs(F[i, k]))


def test_npz_roundtrip_keeps_the_basis(tmp_path, monkeypatch):
    from ace_jax.basis.export import save_npz
    from ace_jax.eval import load
    b = _build(tmp_path, monkeypatch)
    p = tmp_path / "m.npz"
    save_npz(p, b)
    m2, meta, _ = load(p)
    assert m2.rnl_basis == "sbessel"
    a = _atoms(1)
    E1, F1, V1 = _efv(b.model, a, meta=b.meta)
    E2, F2, V2 = _efv(m2, a, meta=meta)
    np.testing.assert_allclose(E2, E1, rtol=1e-12)
    np.testing.assert_allclose(F2, F1, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("learned", [False, True])
def test_lean_matches_the_full_model(tmp_path, monkeypatch, learned):
    """lean keeps an sbessel radial analytic (to_spline tabulates polynomials in x only), learned
    or not, so lean and full agree to roundoff."""
    from ace_jax.fit.radial_model import with_radial
    b = _build(tmp_path, monkeypatch); m, meta = b.model, b.meta
    if learned:
        rng = np.random.default_rng(2)
        m = with_radial(m, m.rnl_Wnlq + 0.1 * jnp.asarray(rng.standard_normal(m.rnl_Wnlq.shape)))
    a = _atoms(2)
    E, F, V = _efv(m, a, meta=meta)
    El, Fl, Vl = _efv(m, a, lean=True, meta=meta)
    np.testing.assert_allclose(El, E, rtol=1e-12)
    np.testing.assert_allclose(Fl, F, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(Vl, V, rtol=1e-10, atol=1e-11)


def test_learner_helpers_take_the_sbessel_basis(tmp_path, monkeypatch):
    """The radial Gram is positive definite, the roughness matrix symmetric and PSD, and widening
    adds g_k columns without changing any radial."""
    from ace_jax.fit.radial_model import _uniform_moment, roughness_matrix, widen_radial
    m = _build(tmp_path, monkeypatch).model
    U = np.asarray(_uniform_moment(m, 401))[0, 0]
    assert np.linalg.eigvalsh(U).min() > 0
    D2 = np.asarray(roughness_matrix(m))
    assert np.allclose(D2, D2.T) and np.linalg.eigvalsh(D2).min() > -1e-8 * np.abs(D2).max()
    n_q = m.rnl_Wnlq.shape[-1]
    w = widen_radial(m, n_q + 4)
    r = jnp.linspace(0.6, 4.9, 50)
    z = jnp.zeros(r.shape, jnp.int32)
    rij = jnp.stack([r, 0 * r, 0 * r], -1)
    np.testing.assert_allclose(np.asarray(w.radial(rij, z, z)[0]), np.asarray(m.radial(rij, z, z)[0]),
                               rtol=1e-12, atol=1e-14)
    with pytest.raises(NotImplementedError, match="sbessel"):
        from ace_jax.eval.splinify import to_spline
        to_spline(m, 1e-8)


def test_the_many_body_term_sees_the_radial_basis(tmp_path, monkeypatch):
    """Guard against a stale readout: the same readout gives different energies on the two bases."""
    a = _atoms(3)
    bp, bs = _build(tmp_path, monkeypatch, "poly"), _build(tmp_path, monkeypatch, "sbessel")
    assert abs(_efv(bp.model, a, meta=bp.meta)[0] - _efv(bs.model, a, meta=bs.meta)[0]) > 1e-3


def test_poly_basis_is_unchanged(tmp_path, monkeypatch):
    """The default basis is the polynomial one, and the built model records it."""
    b = _build(tmp_path, monkeypatch, rnl_basis="poly")
    assert b.model.rnl_basis == "poly" and b.meta["basis"]["rnl_basis"] == "poly"
    with pytest.raises(ValueError, match="rnl_basis"):
        from ace_jax.basis.model import build_model
        build_model([14], 3, 10, rnl_basis="bessel")


def test_radial_basis_is_a_basis_flag_not_a_learner_option():
    """--radial-basis (like --radial-mode) defines the basis: it needs no --learn-radial."""
    from ace_jax.cli import _parse
    a = _parse(["fit", "--train", "x.xyz", "--order", "2", "--max-degree", "6", "--radial-basis", "sbessel",
                "--radial-mode", "onehot", "--out", "o"])
    assert a.radial_basis == "sbessel" and not a.learn_radial
    with pytest.raises(SystemExit):
        _parse(["fit", "--train", "x.xyz", "--order", "2", "--max-degree", "6", "--radial-steps", "5", "--out", "o"])
