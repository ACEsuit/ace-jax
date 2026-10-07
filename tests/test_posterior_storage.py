"""posterior.npz storage: the Cholesky factor of S is stored only when a served path reads it (the kappa
shape), and then as a packed lower triangle; sandwich posteriors (the default shape R) leave it out."""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import pytest  # noqa: E402


def _with_R(post, r=5):
    """The ard_setup posterior with a stand-in shape factor R (storage is all these tests look at)."""
    L = len(np.asarray(post.dinv))
    return post._replace(R=jax.numpy.asarray(np.random.default_rng(0).normal(size=(L, r))))


def test_sandwich_posterior_is_saved_without_chol(ard_setup, tmp_path):
    from ace_jax.fit.ard import ARDPosterior
    *_, post = ard_setup
    post = _with_R(post)
    post.save(tmp_path / "p.npz")
    z = np.load(tmp_path / "p.npz")
    assert "chol" not in z.files and "chol_packed" not in z.files
    back = ARDPosterior.load(tmp_path / "p.npz")
    assert back.chol is None
    F = np.random.default_rng(1).normal(size=(4, 3, len(np.asarray(post.dinv))))
    np.testing.assert_allclose(back.atom_shape(F), post.atom_shape(F), rtol=1e-6)    # R stored float32


def test_kappa_posterior_stores_chol_packed_and_restores_it(ard_setup, tmp_path):
    from ace_jax.fit.ard import ARDPosterior
    *_, post = ard_setup                                        # no R, no Q: the kappa shape
    post.save(tmp_path / "p.npz")
    z = np.load(tmp_path / "p.npz")
    n = len(np.asarray(post.dinv))
    assert "chol" not in z.files and z["chol_packed"].shape == (n * (n + 1) // 2,)
    back = ARDPosterior.load(tmp_path / "p.npz")
    np.testing.assert_array_equal(back.chol, np.asarray(post.chol, np.float32).astype(np.float64))


def test_a_posterior_with_full_chol_still_loads(ard_setup, tmp_path):
    """Files written before the packed form (a dense `chol`) load unchanged."""
    from ace_jax.fit.ard import ARDPosterior
    *_, post = ard_setup
    post.save(tmp_path / "p.npz")
    z = dict(np.load(tmp_path / "p.npz"))
    n = len(np.asarray(post.dinv))
    full = np.zeros((n, n), np.float32)
    full[np.tril_indices(n)] = z.pop("chol_packed")
    np.savez(tmp_path / "old.npz", chol=full, **z)
    np.testing.assert_array_equal(ARDPosterior.load(tmp_path / "old.npz").chol, full.astype(np.float64))


def test_untempered_variance_needs_chol(ard_setup, tmp_path):
    from ace_jax.fit.ard import ARDPosterior
    *_, post = ard_setup
    _with_R(post).save(tmp_path / "p.npz")
    back = ARDPosterior.load(tmp_path / "p.npz")
    with pytest.raises(ValueError, match="Cholesky"):
        back.var_rows(np.ones((2, len(np.asarray(post.dinv)))))
