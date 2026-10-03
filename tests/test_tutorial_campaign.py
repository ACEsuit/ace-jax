"""Curation-tutorial helpers: descriptors on a fixed reference basis, MD pools, novelty."""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR  # noqa: E402

from ace_jax.tutorials import campaign as C  # noqa: E402


def _clouds():
    rng = np.random.default_rng(0)
    train = [0.0 + 0.001 * rng.standard_normal((4, 6)), 0.01 + 0.001 * rng.standard_normal((4, 6)),
             0.02 + 0.001 * rng.standard_normal((4, 6)), np.full((1, 6), 25.0)]
    pool = [np.full((3, 6), x) for x in (0.02, 3.0, 25.05, 12.0)]
    return train, pool


def test_novelty_scores_rank_the_school_fixture():
    train, pool = _clouds()
    s = C.score_novelty(pool, train)
    assert list(np.argsort(s)[::-1][:2]) == [3, 1]
    np.testing.assert_allclose(s[3], np.sqrt(6) * 12.0, rtol=1e-2)


def test_nn_ratio_is_one_on_itself_and_grows_with_distance():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 4))
    assert abs(C.nn_ratio(X, X) - 0.0) < 1e-12              # every query atom is a training atom
    r1, r2 = C.nn_ratio(X, X + 5.0), C.nn_ratio(X, X + 10.0)
    assert r2 > r1 > 1.0


def test_physical_rejects_overlapping_atoms():
    from ase.build import bulk
    a = bulk("Si", "diamond", a=5.43, cubic=True)
    assert C.physical(a)
    b = a.copy(); b.positions[1] = b.positions[0] + [1.0, 0.0, 0.0]
    assert not C.physical(b)


def test_reference_basis_is_the_schools_120_functions():
    assert C.reference_basis().meta["len_basis"] == 120


def test_atom_descriptors_shapes():
    from ase.build import bulk
    rows = C.atom_descriptors([bulk("Si", "diamond", a=5.43, cubic=True)], C.reference_basis())
    assert len(rows) == 1 and rows[0].shape == (8, 120) and np.all(np.isfinite(rows[0]))


def test_md_pool_is_seeded_and_finite():
    from ase.build import bulk
    start = bulk("Si", "diamond", a=5.43, cubic=True)
    kw = dict(temperature=400.0, n_steps=8, every=4, seed=3)
    a = C.md_pool(str(FIXTURE_DIR / "si_fitted.npz"), [start], **kw)
    b = C.md_pool(str(FIXTURE_DIR / "si_fitted.npz"), [start], **kw)
    assert len(a) == 8 // 4                                   # observed after the dynamics, not the start
    from ace_jax.tutorials.labels import structure_key
    assert structure_key(start) not in {structure_key(f) for f in a}
    assert all(np.array_equal(x.positions, y.positions) for x, y in zip(a, b))
    assert all(np.isfinite(x.positions).all() for x in a) and not np.allclose(a[0].positions, a[-1].positions)


def test_nn_ratio_ignores_duplicate_training_atoms():
    # unrattled crystals: symmetric atoms share a descriptor, so the raw NN spacing is 0
    X = np.repeat(np.arange(5.0)[:, None] * np.ones((1, 3)), 4, axis=0)     # 5 distinct rows, 4 copies each
    r = C.nn_ratio(X, X[:1] + 10.0)
    assert np.isfinite(r) and 1.0 < r < 100.0
