import jax
jax.config.update("jax_enable_x64", True)
import pytest  # noqa: E402


def test_require_names_the_extra_when_cueq_is_missing(monkeypatch):
    import builtins
    from ace_jax.eval import cueq
    real = builtins.__import__
    def fake(name, *a, **k):
        if name.startswith("cuequivariance"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert not cueq.available()
    with pytest.raises(cueq.CueqUnavailable, match=r"ace-jax\[cueq\]"):
        cueq.require()


def _si():
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.eval.model import lean
    m, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    return lean(m)


def test_refuses_multi_species_categorical_unfolded_and_pace():
    import dataclasses
    from conftest import FIXTURE_DIR, pace_fixture
    from ace_jax import load
    from ace_jax.eval import cueq
    from ace_jax.eval.model import lean
    m2, _, _ = load(str(FIXTURE_DIR / "sige_nofit.npz"))             # two species, categorical
    m2 = lean(m2)
    assert m2.ctilde.shape[1] == 2
    with pytest.raises(ValueError, match="one species"):
        cueq.check_supported(m2)
    with pytest.raises(ValueError, match="folded"):
        cueq.check_supported(dataclasses.replace(_si(), folded=False))
    with pytest.raises(ValueError, match="linear ACEModel"):
        cueq.check_supported(load(str(pace_fixture(FIXTURE_DIR / "pace" / "si_chebexpcos.yace")))[0])
    cueq.check_supported(_si())                                       # the supported case passes


def test_n_A_of():
    import numpy as np
    from ace_jax.eval import cueq
    assert cueq.n_A_of([np.zeros((0, 1), int), np.array([[0, 3], [2, 1]])]) == 4


def test_polynomial_chunks_cover_every_monomial_once():
    pytest.importorskip("cuequivariance")
    import numpy as np
    from ace_jax.eval import cueq
    m = _si()
    specs, coef = m.aa_specs, np.asarray(m.ctilde[:, 0])
    plans = cueq.polynomial(specs, coef, chunk=7)
    cols = np.concatenate([c for _, c in plans])
    assert sorted(cols.tolist()) == list(range(sum(np.shape(s)[0] for s in specs if np.size(s))))
    assert all(len(c) <= 7 for _, c in plans)
