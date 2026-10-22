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
