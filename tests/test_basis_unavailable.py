import builtins
import sys

import pytest


def test_missing_library_message_is_backend_neutral(monkeypatch):
    from ace_jax.basis import coupling as C
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "ace_jax_coupling":
            raise ModuleNotFoundError("No module named 'ace_jax_coupling'")
        return real_import(name, *a, **k)
    monkeypatch.delenv("ACEJAX_COUPLING_CACHE_ONLY", raising=False)
    monkeypatch.delitem(sys.modules, "ace_jax_coupling", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(C.BasisUnavailable) as e:
        C._lib()
    msg = str(e.value)
    assert "not available on this platform" in msg and "--model" in msg
    assert "julia" not in msg.lower()
