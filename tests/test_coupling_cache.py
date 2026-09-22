"""Coupling cache mechanics (Tier 2, point 2) -- pip-only, no Julia.

The real shim is monkeypatched: these tests pin the key derivation, the
miss/hit lifecycle, and the two invalidation paths (stored-specs mismatch,
pin-hash change).  The end-to-end "a hit never launches Julia" guarantee is
pinned by the ACEJAX_NO_JULIA subprocess test in test_python_authoring.py.
"""

import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pytest

from ace_jax.construct import coupling as C


def _fake_coupling(mb, rnl, ylm):
    """Deterministic stand-in shaped like the real return."""
    n_AA = sum(len(bb) for bb in mb)
    return C.Coupling(
        A2B=np.eye(len(mb), n_AA),
        aa_sig=tuple(tuple((n, l, 0) for n, l in bb) for bb in mb),
        aspec=tuple((i % len(rnl), i % len(ylm)) for i in range(n_AA)),
        aa_specs=tuple(np.arange(n_AA).reshape(1, -1) for _ in mb),
        nnll_spec=tuple(tuple(bb) for bb in mb))


_MB = [[(1, 0), (1, 0)], [(2, 0), (1, 0)]]
_RNL = [(1, 0), (2, 0)]
_YLM = [(0, 0), (1, -1), (1, 1)]


@pytest.fixture
def shim(tmp_path, monkeypatch):
    calls = []

    def counting(mb, rnl, ylm):
        calls.append([list(bb) for bb in mb])
        return _fake_coupling(mb, rnl, ylm)

    monkeypatch.setattr(C, "couple", counting)
    return calls


def test_miss_then_hit(tmp_path, shim):
    c1 = C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert len(shim) == 1                       # miss -> shim ran
    c2 = C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert len(shim) == 1                       # hit -> shim did not run again
    assert np.array_equal(c1.A2B, c2.A2B)
    assert c1.aa_sig == c2.aa_sig and c1.nnll_spec == c2.nnll_spec
    assert c1.aspec == c2.aspec
    assert all(np.array_equal(a, b) for a, b in zip(c1.aa_specs, c2.aa_specs))


def test_reconstructed_types(tmp_path, shim):
    """A hit must return pure-python/numpy types like `couple` does -- no
    numpy scalars leaking into the tuples (they end up in JSON meta)."""
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    c2 = C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert all(type(v) is int for sig in c2.aa_sig for t in sig for v in t)
    assert all(type(v) is int for row in c2.nnll_spec for pair in row for v in pair)
    assert all(type(v) is int for r, y in c2.aspec for v in (r, y))


def test_key_order_sensitive():
    k0 = C.coupling_key(_MB, _RNL, _YLM)
    assert C.coupling_key(_MB, _RNL, _YLM) == k0            # stable
    assert C.coupling_key(_MB[::-1], _RNL, _YLM) != k0      # B-row order matters
    assert C.coupling_key(_MB, _RNL[::-1], _YLM) != k0      # Rnl order matters
    assert C.coupling_key(_MB, _RNL, _YLM[::-1]) != k0      # Ylm order matters


def test_stored_specs_mismatch_recomputes(tmp_path, shim):
    """Tampered entry (wrong key inside) -> recompute, not a wrong hit."""
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    key = C.coupling_key(_MB, _RNL, _YLM)
    p = C._entry_path(tmp_path, key)
    z = dict(np.load(p))
    meta = json.loads(bytes(z["meta_json"]).decode())
    meta["key"] = "0" * 64
    z["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez(p, **z)
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert len(shim) == 2


def test_pin_hash_change_recomputes(tmp_path, shim, monkeypatch):
    """A juliapkg.json pin change invalidates entries (recompute, then the
    new pin is stored)."""
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert len(shim) == 1
    monkeypatch.setattr(C, "juliapkg_hash", lambda: "changed-pin")
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert len(shim) == 2
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert len(shim) == 2                       # rewritten entry carries the new pin


def test_entry_loads_without_pickle(tmp_path, shim):
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    key = C.coupling_key(_MB, _RNL, _YLM)
    with np.load(C._entry_path(tmp_path, key), allow_pickle=False):
        pass                                    # must not raise


def test_unwritable_cache_dir_degrades(tmp_path, shim):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    cpl = C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(blocker))
    assert len(shim) == 1 and cpl.A2B.shape[0] == len(_MB)   # best-effort, no raise


def test_default_cache_dir_env(monkeypatch):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE", "none")
    assert C.default_cache_dir() is None
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE", str(p := "/tmp/cplcache-x"))
    assert C.default_cache_dir() == pathlib.Path(p)
    monkeypatch.delenv("ACEJAX_COUPLING_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert C.default_cache_dir() == pathlib.Path("/xdg/ace-jax/coupling")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert str(C.default_cache_dir()).startswith(os.path.expanduser("~/.cache"))


def test_disabled_cache_calls_shim_directly(shim):
    cpl = C.couple_cached(_MB, _RNL, _YLM, cache_dir="none")
    assert len(shim) == 1 and cpl.A2B.shape[0] == len(_MB)
    C.couple_cached(_MB, _RNL, _YLM, cache_dir="none")
    assert len(shim) == 2                       # nothing persisted anywhere


def test_juliapkg_hash_finds_repo_pin(monkeypatch):
    """The repo's own juliapkg.json is discoverable and hashed without
    juliacall.  (CI's pytest may not have the repo root on sys.path, so the
    test prepends cwd itself; the end-to-end pin re-check on a real entry is
    covered by the ACEJAX_NO_JULIA subprocess test.)"""
    if os.path.exists("juliapkg.json"):
        expected = hashlib.sha256(open("juliapkg.json", "rb").read()).hexdigest()
        monkeypatch.syspath_prepend(os.getcwd())
        assert C.juliapkg_hash() == expected
    h = C.juliapkg_hash()
    assert h is None or len(h) == 64


def test_corrupt_entry_recomputes(tmp_path, shim):
    """A torn entry (half-written zip) must read as a miss, not crash every
    later authoring of that shape."""
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    p = C._entry_path(tmp_path, C.coupling_key(_MB, _RNL, _YLM))
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    cpl = C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert len(shim) == 2 and cpl.A2B.shape[0] == len(_MB)


def test_schema_drift_entry_recomputes(tmp_path, shim):
    """An entry whose meta lacks a field this reader needs is a miss."""
    C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    p = C._entry_path(tmp_path, C.coupling_key(_MB, _RNL, _YLM))
    z = dict(np.load(p))
    meta = json.loads(bytes(z["meta_json"]).decode())
    del meta["aa_sig"]
    z["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez(p, **z)
    cpl = C.couple_cached(_MB, _RNL, _YLM, cache_dir=str(tmp_path))
    assert len(shim) == 2 and cpl.A2B.shape[0] == len(_MB)


def test_concurrent_writes_same_key(tmp_path, shim, monkeypatch):
    """Two writers racing on one key must both publish a valid entry.  The
    race is made deterministic: while the first writer is inside np.savez, a
    second full write of the same entry runs to completion."""
    key = C.coupling_key(_MB, _RNL, _YLM)
    p = C._entry_path(tmp_path, key)
    cpl = _fake_coupling(_MB, _RNL, _YLM)
    real_savez = np.savez
    depth = []

    def racing_savez(fh, **kw):
        if not depth:
            depth.append(1)
            C._write_entry(p, cpl, key, _MB, _RNL, _YLM)     # the other writer
        real_savez(fh, **kw)

    monkeypatch.setattr(np, "savez", racing_savez)
    C._write_entry(p, cpl, key, _MB, _RNL, _YLM)              # must not raise
    monkeypatch.setattr(np, "savez", real_savez)
    got, ok = C._read_entry(p, key)
    assert ok and np.array_equal(got.A2B, cpl.A2B)
    assert not list(tmp_path.glob("*.tmp*"))                  # no leftovers
