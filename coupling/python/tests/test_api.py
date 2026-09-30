"""ace_jax_coupling against unpatched upstream EquivariantTensors (BASE_SHA)."""
import numpy as np
import pytest

import ace_jax_coupling as ajc

FIELDS = ["A2B_rows", "A2B_cols", "A2B_vals", "aa_sig_off", "aa_sig", "aspec",
          "aa_off", "aa_idx", "nnll_off", "nnll"]


def test_bit_exact_vs_upstream_et(cases, reference):
    for name, c in cases.items():
        r = ajc.couple_raw(c["mb"], c["R"], c["Y"])
        assert r.A2B_shape == tuple(int(v) for v in reference[f"{name}__A2B_shape"]), name
        for f in FIELDS:
            got, want = getattr(r, f), reference[f"{name}__{f}"]
            assert got.dtype == want.dtype and got.shape == want.shape, (name, f, got.shape, want.shape)
            assert np.array_equal(got, want), (name, f)


def test_largest_case_is_fast():
    """Guards against a mis-pruned bundle: duplicate library copies once made
    the 6216x17302 case take ~1000 s on Linux and hang on macOS (0.15 s when
    correct).  Run in a subprocess with a timeout so a regression fails, not hangs."""
    import pathlib
    import subprocess
    import sys
    data = pathlib.Path(__file__).parent / "data" / "cases.json"
    code = ("import json, sys, time, ace_jax_coupling as a; "
            f"c = json.load(open({str(data)!r}))['build_NZ3_o4_d10']; "
            "t = time.perf_counter(); a.couple_raw(c['mb'], c['R'], c['Y']); "
            "print(time.perf_counter() - t)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    assert float(r.stdout.split()[-1]) < 10.0


def test_accepts_tuples_lists_and_numpy_ints(cases):
    c = cases["tiny"]
    a = ajc.couple_raw(c["mb"], c["R"], c["Y"])
    mb = [tuple((np.int32(n), np.int64(l)) for n, l in bb) for bb in c["mb"]]
    b = ajc.couple_raw(mb, np.array(c["R"]), [tuple(y) for y in c["Y"]])
    assert all(np.array_equal(getattr(a, f), getattr(b, f)) for f in FIELDS)


def test_deterministic(cases):
    c = cases["SiGe_o4d5w05"]
    a, b = ajc.couple_raw(c["mb"], c["R"], c["Y"]), ajc.couple_raw(c["mb"], c["R"], c["Y"])
    assert all(np.array_equal(getattr(a, f), getattr(b, f)) for f in FIELDS)


@pytest.mark.parametrize("mb,R,Y,msg", [
    ([], [(1, 0)], [(0, 0)], "empty"),
    ([[]], [(1, 0)], [(0, 0)], "empty body"),
    ([[(2, 0)]], [(1, 0)], [(0, 0)], "not in Rnl_spec"),
    ([[(1, 1), (1, 1)]], [(1, 1)], [(1, 0)], "not in Ylm_spec"),
    ([[(1, 0)] * 9], [(1, 0)], [(0, 0)], "order"),
    ([[(1, -1)]], [(1, -1)], [(0, 0)], "negative"),
    ([[(1, 0)]], [(1, 0)], [(1, 2)], r"\|m\|"),
    ([[(1, 0)]], [(1, 0), (1, 0)], [(0, 0)], "duplicate"),
    ([[(1.5, 0)]], [(1, 0)], [(0, 0)], "integer"),
])
def test_invalid_inputs_raise(mb, R, Y, msg):
    with pytest.raises(ValueError, match=msg):
        ajc.couple_raw(mb, R, Y)


def test_raw_invalid_returns_code():
    """A malformed direct ctypes call is rejected by the library (code 2), not a crash."""
    from ace_jax_coupling import _loader
    lib = _loader.lib()
    z = np.zeros(8, np.int64)
    off = np.array([0, 2, 1], np.int64)                      # non-monotone offsets (in bounds)
    p = lambda a: a.ctypes.data_as(_loader.c_void_p)
    code = lib.etc_couple(2, p(off), p(z), p(z), 1, p(z), p(z), 1, p(z), p(z), p(z),
                          *([None] * 10))
    assert code == 2


def test_build_info_does_not_load_library(monkeypatch):
    from ace_jax_coupling import _loader
    monkeypatch.setattr(_loader, "_LIB", None)
    info = ajc.build_info()
    assert info["abi"] == ajc.ABI_VERSION and info["max_order"] == ajc.MAX_ORDER
    assert len(info["et_rev"]) == 40
    assert _loader._LIB is None
