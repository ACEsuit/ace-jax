"""ace_jax_coupling against its untrimmed EquivariantTensors oracle (upstream
main + the nullspace_solver commit, :dense; coupling/julia/reference), and
against unpatched upstream (default :sparse, UMFPACK) up to the documented
freedom: per-row sign, rotation inside a degenerate nnll block."""
import pathlib

import numpy as np
import pytest

import ace_jax_coupling as ajc

FIELDS = ["A2B_rows", "A2B_cols", "A2B_vals", "aa_sig_off", "aa_sig", "aspec",
          "aa_off", "aa_idx", "nnll_off", "nnll"]


def test_matches_upstream_et(cases, reference):
    """Every index array bit-identical to the untrimmed oracle (ET with
    nullspace_solver = :dense; reference made on Linux x86_64); A2B values within
    4 ulp, which leaves room for BLAS rounding across platforms."""
    for name, c in cases.items():
        r = ajc.couple_raw(c["mb"], c["R"], c["Y"])
        assert r.A2B_shape == tuple(int(v) for v in reference[f"{name}__A2B_shape"]), name
        for f in FIELDS:
            got, want = getattr(r, f), reference[f"{name}__{f}"]
            assert got.dtype == want.dtype and got.shape == want.shape, (name, f, got.shape, want.shape)
            if f == "A2B_vals":
                tol = 4 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(want))
                assert np.all(np.abs(got - want) <= tol), (name, f, float(np.abs(got - want).max()))
            else:
                assert np.array_equal(got, want), (name, f)


def test_spans_upstream_umfpack_coupling(cases):
    """Against unpatched upstream ET (et_reference_umfpack.npz, UMFPACK LU): the
    same A basis, AA columns and nnll rows, and per nnll block the same row space
    (a single row may flip sign; a degenerate block may rotate)."""
    ref = np.load(pathlib.Path(__file__).parent / "data" / "et_reference_umfpack.npz")
    for name, c in cases.items():
        r = ajc.couple_raw(c["mb"], c["R"], c["Y"])
        for f in ("aa_sig_off", "aa_sig", "aspec", "aa_off", "aa_idx", "nnll_off", "nnll"):
            assert np.array_equal(getattr(r, f), ref[f"{name}__{f}"]), (name, f)
        shape = tuple(int(v) for v in ref[f"{name}__A2B_shape"])
        assert r.A2B_shape == shape, name
        A, U = np.zeros(shape), np.zeros(shape)
        A[r.A2B_rows, r.A2B_cols] = r.A2B_vals
        U[ref[f"{name}__A2B_rows"], ref[f"{name}__A2B_cols"]] = ref[f"{name}__A2B_vals"]
        no, nl = r.nnll_off, r.nnll
        blocks = {}
        for i in range(shape[0]):
            blocks.setdefault(tuple(map(tuple, nl[no[i]:no[i + 1]])), []).append(i)
        for rows in blocks.values():
            cols = np.flatnonzero(np.abs(A[rows]).sum(0) + np.abs(U[rows]).sum(0))   # the block's own columns
            a, u = A[np.ix_(rows, cols)], U[np.ix_(rows, cols)]
            P, Q = np.linalg.pinv(a) @ a, np.linalg.pinv(u) @ u                       # row-space projectors
            assert np.abs(P - Q).max() < 1e-10, (name, rows[:3])


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


def test_spec_without_invariants_raises_not_aborts():
    """Every body without an L=0 invariant (e.g. a single l=1 body): ET finds no
    coupling.  Must be a ValueError -- an exception escaping the C ABI aborts the
    whole process, so run it in a subprocess."""
    import subprocess
    import sys
    code = ("import ace_jax_coupling as a\n"
            "try:\n"
            "    a.couple_raw([[(1, 1)]], [(1, 0), (1, 1)], [(0, 0), (1, -1), (1, 0), (1, 1)])\n"
            "except ValueError as e:\n"
            "    print('VALUEERROR', e)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "VALUEERROR" in r.stdout, (r.returncode, r.stdout, r.stderr[-1500:])


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
