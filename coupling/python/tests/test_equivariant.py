"""ABI v2: couple_equivariant_raw (L = 0..MAX_L, parity +-1) and the unchanged v1 call."""
import numpy as np
import pytest

import ace_jax_coupling as ajc
from ace_jax_coupling import _loader

Y2 = [(l, m) for l in range(3) for m in range(-l, l + 1)]
R2 = [(n, l) for l in range(3) for n in (1, 2)]
V1 = ["A2B_rows", "A2B_cols", "A2B_vals", "aa_sig_off", "aa_sig", "aspec", "aa_off", "aa_idx", "nnll_off", "nnll"]


def bodies(order=3, even_only=False):
    out = []

    def rec(bb, s):
        if bb and not (even_only and sum(l for _, l in bb) % 2):
            out.append(list(bb))
        if len(bb) == order:
            return
        for i in range(s, len(R2)):
            rec(bb + [R2[i]], i)
    rec([], 0)
    return out


def test_abi_and_limits():
    assert ajc.ABI_VERSION == 2 and ajc.MAX_L == 6
    assert _loader.lib().etc_max_L() == 6


def test_L0_even_is_couple_raw(cases):
    """(L, parity) = (0, +1) is etc_couple, array for array."""
    for name in ("tiny", "SiGe_o2d6", "CrMnFe_o3d6"):
        c = cases[name]
        a, b = ajc.couple_raw(c["mb"], c["R"], c["Y"]), ajc.couple_equivariant_raw(0, 1, c["mb"], c["R"], c["Y"])
        assert a.A2B_shape == b.A2B_shape and (b.L, b.parity) == (0, 1)
        for f in V1:
            assert np.array_equal(getattr(a, f), getattr(b, f)), (name, f)
        assert not b.A2B_m.any()


@pytest.mark.parametrize("L", range(0, 7))
@pytest.mark.parametrize("parity", [1, -1])
def test_layout(L, parity):
    mb = bodies(3)
    try:
        r = ajc.couple_equivariant_raw(L, parity, mb, R2, Y2)
    except ValueError as e:
        assert "NO_INVARIANTS" in str(e) and (L, parity) == (6, -1)
        return
    nB, nAA = r.A2B_shape
    assert nB > 0 and r.A2B_m.dtype == np.int64 and r.A2B_m.shape == r.A2B_vals.shape
    assert r.A2B_m.min() >= 0 and r.A2B_m.max() <= 2 * L and np.all(r.A2B_vals != 0)
    assert r.A2B_rows.max() == nB - 1 and r.A2B_cols.max() <= nAA - 1
    # each (row, m, col) once
    key = (r.A2B_rows * (2 * L + 1) + r.A2B_m) * nAA + r.A2B_cols
    assert len(np.unique(key)) == len(key)
    # every product has parity (-1)^sum(l)
    so = r.aa_sig_off
    for k in range(nAA):
        assert (-1) ** int(r.aa_sig[so[k]:so[k + 1], 1].sum()) == parity
    # rows are orthonormal as (2L+1) x nAA blocks up to the m-multiplicity of each column
    assert len(r.nnll_off) == nB + 1 and len(r.aa_off) == nAA + 1


@pytest.mark.parametrize("L, parity", [(7, 1), (-1, 1), (1, 0), (1, 2), (True, 1), (1.0, 1)])
def test_bad_L_or_parity(L, parity):
    with pytest.raises(ValueError):
        ajc.couple_equivariant_raw(L, parity, bodies(2), R2, Y2)


def test_no_function_of_that_symmetry_is_valueerror():
    with pytest.raises(ValueError, match="NO_INVARIANTS"):
        ajc.couple_equivariant_raw(6, -1, bodies(2), R2, Y2)


def test_raw_bad_L_returns_code():
    """A direct ctypes call with L out of range is rejected (code 2) before the buffers are read."""
    lib = _loader.lib()
    assert lib.etc_couple_equivariant(9, 1, 0, None, None, None, 0, None, None, 0, None, None, None,
                                      *([None] * 11)) == 2


def test_old_library_keeps_v1_and_refuses_v2(monkeypatch):
    """An ABI 1 library (ace-jax-coupling 0.2.0's) still serves couple_raw; the v2 call says why not."""
    lib = _loader.lib()
    monkeypatch.setattr(_loader, "_ABI", 1)
    c = [[(1, 0)], [(1, 1), (1, 1)]], [(1, 0), (1, 1)], [(0, 0), (1, -1), (1, 0), (1, 1)]
    assert ajc.couple_raw(*c).A2B_shape[0] == 2
    with pytest.raises(ajc.CouplingLibError, match="ABI 1"):
        ajc.couple_equivariant_raw(1, -1, *c)
    assert lib is _loader.lib()
