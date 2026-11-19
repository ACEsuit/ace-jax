"""Equivariant couplings (L, parity) from the compiled library's ABI v2.

The physics check is here, not in the library's own tests: with ace-jax's real
Y_lm, B^(L,p) = A2B . AA must rotate with the Wigner matrix of the SAME
harmonics (component m of the L-vector is Y_L's m), and pick up `parity` under
inversion.  That fixes the convention the Hamiltonian readout relies on."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from conftest import require_coupling_lib

from ace_jax.basis import coupling as C
from ace_jax.basis.spec import ylm_spec
from ace_jax.eval.harmonics import real_spherical_harmonics

MAXN, MAXL = 2, 2
RNL = [(n, l) for l in range(MAXL + 1) for n in range(1, MAXN + 1)]
YLM = ylm_spec(MAXL)
LPS = [(0, 1), (0, -1), (1, -1), (1, 1), (2, 1), (2, -1), (3, -1), (4, 1)]


def bodies(order=3):
    out = []

    def rec(bb, s):
        if bb:
            out.append(list(bb))
        if len(bb) == order:
            return
        for i in range(s, len(RNL)):
            rec(bb + [RNL[i]], i)
    rec([], 0)
    return out


MB = bodies()


def need_v2():
    ajc = require_coupling_lib()
    if not hasattr(ajc, "couple_equivariant_raw"):
        pytest.skip("ace-jax-coupling < 0.3.0: no equivariant couplings")


def a_basis(xyz, aspec):
    """Pooled A_{nlm} = sum_j R_n(r_j) Y_lm(r_j) for a toy radial R_n = r^n exp(-r)."""
    r = np.linalg.norm(xyz, axis=1)
    Y = np.asarray(real_spherical_harmonics(xyz, MAXL))
    R = np.stack([r ** n * np.exp(-r) * (1 + 0.3 * l) for n, l in RNL], axis=1)
    return np.array([(R[:, ri] * Y[:, yi]).sum() for ri, yi in aspec])


def aa_products(A, aa_specs):
    return np.concatenate([np.prod(A[np.asarray(g)], axis=1) for g in aa_specs if len(g)])


def features(xyz, aspec, aa_specs, a2b):
    return np.einsum("bmk,k->bm", a2b.dense(), aa_products(a_basis(xyz, aspec), aa_specs))


def wigner(L, Q):
    """D with Y_L(Q r) = D Y_L(r), from the harmonics themselves."""
    pts = np.random.default_rng(1).normal(size=(60, 3))
    sl = slice(L * L, (L + 1) ** 2)
    Y0 = np.asarray(real_spherical_harmonics(pts, max(L, 1)))[:, sl]
    Y1 = np.asarray(real_spherical_harmonics(pts @ Q.T, max(L, 1)))[:, sl]
    return np.linalg.lstsq(Y0, Y1, rcond=None)[0].T


def rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


@pytest.mark.parametrize("L, parity", LPS)
def test_rotation_and_inversion(L, parity):
    need_v2()
    cpl = C.couple_equivariant(L, parity, MB, RNL, YLM)
    nB = cpl.A2B.shape[0]
    assert nB > 0 and cpl.A2B.shape[1] == 2 * L + 1
    xyz = np.random.default_rng(0).normal(size=(7, 3))
    B = features(xyz, cpl.aspec, cpl.aa_specs, cpl.A2B)
    assert np.abs(B).max() > 1e-3
    Q = rotation(2)
    D = wigner(L, Q) if L else np.ones((1, 1))
    assert np.allclose(features(xyz @ Q.T, cpl.aspec, cpl.aa_specs, cpl.A2B), B @ D.T, atol=1e-11)
    assert np.allclose(features(-xyz, cpl.aspec, cpl.aa_specs, cpl.A2B), parity * B, atol=1e-12)


def test_L0_even_is_couple():
    need_v2()
    a, b = C.couple(MB, RNL, YLM), C.couple_equivariant(0, 1, MB, RNL, YLM)
    assert np.array_equal(a.A2B, b.A2B.dense()[:, 0, :])
    assert a.aa_sig == b.aa_sig and a.aspec == b.aspec and a.nnll_spec == b.nnll_spec
    assert all(np.array_equal(x, y) for x, y in zip(a.aa_specs, b.aa_specs))


def test_no_function_is_empty_not_an_error():
    need_v2()
    cpl = C.couple_equivariant(6, -1, bodies(2), RNL, YLM)
    assert cpl.A2B.shape == (0, 13, 0) and cpl.A2B.dense().shape == (0, 13, 0)
    assert cpl.aa_sig == () and cpl.aspec == () and cpl.nnll_spec == ()


def test_set_shares_one_aa():
    """One A list and one AA evaluation order for all (L, parity); each part
    gives the features of its single-call coupling."""
    need_v2()
    lps = [(0, 1), (1, -1), (1, 1), (2, 1), (6, -1)]
    s = C.couple_equivariant_set(lps, bodies(2), RNL, YLM, cache_dir="none")
    xyz = np.random.default_rng(3).normal(size=(6, 3))
    n_aa = sum(len(g) for g in s.aa_specs)
    assert len(s.aa_sig) == n_aa and len(set(s.aa_sig)) == n_aa and len(set(s.aspec)) == len(s.aspec)
    assert [g.shape[1] for g in s.aa_specs] == list(range(1, len(s.aa_specs) + 1))
    for lp in lps:
        one = C.couple_equivariant(*lp, bodies(2), RNL, YLM)
        assert s.A2B[lp].shape == (one.A2B.shape[0], 2 * lp[0] + 1, n_aa)
        assert s.nnll[lp] == one.nnll_spec
        if one.A2B.shape[0]:
            assert np.allclose(features(xyz, s.aspec, s.aa_specs, s.A2B[lp]),
                               features(xyz, one.aspec, one.aa_specs, one.A2B), atol=1e-13)
    assert s.A2B[(6, -1)].shape[0] == 0


# ------------------------------------------------------------- no library needed
def _fake(L, parity, mb, rnl, ylm):
    k = 2 * L + 1
    a2b = C.A2B3((1, k, 2), np.zeros(k, np.int64), np.arange(k), np.zeros(k, np.int64), np.full(k, 0.5 * parity))
    return C.EquivariantCoupling(L, parity, a2b, aa_sig=(((1, 0, 0),), ((1, 0, 0), (1, 0, 0))), aspec=((0, 0),),
                                 aa_specs=(np.zeros((1, 1), np.int64), np.zeros((1, 2), np.int64)),
                                 nnll_spec=(((1, 0),),))


def test_keys_differ():
    mb, rnl, ylm = [[(1, 0)], [(1, 0), (1, 0)]], [(1, 0)], [(0, 0)]
    keys = {C.equivariant_key(L, p, mb, rnl, ylm) for L in range(3) for p in (1, -1)}
    assert len(keys) == 6 and C.coupling_key(mb, rnl, ylm) not in keys


def test_cache_miss_then_hit(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(C, "couple_equivariant", lambda *a: (calls.append(a[:2]), _fake(*a))[1])
    mb, rnl, ylm = [[(1, 0)], [(1, 0), (1, 0)]], [(1, 0)], [(0, 0)]
    a = C.couple_equivariant_cached(1, -1, mb, rnl, ylm, cache_dir=str(tmp_path))
    b = C.couple_equivariant_cached(1, -1, mb, rnl, ylm, cache_dir=str(tmp_path))
    assert calls == [(1, -1)]
    assert (b.L, b.parity) == (1, -1) and b.A2B.shape == a.A2B.shape
    for f in ("rows", "ms", "cols", "vals"):
        assert np.array_equal(getattr(a.A2B, f), getattr(b.A2B, f))
    assert a.aa_sig == b.aa_sig and a.aspec == b.aspec and a.nnll_spec == b.nnll_spec
    C.couple_equivariant_cached(1, 1, mb, rnl, ylm, cache_dir=str(tmp_path))       # another parity: a miss
    C.couple_equivariant_cached(2, -1, mb, rnl, ylm, cache_dir=str(tmp_path))      # another L: a miss
    assert calls == [(1, -1), (1, 1), (2, -1)]
    assert sorted(p.name[:5] for p in tmp_path.iterdir()) == ["cple-"] * 3


def test_cache_entry_from_another_backend_is_a_miss(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(C, "couple_equivariant", lambda *a: (calls.append(1), _fake(*a))[1])
    mb, rnl, ylm = [[(1, 0)]], [(1, 0)], [(0, 0)]
    C.couple_equivariant_cached(1, -1, mb, rnl, ylm, cache_dir=str(tmp_path))
    monkeypatch.setattr(C, "backend_id", lambda: "ace-jax-coupling==9.9.9")
    C.couple_equivariant_cached(1, -1, mb, rnl, ylm, cache_dir=str(tmp_path))
    assert len(calls) == 2


def test_cache_only_miss_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    with pytest.raises(RuntimeError, match="CACHE_ONLY"):
        C.couple_equivariant_cached(1, -1, [[(1, 0)]], [(1, 0)], [(0, 0)], cache_dir=str(tmp_path))
