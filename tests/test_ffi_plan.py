"""The CUDA kernel's DAG pass plan (`ffi.pull_plan`) without a GPU: a NumPy copy of `ace_dag` in
ace_ffi_gpu.cu, a thread at a time between the kernel's barriers, on one row with a random A. Its energy and
dE/dA must equal the plain DAG's (values by node, then reverse mode). Keep `emulate` in step with the kernel:
it checks the lists the kernel reads (tiles, ranges, packed items, piece slots), not the CUDA arithmetic."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from ace_jax.eval import ffi, load
from ace_jax.eval.model import lean
from conftest import FIXTURE_DIR as FIXTURES

NAMES = ["si_fitted", "si_ace_model", "sige_nofit", "ace_cantor5_small", "si_m710"]


def unpack(attrs, consts, zc):
    grp, fac, lev = (np.asarray(consts[k], np.int64) for k in (3, 4, 7))
    ct, lcz = np.asarray(consts[5])[zc], np.asarray(consts[9])[zc]
    return grp.ravel(), fac, lev, ct, lcz, len(ct) - len(fac)


def reference(attrs, consts, zc, A):
    """(E, dE/dA) of sum_n ct_n L_n, L_n = L_parent(n) A[factor(n)]."""
    grp, fac, lev, ct, _, na = unpack(attrs, consts, zc)
    parent = np.repeat(grp[0::2], np.diff(np.r_[0, grp[1::2]]))
    L = np.r_[A, np.zeros(len(fac))]
    for k in range(len(fac)):
        L[na + k] = L[parent[k]] * A[fac[k]]
    w, dA = ct.copy(), np.zeros(na)
    for k in reversed(range(len(fac))):
        w[parent[k]] += w[na + k] * A[fac[k]]
        dA[fac[k]] += w[na + k] * L[parent[k]]
    return ct @ L, dA + w[:na]


def emulate(attrs, consts, zc, A_in):
    """ace_dag for one row (GA = 1): names as in the kernel."""
    grp, fac, lev, cz, lcz, na = unpack(attrs, consts, zc)
    P = np.asarray(consts[8], np.int64)
    T, mode, n_lev = int(attrs["threads"]), int(attrs["lv"]), len(lev) - 1
    W, mids, st_lev = T + 1, n_lev > 1, max(n_lev - 2, 0)
    n_st = int(attrs["n_st"]) if mids else 0
    nL = na + n_st + 1
    nV = nL if mode else na + 1
    tiles, lcut, litems, ib, ist, pst, fst, items = (P[P[k]:] for k in range(3, 11))
    n_tiles, n_slots, n_lists = P[:3]
    kids = lambda q: range(grp[2 * q - 1] if q else 0, grp[2 * q + 1])
    Lv = np.zeros(nV); Lv[:na] = A_in; Lv[nV - 1] = 1
    A = Lv
    for l in range(st_lev if mode else 0):                  # the kept levels
        for q in range(lev[l], lev[l + 1]):
            for k in kids(q):
                Lv[na + k] = Lv[grp[2 * q]] * A[fac[k]]
    wv = np.zeros(nL); wv[:nL - 1] = cz[:nL - 1]
    pw = np.zeros(int(max(pst[:n_lists * W].max(), fst[:n_lists * (na + 1)].max())) + 1)

    def pieces(L, term):                                    # a list's ranges into their piece slots
        it = items[ib[L]:]
        for t in range(T):
            pc, acc = pst[L * W + t], 0.0
            for o in range(t, t + T * (ist[L * W + t + 1] - ist[L * W + t]), T):
                if mode == 0:
                    v = int(it[o]); last = v < 0
                    acc += term(o, v, None)
                else:
                    x, y = int(it[2 * o]), int(it[2 * o + 1]); last = y < 0
                    acc += term(o, x, ~y if last else y)
                if last:
                    pw[pc] = acc; acc = 0.0; pc += 1

    for ti in range(n_tiles):
        u0, nt, base, g0, g1 = tiles[5 * ti:5 * ti + 5]
        lc = lcut[ti * W:(ti + 1) * W]
        wms = cz[(na + n_st if mids else 0) + u0:][:nt].copy()
        add = np.zeros(nt)                                  # the leaves into their parents
        for t in range(T):
            acc = 0.0
            for o in range(t, t + T * (lc[t + 1] - lc[t]), T):
                v = int(litems[base + o])
                acc += lcz[base + o] * A[v & 32767]
                if v < 0:
                    add[(v >> 15) & 65535] += acc; acc = 0.0
        wms += add
        if mids:
            for q in range(g0, g1):                         # the mids into theirs
                wv[grp[2 * q]] += sum(wms[k - n_st - u0] * A[fac[k]] for k in kids(q))
            if mode == 0:
                pieces(n_lev + ti, lambda o, v, _, wms=wms: wms[(v >> 20) & 2047] * A[v & 1023] * A[(v >> 10) & 1023])
            else:
                pieces(n_lev + ti, lambda o, x, y, wms=wms: wms[x] * Lv[y])
        else:
            wv[u0:u0 + nt] = wms
    for l in reversed(range(st_lev)):                       # the kept levels into theirs
        for q in range(lev[l], lev[l + 1]):
            wv[grp[2 * q]] += sum(wv[na + k] * A[fac[k]] for k in kids(q))
    e = A[:na] @ wv[:na]
    w = lcz[n_slots:]
    for l in range(n_lev):                                  # dE/dA: the pieces
        if l == n_lev - 1 and mode == 0:
            pieces(l, lambda o, v, _: w[o] * A[v & 1023] * A[(v >> 10) & 1023] * A[(v >> 20) & 1023])
        elif l == n_lev - 1:
            pieces(l, lambda o, x, y: w[o] * Lv[x] * Lv[y])
        elif mode == 0:
            pieces(l, lambda o, v, _: wv[(v >> 10) & 0x1fffff] * A[v & 1023])
        else:
            pieces(l, lambda o, x, y: wv[x] * Lv[y])
    wA = wv[:na].copy()
    for l in range(n_lists):
        for a in range(na):
            wA[a] += pw[fst[l * (na + 1) + a]:fst[l * (na + 1) + a + 1]].sum()
    return e, wA


@pytest.fixture(scope="module", params=NAMES)
def model(request):
    m, _, _ = load(str(FIXTURES / f"{request.param}.npz"))
    return lean(m, spline_tol=None, radial_table=True)


@pytest.mark.parametrize("tile", [None, 8])
@pytest.mark.parametrize("threads", [None, 32, 96, 256])
@pytest.mark.parametrize("lv", [None, 1])
def test_pull_plan_gives_the_dag(model, monkeypatch, lv, threads, tile):
    """Every form the kernel runs: values recomputed from A (lv 0, the fixtures' default) or kept, the block
    sizes `pull_threads` chooses plus one that is not a power of two, and several tiles (the fixtures' leaf
    parents fit one)."""
    monkeypatch.setattr(ffi, "PULL_LV", lv)
    monkeypatch.setattr(ffi, "PULL_THREADS", threads)
    if tile:
        monkeypatch.setattr(ffi, "PULL_TILE", tile)
    attrs, consts = ffi.kernel_inputs(model, gpu=True)
    assert int(attrs["threads"]) == (threads or ffi.pull_threads(len(consts[4])))
    rng = np.random.default_rng(0)
    for zc in range(np.asarray(consts[5]).shape[0]):
        A = rng.normal(size=np.asarray(consts[5]).shape[1] - len(consts[4]))
        (e0, d0), (e1, d1) = reference(attrs, consts, zc, A), emulate(attrs, consts, zc, A)
        assert abs(e1 - e0) <= 1e-11 * max(abs(e0), 1.0)
        assert np.abs(d1 - d0).max() <= 1e-11 * max(np.abs(d0).max(), 1.0)


def test_emulation_sees_a_wrong_item(model):
    """The check has teeth: one leaf item's factor changed is a different energy or dE/dA."""
    attrs, consts = ffi.kernel_inputs(model, gpu=True)
    P = np.array(consts[8]); na = np.asarray(consts[5]).shape[1] - len(consts[4])
    A = np.random.default_rng(1).normal(size=na)
    at = P[5] + int(np.flatnonzero(np.asarray(consts[9])[0][:P[1]])[0])    # a leaf item with a coefficient
    P[at] ^= 1
    (e0, d0), (e1, d1) = reference(attrs, consts, 0, A), emulate(attrs, consts[:8] + (P,) + consts[9:], 0, A)
    assert abs(e1 - e0) > 1e-8 * max(abs(e0), 1.0) or np.abs(d1 - d0).max() > 1e-8 * np.abs(d0).max()
