"""couple_raw: validated, two-phase call into etc_couple."""
import numbers
from typing import NamedTuple

import numpy as np

from . import _loader

MAX_ORDER = 8
_OK, _SHORT, _INVALID, _ORDER = 0, 1, 2, 3


class RawCoupling(NamedTuple):
    A2B_shape: tuple
    A2B_rows: np.ndarray
    A2B_cols: np.ndarray
    A2B_vals: np.ndarray
    aa_sig_off: np.ndarray
    aa_sig: np.ndarray
    aspec: np.ndarray
    aa_off: np.ndarray
    aa_idx: np.ndarray
    nnll_off: np.ndarray
    nnll: np.ndarray


def _int(v, what):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise ValueError(f"{what}: expected an integer, got {v!r}")
    return int(v)


def _pairs(xs, what):
    out = []
    for x in xs:
        x = tuple(x)
        if len(x) != 2:
            raise ValueError(f"{what}: expected pairs, got {x!r}")
        out.append((_int(x[0], what), _int(x[1], what)))
    return out


def _validate(mb, R, Y):
    if len(mb) == 0:
        raise ValueError("mb_spec is empty")
    if len(set(R)) != len(R):
        raise ValueError("Rnl_spec has duplicate (n, l) entries")
    if len(set(Y)) != len(Y):
        raise ValueError("Ylm_spec has duplicate (l, m) entries")
    for n, l in R:
        if l < 0:
            raise ValueError(f"Rnl_spec: negative l in {(n, l)}")
    for l, m in Y:
        if l < 0 or abs(m) > l:
            raise ValueError(f"Ylm_spec: need l >= 0 and |m| <= l, got {(l, m)}")
    rs, ys = set(R), set(Y)
    for i, bb in enumerate(mb):
        if len(bb) == 0:
            raise ValueError(f"mb_spec[{i}] is an empty body")
        if len(bb) > MAX_ORDER:
            raise ValueError(f"mb_spec[{i}]: correlation order {len(bb)} > {MAX_ORDER}")
        for n, l in bb:
            if l < 0:
                raise ValueError(f"mb_spec[{i}]: negative l in {(n, l)}")
            if (n, l) not in rs:
                raise ValueError(f"mb_spec[{i}]: {(n, l)} not in Rnl_spec")
            for m in range(-l, l + 1):
                if (l, m) not in ys:
                    raise ValueError(f"mb_spec[{i}]: {(l, m)} not in Ylm_spec")


def couple_raw(mb_spec, Rnl_spec, Ylm_spec) -> RawCoupling:
    """L = 0 real-basis coupling of EquivariantTensors for the given specs.
    mb_spec: bodies of (n, l); Rnl_spec: (n, l); Ylm_spec: (l, m).  All
    returned indices are 0-based; layout documented on `RawCoupling`'s fields
    (docs/coupling-etshim-spec.md)."""
    mb = [_pairs(bb, "mb_spec") for bb in mb_spec]
    R, Y = _pairs(Rnl_spec, "Rnl_spec"), _pairs(Ylm_spec, "Ylm_spec")
    _validate(mb, R, Y)
    i64 = lambda xs: np.ascontiguousarray(np.asarray(xs, np.int64))
    off = i64(np.cumsum([0] + [len(bb) for bb in mb]))
    flat = [b for bb in mb for b in bb]
    mn, ml = i64([n for n, _ in flat]), i64([l for _, l in flat])
    rn, rl = i64([n for n, _ in R]), i64([l for _, l in R])
    yl, ym = i64([l for l, _ in Y]), i64([m for _, m in Y])
    p = lambda a: a.ctypes.data
    h = _loader.lib()
    sizes = np.zeros(7, np.int64)

    def call(bufs):
        return h.etc_couple(len(mb), p(off), p(mn), p(ml), len(R), p(rn), p(rl), len(Y), p(yl), p(ym),
                            p(sizes), *[p(b) if b is not None else None for b in bufs])

    with _loader.LOCK:
        code = call([None] * 10)
        if code not in (_OK, _SHORT):
            raise _error(code)
        nnz, nB, nAA, nsig, nA, naa, nnl = (int(v) for v in sizes)
        bufs = [np.zeros(nnz, np.int64), np.zeros(nnz, np.int64), np.zeros(nnz, np.float64),
                np.zeros(nAA + 1, np.int64), np.zeros(3 * nsig, np.int64), np.zeros(2 * nA, np.int64),
                np.zeros(nAA + 1, np.int64), np.zeros(naa, np.int64),
                np.zeros(nB + 1, np.int64), np.zeros(2 * nnl, np.int64)]
        code = call(bufs)
        if code != _OK:
            raise _error(code)
    return RawCoupling((nB, nAA), bufs[0], bufs[1], bufs[2], bufs[3], bufs[4].reshape(-1, 3),
                       bufs[5].reshape(-1, 2), bufs[6], bufs[7], bufs[8], bufs[9].reshape(-1, 2))


def _error(code):
    if code == _INVALID:
        return ValueError("coupling library rejected the specs (INVALID_INPUT)")
    if code == _ORDER:
        return ValueError(f"correlation order > {MAX_ORDER} (ORDER_TOO_HIGH)")
    return _loader.CouplingLibError(f"etc_couple returned unexpected code {code}")
