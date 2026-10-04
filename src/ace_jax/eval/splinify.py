"""Analytic tensor radials -> the spline branch (`to_spline`), for deployment.

The analytic radial is R_n(r) = env(x) sum_q W[zi, zj, n, q] P_q(x) with
x = T_{zi,zj}(r) (`ACEModel._radial_one`), so its envelope-free part
S_n(x) = sum_q W P_q(x) is a polynomial on [-1, 1].  `to_spline` tabulates S on a
uniform grid and interpolates it with the cubic B-spline `spline_eval` evaluates
(`construct.radial_ace1.cubic_bspline_coefs`, the builder of the Python-authored
and Julia `splinify` tables), with the polynomial's exact end curvature as the
boundary condition, so the error is O(h^4) up to the ends.  Per edge the
spline is a 4-row gather instead of the n_q-term recursion and an
(n_rnl x n_q) contraction, and a table keeps ACE1's zero pattern exactly (a
radial column zero for a z_j tabulates to exact zeros), so `lean`'s
species-compact l-blocks apply to a learned radial too.

It is an approximation, not a restructuring: the result agrees with the
analytic model to the requested tolerance, not to roundoff.  Fitting keeps the
analytic model; this is for evaluation and export (`lean` applies it).
"""
import dataclasses
import hashlib
import threading
from collections import OrderedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# The splining policy of lean / lean_keep_basis / ACECalculator / export_lammps,
# decided in `spline_plan` alone.  Their default spline_tol is "auto": spline
# an analytic tensor radial only when it was learned (`radial_learned`), at
# DEFAULT_SPLINE_TOL.  A float is the explicit opt-in (every analytic radial,
# learned or not: Julia ace_model exports, Python-authored models); None never
# splines.
DEFAULT_SPLINE_TOL = 1e-10
AUTO = "auto"


def spline_plan(model, spline_tol=AUTO):
    """(tol, radials) that `lean` will pass to `to_spline`, or None for no
    splining.  "auto": ("rnl",) at DEFAULT_SPLINE_TOL when R_nl is analytic
    and `radial_learned`; the pair radial is never learned, so "auto" keeps
    an analytic pair radial exact.  A float: every analytic radial at that
    tol.  None: nothing."""
    rk, pk = getattr(model, "radial_kind", None), getattr(model, "pair_radial_kind", None)
    if spline_tol is None:
        return None
    if isinstance(spline_tol, str):
        if spline_tol != AUTO:
            raise ValueError(f"spline_tol must be 'auto', a float or None, got {spline_tol!r}")
        if rk == "analytic" and getattr(model, "radial_learned", False):
            return DEFAULT_SPLINE_TOL, ("rnl",)
        return None
    radials = tuple(r for r, k in (("rnl", rk), ("pair", pk)) if k == "analytic")
    return (float(spline_tol), radials) if radials else None
TOL_FLOOR = 1e-14                 # float64: below this the polynomial sum's own roundoff shows
START_INTERVALS = 32
MAX_INTERVALS = 1 << 16
CHECK_PER_INTERVAL = 10
CHECK_CHUNK = 1 << 14             # check-grid points per block: bounds _rel_err's temporaries

# Interval counts are rounded up to BUCKETS, round(2^(k/4)): a grid is at most
# 19% (2^(1/4)) finer than it needs, 9% on average, and radials that need
# similar grids -- successive learned radials, or one swapped for another --
# share the table shape and the static grid, so the compiled step is reused
# instead of retraced.  Quarter-octave steps are ~4 buckets per doubling:
# multiples of 256 would cost 20% at 1258 intervals and give ~10 buckets over
# the 1258-3847 range that 1e-10 needs; 2^(k/4) gives 7, at a flat <= 19%.
BUCKETS = tuple(sorted({int(round(2 ** (k / 4))) for k in range(20, 65)}))   # 32 .. 65536


def _bucket(n):
    for b in BUCKETS:
        if b >= n:
            return b
    return BUCKETS[-1]


# Converted tables, keyed on the content of what determines them (Wnlq, the
# recursion, the R_nl envelope the error is measured with, the measured
# columns, n_intervals, tol, deriv_tol), never on object identity: an
# ACECalculator re-leans on every `calc.model` swap, and a swap that changes
# only readout weights (ctilde, WB, Wpair, E0) must not re-spline, while any
# radial edit must.  An LRU bounded by bytes (the float64 tables); a lock, so
# calculators in threads can share it.
CACHE_BYTES = 256 << 20
_CACHE = OrderedDict()
_LOCK = threading.Lock()


def clear_cache():
    with _LOCK:
        _CACHE.clear()


def _cache_bytes():
    return sum(v[0].nbytes for v in _CACHE.values())


def _key(*parts):
    h = hashlib.sha256()
    for p in parts:
        if p is None or isinstance(p, (int, float, str)):
            h.update(repr(p).encode())
        else:
            a = np.ascontiguousarray(np.asarray(p, np.float64))
            h.update(repr(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def _memo(key, build):
    """build() through the LRU under `key` (`_key`); the value is a tuple whose
    first entry is the table, which is what the byte bound counts.  Built
    outside the lock: two threads missing at once both build, and the second
    insert wins (same content)."""
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    out = build()
    with _LOCK:
        _CACHE[key] = out
        while len(_CACHE) > 1 and _cache_bytes() > CACHE_BYTES:
            _CACHE.popitem(last=False)
    return out


def _fit_cached(W, polys, n_intervals, tol, deriv_tol, env_params, cols):
    """`_fit` through the LRU; env_params (NZ, NZ, 5) or None (pair radial)."""
    key = _key("rnl" if env_params is not None else "pair", W, *polys, env_params, cols,
               None if n_intervals is None else int(n_intervals), float(tol),
               None if deriv_tol is None else float(deriv_tol))
    return _memo(key, lambda: _fit(W, polys, n_intervals, tol, deriv_tol, env_params, cols))


def _poly_d012(x, A, B, C):
    """P_q, P_q' and P_q'' of the 3-term recurrence at x, each (n_x, n_q)."""
    x = np.asarray(x, np.float64)
    A, B, C = (np.asarray(v, np.float64) for v in (A, B, C))
    n = len(A)
    P, D, D2 = (np.zeros(x.shape + (n,)) for _ in range(3))
    P[..., 0] = A[0]
    if n > 1:
        P[..., 1] = A[1] * x + B[1]
        D[..., 1] = A[1]
    for k in range(2, n):
        a = A[k] * x + B[k]
        P[..., k] = a * P[..., k - 1] + C[k] * P[..., k - 2]
        D[..., k] = A[k] * P[..., k - 1] + a * D[..., k - 1] + C[k] * D[..., k - 2]
        D2[..., k] = 2 * A[k] * D[..., k - 1] + a * D2[..., k - 1] + C[k] * D2[..., k - 2]
    return P, D, D2


def _env_d01(x, e):
    """PolyEnvelope2sX s (x - x1)^p1 (x2 - x)^p2 and its x-derivative; e (5,)."""
    x1, x2, p1, p2, sc = e
    inside = (x > x1) & (x < x2)
    a = np.where(inside, x - x1, 1.0)
    b = np.where(inside, x2 - x, 1.0)
    v = np.where(inside, sc * a ** p1 * b ** p2, 0.0)
    return v, np.where(inside, v * (p1 / a - p2 / b), 0.0)


def _bspline_eval(x, c, n_int, deriv=False, x0=-1.0, h=None):
    """`radial.spline_eval` in numpy on the grid (x0, h, n_int+1), by default
    to_spline's (-1, 2/n_int); c (n_int+3, F) -> (n_x, F), or its x-derivative
    with deriv=True."""
    h = 2.0 / n_int if h is None else h
    k = (x - x0) / h + 1.0                     # x - (-1.0) is x + 1.0 bit for bit
    ix = np.clip(np.floor(k).astype(int), 1, n_int)
    t = k - ix
    ct = 1.0 - t
    if deriv:
        w = np.stack([-3.0 * ct ** 2, -12.0 * t + 9.0 * t ** 2,
                      12.0 * ct - 9.0 * ct ** 2, 3.0 * t ** 2], axis=-1) / (6.0 * h)
    else:
        w = np.stack([ct ** 3, 4.0 - 6.0 * t ** 2 + 3.0 * t ** 3,
                      4.0 - 6.0 * ct ** 2 + 3.0 * ct ** 3, t ** 3], axis=-1) / 6.0
    g = c[ix[:, None] + np.arange(-1, 3), :]                                   # (n_x, 4, F)
    return np.einsum("xk,xkf->xf", w, g)


def _table(W, polys, n_int):
    """Spline coefficients (NZ, NZ, n_int+3, F) of S = W . P on n_int intervals."""
    from ..basis.radial_ace1 import cubic_bspline_coefs
    NZ, _, F, n_q = W.shape
    xs = np.linspace(-1.0, 1.0, n_int + 1)
    P, _, _ = _poly_d012(xs, *polys)
    _, _, D2 = _poly_d012(np.array([-1.0, 1.0]), *polys)
    h2 = (2.0 / n_int) ** 2
    out = np.zeros((NZ, NZ, n_int + 3, F))
    for i in range(NZ):
        for j in range(NZ):
            if F and np.any(W[i, j]):                          # a zero block stays exact zeros
                E = D2 @ W[i, j].T * h2                        # h^2 S'' at the two ends
                out[i, j] = cubic_bspline_coefs(P @ W[i, j].T, (E[0], E[1]))
    return out


def _rel_err(W, polys, c, n_int, env_params=None, cols=None):
    """(value, derivative) relative errors: max over (zi, zj, column) of
    max_x |spline - exact| / max_x |exact|, for R = env * S (env given) or S,
    and the same for dR/dx (what forces see; O(h^3) where values are O(h^4)).
    On CHECK_PER_INTERVAL points per interval (the knots and 9 interior
    points), in CHECK_CHUNK blocks; only `cols` (default all) are measured."""
    x_all = np.linspace(-1.0, 1.0, CHECK_PER_INTERVAL * n_int + 1)
    cols = np.arange(W.shape[2]) if cols is None else np.asarray(cols)
    if not cols.size:
        return 0.0, 0.0
    worst, dworst = 0.0, 0.0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wij = W[i, j][cols]
            if not np.any(Wij):
                continue
            cij = c[i, j][:, cols]
            nrm = np.zeros(len(cols)); err = np.zeros(len(cols))
            dnrm = np.zeros(len(cols)); derr = np.zeros(len(cols))
            for lo in range(0, len(x_all), CHECK_CHUNK):
                x = x_all[lo:lo + CHECK_CHUNK]
                P, D, _ = _poly_d012(x, *polys)
                S, dS = P @ Wij.T, D @ Wij.T
                eS = _bspline_eval(x, cij, n_int) - S
                edS = _bspline_eval(x, cij, n_int, deriv=True) - dS
                if env_params is not None:
                    v, dv = _env_d01(x, env_params[i, j])
                    S, dS = v[:, None] * S, dv[:, None] * S + v[:, None] * dS
                    eS, edS = v[:, None] * eS, dv[:, None] * eS + v[:, None] * edS
                nrm = np.maximum(nrm, np.abs(S).max(0)); err = np.maximum(err, np.abs(eS).max(0))
                dnrm = np.maximum(dnrm, np.abs(dS).max(0)); derr = np.maximum(derr, np.abs(edS).max(0))
            live, dlive = nrm > 0, dnrm > 0
            if live.any():
                worst = max(worst, float((err[live] / nrm[live]).max()))
            if dlive.any():
                dworst = max(dworst, float((derr[dlive] / dnrm[dlive]).max()))
    return worst, dworst


def _try(W, polys, n_int, env_params, cols):
    c = _table(W, polys, n_int)
    return (c, n_int, *_rel_err(W, polys, c, n_int, env_params, cols))


def _fit(W, polys, n_intervals, tol, deriv_tol, env_params, cols):
    """(coefs, n_int, err, deriv_err): the given n_intervals, or the smallest
    bucket (`BUCKETS`) meeting tol (and deriv_tol when given).  Doubles from
    START_INTERVALS to bracket it, then, since the last doubling can overshoot
    by up to 16x (the error is O(h^4)), tries buckets from the one the h^4
    rate predicts up to the bracket."""
    if n_intervals:
        return _try(W, polys, int(n_intervals), env_params, cols)
    ok = lambda r: r[2] <= tol and (deriv_tol is None or r[3] <= deriv_tol)   # noqa: E731
    lo, n_int = None, START_INTERVALS
    while True:
        r = _try(W, polys, n_int, env_params, cols)
        if ok(r):
            break
        if n_int >= MAX_INTERVALS:
            raise ValueError(f"to_spline: relative error {r[2]:.2e} (derivative {r[3]:.2e}) "
                             f"misses tol {tol:.1e} at {n_int} intervals; loosen tol")
        lo = r
        n_int = _bucket(2 * n_int)
    if lo is not None:
        rate = max(lo[2] / tol, 1.0) ** 0.25
        if deriv_tol is not None:
            rate = max(rate, max(lo[3] / deriv_tol, 1.0) ** (1 / 3))
        for b in BUCKETS:
            if b >= r[1]:
                break
            if b > lo[1] and b >= 1.05 * lo[1] * rate:
                t = _try(W, polys, b, env_params, cols)
                if ok(t):
                    return t
    return r


def to_spline(model, n_intervals=None, tol=DEFAULT_SPLINE_TOL, deriv_tol=None,
              return_info=False, radials=("rnl", "pair")):
    """Convert an ACEModel's analytic radials to the spline branch.

    Converts every analytic radial named in `radials` ("rnl", "pair"), learned
    or not (an ACEpotentials `ace_model` export or a built basis too): this is
    the conversion itself.  Whether `lean` applies it is `spline_plan`'s call
    (by default only for learned radials).

    The tensor radial (`rnl_Wnlq`, radial_kind "analytic") and the pair radial
    (`pair_Wnlq`, pair_radial_kind "analytic") are each tabulated as
    S(x) = sum_q W P_q(x) on a uniform grid of n intervals over x in [-1, 1],
    (x0, h, n + 1) = (-1, 2 / n, n + 1), and interpolated with the cubic
    B-spline `spline_eval` evaluates, clamped to the polynomial's own second
    derivative at the ends.  The envelope, the transform and every other array
    are unchanged.

    Grid: `n_intervals` pins it.  None (default) takes the smallest bucket of
    `BUCKETS` (quarter-octave steps, <= 19% finer than needed) whose error is
    <= `tol`, separately for the tensor and the pair radial; bucketing keeps
    the table shape, hence the compiled step, across radial swaps.

    Error: the largest, over species pairs and the radial columns the A basis
    reads (`aspec_r`; all pair columns), of max|spline - exact| / max|exact| on
    10 points per interval.  For R_nl it includes the envelope (a function of
    x), for the pair radial it is envelope-free (its envelope is in r and
    multiplies both alike).  `tol` bounds VALUES.  The derivative error, which
    forces see, is O(h^3) where values are O(h^4): it is measured the same way
    on d/dx and reported (`return_info`), and gated too when `deriv_tol` is
    given.  At 1e-10 the lean energies agree with the full model to
    up to ~1e-9 relative and forces to up to ~2.3e-8 of max|F| on the
    benchmark models (docs/dev/learned-radial-splining.md).

    tol must be in [TOL_FLOOR, inf); it is floored at 10 eps of the model's
    dtype (a float32 table cannot do better).  Returns (model, max_rel_err), or
    with return_info=True (model, info): max_rel_err, max_rel_deriv_err,
    n_intervals {"rnl", "pair"} and the effective tol.  A spline or
    `spline_factorised` R_nl, and a spline pair radial, are kept as they are.
    The result is an ordinary full model (float64 tables cast to the model's
    dtype), trainable as a spline; it agrees with `model` to `tol`, not to
    roundoff.  Needs the full model (`require_full`).

    Cached (an LRU of at most CACHE_BYTES, `clear_cache()`) on the content of
    each radial's Wnlq, recursion, (R_nl) envelope and measured columns plus
    n_intervals and the tolerances, so converting a model whose radials are
    unchanged -- e.g. after a readout-only swap -- costs a hash, not a fit."""
    from .model import ACEModel
    if not isinstance(model, ACEModel):
        raise TypeError(f"to_spline needs an ACEModel, got {type(model).__name__}")
    if not (tol >= TOL_FLOOR):
        raise ValueError(f"to_spline: tol={tol!r} must be >= {TOL_FLOOR:g} (float64 roundoff "
                         "of the polynomial sum bounds what a table can reach)")
    if deriv_tol is not None and not (deriv_tol >= TOL_FLOOR):
        raise ValueError(f"to_spline: deriv_tol={deriv_tol!r} must be >= {TOL_FLOOR:g}")
    model.require_full("to_spline")
    kinds = ("spline", "spline_factorised", "analytic")
    if model.radial_kind not in kinds or model.pair_radial_kind not in ("spline", "analytic"):
        raise ValueError(f"to_spline: unsupported radial kinds {model.radial_kind!r}, "
                         f"{model.pair_radial_kind!r}")
    kw, errs, derrs, nints = {}, [0.0], [0.0], {}

    def floor(dt, t):
        return None if t is None else max(float(t), 10.0 * float(np.finfo(dt).eps))

    teff = None
    if model.radial_kind == "analytic" and "rnl" in radials:
        W = np.asarray(model.rnl_Wnlq, np.float64)
        polys = tuple(np.asarray(v, np.float64) for v in (model.polys_A, model.polys_B, model.polys_C))
        envp = np.asarray(model.rnl_envelope, np.float64)
        dt = model.rnl_Wnlq.dtype
        teff = floor(dt, tol)
        cols = np.unique(np.asarray(model.aspec_r))          # the columns A reads
        c, n_int, err, derr = _fit_cached(W, polys, n_intervals, teff, floor(dt, deriv_tol),
                                          envp, cols)
        kw.update(radial_kind="spline", rnl_coefs=jnp.asarray(c, dt),
                  rnl_Wnlq=jnp.zeros((1, 1, 1, 1), dt),
                  rnl_grid=(-1.0, 2.0 / n_int, n_int + 1))
        errs.append(err); derrs.append(derr); nints["rnl"] = n_int
    if model.pair_radial_kind == "analytic" and "pair" in radials:
        W = np.asarray(model.pair_Wnlq, np.float64)
        polys = tuple(np.asarray(v, np.float64)
                      for v in (model.pair_polys_A, model.pair_polys_B, model.pair_polys_C))
        dt = model.pair_Wnlq.dtype
        teff = floor(dt, tol)
        c, n_int, err, derr = _fit_cached(W, polys, n_intervals, teff, floor(dt, deriv_tol),
                                          None, None)
        kw.update(pair_radial_kind="spline", pair_coefs=jnp.asarray(c, dt),
                  pair_Wnlq=jnp.zeros((1, 1, 1, 1), dt),
                  pair_grid=(-1.0, 2.0 / n_int, n_int + 1))
        errs.append(err); derrs.append(derr); nints["pair"] = n_int
    # the column widths are unchanged, so matmul-form A selectors stay valid
    out = dataclasses.replace(model, **kw) if kw else model
    if return_info:
        return out, {"max_rel_err": max(errs), "max_rel_deriv_err": max(derrs),
                     "n_intervals": nints, "tol": teff}
    return out, max(errs)


# ------------------------------------------------------------------ radial tables in r
# Opt-in (`lean(..., radial_table=)`, ACECalculator, export_lammps): the whole
# per-edge radial stage -- ACE's R_nl with its Agnesi transform and envelope,
# and the pair radial; PACE's radial basis g_k -- as one cubic B-spline table
# per species pair on a uniform grid in r, so an edge costs a 4-row gather
# instead of the transcendentals (a CPU speed-up; docs in `radial_table`).
DEFAULT_RADIAL_TABLE = 4000       # intervals: errors ~1e-11..1e-9; speed flat from 1k to 16k
DEFAULT_TABLE_R_MIN = 0.5         # A: below it the end cubic extrapolates
RTAB_CHECK_PER_INTERVAL = 10      # error-check points per interval (knots + 9 interior)
RTAB_CHUNK = 1 << 14              # points per jitted sample call (one compiled shape)
RTAB_VANISH = 1e-12               # |R| beyond rc, relative to max|R|, that the mask may drop


def table_intervals(radial_table):
    """The `radial_table=` option as an interval count: None / False -> None
    (off), True -> DEFAULT_RADIAL_TABLE, a positive int -> itself."""
    if radial_table is None or radial_table is False:
        return None
    if radial_table is True:
        return DEFAULT_RADIAL_TABLE
    if not isinstance(radial_table, (int, np.integer)):
        raise TypeError(f"radial_table must be None, True or a number of intervals, "
                        f"got {radial_table!r}")
    if radial_table < 1:
        raise ValueError(f"radial_table must be >= 1 interval, got {radial_table!r}")
    return int(radial_table)


def _rtab_vd(m, which, r, zi, zj):
    """The exact radial `m.radial_table_values(which, ...)` and its r-derivative."""
    return jax.jvp(lambda q: m.radial_table_values(which, q, zi, zj), (r,), (jnp.ones_like(r),))


def _rtab_d2(m, which, r, zi, zj):
    return jax.jvp(lambda q: _rtab_vd(m, which, q, zi, zj)[1], (r,), (jnp.ones_like(r),))[1]


_rtab_vd_jit = eqx.filter_jit(_rtab_vd)
_rtab_d2_jit = eqx.filter_jit(_rtab_d2)


def _rtab_sample(m, which, r, zi, zj):
    """(values, d/dr) of the exact radial at r (n,) for one species pair, float64
    numpy, in RTAB_CHUNK calls of one compiled shape."""
    vs, ds = [], []
    dt = m.E0.dtype
    for lo in range(0, len(r), RTAB_CHUNK):
        x = r[lo:lo + RTAB_CHUNK]
        xp = np.concatenate([x, np.full(RTAB_CHUNK - len(x), x[-1])])
        z = jnp.zeros(RTAB_CHUNK, jnp.int32)
        v, d = _rtab_vd_jit(m, which, jnp.asarray(xp, dt), z + zi, z + zj)
        vs.append(np.asarray(v, np.float64)[:len(x)]); ds.append(np.asarray(d, np.float64)[:len(x)])
    return np.concatenate(vs), np.concatenate(ds)


def _rtab_fit(m, which, rc, r_min, r_max, n_int):
    """(coefs (NZ, NZ, n_int+3, F), max_rel_err, max_rel_deriv_err) of the table
    of `m.radial_table_values(which, ...)` on (r_min, h, n_int+1), h =
    (r_max - r_min) / n_int, clamped to its own h^2 f'' at both ends (autodiff;
    at r_max just inside, the left limit, as f'' may jump at a cutoff).  Errors
    as `_rel_err`: max over (zi, zj, column) of max|table - exact| / max|exact|,
    and the same for d/dr, on RTAB_CHECK_PER_INTERVAL points per interval, the
    table masked at rc[zi, zj] as the models evaluate it.  Raises when the exact
    radial does not vanish from rc on (beyond RTAB_VANISH relative): the mask
    would truncate it."""
    from ..basis.radial_ace1 import cubic_bspline_coefs
    nz = rc.shape[0]
    h = (r_max - r_min) / n_int
    C = RTAB_CHECK_PER_INTERVAL
    x = r_min + (h / C) * np.arange(C * n_int + 1)          # every C-th point is a knot
    x = np.concatenate([x, [r_max + 0.25, r_max + 1.0]])     # beyond the table: must be zero
    ends = jnp.asarray([r_min, r_max - 1e-6 * h], m.E0.dtype)
    out, worst, dworst = None, 0.0, 0.0
    for i in range(nz):
        for j in range(nz):
            v, d = _rtab_sample(m, which, x, i, j)
            beyond = x >= rc[i, j]
            # to roundoff: at rc the transform can land an ulp short of the envelope's zero
            if np.abs(v[beyond]).max(initial=0.0) > RTAB_VANISH * np.abs(v).max(initial=0.0):
                raise ValueError(
                    f"radial_table: the radial ({which}) of species pair ({i}, {j}) does not "
                    f"vanish at its cutoff rc = {rc[i, j]:g} (max |R| {np.abs(v[beyond]).max():.2e} "
                    "beyond it), so a table masked there would truncate it")
            if out is None:
                out = np.zeros((nz, nz, n_int + 3, v.shape[1]))
            y = v[:C * n_int + 1:C]
            if not np.any(y):
                continue                                     # a zero block stays exact zeros
            z = jnp.zeros(2, jnp.int32)
            e = np.asarray(_rtab_d2_jit(m, which, ends, z + i, z + j), np.float64) * h * h
            c = out[i, j] = cubic_bspline_coefs(y, (e[0], e[1]))
            live = ~beyond
            ev = np.where(live[:, None], _bspline_eval(x, c, n_int, x0=r_min, h=h), 0.0) - v
            ed = np.where(live[:, None], _bspline_eval(x, c, n_int, True, x0=r_min, h=h), 0.0) - d
            nrm, dnrm = np.abs(v).max(0), np.abs(d).max(0)
            ok, dok = nrm > 0, dnrm > 0
            if ok.any():
                worst = max(worst, float((np.abs(ev).max(0)[ok] / nrm[ok]).max()))
            if dok.any():
                dworst = max(dworst, float((np.abs(ed).max(0)[dok] / dnrm[dok]).max()))
    return out, worst, dworst


def radial_table(model, n_intervals=DEFAULT_RADIAL_TABLE, r_min=DEFAULT_TABLE_R_MIN,
                 return_info=False):
    """Tabulate an ACEModel's or PACEModel's per-edge radial stage in r.

    ACE: R_nl -- every column `radial()` returns, the Agnesi transform and the
    envelope folded in -- and the pair radial, one table `rtab_coefs` (NZ, NZ,
    n + 3, n_rnl + n_pair); with species-compact l-blocks (`block_dense`) also
    `blk_rtab_coefs` for the blocked dense path, [compact R | R_pair].  PACE:
    the radial basis g_k, `rtab_coefs` (NZ, NZ, n + 3, nradbase); the core
    repulsion stays analytic (tabulating it gained nothing).  `radial()`,
    `pair_radial()`, `_blocked_radial()` and `edge_basis_factors()` then read
    the table: a 4-row gather per edge instead of transcendentals (1.1-1.4x on
    one CPU core; docs/dev/benchmarks.md, CHANGELOG).

    One uniform grid in r, (x0, h, n) = (r_min, (r_max - r_min) / n_intervals,
    n_intervals + 1), r_max the largest per-pair cutoff (ACE: the pair
    envelope's rcut, `pad_cutoff`; PACE: the bond rcut), static like
    `rnl_grid`.  Samples come from the model's own exact JAX code at the knots,
    clamped to its own end curvature (autodiff), so no radial maths is
    duplicated here.  Evaluation masks with r < rc[zi, zj], so the radial is
    exactly zero from each pair's cutoff on (skin-list edges between rcut and
    rcut + skin contribute nothing); building refuses a radial that does not
    vanish there.  Below r_min the end cubic extrapolates: it is no longer the
    model there (the ACE pair radial is singular as r -> 0, which is why r_min
    exists).  0.5 A is far inside any physical first-neighbour distance.

    An approximation, not a restructuring: at 4000 intervals energies agree with
    the analytic model to ~1e-11 relative and forces to ~1e-9..1e-8 of max|F|
    on the fixtures and benchmark models (tests/test_radial_table.py).  For
    evaluation only: the basis and fitting paths need the exact radial (a lean
    ACE model is `energy_only` already).

    Returns the model with the table (an existing table is rebuilt), or with
    return_info=True (model, info): n_intervals, r_min, r_max and the largest
    relative value and r-derivative errors over species pairs and columns
    (`_rtab_fit`).  Cached in the `to_spline` LRU on the content of what the
    radial reads (`radial_table_key`), n_intervals and r_min: a readout-only
    swap costs a hash, not a fit."""
    from .model import ACEModel
    from .pace_model import PACEModel
    if not isinstance(model, (ACEModel, PACEModel)):
        raise TypeError(f"radial_table needs an ACEModel or a PACEModel, got {type(model).__name__}")
    n = table_intervals(n_intervals)
    if n is None:
        raise ValueError("radial_table: n_intervals must be a positive number of intervals")
    base = model.without_radial_table()
    rc = np.asarray(base.table_cutoffs(), np.float64)
    r_max, r_min = float(rc.max()), float(r_min)
    if not 0.0 < r_min < float(rc.min()):
        raise ValueError(f"radial_table: r_min = {r_min:g} must be in (0, {rc.min():g}), "
                         "the smallest pair cutoff")
    dt = base.E0.dtype
    kw, errs, derrs = {}, [0.0], [0.0]
    for which, field in base.radial_table_targets():
        key = _key("rtab", type(base).__name__, which, str(dt), *base.radial_table_key(), n, r_min)
        c, err, derr = _memo(key, lambda w=which: _rtab_fit(base, w, rc, r_min, r_max, n))
        kw[field] = jnp.asarray(c, dt)
        errs.append(err); derrs.append(derr)
    out = dataclasses.replace(base, rtab_grid=(r_min, (r_max - r_min) / n, n + 1), **kw)
    if return_info:
        return out, {"n_intervals": n, "r_min": r_min, "r_max": r_max,
                     "max_rel_err": max(errs), "max_rel_deriv_err": max(derrs)}
    return out
