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
from collections import OrderedDict

import jax.numpy as jnp
import numpy as np

START_INTERVALS = 32
MAX_INTERVALS = 1 << 16
CHECK_PER_INTERVAL = 10

# Converted tables, keyed on the content of what determines them (Wnlq, the
# recursion, the R_nl envelope the error is measured with, n_intervals, tol),
# never on object identity: an ACECalculator re-leans on every `calc.model`
# swap, and a swap that changes only readout weights (ctilde, WB, Wpair, E0)
# must not re-spline, while any radial edit must.  A small LRU: a few models
# in flight at once, each entry one float64 table (a few MB).
CACHE_SIZE = 8
_CACHE = OrderedDict()


def clear_cache():
    _CACHE.clear()


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


def _fit_cached(W, polys, n_intervals, tol, env_params):
    """`_fit` through the LRU; env_params (NZ, NZ, 5) or None (pair radial)."""
    from .radial import env_poly2sx
    key = _key("rnl" if env_params is not None else "pair", W, *polys, env_params,
               None if n_intervals is None else int(n_intervals), float(tol))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    env_fn = None
    if env_params is not None:
        env_fn = lambda x: np.asarray(env_poly2sx(jnp.asarray(x)[None, None, :],   # noqa: E731
                                                  jnp.asarray(env_params)[:, :, None, :]))
    out = _fit(W, polys, n_intervals, tol, env_fn)
    _CACHE[key] = out
    while len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)
    return out


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


def _bspline_eval(x, c, n_int):
    """`radial.spline_eval` in numpy on the grid (x0, h, n) = (-1, 2/n_int, n_int+1);
    c (..., n_int+3, F) -> (..., n_x, F)."""
    h = 2.0 / n_int
    k = (x + 1.0) / h + 1.0
    ix = np.clip(np.floor(k).astype(int), 1, n_int)
    t = k - ix
    ct = 1.0 - t
    w = np.stack([ct ** 3, 4.0 - 6.0 * t ** 2 + 3.0 * t ** 3,
                  4.0 - 6.0 * ct ** 2 + 3.0 * ct ** 3, t ** 3], axis=-1) / 6.0   # (n_x, 4)
    g = c[..., ix[:, None] + np.arange(-1, 3), :]                                 # (..., n_x, 4, F)
    return np.einsum("xk,...xkf->...xf", w, g)


def _table(W, polys, n_int):
    """Spline coefficients (NZ, NZ, n_int+3, F) of S = W . P on n_int intervals."""
    from ..construct.radial_ace1 import cubic_bspline_coefs
    NZ, _, F, n_q = W.shape
    xs = np.linspace(-1.0, 1.0, n_int + 1)
    P, _, _ = _poly_d012(xs, *polys)
    _, _, D2 = _poly_d012(np.array([-1.0, 1.0]), *polys)
    h2 = (2.0 / n_int) ** 2
    S = np.einsum("xq,abnq->abxn", P, W)                       # nodal values
    E = np.einsum("eq,abnq->aben", D2, W) * h2                 # h^2 S'' at the two ends
    out = np.zeros((NZ, NZ, n_int + 3, F))
    for i in range(NZ):
        for j in range(NZ):
            if F and np.any(W[i, j]):                          # a zero block stays exact zeros
                out[i, j] = cubic_bspline_coefs(S[i, j], (E[i, j, 0], E[i, j, 1]))
    return out


def _rel_err(W, polys, c, n_int, env=None):
    """max over (zi, zj, column) of max_x |spline - exact| / max_x |exact|, with
    both multiplied by env (NZ, NZ, n_x) when given, on a check grid of
    CHECK_PER_INTERVAL points per interval (the knots and 9 interior points)."""
    x = np.linspace(-1.0, 1.0, CHECK_PER_INTERVAL * n_int + 1)
    P, _, _ = _poly_d012(x, *polys)
    worst = 0.0
    for i in range(W.shape[0]):                                # per pair: (n_x, F) at a time
        for j in range(W.shape[1]):
            if not W.shape[2] or not np.any(W[i, j]):
                continue
            exact = P @ W[i, j].T
            diff = _bspline_eval(x, c[i, j], n_int) - exact
            if env is not None:
                exact, diff = exact * env[i, j, :, None], diff * env[i, j, :, None]
            nrm = np.abs(exact).max(axis=0)
            live = nrm > 0
            if live.any():
                worst = max(worst, float((np.abs(diff).max(axis=0)[live] / nrm[live]).max()))
    return worst


def _try(W, polys, n_int, env_fn):
    c = _table(W, polys, n_int)
    env = None if env_fn is None else env_fn(np.linspace(-1.0, 1.0, CHECK_PER_INTERVAL * n_int + 1))
    return c, n_int, _rel_err(W, polys, c, n_int, env)


def _fit(W, polys, n_intervals, tol, env_fn):
    """(coefs, n_int, err): the given n_intervals, or doubled from
    START_INTERVALS until err <= tol.  The last doubling can overshoot tol by
    up to 16x (the error is O(h^4)), so one more trial at the n the h^4 rate
    predicts from the last failing grid is kept when it also meets tol: a
    smaller table for the same guarantee."""
    if n_intervals:
        return _try(W, polys, int(n_intervals), env_fn)
    lo, n_int = None, START_INTERVALS
    while True:
        c, n_int, err = _try(W, polys, n_int, env_fn)
        if err <= tol:
            break
        if n_int >= MAX_INTERVALS:
            raise ValueError(f"to_spline: relative error {err:.2e} > tol {tol:.1e} at {n_int} "
                             "intervals; loosen tol (float64 roundoff of the polynomial "
                             "sum bounds what a table can reach)")
        lo = (n_int, err)
        n_int *= 2
    if lo is not None:
        n_pred = int(np.ceil(1.1 * lo[0] * (lo[1] / tol) ** 0.25))
        if lo[0] < n_pred < n_int:
            trial = _try(W, polys, n_pred, env_fn)
            if trial[2] <= tol:
                return trial
    return c, n_int, err


def to_spline(model, n_intervals=None, tol=1e-10):
    """Convert an ACEModel's analytic radials to the spline branch.

    The tensor radial (`rnl_Wnlq`, radial_kind "analytic") and the pair radial
    (`pair_Wnlq`, pair_radial_kind "analytic") are each tabulated as
    S(x) = sum_q W P_q(x) on a uniform grid of `n_intervals` intervals over
    x in [-1, 1], (x0, h, n) = (-1, 2 / n_intervals, n_intervals + 1), and
    interpolated with the cubic B-spline `spline_eval` evaluates, clamped to
    the polynomial's own second derivative at the ends.  The envelope, the
    transform and every other array are unchanged.

    n_intervals None (default) doubles from 32 until the error is <= `tol`,
    separately for the tensor and the pair radial.  The error is the largest,
    over species pairs and radial columns, of max|spline - exact| / max|exact|
    on a grid of 10 points per interval: for R_nl including its envelope
    (env(x) is a function of x), for the pair radial envelope-free (its envelope
    is a function of r and multiplies both alike).

    Returns (model, max_rel_err): max_rel_err is the larger of the two
    radials' errors (0.0 when nothing was analytic).  A spline or
    `spline_factorised` R_nl, and a spline pair radial, are kept as they are.
    The result is an ordinary full model (float64 tables cast to the model's
    dtype), trainable as a spline; it agrees with `model` to `tol`, not to
    roundoff.  Needs the full model (`require_full`).

    Cached (`CACHE_SIZE`-entry LRU, `clear_cache()`) on the content of each
    radial's Wnlq, polynomial recursion and (R_nl) envelope plus n_intervals
    and tol, so converting a model whose radials are unchanged -- e.g. after
    a readout-only swap -- costs a hash, not a fit."""
    from .model import ACEModel
    if not isinstance(model, ACEModel):
        raise TypeError(f"to_spline needs an ACEModel, got {type(model).__name__}")
    model.require_full("to_spline")
    kinds = ("spline", "spline_factorised", "analytic")
    if model.radial_kind not in kinds or model.pair_radial_kind not in ("spline", "analytic"):
        raise ValueError(f"to_spline: unsupported radial kinds {model.radial_kind!r}, "
                         f"{model.pair_radial_kind!r}")
    kw, errs = {}, [0.0]
    if model.radial_kind == "analytic":
        W = np.asarray(model.rnl_Wnlq, np.float64)
        polys = tuple(np.asarray(v, np.float64) for v in (model.polys_A, model.polys_B, model.polys_C))
        envp = np.asarray(model.rnl_envelope, np.float64)
        c, n_int, err = _fit_cached(W, polys, n_intervals, tol, envp)
        dt = model.rnl_Wnlq.dtype
        kw.update(radial_kind="spline", rnl_coefs=jnp.asarray(c, dt),
                  rnl_Wnlq=jnp.zeros((1, 1, 1, 1), dt),
                  rnl_grid=(-1.0, 2.0 / n_int, n_int + 1))
        errs.append(err)
    if model.pair_radial_kind == "analytic":
        W = np.asarray(model.pair_Wnlq, np.float64)
        polys = tuple(np.asarray(v, np.float64)
                      for v in (model.pair_polys_A, model.pair_polys_B, model.pair_polys_C))
        c, n_int, err = _fit_cached(W, polys, n_intervals, tol, None)
        dt = model.pair_Wnlq.dtype
        kw.update(pair_radial_kind="spline", pair_coefs=jnp.asarray(c, dt),
                  pair_Wnlq=jnp.zeros((1, 1, 1, 1), dt),
                  pair_grid=(-1.0, 2.0 / n_int, n_int + 1))
        errs.append(err)
    if not kw:
        return model, 0.0
    # the column widths are unchanged, so matmul-form A selectors stay valid
    return dataclasses.replace(model, **kw), max(errs)
