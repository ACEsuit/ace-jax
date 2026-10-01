"""Block-structured L-BFGS for VarPro outer loops with several parameter blocks.

A VarPro outer objective f(blocks, *args, *statics) over parameter blocks of very
different scale (e.g. tensor radials and a second, nonlinear family such as density
or species-embedding weights) defeats a plain L-BFGS: one step size cannot suit both
blocks and the line search fails.  Two remedies, measured on the radial + sqrt-density
problem (docs/learn-radial-density-results.md, "Optimiser"):

* curvature-matched block scaling -- optimise u = [r_0 x_0 ; r_1 x_1 ; ...] with
  r_0 = 1 and r_b = sqrt(|h_b| / |h_0|), h_b the curvature along one random probe per
  block (a Hessian-vector product, forward over reverse), so a unit step in u moves
  every block by a curvature-matched amount; magnitudes because the outer objective
  is often non-convex along the probes; clamped at r_b >= r_min so a nearly flat
  block is never amplified more than 1 / r_min.  (The RMS-gradient ratio it replaced
  gave r ~ 3e-3 on Cantor and broke the line search.)  Probes are restricted to each
  block's live entries and, for blocks with a per-row scale gauge (rows = last axis,
  e.g. normalised radials), projected off each row's own direction, which is flat by
  construction and would bias the estimate low.
* fallback -- when a joint line search fails, the rest of that round is spent on
  one block at a time (alternating), instead of ending the run; the next round tries
  joint again.  mode="alternating" uses blocks throughout.

The scales are traced, so the whole run goes through radial_learn's single compiled
`_lbfgs_step` (one compile per block kind).  `f` and `statics` must be hashable
(e.g. a module-level function and a tuple of configs).
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import radial_learn

R_MIN = 0.1
MODES = ("joint", "alternating")


def _split(x, sizes):
    idx = np.cumsum([0] + list(sizes))
    return [x[idx[k]:idx[k + 1]] for k in range(len(sizes))]


def pack(blocks, r, sel=None):
    """(xb, other): the optimised variables and the frozen rest, each block scaled by r.
    sel=None packs every block (joint); an int packs that block alone."""
    flat = [ri * b.ravel() for ri, b in zip(r, blocks)]
    if sel is None:
        return jnp.concatenate(flat), jnp.zeros(0)
    rest = [x for k, x in enumerate(flat) if k != sel]
    return flat[sel], (jnp.concatenate(rest) if rest else jnp.zeros(0))


def unpack(xb, other, r, sel, shapes):
    """Inverse of `pack`: the tuple of blocks."""
    sizes = [math.prod(s) for s in shapes]
    if sel is None:
        parts = _split(xb, sizes)
    else:
        rest = _split(other, [s for k, s in enumerate(sizes) if k != sel])
        parts = rest[:sel] + [xb] + rest[sel:]
    return tuple((p / r[k]).reshape(shapes[k]) for k, p in enumerate(parts))


def _packed(xb, other, r, uargs, uf, sel, shapes, ustatics):
    """lbfgs_loop's f: the caller's objective on the unpacked blocks."""
    return uf(unpack(xb, other, r, sel, shapes), *uargs, *ustatics)


@partial(jax.jit, static_argnames=("uf", "shapes", "ustatics"))
def _hvp(x, v, r, uargs, *, uf, shapes, ustatics):
    g = lambda y: jax.grad(_packed)(y, jnp.zeros(0), r, uargs, uf, None, shapes, ustatics)
    return jax.jvp(g, (x,), (v,))[1]


def tangent_probe(X, live, key, row_gauge=False):
    """Random probe over the live entries of block X; with row_gauge, each row's
    (last axis) component along X itself is removed -- that direction is the row's
    scale gauge, flat by construction."""
    v = jax.random.normal(key, X.shape, X.dtype) * live
    if not row_gauge:
        return v
    xx = jnp.sum(X * X, -1, keepdims=True)
    return v - jnp.where(xx > 0, jnp.sum(v * X, -1, keepdims=True) / jnp.where(xx > 0, xx, 1.0), 0.0) * X


def curvature_scales(f, blocks, *, args=(), statics=(), live=None, row_gauge=None, key=None, r_min=R_MIN):
    """(r, h): per-block scales r (r_0 = 1, r_b = max(sqrt(|h_b| / |h_0|), r_min)) from
    one Hessian-vector product per block at r = 1, and the probe curvatures h.  A zero
    or non-finite curvature leaves that block's scale at 1."""
    n = len(blocks)
    live = live or [1.0] * n
    row_gauge = row_gauge or [False] * n
    key = jax.random.PRNGKey(0) if key is None else key
    shapes = tuple(tuple(b.shape) for b in blocks)
    one = jnp.ones(n)
    x, _ = pack(blocks, one)
    sizes = [math.prod(s) for s in shapes]
    off = np.cumsum([0] + sizes)
    h = []
    for k, kb in enumerate(jax.random.split(key, n)):
        vb = tangent_probe(blocks[k], live[k], kb, row_gauge[k]).ravel()
        v = jnp.zeros(x.shape[0]).at[off[k]:off[k + 1]].set(vb)
        Hv = _hvp(x, v, one, tuple(args), uf=f, shapes=shapes, ustatics=tuple(statics))
        h.append(float(vb @ Hv[off[k]:off[k + 1]]) / max(float(vb @ vb), 1e-300))
    r = [1.0]
    for hb in h[1:]:
        ok = np.isfinite(h[0]) and np.isfinite(hb) and h[0] != 0 and hb != 0
        r.append(max(float(np.sqrt(abs(hb) / abs(h[0]))), r_min) if ok else 1.0)
    return np.asarray(r), h


def _alternate(m, n):
    """m steps over n blocks, one block at a time, the remainder to the first blocks."""
    return [(k, m // n + (1 if k < m % n else 0)) for k in range(n)]


def block_lbfgs(f, blocks, *, steps, round_steps=10, mode="joint", args=(), statics=(), live=None,
                row_gauge=None, normalise=None, after_round=None, tol=1e-6, patience=3, seed=0,
                r_min=R_MIN, log=None):
    """Minimise f(blocks, *args, *statics) over a tuple of array blocks by L-BFGS in
    rounds of `round_steps`.  mode "joint": curvature-matched scaling (recomputed each
    round), falling back to one block at a time for the rest of a round when a line
    search fails; "alternating": one block at a time throughout, unscaled.
    `normalise(blocks) -> blocks` re-applies gauges after each round;
    `after_round(blocks, args) -> args` may refresh traced args (e.g. re-profile theta;
    shapes must not change, so nothing recompiles).  Stops at `steps`, or at a round
    in which no block used its full budget, or on a non-finite value.
    Returns (blocks, info): trace, reasons, round_lengths, precond, curvature,
    fallbacks, steps."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    blocks = tuple(jnp.asarray(b) for b in blocks)
    n, shapes = len(blocks), tuple(tuple(b.shape) for b in blocks)
    args = tuple(args)
    info = {"trace": [], "reasons": [], "round_lengths": [], "precond": [], "curvature": [], "fallbacks": 0,
            "steps": 0}
    done, rnd = 0, 0
    while done < int(steps):
        rnd += 1
        m = min(int(round_steps), int(steps) - done)
        if mode == "joint":
            r, h = curvature_scales(f, blocks, args=args, statics=statics, live=live, row_gauge=row_gauge,
                                    key=jax.random.PRNGKey(seed * 1000 + rnd), r_min=r_min)
            info["curvature"].append(h)
        else:
            r = np.ones(n)
        rj = jnp.asarray(r)
        info["precond"].append(r.tolist())
        plan = [(None, m)] if mode == "joint" else _alternate(m, n)
        reasons, obj, fell_back = [], float("nan"), False
        while plan:
            sel, nb = plan.pop(0)
            if nb == 0:
                continue
            xb, other = pack(blocks, rj, sel)
            xb, obj, trace, reason = radial_learn.lbfgs_loop(
                _packed, xb, steps=nb, tol=tol, patience=patience, args=(other, rj, args),
                statics=(f, sel, shapes, tuple(statics)))
            blocks = unpack(xb, other, rj, sel, shapes)
            info["trace"].extend(trace)
            info["round_lengths"].append(len(trace))
            reasons.append(reason)
            left = nb - len(trace) - 1
            if sel is None and reason == "linesearch" and left > 0:
                plan = _alternate(left, n)
                info["fallbacks"] += 1
                fell_back = True
        info["reasons"].append("+".join(reasons))
        done += m
        if normalise is not None:
            blocks = tuple(normalise(blocks))
        if after_round is not None:
            args = tuple(after_round(blocks, args))
        if log is not None:
            log(f"block_lbfgs: round {rnd} steps={done}/{int(steps)} obj={obj:.6e} reasons={reasons} "
                f"precond={[f'{x:.3g}' for x in r]}")
        if "nonfinite" in reasons or all(x != "steps" for x in (reasons[1:] if fell_back else reasons)):
            break
    info["steps"] = done
    return blocks, info
