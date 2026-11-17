"""L-BFGS with a zoom line search (optax), one compiled step reused across calls."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax


@partial(jax.jit, static_argnames=("f", "statics", "memory_size"))
def _lbfgs_step(x, state, args, *, f, statics, memory_size):
    """One optax L-BFGS + zoom-line-search step of f(y, *args, *statics).

    `f` and `statics` are static (baked into the compiled program; `f` must be
    a stable, hashable callable such as a module-level function, and `statics`
    a tuple of hashable objects), `x`, `state` and `args` are traced.  Reusing
    this single jitted step across rounds -- rather than re-jitting a fresh
    closure over the (theta, model, ...) each round -- means only a change in
    shape/dtype (never in value) of `x`/`state`/`args`, or in `f`/`statics`/
    `memory_size` themselves, triggers a recompile."""
    fa = lambda y: f(y, *args, *statics)
    opt = optax.lbfgs(memory_size=memory_size)
    value, grad = optax.value_and_grad_from_state(fa)(x, state=state)
    updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=fa)
    return optax.apply_updates(x, updates), state, value, grad


def lbfgs_loop(f, x0, *, steps, args=(), statics=(), tol=1e-6, patience=3, memory_size=10, info=None):
    """Minimise f(x, *args, *statics) -> scalar by optax L-BFGS with zoom line
    search, via the single compiled `_lbfgs_step` (no per-call closure is
    jitted here, so calling this repeatedly with the same `f`/`statics` and
    matching `x0`/`args` shapes -- e.g. once per VarPro round -- compiles
    once and reuses the executable thereafter).  `args` may change VALUE
    between calls (e.g. re-profiled theta) without forcing a recompile, since
    only their shape/dtype is baked into the trace; `f`, `statics` and
    `memory_size` must stay fixed (they are static, part of the cache key).

    Stops after `steps` iterations ("steps"); when the relative decrease stays
    below `tol` for `patience` consecutive iterations ("converged"); when an
    iteration fails to decrease f ("linesearch"); or when f, its gradient or
    the iterate go non-finite ("nonfinite").  Always returns the best finite
    iterate seen: (x_best, f_best, trace, reason), trace = f after each
    accepted step (strictly decreasing).  `info`, a dict if given, receives "f0" = f(x0) (for a
    progress report: the decrease over this call, at fixed `args`)."""
    if int(steps) <= 0:
        return x0, float(f(x0, *args, *statics)), [], "steps"
    opt = optax.lbfgs(memory_size=memory_size)
    # optax.lbfgs's init leaves a few ZoomLinesearchInfo fields weak-typed
    # (Python-literal defaults), while every _lbfgs_step call returns them
    # strongly typed; left alone, that mismatch is a second, permanent
    # abstract signature -- one extra (harmless but avoidable) compile the
    # first time this shape/dtype combination is ever seen.  Stripping it
    # up front means the very first iteration of the very first round
    # already matches every later call's signature: one compile, not two.
    state = jax.tree.map(lambda a: jax.lax.convert_element_type(a, a.dtype)
                         if hasattr(a, "dtype") else a, opt.init(x0))
    x = x0
    x_best, f_best, trace, prev, small = x0, None, [], None, 0
    for i in range(int(steps)):
        x_new, state, value, grad = _lbfgs_step(x, state, args, f=f, statics=statics,
                                                 memory_size=memory_size)
        value = float(value)               # f(x) at the pre-step x (f(x0) when i == 0)
        if i == 0:
            f_best = prev = value
            if info is not None:
                info["f0"] = value
        if not (np.isfinite(value) and bool(jnp.all(jnp.isfinite(grad)))):
            return x_best, f_best, trace, "nonfinite"
        fx = float(optax.tree_utils.tree_get(state, "value"))
        if not (np.isfinite(fx) and bool(jnp.all(jnp.isfinite(x_new)))):
            return x_best, f_best, trace, "nonfinite"
        if fx >= prev:                       # rejected step: not recorded, best kept
            return x_best, f_best, trace, "linesearch"
        x = x_new
        trace.append(fx)
        x_best, f_best = x, fx
        small = small + 1 if (prev - fx) <= tol * max(abs(prev), 1e-300) else 0
        prev = fx
        if small >= patience:
            return x_best, f_best, trace, "converged"
    return x_best, f_best, trace, "steps"
