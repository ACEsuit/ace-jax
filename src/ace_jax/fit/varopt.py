"""Objective-agnostic outer VarOpt driver.  The inner path (linear coefficients,
and here also theta, profiled/projected out) supplies objective(psi) and its
outer gradient grad(psi); this drives psi by Adam ascent and picks between
candidate solutions on a held-out score.  See docs/superpowers/specs/
2026-09-21-learned-species-embedding-varopt-design.md."""
import jax.numpy as jnp
import numpy as np
import optax


def learn(psi0, objective, grad, *, steps=30, lr=0.05):
    """Maximize objective(psi) by Adam using grad(psi) (ascent = descend on -grad).
    Returns (psi_star, {"trace": [objective per step]})."""
    opt = optax.adam(lr)
    state = opt.init(psi0)
    psi = psi0
    trace = [float(objective(psi))]                     # objective per step
    for _ in range(int(steps)):
        g = jnp.asarray(grad(psi))
        updates, state = opt.update(-g, state)           # ascent
        psi = optax.apply_updates(psi, updates)
        trace.append(float(objective(psi)))
    return psi, {"trace": trace}


def select_by_holdout(candidates, score):
    """candidates: dict label->psi; score: psi->float (lower is better). Returns
    (best_label, best_psi); ties resolve to insertion order."""
    best = None
    for label, psi in candidates.items():
        s = float(score(psi))
        if best is None or s < best[2]:
            best = (label, psi, s)
    return best[0], best[1]


def varopt_ps(ps, objective_and_grad, *, steps=30, lr=0.05, val_score=None):
    """Outer VarOpt over `ps.varopt_vector()` only -- LML/fixed blocks are untouched.
    `objective_and_grad(x) -> (value, grad)` closes over the inner MAP (re-materialising
    ps at each step is the caller's job, inside its own closure). No-op (returns ps
    unchanged) when there are no VarOpt blocks or steps==0. With val_score, applies the
    held-out gate between {"learned": x_star, "init": x0} (ties -> insertion order, so
    "learned" wins a tie). Returns the updated ParamSet."""
    x0 = ps.varopt_vector()
    if x0.size == 0 or steps == 0:
        return ps
    obj = lambda x: objective_and_grad(x)[0]
    grad = lambda x: objective_and_grad(x)[1]
    x_star, _info = learn(x0, obj, grad, steps=steps, lr=lr)
    if val_score is None:
        return ps.set_varopt_vector(x_star)
    cands = {"learned": x_star, "init": x0}
    _label, x_sel = select_by_holdout(cands, val_score)
    return ps.set_varopt_vector(x_sel)
