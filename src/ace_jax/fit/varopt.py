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
