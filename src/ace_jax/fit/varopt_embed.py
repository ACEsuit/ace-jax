"""GP-kernel inner path for the outer VarOpt (species embedding E).  The linear
coefficients and GP weights are projected out inside the LML; theta is profiled by
the existing MAP; the outer E-gradient is the envelope partial d/dE at theta*:
∂LML/∂E - ∂anchor/∂E (no backprop through the inner MAP -- see the spec and
FINDINGS_embed_varopt_spike.md)."""
import jax
import jax.numpy as jnp
import numpy as np

from .embedding import anchor_penalty, normalize_rows
from .hypers import from_array, log_prior, to_array
from .objective import make_lml_embed
from .ladder import run_map


def theta_map_at(prob, ds, E, steps=300, seed=0):
    """Inner theta-MAP at a fixed embedding E: reuse the ladder MAP on the LML with
    ind.embed = normalize_rows(E).  Returns the theta-array a_star."""
    from .objective import make_lml
    ind_E = prob.ind._replace(embed=normalize_rows(E))
    lml = make_lml(prob._replace(ind=ind_E), ds)
    theta_star = run_map(lml, prob.prior, steps=steps, seed=seed)
    return to_array(theta_star)


def embed_objective_and_grad(prob, ds, lam):
    """Returns (objective, grad, lml_E).  objective(E, a) = profiled log-posterior
    minus anchor at a fixed theta-array a; grad(E, a) = envelope partial
    ∂LML/∂E|a - ∂anchor/∂E.  Both take the caller's theta* (from theta_map_at)."""
    lml_E = make_lml_embed(prob, ds)
    grad_lml = jax.jit(jax.grad(lml_E, argnums=0))
    grad_anchor = jax.jit(jax.grad(anchor_penalty, argnums=0))

    def objective(E, a):
        return (float(lml_E(E, a)) + float(log_prior(from_array(a), prob.prior))
                - float(anchor_penalty(E, lam)))

    def grad(E, a):
        return np.asarray(grad_lml(E, a)) - np.asarray(grad_anchor(E, lam))
    return objective, grad, lml_E


def learn_embedding(prob, ds, E0, *, lam=1.0, steps=30, lr=0.05, inner_steps=300,
                    val_score=None, seed=0):
    """Outer VarOpt over the embedding, profiled in theta.  Returns
    (E_star_normalized, info).  steps=0 -> return normalize_rows(E0) unchanged."""
    from .varopt import learn, select_by_holdout
    E0 = jnp.asarray(E0, jnp.float64)
    NZ = E0.shape[0]
    if steps <= 0:
        return normalize_rows(E0), {"steps": 0, "selected": "init", "trace": []}

    objective, grad, _ = embed_objective_and_grad(prob, ds, lam)

    # Profiled outer step: re-MAP theta at the current E, then one envelope step.
    def outer_objective(psi):
        E = psi.reshape(E0.shape)
        a = theta_map_at(prob, ds, E, steps=inner_steps, seed=seed)
        return objective(E, a)

    def outer_grad(psi):
        E = psi.reshape(E0.shape)
        a = theta_map_at(prob, ds, E, steps=inner_steps, seed=seed)
        return grad(E, a).reshape(-1)

    psi0 = E0.reshape(-1)
    psi_star, oinfo = learn(psi0, outer_objective, outer_grad, steps=steps, lr=lr)
    E_learned = normalize_rows(psi_star.reshape(E0.shape))

    info = {"steps": int(steps), "trace": oinfo["trace"], "selected": "learned"}
    if val_score is None:
        return E_learned, info
    # Held-out gate: keep the best of {learned, mace=E0, eye}; ties -> eye.
    eye = normalize_rows(jnp.eye(NZ))
    cands = {"eye": eye, "mace": normalize_rows(E0), "learned": E_learned}   # eye first -> tie winner
    label, E_sel = select_by_holdout(cands, val_score)
    info["selected"] = label
    return E_sel, info
