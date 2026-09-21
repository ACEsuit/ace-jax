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
