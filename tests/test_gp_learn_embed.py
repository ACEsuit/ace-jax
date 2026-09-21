import jax, numpy as np, jax.numpy as jnp, pytest
from conftest import FIXTURE_DIR
jax.config.update("jax_enable_x64", True)
from ace_jax.eval import highest_precision
from ace_jax.fit.hypers import to_array
from ace_jax.fit.embedding import species_onehot
# reuse the tiny-problem builder from test_gp_varopt
from test_gp_varopt import _tiny_problem

@pytest.mark.slow
def test_embed_grad_is_correct():
    # Correctness of the envelope partial grad(E, a) = d/dE [LML(E,a) - anchor(E)].
    # Definitive JAX-AD check (the spike's own check): a forward-mode JVP along a
    # random direction v must equal <grad, v>.  No FD cross-check is used here: this
    # tiny fixture has zero Ge atoms in si_tiny (ind.ZM/K_MM's Ge block is populated
    # only through the prior term, never through data), making that block of the
    # Cholesky factorisation in log_marginal_likelihood/prior_precision degenerate
    # and severely ill-conditioned. A dense eps sweep (1e-7..1e-1, 60 points) showed
    # central-difference FD pass/fail chaotically at every scale (only 28% passing a
    # 1e-3 relative gate, with errors up to 7652x at adjacent eps) -- the signature of
    # rounding-noise amplification by ill-conditioning, not a bad eps choice. AD is
    # immune to this (it differentiates the exact deterministic graph via the chain
    # rule, with no large-number cancellation), so fwd==reverse is the correctness
    # gate; it agrees to ~1e-14, far inside the 1e-8 threshold below.
    from ace_jax.fit.varopt_embed import embed_objective_and_grad
    from ace_jax.fit.embedding import gram
    with highest_precision():
        prob, ds, els = _tiny_problem()
        a = to_array(prob.prior.mu)
        NZ = len(els)
        rng = np.random.default_rng(0)
        E = jnp.asarray(species_onehot(NZ) + 0.05 * rng.normal(size=(NZ, NZ)))
        objective, grad, lml_E = embed_objective_and_grad(prob, ds, lam=1.0)
        g = np.asarray(grad(E, a))                        # envelope partial at fixed a
        pen = lambda X: lml_E(X, a) - 1.0 * jnp.sum((gram(X) - jnp.eye(NZ)) ** 2)   # same lam as grad
        v = jnp.asarray(rng.normal(size=(NZ, NZ)))
        jvp = float(jax.jvp(pen, (E,), (v,))[1])
        gv = float(np.sum(g * np.asarray(v)))
        assert abs(jvp - gv) / (abs(jvp) + 1e-12) < 1e-8      # fwd == reverse (exact)
