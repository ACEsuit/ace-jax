import jax.numpy as jnp
from ace_jax.fit.paramset import ParamBlock, ParamSet

def _ps():
    return ParamSet((
        ParamBlock("a", jnp.array([1.0, 2.0]), "lml"),
        ParamBlock("E", jnp.zeros((2, 2)), "varopt"),
        ParamBlock("k", jnp.array([9.0]), "fixed"),
    ))

def test_lml_vector_roundtrip_only_lml_blocks():
    ps = _ps()
    assert ps.lml_vector().shape == (2,)                 # only "a"
    ps2 = ps.set_lml_vector(jnp.array([3.0, 4.0]))
    assert jnp.allclose(ps2.block("a").value, jnp.array([3.0, 4.0]))
    assert jnp.allclose(ps2.block("k").value, jnp.array([9.0]))   # fixed untouched
    assert jnp.allclose(ps2.block("E").value, jnp.zeros((2, 2)))  # varopt untouched

def test_varopt_vector_roundtrip_flattens_matrix():
    ps = _ps()
    assert ps.varopt_vector().shape == (4,)              # 2x2 flattened
    ps2 = ps.set_varopt_vector(jnp.arange(4.0))
    assert jnp.allclose(ps2.block("E").value, jnp.arange(4.0).reshape(2, 2))
    assert jnp.allclose(ps2.block("a").value, jnp.array([1.0, 2.0]))  # lml untouched

def test_materialise_roundtrips_hypers():
    from ace_jax.fit.hypers import default_prior, to_array, from_array
    from ace_jax.fit.paramset import from_hypers
    pr = default_prior(2.5); h = from_array(to_array(pr.mu))
    ps = from_hypers(h, pr)
    h2, embed = ps.materialise()
    assert embed is None
    assert jnp.allclose(to_array(h2), to_array(h))
    # optimiser sees exactly the 10 hyper values as the lml vector
    assert jnp.allclose(ps.lml_vector(), to_array(h))

def test_varopt_ps_noop_when_no_varopt_blocks(tiny_linear_problem):
    from ace_jax.fit.paramset import from_hypers
    from ace_jax.fit.varopt import varopt_ps
    prob, ds = tiny_linear_problem
    ps = from_hypers(prob.prior.mu, prob.prior)              # no varopt blocks
    out = varopt_ps(ps, lambda x: (0.0, x * 0.0), steps=3)
    assert jnp.allclose(out.varopt_vector(), jnp.zeros((0,)))  # nothing to do

def test_varopt_ps_moves_toward_objective_optimum():
    from ace_jax.fit.varopt import varopt_ps
    ps = _ps()                                                # has varopt block "E" (2x2, zeros)
    target = jnp.array([1.0, 2.0, 3.0, 4.0])
    def objective_and_grad(x):
        diff = x - target
        return float(-jnp.sum(diff ** 2)), -2 * diff          # ascent target: maximize -||x-target||^2
    out = varopt_ps(ps, objective_and_grad, steps=300, lr=0.1)
    assert jnp.allclose(out.varopt_vector(), target, atol=1e-2)
    assert jnp.allclose(out.block("a").value, jnp.array([1.0, 2.0]))  # lml block untouched
    assert jnp.allclose(out.block("k").value, jnp.array([9.0]))       # fixed block untouched

def test_varopt_ps_holdout_gate_can_reject_learned_for_init():
    from ace_jax.fit.varopt import varopt_ps
    ps = _ps()                                                # varopt block "E" init is all zeros
    objective_and_grad = lambda x: (0.0, jnp.ones_like(x))    # pushes learned x away from zero
    val_score = lambda x: float(jnp.sum(x ** 2))               # smaller norm is better -> init wins
    out = varopt_ps(ps, objective_and_grad, steps=5, lr=0.1, val_score=val_score)
    assert jnp.allclose(out.varopt_vector(), jnp.zeros((4,)))  # gate rejected the learned point
