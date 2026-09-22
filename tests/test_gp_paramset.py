import jax.numpy as jnp
import pytest
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


# --- Task 14: --sigma-type / --route CLI wiring (build_fit_paramset + parse_route) ---

def test_build_fit_paramset_default_is_all_lml_hypers():
    from ace_jax.fit.hypers import default_prior, to_array
    from ace_jax.fit.paramset import build_fit_paramset, from_hypers
    pr = default_prior(2.5)
    ps = build_fit_paramset(pr.mu, pr)
    # Default (no sigma_type, no route, no embed) == plain from_hypers: a single
    # all-LML hypers block, so run_map_ps reduces to the flat run_map (Task 3).
    assert [b.name for b in ps.blocks] == ["hypers"]
    assert ps.block("hypers").route == "lml"
    assert jnp.allclose(ps.lml_vector(), to_array(pr.mu))
    assert jnp.allclose(ps.lml_vector(), from_hypers(pr.mu, pr).lml_vector())


def test_sigma_type_flag_adds_lml_block_with_free_rows():
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.paramset import build_fit_paramset
    pr = default_prior(2.5)
    ps = build_fit_paramset(pr.mu, pr, n_types=3, sigma_type=True)
    st = ps.block("sigma_type")
    assert st.route == "lml"
    assert st.value.shape == (2, 3)          # (n_types - 1) free rows x (E, F, V)
    assert ps.n_types() == 3


def test_sigma_type_flag_off_leaves_no_block_even_with_types():
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.paramset import build_fit_paramset
    pr = default_prior(2.5)
    ps = build_fit_paramset(pr.mu, pr, n_types=3, sigma_type=False)
    assert all(b.name != "sigma_type" for b in ps.blocks)
    assert ps.n_types() == 1


def test_route_overrides_block_routes():
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.paramset import build_fit_paramset
    pr = default_prior(2.5)
    embed = jnp.ones((3, 4))
    ps = build_fit_paramset(pr.mu, pr, embed=embed, n_types=2, sigma_type=True,
                            route='{"embed": "varopt", "sigma_type": "lml"}')
    assert ps.block("embed").route == "varopt"      # frozen-by-default embed flipped on
    assert ps.block("sigma_type").route == "lml"
    assert ps.block("hypers").route == "lml"
    # a route naming an absent block is a harmless no-op on the set
    ps2 = build_fit_paramset(pr.mu, pr, route='{"embed": "fixed"}')
    assert [b.name for b in ps2.blocks] == ["hypers"]


def test_route_embed_to_lml_with_sigma_type_raises():
    # embed routed to "lml" while a sigma_type block is present would corrupt
    # _sigma_type_decode's [hypers | sigma_type free rows] layout -- must raise.
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.paramset import build_fit_paramset
    pr = default_prior(2.5)
    embed = jnp.ones((3, 4))
    with pytest.raises(ValueError):
        build_fit_paramset(pr.mu, pr, embed=embed, n_types=2, sigma_type=True,
                           route='{"embed": "lml"}')


def test_parse_route_parses_and_validates():
    from ace_jax.fit.paramset import parse_route
    assert parse_route(None) == {}
    assert parse_route("") == {}
    assert parse_route('{"sigma_type": "lml"}') == {"sigma_type": "lml"}
    assert parse_route({"embed": "fixed"}) == {"embed": "fixed"}   # dict passthrough
    with pytest.raises(ValueError):
        parse_route('{"embed": "bogus"}')                          # route not in ROUTES
    with pytest.raises(ValueError):
        parse_route('["not", "an", "object"]')                     # not a mapping
