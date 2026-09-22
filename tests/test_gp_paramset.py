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
