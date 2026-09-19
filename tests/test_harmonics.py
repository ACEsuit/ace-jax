"""Pure-JAX spherical harmonics vs sphericart, and the no-custom-call guarantee.

sphericart is a dev dependency here: it validates the recursion but must not
appear in the model path, or the exported StableHLO carries an FFI custom call
target that lammps-jax would have to resolve at run time.
"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval.harmonics import real_spherical_harmonics


@pytest.mark.parametrize("L", range(7))
def test_matches_sphericart(L):
    scj = pytest.importorskip("sphericart.jax")
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 3)) * rng.uniform(0.2, 5.0, (500, 1))
    a = np.asarray(real_spherical_harmonics(jnp.asarray(v), L))
    b = np.asarray(scj.spherical_harmonics(jnp.asarray(v), L))
    rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
    assert rel < 1e-13, f"L={L} rel {rel:.3e}"


def test_on_axis_is_finite():
    """The z axis is where a naive cos(m phi) formulation blows up."""
    v = jnp.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    Y = np.asarray(real_spherical_harmonics(v, 6))
    assert np.all(np.isfinite(Y))


def test_gradient_is_finite_on_axis():
    v = jnp.asarray([[0.0, 0.0, 1.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    g = jax.grad(lambda a: jnp.sum(real_spherical_harmonics(a, 5) ** 2))(v)
    assert np.all(np.isfinite(np.asarray(g)))


def test_no_custom_call_in_lowered_hlo():
    """The whole point: stock HLO, no FFI target for LAMMPS to resolve."""
    import re
    f = jax.jit(lambda x: real_spherical_harmonics(x, 4))
    hlo = f.lower(jnp.ones((8, 3))).compile().as_text()
    targets = sorted(set(re.findall(r'custom_call_target\s*=\s*"([^"]+)"', hlo)))
    assert targets == [], f"unexpected custom call targets: {targets}"
