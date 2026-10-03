"""The product basis AA = prod_k A[spec[:, k]] has a CPU form (explicit,
feature-major multiplies; docs/dev/cpu-gap-profile.md) and the default form
(`jnp.prod` over gathered factors, the layout the GPU was tuned for).  Both must
give the same values and the same reverse-mode gradients."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import edge_model as em


def _specs(rng, n_a):
    return tuple(jnp.asarray(rng.integers(0, n_a, (nv, o)), jnp.int32)
                 for o, nv in [(1, 7), (2, 23), (3, 31), (4, 17)])


@pytest.mark.parametrize("major", ["node", "feature"])
def test_cpu_form_matches_default(major):
    rng = np.random.default_rng(0)
    n, n_a = 13, 11
    specs = _specs(rng, n_a)
    X = jnp.asarray(rng.standard_normal((n, n_a) if major == "node" else (n_a, n)))
    cpu, ref = ((em._product_basis_cpu, em._product_basis_default) if major == "node" else
                (em._product_basis_t_cpu, em._product_basis_t_default))
    w = jnp.asarray(rng.standard_normal(ref(X, specs).shape))
    np.testing.assert_array_equal(cpu(X, specs), ref(X, specs))     # same multiply order
    g = lambda f: jax.grad(lambda X: jnp.sum(w * f(X, specs)))(X)
    np.testing.assert_allclose(g(cpu), g(ref), rtol=1e-13, atol=1e-14)


def test_cpu_selects_explicit_form():
    """On the CPU the compiled program is the explicit form: no conditional left over
    from the platform switch, and no reduce (jnp.prod's lowering)."""
    rng = np.random.default_rng(1)
    specs = _specs(rng, 11)
    A = jnp.asarray(rng.standard_normal((13, 11)))
    for f, X in [(em.product_basis, A), (em.product_basis_t, A.T)]:
        loss = lambda X, f=f: jnp.sum(f(X, specs))
        hlo = jax.jit(jax.grad(loss)).lower(X).compile().as_text()
        assert "conditional" not in hlo and " reduce(" not in hlo


def test_cpu_readout_matches_default():
    rng = np.random.default_rng(2)
    n, n_a = 13, 11
    specs = _specs(rng, n_a)
    A = jnp.asarray(rng.standard_normal((n, n_a)))
    C = jnp.asarray(rng.standard_normal((sum(int(s.shape[0]) for s in specs), n)))
    cpu, ref = em._product_basis_dot_cpu, em._product_basis_dot_default
    np.testing.assert_allclose(cpu(A, specs, C), ref(A, specs, C), rtol=1e-13, atol=1e-14)
    g = lambda f: jax.grad(lambda A, C: jnp.sum(f(A, specs, C) ** 2), argnums=(0, 1))(A, C)
    for a, b in zip(g(cpu), g(ref)):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(em.product_basis_dot(A, specs, C), ref(A, specs, C), rtol=1e-13, atol=1e-14)
