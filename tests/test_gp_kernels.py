import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.fit.hypers import Hypers
from ace_jax.fit.kernels import K_MM, KernelSpec, delta, grad_k_rows, k_rows, kernel

THETA = Hypers(log_ell=np.log(0.7), log_A=np.log(0.2), log_alpha=np.log(1.0),
               log_r0=np.log(2.3), log_eps=np.log(0.3), log_rho=np.log(3.0),
               log_sigma_c=0.0, log_sigma_E=0.0, log_sigma_F=0.0, log_sigma_V=0.0)
D = 12


def _data(n, seed):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.normal(size=(n, D)))
    S = jnp.asarray(rng.uniform(1.8, 3.5, size=n))
    Z = jnp.asarray(rng.integers(0, 2, size=n), jnp.int32)
    return X, S, Z


@pytest.mark.parametrize("kind", ["cosine", "matern32"])
@pytest.mark.parametrize("bump", [True, False])
def test_kmm_symmetric_positive_definite(kind, bump):
    spec = KernelSpec(kind=kind, bump=bump, D=D)
    XM, SM, ZM = _data(40, 0)
    K = K_MM(THETA, spec, XM, SM, ZM, jitter=0.0)
    assert float(jnp.abs(K - K.T).max()) < 1e-12
    ev = jnp.linalg.eigvalsh(K)
    assert float(ev.min()) > -1e-10 * float(ev.max())


def test_species_delta_and_diagonal():
    spec = KernelSpec(kind="cosine", bump=True, D=D)
    X, S, Z = _data(5, 1)
    K = k_rows(THETA, spec, X, S, Z, X, S, Z)
    for i in range(5):
        assert abs(float(K[i, i]) - float(delta(S[i], THETA)) ** 2) < 1e-12
        for j in range(5):
            if int(Z[i]) != int(Z[j]):
                assert float(K[i, j]) == 0.0


def test_delta_vanishes_for_isolated_atom_with_finite_gradient():
    g = jax.grad(lambda s: delta(s, THETA))
    assert float(delta(jnp.asarray(1e30), THETA)) == 0.0
    assert np.isfinite(float(g(jnp.asarray(1e30))))
    assert float(delta(jnp.asarray(2.3), THETA)) > 0.0


@pytest.mark.parametrize("kind", ["cosine", "matern32"])
def test_grad_k_rows_matches_finite_difference(kind):
    spec = KernelSpec(kind=kind, bump=True, D=D)
    X, S, Z = _data(3, 2)
    XM, SM, ZM = _data(4, 3)
    Z = jnp.zeros(3, jnp.int32); ZM = jnp.zeros(4, jnp.int32)
    dKx, dKs = grad_k_rows(THETA, spec, X, S, Z, XM, SM, ZM)
    h = 1e-6
    for i in range(3):
        for m in range(4):
            for d in range(D):
                fd = (kernel(THETA, spec, X[i].at[d].add(h), S[i], Z[i], XM[m], SM[m], ZM[m])
                      - kernel(THETA, spec, X[i].at[d].add(-h), S[i], Z[i], XM[m], SM[m], ZM[m])) / (2 * h)
                assert abs(float(dKx[i, m, d]) - float(fd)) < 1e-6
            fd = (kernel(THETA, spec, X[i], S[i] + h, Z[i], XM[m], SM[m], ZM[m])
                  - kernel(THETA, spec, X[i], S[i] - h, Z[i], XM[m], SM[m], ZM[m])) / (2 * h)
            assert abs(float(dKs[i, m]) - float(fd)) < 1e-6


def test_gradient_defined_at_coincident_points():
    """Inducing points are drawn from the training environments, so x == xm occurs."""
    spec = KernelSpec(kind="matern32", bump=True, D=D)
    X, S, Z = _data(2, 4)
    dKx, dKs = grad_k_rows(THETA, spec, X, S, Z, X, S, Z)
    assert bool(jnp.all(jnp.isfinite(dKx))) and bool(jnp.all(jnp.isfinite(dKs)))


def test_unknown_kernel_kind_raises():
    X, S, Z = _data(2, 3)
    with pytest.raises(ValueError, match="unknown kernel kind"):
        kernel(THETA, KernelSpec(kind="rbf", D=D), X[0], S[0], Z[0], X[1], S[1], Z[1])
