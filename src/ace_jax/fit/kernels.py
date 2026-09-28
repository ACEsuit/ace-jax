"""k(x, x') = delta(s) delta(s') * kappa(x, x') * psi(|x - x'| / rho) * (e(z).e(z')).

delta is ONE-point (a property of a single environment: its non-stationary
prior amplitude); kappa and psi are TWO-point (descriptor-space distances).
x are compact site descriptors already multiplied by the fixed per-component
scale (inducing.py).  See the spec, "Kernel family", for why psi is a smooth
squared exponential and not a compact-support function."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp

_TINY = 1e-30   # inside sqrt: |x - x'| has an undefined gradient at 0, and x == x' occurs


@dataclass(frozen=True)
class KernelSpec:
    kind: str = "cosine"     # "cosine" | "matern32"
    bump: bool = True        # multiply by the SE magnitude factor psi_se
    D: int = 1               # descriptor width, informational (the kernels read x.shape[-1])
    # lower floor on the delta(s) coordinate: delta is evaluated at max(s, s_floor), so
    # the learned amplitude cannot extrapolate to zero below the training range of s
    # (compressed environments), where the GP's prior variance is its OOD uncertainty.
    # A lower floor only: isolated/padding sites (s ~ 1e3 r0) keep delta = 0.
    s_floor: float | None = None


def _sqdist(x, xm):
    return jnp.sum((x - xm) ** 2)


def kappa_cosine(x, xm, ell):
    """Squared-exponential on the unit sphere: exp(-|x^ - xm^|^2 / 2 ell^2).
    Depends on theta only through a scalar function of the theta-independent
    inner product, and is positive definite for any sign of x . xm.
    d x^/dx carries a 1/sqrt(_TINY) ~ 1e15 factor at x = 0; every zero
    descriptor here (isolated atom, padding) also has delta(s) = 0, which
    annihilates it."""
    xh = x / jnp.sqrt(_sqdist(x, 0.0) + _TINY)
    xmh = xm / jnp.sqrt(_sqdist(xm, 0.0) + _TINY)
    return jnp.exp(-_sqdist(xh, xmh) / (2.0 * ell * ell))


def _rms_dist(x, xm):
    """RMS per-component distance: after 1/std scaling raw distances are
    ~sqrt(2D), so dividing by sqrt(D) keeps ell and rho O(1) for any basis."""
    return jnp.sqrt(_sqdist(x, xm) / x.shape[-1] + _TINY)


def kappa_matern32(x, xm, ell):
    u = jnp.sqrt(3.0) * _rms_dist(x, xm) / ell
    return (1.0 + u) * jnp.exp(-u)


def delta(s, theta):
    """A exp[-alpha u] / (1 + exp[-(alpha+eps) u]), u = s - r0, in the
    log-domain form A exp(-alpha u - softplus(-(alpha+eps) u)) so that neither
    limit of u overflows and the gradient stays finite at the padding value
    s ~ 1e3 r0."""
    A, alpha, r0, eps = (jnp.exp(theta.log_A), jnp.exp(theta.log_alpha),
                         jnp.exp(theta.log_r0), jnp.exp(theta.log_eps))
    u = s - r0
    return A * jnp.exp(-alpha * u - jax.nn.softplus(-(alpha + eps) * u))


def psi_se(x, xm, rho):
    """Smooth magnitude factor exp(-d^2 / 2 rho^2) on the RMS per-component
    distance.  Deliberately NOT compactly supported: Askey/Wendland factors
    have a cusp at d = 0 (gradient a unit vector in an arbitrary direction),
    are therefore not mean-square differentiable, and inducing points sit
    exactly on training environments -- force rows would be ill-defined."""
    return jnp.exp(-(_sqdist(x, xm) / x.shape[-1]) / (2.0 * rho * rho))


def kernel(theta, spec, x, s, z, xm, sm, zm, embed):
    ell = jnp.exp(theta.log_ell)
    if spec.kind == "cosine":
        kap = kappa_cosine(x, xm, ell)
    elif spec.kind == "matern32":
        kap = kappa_matern32(x, xm, ell)
    else:
        raise ValueError(f"unknown kernel kind {spec.kind!r}")
    if spec.s_floor is not None:
        s, sm = jnp.maximum(s, spec.s_floor), jnp.maximum(sm, spec.s_floor)
    k = delta(s, theta) * delta(sm, theta) * kap
    if spec.bump:
        k = k * psi_se(x, xm, jnp.exp(theta.log_rho))
    return k * jnp.dot(embed[z], embed[zm])      # coregionalization; eye(NZ) -> (z==zm)


def k_rows(theta, spec, X, S, Z, XM, SM, ZM, embed):
    f = lambda x, s, z: jax.vmap(lambda xm, sm, zm: kernel(theta, spec, x, s, z, xm, sm, zm, embed))(XM, SM, ZM)
    return jax.vmap(f)(X, S, Z)                                            # (N, M)


def grad_k_rows(theta, spec, X, S, Z, XM, SM, ZM, embed):
    g = jax.grad(kernel, argnums=(2, 3))         # w.r.t. (x, s); embed is a frozen constant
    f = lambda x, s, z: jax.vmap(lambda xm, sm, zm: g(theta, spec, x, s, z, xm, sm, zm, embed))(XM, SM, ZM)
    dKx, dKs = jax.vmap(f)(X, S, Z)                                       # (N, M, D), (N, M)
    return dKx, dKs


def K_MM(theta, spec, XM, SM, ZM, embed, jitter=1e-8):
    K = k_rows(theta, spec, XM, SM, ZM, XM, SM, ZM, embed)
    K = 0.5 * (K + K.T)
    return K + jitter * jnp.mean(jnp.diag(K)) * jnp.eye(K.shape[0])
