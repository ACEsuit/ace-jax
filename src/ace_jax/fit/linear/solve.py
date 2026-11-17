"""Matrix-free least squares."""
import jax
import jax.numpy as jnp


def lsqr(matvec, rmatvec, b, n, maxiter=200):
    """Least-squares min ||A x - b|| via LSQR.  matvec: R^n -> R^m (A v),
    rmatvec: R^m -> R^n (A^T u).  Returns x in R^n."""
    beta = jnp.linalg.norm(b)
    u = b / jnp.where(beta > 0, beta, 1.0)
    v0 = rmatvec(u)
    alpha = jnp.linalg.norm(v0)
    v = v0 / jnp.where(alpha > 0, alpha, 1.0)
    w = v
    x = jnp.zeros(n)
    phibar, rhobar = beta, alpha

    def step(carry, _):
        u, v, w, x, phibar, rhobar, alpha = carry
        u2 = matvec(v) - alpha * u
        beta = jnp.linalg.norm(u2)
        u2 = u2 / jnp.where(beta > 0, beta, 1.0)
        v2 = rmatvec(u2) - beta * v
        alpha2 = jnp.linalg.norm(v2)
        v2 = v2 / jnp.where(alpha2 > 0, alpha2, 1.0)
        rho = jnp.sqrt(rhobar ** 2 + beta ** 2)
        c, sn = rhobar / rho, beta / rho
        theta = sn * alpha2
        rhobar2 = -c * alpha2
        phi = c * phibar
        phibar2 = sn * phibar
        x = x + (phi / rho) * w
        w = v2 - (theta / rho) * w
        return (u2, v2, w, x, phibar2, rhobar2, alpha2), None

    return jax.lax.scan(step, (u, v, w, x, phibar, rhobar, alpha), None, length=maxiter)[0][3]
