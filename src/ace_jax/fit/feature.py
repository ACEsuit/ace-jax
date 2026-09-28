"""The residual GP's feature map phi_res(B): the coordinates the two-point
kernel factors kappa, psi act on (delta still uses the summary s).

Two modes, selected by the projection Pmap (D, d) and the warp:
  - isotropic (default): Pmap = diag(scale), warp "none"  ->  U = B * scale, d = D
    (the original kernel; residual_rows is bit-identical here).
  - density (Finnis-Sinclair): Pmap selects the ACE pair-density channels, warp
    "sqrt" -> the FS |.|^{1/2} embedding.  ACE pair-basis columns are signed
    (they are radial projections, not a nonnegative density), so a plain
    sqrt(relu(.)) would discard every negative component and has a kink + an
    infinite slope at 0.  Instead use the smooth SIGNED square root

        u = B / (B^2 + eps)^{1/4}

    which keeps the sign, is C^infinity (twice differentiable -> forces stay
    continuous and the derivative-DTC is well defined, Ruling R30), tends to
    sign(B) sqrt(|B|) for |B| >> sqrt(eps) (the FS sqrt-density scaling on both
    signs) and to B / eps^{1/4} (linear) near 0.  d = n_pair, so the
    derivative-DTC is affordable.
"""
import jax.numpy as jnp

EPS = 1e-6


def apply(X, Pmap, warp):
    """B-compact descriptors X (..., D) -> kernel coordinates U (..., d)."""
    return warp_u(X @ Pmap, warp)


def warp_u(U0, warp):
    """Projected coordinates U0 = X @ Pmap -> kernel coordinates U (the warp)."""
    return U0 * (U0 * U0 + EPS) ** -0.25 if warp == "sqrt" else U0


def dwarp(U0, warp):
    """d U / d U0, elementwise.  For u = U0 (U0^2 + eps)^{-1/4}:
    du/dU0 = (U0^2 + eps)^{-5/4} (U0^2/2 + eps)  (finite at U0 = 0)."""
    if warp == "sqrt":
        q = U0 * U0 + EPS
        return q ** -1.25 * (0.5 * U0 * U0 + EPS)
    return jnp.ones_like(U0)
