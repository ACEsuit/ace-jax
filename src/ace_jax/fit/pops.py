"""Native POPS misspecification UQ for the linear (M=0) arm, on the design
whitened per quantity by w/sigma_q so E and F share one homoscedastic scale."""
import jax.numpy as jnp


def whiten(phi, resid, w, sigma):
    """Whiten rows of a design ``phi`` and residual ``resid`` by ``w/sigma``.

    ``phi`` is (n, D), ``resid`` and ``w`` are (n,), ``sigma`` is a scalar (or
    broadcastable to (n,)) per-quantity noise scale. Returns
    ``((w/sigma) * phi, (w/sigma) * resid)`` with the scale broadcast over
    phi's rows.
    """
    s = w / sigma
    return phi * s[:, None], resid * s
