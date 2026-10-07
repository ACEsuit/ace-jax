"""The square root R0 of the ARD prior precision, Lambda = R0^T diag(lambda) R0.

R0 = blockdiag(diag(Gamma), chol(K_MM)^T): diagonal on the L linear columns (the readout's smoothness
prior Gamma, then the joint-E0 columns' sqrt(e0_prec): ard.ard_gamma), upper triangular on the M inducing
columns of a GP-arm fit (K_MM at theta_MAP).  The ARD posterior works in the prior-scaled system
S = R0^-T A R0^-1, where the R0 terms of log|A| and log|Lambda| cancel for any invertible R0
(docs/dev/gp-discrepancy-exploration.md, 0.1).  M = 0 is the diagonal D = diag(Gamma) of the linear arm:
every method then evaluates today's dinv expression verbatim, so --uq ard stays bit-identical."""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular


class PriorRoot(NamedTuple):
    dinv: np.ndarray                 # (L,) 1 / Gamma on the linear columns
    U: object = None                 # (M, M) upper triangular chol(K_MM)^T, or None (M = 0)

    @staticmethod
    def diag(dinv):
        return PriorRoot(np.asarray(dinv, np.float64), None)

    @property
    def M(self):
        return 0 if self.U is None else int(self.U.shape[0])

    @property
    def width(self):
        return len(self.dinv) + self.M

    def _split(self, a, axis):
        L = len(self.dinv)
        a = jnp.moveaxis(jnp.asarray(a), axis, 0)
        return jnp.moveaxis(a[:L], 0, axis), jnp.moveaxis(a[L:], 0, axis)

    def rows(self, X):
        """X R0^-1 on the last axis."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return X * d
        Xl, Xg = self._split(X, -1)
        sh = Xg.shape
        g = solve_triangular(self.U, Xg.reshape(-1, self.M).T, lower=False, trans="T").T.reshape(sh)
        return jnp.concatenate([Xl * d, g], -1)

    def dual(self, G):
        """R0^-T G on the first axis (G (Dt,) or (Dt, k))."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return d * G if G.ndim == 1 else d[:, None] * G
        Gl, Gg = self._split(G, 0)
        lin = d * Gl if Gl.ndim == 1 else d[:, None] * Gl
        return jnp.concatenate([lin, solve_triangular(self.U, Gg, lower=False, trans="T")], 0)

    def primal(self, x):
        """R0^-1 x on the first axis (the mean coefficients from the scaled solution)."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return d * x
        xl, xg = self._split(x, 0)
        return jnp.concatenate([d * xl if xl.ndim == 1 else d[:, None] * xl,
                                solve_triangular(self.U, xg, lower=False)], 0)

    def lift(self, z):
        """R0^T z on the first axis (press_scores' push-through: g~ = R0^T L z)."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return z / d if z.ndim == 1 else z / d[:, None]
        zl, zg = self._split(z, 0)
        return jnp.concatenate([zl / d if zl.ndim == 1 else zl / d[:, None], self.U.T @ zg], 0)

    def gram(self, Mx):
        """R0^-T Mx R0^-1."""
        d = jnp.asarray(self.dinv)
        if self.M == 0:
            return d[:, None] * Mx * d[None, :]
        return self.rows(self.dual(Mx))


def prior_root(prob, theta):
    """The ARD prior root of prob at theta: Gamma (ard_gamma) on the linear columns, chol(K_MM(theta))^T on
    the M inducing columns (None when M = 0)."""
    from .ard import ard_gamma
    from .kernels import K_MM
    dinv = 1.0 / ard_gamma(prob)
    if prob.ind.XM.shape[0] == 0:
        return PriorRoot.diag(dinv)
    ind = prob.ind
    K = K_MM(theta, prob.spec, ind.XM, ind.SM, ind.ZM, ind.embed)
    return PriorRoot(dinv, jnp.linalg.cholesky(K).T)
