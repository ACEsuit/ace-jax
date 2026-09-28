"""Prototype: PACE's SBessel radial as one elementwise sin + a constant matrix
(docs/ace-vs-pace-gap.md §4.3).  Not in src/.

`pace_radial._sbessel` builds g_0..g_{K-1} by two chained recurrences (the
sin/cos rotation for sin(kx), then PACE's orthogonalisation g_n = (f_n +
a_n g_{n-1}) / b_n) and stacks the columns.  XLA fuses the stack into one
`loop_concatenate` kernel in which every output column re-evaluates its whole
recurrence chain, so the kernel's work grows ~K^2 per edge: on SiGe_medium
(K = nradbase = 13) it was the largest kernel of the PACE force call (1.76 ms
of 6.6 ms, A100).  The orthogonalisation is linear with constant coefficients,
so

    g = (S @ M) / rc^1.5,   S[:, k] = sinc((k+1) x), k = 0..K

with M (K+1, K) a constant matrix; S is one elementwise sin per entry.
"""
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from ace_jax.eval import pace_radial as pr
from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.eval.pace_model import PACEModel


def sbessel_matrix(K):
    """M (K+1, K): g_n = sum_k S_k M[k, n] (times rc^-1.5), S_k = sinc((k+1) x)."""
    Kf = max(K, 1)
    P = np.zeros((Kf + 1, Kf))                      # f_n = pre_n (S_n + S_{n+1})
    for n in range(Kf):
        pre = ((-1) ** n * math.sqrt(2) * math.pi * (n + 1) * (n + 2)
               / math.sqrt((n + 1) ** 2 + (n + 2) ** 2))
        P[n, n] += pre
        P[n + 1, n] += pre
    L = np.zeros((Kf, Kf))                           # g = f @ L
    L[0, 0] = 1.0
    d_prev = 1.0
    for n in range(1, Kf):
        en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = 1 - en / d_prev
        L[:, n] = (np.eye(Kf)[:, n] + math.sqrt(en / d_prev) * L[:, n - 1]) / math.sqrt(dn)
        d_prev = dn
    return (P @ L)[:, :K] if K >= 1 else (P @ L)[:, :1]


def sbessel_matmul(r, rc, K):
    x = r * math.pi / rc
    xs = jnp.where(x == 0, 1.0, x)
    k = jnp.arange(1, max(K, 1) + 2, dtype=r.dtype)
    kx = k * xs[..., None]
    S = jnp.where((x == 0)[..., None], 1.0, jnp.sin(kx) / kx)
    M = jnp.asarray(sbessel_matrix(K), r.dtype)
    g = jnp.matmul(S, M, precision=jax.lax.Precision.HIGHEST)
    return g / (rc ** 1.5)[..., None]


def radbase_fast(r, name, inner, lam, rc, dcut, cut_in, dcut_in, K):
    """`pace_radial.radbase` with the SBessel branch replaced."""
    if name != "SBessel":
        return pr.radbase(r, name, inner, lam, rc, dcut, cut_in, dcut_in, K)
    if inner == "zbl":
        one = jnp.ones_like(dcut_in)
        cut_in = jnp.where(dcut_in == 0, one, 0 * one)
    inside = (r > cut_in - dcut_in) & (r < rc)
    rs = jnp.where(inside, r, 0.5 * rc)
    g = sbessel_matmul(rs, rc, K)
    if inner in ("distance", "zbl"):
        g = g * (1.0 - pr.cutoff_func_poly(rs, cut_in, dcut_in))[..., None]
    return jnp.where(inside[..., None], g, 0.0)


class PACEFast(PACEModel):
    def edge_basis_factors(self, rij, zi, zj, mask=None):
        bp, valid, r, rij_s = self._geometry(rij, zi, zj, mask)
        lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
        g = radbase_fast(r, self.radbasename, self.inner_cutoff_type, lam, rc, dcut,
                         cin, dcin, self.nradbase)
        return jnp.where(valid[:, None], g, 0.0), real_spherical_harmonics(rij_s, self.lmax)


def make(variant, m):
    if variant in ("", "baseline"):
        return m
    if variant != "sbessel_mm":
        raise ValueError(variant)
    return PACEFast(**{f.name: getattr(m, f.name) for f in dataclasses.fields(m)})
