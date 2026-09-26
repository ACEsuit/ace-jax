"""Model-level helpers for learnable tensor-basis radials (analytic branch).

The analytic tensor radial is

    R_n(r; zi, zj) = env(x) * sum_q Wnlq[zi, zj, n, q] P_q(x),   x = T_{zi,zj}(r),

so for a fixed transform, envelope and polynomial set it is LINEAR in Wnlq.
These helpers swap, widen and convert Wnlq, and build the fixed quadratic forms
the learner needs: the empirical radial Gram Q (gauge normalisation) and the
roughness matrix D2.  See docs/specs/2026-09-26-learned-radial-varpro-design.md.
"""
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..construct.radial_init import from_table, legendre_3term
from ..eval.radial import agnesi_normalized, env_poly2sx, poly_recursion, spline_eval
from .data import flat_edges


def require_analytic(model):
    if model.radial_kind != "analytic":
        raise ValueError(
            f"learned radials need the analytic radial branch, got radial_kind="
            f"{model.radial_kind!r}; convert with fit.radial_model.to_analytic(model, n_q)")


def with_radial(model, W):
    """`model` with its tensor-radial weights replaced by W (same shape)."""
    require_analytic(model)
    W = jnp.asarray(W)
    if W.shape != model.rnl_Wnlq.shape:
        raise ValueError(f"W shape {W.shape} != rnl_Wnlq shape {model.rnl_Wnlq.shape}")
    return eqx.tree_at(lambda m: m.rnl_Wnlq, model, W)


def _legendre(n_q):
    return tuple(jnp.asarray(a) for a in legendre_3term(n_q))


def widen_radial(model, n_q):
    """Extend the polynomial span to n_q normalized-Legendre polynomials by
    zero-padding Wnlq: every radial (hence every descriptor) is unchanged, but
    the learner can now reach degrees up to n_q - 1."""
    require_analytic(model)
    old = model.rnl_Wnlq.shape[-1]
    if n_q < old:
        raise ValueError(f"widen_radial: n_q={n_q} < current n_q={old}")
    A, B, C = _legendre(n_q)
    for name, ref in (("polys_A", A), ("polys_B", B), ("polys_C", C)):
        if not np.allclose(np.asarray(getattr(model, name)), np.asarray(ref[:old]),
                           rtol=1e-12, atol=1e-14):
            raise ValueError(f"widen_radial assumes normalized-Legendre polys; {name} differs")
    W = jnp.pad(model.rnl_Wnlq, ((0, 0), (0, 0), (0, 0), (0, n_q - old)))
    return dataclasses.replace(model, rnl_Wnlq=W, polys_A=A, polys_B=B, polys_C=C)


def poly_env(model, r, zi, zj):
    """(E, n_q): env(x) P_q(x) per edge; the tensor radials are
    einsum('eq,enq->en', poly_env, Wnlq[zi, zj])."""
    x = agnesi_normalized(r, model.rnl_transform[zi, zj])
    env = env_poly2sx(x, model.rnl_envelope[zi, zj])
    return env[:, None] * poly_recursion(x, model.polys_A, model.polys_B, model.polys_C)


def to_analytic(model, n_q, n_x=2001):
    """Convert a splined tensor radial to the analytic branch.  Each pair's
    spline S(x) (the envelope-free part) is sampled on a uniform x-grid, env*S
    is projected onto env*P_q(x) (radial_init.from_table), and the branch is
    swapped.  An analytic model is only widened.  Returns (model, relres) with
    relres (NZ, NZ, n_rnl) the per-radial relative projection residual."""
    if model.radial_kind == "analytic":
        wide = widen_radial(model, max(n_q, model.rnl_Wnlq.shape[-1]))
        return wide, np.zeros(model.rnl_Wnlq.shape[:3])
    if model.radial_kind != "spline":
        raise ValueError(f"to_analytic: unsupported radial_kind {model.radial_kind!r}")
    x = np.linspace(-1.0, 1.0, n_x)
    x0, h, n = model.rnl_grid
    NZ = model.rnl_coefs.shape[0]
    envp = np.asarray(model.rnl_envelope)
    env = np.asarray(env_poly2sx(jnp.asarray(x)[None, None, :],
                                 jnp.asarray(envp)[:, :, None, :]))             # (NZ, NZ, n_x)
    S = np.stack([np.stack([np.asarray(spline_eval(jnp.asarray(x), model.rnl_coefs[i, j], x0, h, n))
                            for j in range(NZ)]) for i in range(NZ)])         # (NZ, NZ, n_x, n_rnl)
    polys = legendre_3term(n_q)
    W, rel = from_table(x, env[..., None] * S, envp, polys)
    A, B, C = (jnp.asarray(a) for a in polys)
    out = dataclasses.replace(model, radial_kind="analytic", rnl_Wnlq=jnp.asarray(W),
                              polys_A=A, polys_B=B, polys_C=C)
    return out, rel
