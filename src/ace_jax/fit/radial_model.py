"""Model-level helpers for learnable tensor-basis radials (analytic branch).

The analytic tensor radial is

    R_n(r; zi, zj) = env(x) * sum_q Wnlq[zi, zj, n, q] P_q(x),   x = T_{zi,zj}(r),

so for a fixed transform, envelope and polynomial set it is LINEAR in Wnlq.
These helpers swap, widen and convert Wnlq, and build the fixed quadratic forms
the learner needs: the empirical radial Gram Q (gauge normalisation) and the
roughness matrix D2.  See docs/dev/specs/2026-09-26-learned-radial-varpro-design.md.
"""
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..basis.radial_init import from_table, legendre_3term
from ..eval.radial import env_poly2sx, poly_recursion, spline_eval
from ..eval.splinify import to_spline  # noqa: F401  (to_analytic's inverse, for deployment)
from .data import flat_edges


def require_analytic(model):
    model.require_full("editing the radial")
    if model.radial_kind != "analytic":
        raise ValueError(
            f"learned radials need the analytic radial branch, got radial_kind="
            f"{model.radial_kind!r}; convert with fit.radial_model.to_analytic(model, n_q)")


def with_radial(model, W, learned=True):
    """`model` with its tensor-radial weights replaced by W (same shape),
    marked `radial_learned` (the learner's weights; `lean`'s default splines
    those).  learned=False for weights that are not learned."""
    require_analytic(model)
    W = jnp.asarray(W)
    if W.shape != model.rnl_Wnlq.shape:
        raise ValueError(f"W shape {W.shape} != rnl_Wnlq shape {model.rnl_Wnlq.shape}")
    out = eqx.tree_at(lambda m: m.rnl_Wnlq, model, W)
    return out if out.radial_learned == learned else dataclasses.replace(out, radial_learned=learned)


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
    if model.rnl_basis == "sbessel":          # g_k for k < old are unchanged by asking for more
        W = jnp.pad(model.rnl_Wnlq, ((0, 0), (0, 0), (0, 0), (0, n_q - old)))
        return dataclasses.replace(model, rnl_Wnlq=W)
    A, B, C = _legendre(n_q)
    for name, ref in (("polys_A", A), ("polys_B", B), ("polys_C", C)):
        if not np.allclose(np.asarray(getattr(model, name)), np.asarray(ref[:old]),
                           rtol=1e-12, atol=1e-14):
            raise ValueError(f"widen_radial assumes normalized-Legendre polys; {name} differs")
    W = jnp.pad(model.rnl_Wnlq, ((0, 0), (0, 0), (0, 0), (0, n_q - old)))
    return dataclasses.replace(model, rnl_Wnlq=W, polys_A=A, polys_B=B, polys_C=C)


def poly_env(model, r, zi, zj):
    """(E, n_q): the tensor radials' basis per edge (env(x) P_q(x), or g_k(r) for rnl_basis
    "sbessel": `ACEModel.rnl_basis_values`); the radials are
    einsum('eq,enq->en', poly_env, Wnlq[zi, zj])."""
    return model.rnl_basis_values(r, zi, zj)


def to_analytic(model, n_q, n_x=2001):
    """Convert a splined tensor radial to the analytic branch.  Each pair's
    spline S(x) (the envelope-free part) is sampled on a uniform x-grid, env*S
    is projected onto env*P_q(x) (radial_init.from_table), and the branch is
    swapped.  An analytic model is only widened.  Returns (model, relres) with
    relres (NZ, NZ, n_rnl) the per-radial relative projection residual."""
    model.require_full("to_analytic")
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
    # a projection of an exported table: not learned
    out = dataclasses.replace(model, radial_kind="analytic", rnl_Wnlq=jnp.asarray(W),
                              polys_A=A, polys_B=B, polys_C=C, radial_learned=False)
    return out, rel


def _uniform_moment(model, n_uniform, r_min=0.5):
    """(NZ, NZ, n_q, n_q): mean over a uniform x-grid of p p^T, p = env(x) P(x); for rnl_basis
    "sbessel", over a uniform r-grid on [r_min, rc) of the basis g_k(r)."""
    if model.rnl_basis == "sbessel":
        NZ = model.rnl_Wnlq.shape[0]
        rc = np.asarray(model.pair_envelope[..., 0])
        out = []
        for i in range(NZ):
            row = []
            for j in range(NZ):
                r = jnp.linspace(r_min, rc[i, j], n_uniform + 1)[:-1]
                p = model.rnl_basis_values(r, jnp.full(r.shape, i), jnp.full(r.shape, j))
                row.append(p.T @ p / n_uniform)
            out.append(jnp.stack(row))
        return jnp.stack(out)
    x = jnp.linspace(-1.0, 1.0, n_uniform)
    P = poly_recursion(x, model.polys_A, model.polys_B, model.polys_C)          # (n_x, n_q)
    env = env_poly2sx(x[None, None, :], model.rnl_envelope[:, :, None, :])     # (NZ, NZ, n_x)
    p = env[..., None] * P                                                      # (NZ, NZ, n_x, n_q)
    return jnp.einsum("abxq,abxp->abqp", p, p) / n_uniform


def radial_gram(model, ds, n_prior=10.0, n_uniform=401):
    """Q (NZ, NZ, n_q, n_q): per (centre, neighbour) species pair, the second
    moment E_e[p(r_e) p(r_e)^T] of the enveloped polynomials p = env * P over
    the dataset's live edges, so ||R_n||^2 under the empirical pair-distance
    density is W_n^T Q W_n.  Each pair is shrunk towards the uniform-in-x
    moment with `n_prior` pseudo-edges, so a pair absent from the data (or
    with very few edges) still has a positive-definite Q.  Depends on the
    transform/envelope/polys only, never on Wnlq."""
    require_analytic(model)
    NZ, n_q = model.rnl_Wnlq.shape[0], model.rnl_Wnlq.shape[-1]
    S = jnp.zeros((NZ * NZ, n_q, n_q))
    cnt = jnp.zeros(NZ * NZ)
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        rij, send, recv, mask = flat_edges(b.rij, b.nbr, b.nbr_mask)
        zi, zj = b.node_z[send], b.node_z[recv]
        r = jnp.where(mask, jnp.linalg.norm(rij, axis=-1), 1.0)      # padded slots: any finite r
        p = poly_env(model, r, zi, zj) * mask[:, None]
        pair = zi * NZ + zj
        S = S + jax.ops.segment_sum(p[:, :, None] * p[:, None, :], pair, num_segments=NZ * NZ)
        cnt = cnt + jax.ops.segment_sum(mask.astype(S.dtype), pair, num_segments=NZ * NZ)
    S = S.reshape(NZ, NZ, n_q, n_q)
    cnt = cnt.reshape(NZ, NZ)[..., None, None]
    return (S + n_prior * _uniform_moment(model, n_uniform)) / (cnt + n_prior)


def row_active(W):
    """(NZ, NZ, n_rnl) bool: radials that are not identically zero.  The onehot
    (identity) init has structurally-zero rows for NZ > 1; they stay frozen."""
    return jnp.any(jnp.asarray(W) != 0.0, axis=-1)


def normalise(V, Q, active):
    """W = V with each active radial (zi, zj, n) scaled to unit norm under Q
    (||R||^2 = V Q V^T); inactive rows are returned as exact zeros.  Invariant
    to a positive rescaling of any row -- the gauge the readout and the prior
    would otherwise trade against.  The `where` keeps 0/0 out of the gradient."""
    nrm2 = jnp.einsum("abnq,abqp,abnp->abn", V, Q, V)
    safe = jnp.where(active, nrm2, 1.0)
    return jnp.where(active[..., None], V * jax.lax.rsqrt(safe)[..., None], 0.0)


def roughness_matrix(model):
    """D2 (n_q, n_q) = int_{-1}^{1} P_q''(x) P_p''(x) dx by Gauss-Legendre with
    n_q + 2 nodes or more (exact for these polynomial degrees).  For rnl_basis "sbessel",
    int_0^rc g_q''(r) g_p''(r) dr over the first pair's cutoff (all pairs share it in a built
    basis), by Gauss-Legendre with 4 n_q + 64 nodes."""
    if model.rnl_basis == "sbessel":
        n_q = model.rnl_Wnlq.shape[-1]
        rc = float(model.pair_envelope[0, 0, 0])
        xg, wg = np.polynomial.legendre.leggauss(4 * n_q + 64)
        rg, wr = 0.5 * rc * (xg + 1.0), 0.5 * rc * wg
        z = jnp.zeros((), jnp.int32)
        g = lambda r: model.rnl_basis_values(r[None], z[None], z[None])[0]        # noqa: E731
        d2 = jax.vmap(jax.jacfwd(jax.jacfwd(g)))(jnp.asarray(rg))
        D2 = (d2 * jnp.asarray(wr)[:, None]).T @ d2
        return (D2 + D2.T) / 2
    n_q = model.polys_A.shape[0]
    xg, wg = np.polynomial.legendre.leggauss(max(64, n_q + 2))
    p = lambda x: poly_recursion(x, model.polys_A, model.polys_B, model.polys_C)
    d2 = jax.vmap(jax.jacfwd(jax.jacfwd(p)))(jnp.asarray(xg))     # (n_nodes, n_q)
    D2 = (d2 * jnp.asarray(wg)[:, None]).T @ d2
    # D2 = A^T diag(wg) A is symmetric by construction, but the two triangles
    # accumulate in different summation orders inside the matmul; measured
    # asymmetry on the Si fixture (entries up to ~1.1e7) was 4.66e-10 absolute
    # (~4e-17 relative to the matrix's own scale, i.e. float64 rounding, not a
    # real signal), which trips a naive atol=1e-10 symmetry check on the
    # near-zero entries. Symmetrize explicitly rather than loosen that check.
    return (D2 + D2.T) / 2


def roughness(W, D2, wn):
    """sum_{zi, zj, n} wn[n] * W[zi, zj, n] D2 W[zi, zj, n]^T -- the curvature
    of the polynomial part of each radial, weighted per radial by wn."""
    return jnp.einsum("n,abnq,qp,abnp->", wn, W, D2, W)


def spectral_weights(n_q, p):
    """(n_q,) weights (1 + q)^p for the spectral prior on the radial change."""
    q = jnp.arange(n_q, dtype=jnp.float64)
    return (1.0 + q) ** jnp.asarray(p, jnp.float64)


def spectral_penalty(W, W_ref, sw):
    """sum_{zi,zj,n,q} sw[q] * (W - W_ref)[zi,zj,n,q]^2 -- penalises the change from the
    reference radials, more strongly at high Legendre degree."""
    dW = W - W_ref
    return jnp.einsum("q,abnq,abnq->", sw, dW, dW)


def uniform_gram(model, r_lo, r_hi, n=400):
    """U (NZ, NZ, n_q, n_q): per species pair, the mean over a uniform r grid
    on [r_lo, r_hi] of p p^T with p = env(x(r)) P(x(r)) (poly_env) --
    ||R_n||^2 under a uniform-in-r measure is W_n^T U[zi, zj] W_n.  Unlike
    radial_gram (empirical pair-distance density, dataset-dependent), this
    measure is flat in r, so it costs the change equally everywhere on
    [r_lo, r_hi] including gaps between coordination shells where the data
    density (hence radial_gram) is near zero."""
    require_analytic(model)
    NZ, n_q = model.rnl_Wnlq.shape[0], model.rnl_Wnlq.shape[-1]
    r = jnp.linspace(r_lo, r_hi, n)
    zi, zj = jnp.meshgrid(jnp.arange(NZ), jnp.arange(NZ), indexing="ij")   # (NZ, NZ)
    zi = jnp.broadcast_to(zi[:, :, None], (NZ, NZ, n)).reshape(-1)
    zj = jnp.broadcast_to(zj[:, :, None], (NZ, NZ, n)).reshape(-1)
    r_rep = jnp.tile(r, NZ * NZ)
    p = poly_env(model, r_rep, zi, zj).reshape(NZ, NZ, n, n_q)
    return jnp.einsum("abxq,abxp->abqp", p, p) / n


def gap_penalty(W, W_ref, U):
    """sum_{zi,zj,n} (W - W_ref)[zi,zj,n] U[zi,zj] (W - W_ref)[zi,zj,n]^T --
    the change of every radial from W_ref, measured under the uniform-in-r
    measure U (uniform_gram) rather than the empirical pair-distance density
    (radial_gram): a change that sits in a low-density gap between
    coordination shells costs the same here as an equal change where the
    data is dense."""
    dW = W - W_ref
    return jnp.einsum("abnq,abqp,abnp->", dW, U, dW)


def data_r_range(ds):
    """(r_min, r_max) over all live edges of ds."""
    r_min, r_max = float("inf"), float("-inf")
    for i in range(ds.n_batches):
        b = jax.tree.map(lambda a: a[i], ds)
        rij, _, _, mask = flat_edges(b.rij, b.nbr, b.nbr_mask)
        m = np.asarray(mask)
        if not m.any():
            continue
        r = np.asarray(jnp.linalg.norm(rij, axis=-1))[m]
        r_min = min(r_min, float(r.min()))
        r_max = max(r_max, float(r.max()))
    return r_min, r_max


def rnl_degrees(meta, wL=1.5):
    """(n_rnl,) polynomial degree of each tensor radial under the identity
    (onehot) convention, n' = (n - 1) // NZ, from the model's (n, l) spec.
    Rebuilds the spec with build_spec; raises if its length disagrees with
    meta["n_rnl"] (e.g. a Julia export with a different wL)."""
    from ..basis.spec import build_spec
    NZ = len(meta["elements"])
    wL = meta.get("basis", {}).get("wL", wL)
    _, Rnl, _ = build_spec(NZ, meta["order"], meta["totaldegree"], wL)
    if len(Rnl) != meta["n_rnl"]:
        raise ValueError(f"rnl_degrees: rebuilt spec has {len(Rnl)} radials, meta n_rnl={meta['n_rnl']}")
    return np.array([(n - 1) // NZ for n, _ in Rnl], dtype=int)
