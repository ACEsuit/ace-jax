"""Radial embedding: transform, envelope, and the splined Rnl.

Mirrors ACEpotentials' SplineRnlrzzBasis, which evaluates

    Rnl(r, zi, zj) = spline_{zi,zj}(x) * envelope(r, x),   x = T_{zi,zj}(r)

The transform and envelope stay analytic in Julia too; only x -> Wnlq*polys(x)
is splined.  Exporting Julia's own spline coefficients means we reproduce what
Julia evaluates rather than re-approximating it.

No module-level jax.config here, or anywhere in this package: precision is the
caller's to choose.
"""

import jax
import jax.numpy as jnp


def agnesi_normalized(r, params):
    """NormalizedTransform(GeneralizedAgnesiTransform); params = (p,q,a,rin,r0,yin,ycut).

    y = 1                                   for r <= rin
      = 1 / (1 + a s^q / (1 + s^(q-p)))     otherwise,  s = (r-rin)/(r0-rin)
    x = clamp(-1 + 2 (y - yin)/(ycut - yin), -1, 1)
    """
    p, q, a, rin, r0, yin, ycut = [params[..., i] for i in range(7)]
    s = (r - rin) / (r0 - rin)
    # guard s=0: s**(q-p) is fine but s**q/(1+s^(q-p)) is 0/1 -> y=1, matching r<=rin
    s_safe = jnp.where(s > 0, s, 1.0)
    y_gen = 1.0 / (1.0 + a * s_safe**q / (1.0 + s_safe ** (q - p)))
    y = jnp.where(r > rin, y_gen, 1.0)
    x = -1.0 + 2.0 * (y - yin) / (ycut - yin)
    return jnp.clip(x, -1.0, 1.0)


def env_poly2sx(x, params):
    """PolyEnvelope2sX; params = (x1, x2, p1, p2, s).  s (x-x1)^p1 (x2-x)^p2 on (x1,x2)."""
    x1, x2, p1, p2, s = [params[..., i] for i in range(5)]
    inside = (x > x1) & (x < x2)
    a = jnp.where(inside, x - x1, 1.0)
    b = jnp.where(inside, x2 - x, 1.0)
    return jnp.where(inside, s * a**p1 * b**p2, 0.0)


def env_ace1_poly1sr(r, params):
    """ACE1_PolyEnvelope1sR; params = (rcut, r0, p).

    s^-p - sc^-p + p sc^(-p-1) (s - sc),  s = r/r0, sc = rcut/r0;  0 for r > rcut.
    """
    rcut, r0, p = params[..., 0], params[..., 1], params[..., 2]
    inside = r <= rcut
    s = jnp.where(inside & (r > 0), r / r0, 1.0)
    sc = rcut / r0
    val = s ** (-p) - sc ** (-p) + p * sc ** (-p - 1.0) * (s - sc)
    return jnp.where(inside, val, 0.0)


def spline_eval(x, coefs, x0, h, n):
    """Uniform cubic B-spline, matching Interpolations.cubic_spline_interpolation.

    `coefs` has shape (ncoef, F) with ncoef = n + 2: one pad coefficient at each
    end, laid out so that Julia's OffsetArray index 0..n+1 is row 0..n+1 here.
    Evaluation is a 4-coefficient gather and a cubic, as the plan anticipated.
    """
    k = (x - x0) / h + 1.0                       # Julia 1-based grid coordinate
    ix = jnp.clip(jnp.floor(k).astype(jnp.int32), 1, n - 1)
    t = k - ix
    ct = 1.0 - t
    w = jnp.stack([ct**3 / 6.0,
                   (4.0 - 6.0 * t**2 + 3.0 * t**3) / 6.0,
                   (4.0 - 6.0 * ct**2 + 3.0 * ct**3) / 6.0,
                   t**3 / 6.0], axis=-1)          # (..., 4)
    # Julia coefs[ix-1 .. ix+2] with OffsetArray base 0 -> rows ix-1 .. ix+2
    idx = ix[..., None] + jnp.arange(-1, 3)       # (..., 4)
    g = coefs[idx]                                # (..., 4, F)
    return jnp.einsum("...k,...kf->...f", w, g)


def env_poly1sr(r, params):
    """PolyEnvelope1sR; params = (rcut, p).  ((r/rcut)^-p - 1)(1 - r/rcut), 0 beyond rcut.

    Distinct from ACE1_PolyEnvelope1sR: `ace_model`'s pair basis uses this one,
    `ace1_model`'s uses the ACE1 form.
    """
    rcut, p = params[..., 0], params[..., 1]
    inside = r < rcut
    s = jnp.where(inside & (r > 0), r / rcut, 1.0)
    return jnp.where(inside, (s ** (-p) - 1.0) * (1.0 - s), 0.0)


def poly_recursion(y, A, B, C):
    """OrthPolyBasis1D3T: P0 = A0, P1 = A1 y + B1, Pk = (Ak y + Bk) P_{k-1} + Ck P_{k-2}.

    Three coefficient vectors are the whole basis, which is why the analytic
    branch exports so little.  Validated against Julia in earlier at 1.4e-15.
    """
    n = A.shape[0]
    out = [jnp.broadcast_to(A[0], y.shape), A[1] * y + B[1]]
    for k in range(2, n):
        out.append((A[k] * y + B[k]) * out[k - 1] + C[k] * out[k - 2])
    return jnp.stack(out, axis=-1)                       # (..., n_polys)
