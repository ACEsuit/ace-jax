"""Real spherical harmonics as a pure-JAX polynomial recursion.

Why not sphericart-jax: its lowering registers an FFI op, so the harmonics land
in exported StableHLO as a custom call target (`cpu_spherical_f64` /
`cuda_spherical_f64`).  lammps-jax resolves such targets at run time from
LAMMPS_JAX_FFI_HANDLERS, and contrib/ffi-replay exists for libraries whose
compiled kernels only live inside the exporting process.  A stock-HLO
implementation avoids that machinery entirely, and drops the `sphericart-jax`
pin that forces jax==0.10.1 (the LAMMPS venv runs 0.11.1).

Both conventions are implemented: `ace1_model` uses `Ytype = :spherical`
(ace1_compat.jl:407) while `ace_model` defaults to `:solid`
(ace_heuristics.jl:149).

Convention (matching SpheriCart / P4ML `real_sphericalharmonics(L;
normalisation = :L2)`), with u = z/r and (x + iy)^m = A_m + i B_m:

    Y_l^m = N_lm * Rbar_l^|m|(u) * (A_|m| or B_|m|) / r^|m|   (x sqrt2 for m != 0)
    N_lm  = sqrt((2l+1)/(4 pi) * (l-|m|)! / (l+|m|)!)

`Rbar_l^m(u) = P_l^m(u) / (1-u^2)^(m/2)` is a polynomial in u, so pulling the
sin^m(theta) factor out against A_m/B_m -- which carry s^m -- removes the
removable singularity on the z axis.  No Condon-Shortley phase, matching
SpheriCart.

Solid harmonics are r^l * Y_lm (verified against sphericart to 7e-15).  Rather
than computing the spherical form and multiplying, the two differ only in the
radial power: r^-|m| for spherical, r^(l-|m|) for solid.  Since l >= |m| the
solid branch involves NO division at all, so it is the better-conditioned of
the two.

Storage order is SpheriCart's: index l*l + l + m, m running -l..l.
"""

import math

import jax.numpy as jnp


def _norm(l, m):
    """sqrt((2l+1)/(4 pi) * (l-m)!/(l+m)!), m >= 0, computed in log space."""
    return math.exp(0.5 * (math.log(2 * l + 1) - math.log(4.0 * math.pi)
                           + math.lgamma(l - m + 1) - math.lgamma(l + m + 1)))


def _harmonics(xyz, l_max, solid):
    """Shared core; `solid` selects the radial power r^(l-|m|) vs r^-|m|."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r2 = x * x + y * y + z * z
    r = jnp.sqrt(jnp.where(r2 > 0, r2, 1.0))
    u = jnp.where(r2 > 0, z / r, 0.0)                    # cos(theta)

    # A_m + i B_m = (x + i y)^m, polynomial so no singularity on the z axis
    A = [jnp.ones_like(x)]
    B = [jnp.zeros_like(x)]
    for m in range(1, l_max + 1):
        A.append(x * A[m - 1] - y * B[m - 1])
        B.append(x * B[m - 1] + y * A[m - 1])

    # Rbar_l^m(u) = P_l^m(u) / (1-u^2)^(m/2), polynomial in u.
    #   Rbar_m^m     = (2m-1)!!                       (no Condon-Shortley phase)
    #   Rbar_{m+1}^m = (2m+1) u Rbar_m^m
    #   Rbar_l^m     = [(2l-1) u Rbar_{l-1}^m - (l+m-1) Rbar_{l-2}^m] / (l-m)
    Rbar = {}
    dfact = 1.0                                          # (2m-1)!!
    for m in range(l_max + 1):
        if m > 0:
            dfact *= (2 * m - 1)
        Rbar[(m, m)] = jnp.full_like(u, dfact)
        if m + 1 <= l_max:
            Rbar[(m + 1, m)] = (2 * m + 1) * u * Rbar[(m, m)]
        for l in range(m + 2, l_max + 1):
            Rbar[(l, m)] = ((2 * l - 1) * u * Rbar[(l - 1, m)]
                            - (l + m - 1) * Rbar[(l - 2, m)]) / (l - m)

    # powers of r: r^-m for spherical (one reciprocal), r^k for solid (no division)
    if solid:
        rpow = [jnp.ones_like(x)]
        for k in range(1, l_max + 1):
            rpow.append(rpow[k - 1] * r)
        radial = lambda l, am: rpow[l - am]
    else:
        inv_r = jnp.where(r2 > 0, 1.0 / r, 0.0)
        inv_rm = [jnp.ones_like(x)]
        for m in range(1, l_max + 1):
            inv_rm.append(inv_rm[m - 1] * inv_r)
        radial = lambda l, am: inv_rm[am]

    root2 = math.sqrt(2.0)
    out = [None] * ((l_max + 1) ** 2)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            c = _norm(l, am) * (root2 if m != 0 else 1.0)
            ang = A[am] if m >= 0 else B[am]
            out[l * l + l + m] = c * Rbar[(l, am)] * ang * radial(l, am)
    return jnp.stack(out, axis=-1)


def real_spherical_harmonics(xyz, l_max):
    """Real spherical harmonics, (..., 3) -> (..., (l_max+1)^2), order l*l+l+m."""
    return _harmonics(xyz, l_max, solid=False)


def real_solid_harmonics(xyz, l_max):
    """Real solid harmonics r^l * Y_lm, same shape and ordering."""
    return _harmonics(xyz, l_max, solid=True)
