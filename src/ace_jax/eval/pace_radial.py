"""Pure-JAX ports of ML-PACE radial, core-repulsion and embedding functions.

Source of truth: lammps-user-pace @ 99aa6e6, ML-PACE/ace-evaluator/
  ace_radial.cpp          radbase 241, chebExpCos 294, chebPow 347, chebLinear 366,
                          simplified_bessel 423, cutoff_func_poly 220, radcore 615, ZBL 653
  ace_abstract_basis.cpp  Fexp 37, FexpShiftedScaled 67
Values are checked against the C++ in tests/test_pace_radial.py.  Every branch
not taken is evaluated on a safe argument (double-where), so gradients stay
finite at r >= rcut, x = 0 and m = 1.
"""
import math

import jax.numpy as jnp

PI = math.pi


def _cheb(x, n):
    """T_0..T_n stacked on the last axis."""
    T = [jnp.ones_like(x), x]
    for _ in range(n - 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return jnp.stack(T[:n + 1], axis=-1)


def cutoff_func_poly(r, r_in, delta):
    """1 for r <= r_in - delta, 0 for r >= r_in, quintic in between."""
    d = jnp.where(delta == 0, 1.0, delta)
    x = 1.0 - 2.0 * (1.0 + (r - r_in) / d)
    fc = 0.5 + 7.5 / 2.0 * (x / 4.0 - x ** 3 / 6.0 + x ** 5 / 20.0)
    return jnp.where(r <= r_in - delta, 1.0, jnp.where(r >= r_in, 0.0, fc))


def _cheb_exp_cos(r, lam, rc, dcut, K):
    y1, y2 = jnp.exp(-lam * r / rc), jnp.exp(-lam)
    x = 1.0 - 2.0 * ((y1 - y2) / (1.0 - y2))
    T = _cheb(x, K)
    g = jnp.concatenate([T[..., :1], 0.5 - 0.5 * T[..., 1:K]], axis=-1)
    env = 0.5 * (1.0 + jnp.cos(PI * r / rc))
    fc = jnp.where(r > rc - dcut, 0.5 * (1.0 + jnp.cos(PI * (r - rc + dcut) / dcut)), 1.0)
    return g * (env * fc)[..., None]


def _cheb_pow(r, lam, rc, K):
    y = (1.0 - r / rc) ** lam
    T = _cheb(2.0 * (1.0 - y) - 1.0, K)
    return 0.5 - 0.5 * T[..., 1:K + 1]


def _cheb_linear(r, rc, K):
    T = _cheb(1.0 - r / rc, K)
    return 0.5 - 0.5 * T[..., 1:K + 1]


def _sinc(x):
    xs = jnp.where(x == 0, 1.0, x)
    return jnp.where(x == 0, 1.0, jnp.sin(xs) / xs)


def _sbessel(r, rc, K):
    def f(n):
        pre = ((-1) ** n * math.sqrt(2) * PI * (n + 1) * (n + 2)
               / math.sqrt((n + 1) ** 2 + (n + 2) ** 2))
        return pre / rc ** 1.5 * (_sinc(r * (n + 1) * PI / rc) + _sinc(r * (n + 2) * PI / rc))
    g = [f(0)]
    d_prev = 1.0
    for n in range(1, K):
        en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = 1 - en / d_prev
        g.append((f(n) + math.sqrt(en / d_prev) * g[-1]) / math.sqrt(dn))
        d_prev = dn
    return jnp.stack(g, axis=-1)


def radbase(r, name, inner, lam, rc, dcut, cut_in, dcut_in, K):
    """g_k(r), k < K, zero outside (cut_in - dcut_in, rc)."""
    if inner == "zbl":                       # ace_radial.cpp:246, ported as-is
        cut_in = jnp.where(dcut_in == 0, 1.0, 0.0)
    inside = (r > cut_in - dcut_in) & (r < rc)
    rs = jnp.where(inside, r, 0.5 * rc)      # safe argument for the untaken branch
    if name == "ChebExpCos":
        g = _cheb_exp_cos(rs, lam, rc, dcut, K)
    elif name == "ChebPow":
        g = _cheb_pow(rs, lam, rc, K)
    elif name == "ChebLinear":
        g = _cheb_linear(rs, rc, K)
    elif name == "SBessel":
        g = _sbessel(rs, rc, K)
    else:
        raise NotImplementedError(f"radial basis {name!r}")
    if inner in ("distance", "zbl"):
        g = g * (1.0 - cutoff_func_poly(rs, cut_in, dcut_in))[..., None]
    return jnp.where(inside[..., None], g, 0.0)


def radcore(r, pre, lamhc, rc, r_in, delta_in, inner):
    """Hard-core repulsion c_r(r) for the density / distance regimes."""
    pre, lam = jnp.abs(pre), jnp.abs(lamhc)
    lr2 = lam * r * r
    ok = lr2 < 50.0
    rs = jnp.where(ok & (r > 0), r, 1.0)
    cr = pre * jnp.exp(-lam * rs * rs) / rs * 0.5 * (1.0 + jnp.cos(PI * rs / rc))
    cr = jnp.where(ok, cr, 0.0)
    if inner == "distance":
        cr = cr * cutoff_func_poly(r, r_in, delta_in)
    return cr


_ZC = (0.18175, 0.50986, 0.28022, 0.02817)
_ZE = (-3.19980, -0.94229, -0.40290, -0.20162)
_KZBL = 14.399645351950543


def _zbl_E(r, a):
    """E(r) = phi(r/a)/r and its first two r-derivatives (fun_E_ij_and_deriv2)."""
    x = r / a
    ph = sum(c * jnp.exp(e * x) for c, e in zip(_ZC, _ZE))
    dph = sum(c * e * jnp.exp(e * x) for c, e in zip(_ZC, _ZE))
    d2ph = sum(c * e * e * jnp.exp(e * x) for c, e in zip(_ZC, _ZE))
    E = ph / r
    dE = -ph / r ** 2 + dph / (a * r)
    d2E = 2 * ph / r ** 3 - 2 * dph / (a * r ** 2) + d2ph / (a * a * r)
    return E, dE, d2E


def pace_zbl(r, Zi, Zj, rc, dcut, pre):
    """ACERadialFunctions::ZBL(r, Zi, Zj, cut_out=rc, cut_in=rc-dcut, pre)."""
    cut_out, cut_in = rc, rc - dcut
    a = 0.46850 / (Zi ** 0.23 + Zj ** 0.23)
    rs = jnp.where((r > 0) & (r < cut_out), r, 0.5 * cut_out)
    E, _, _ = _zbl_E(rs, a)
    Ec, dEc, d2Ec = _zbl_E(cut_out, a)
    dr = cut_out - cut_in
    A = (-3 * dEc + dr * d2Ec) / dr ** 2
    B = (2 * dEc - dr * d2Ec) / dr ** 3
    C = -Ec + 0.5 * dr * dEc - dr ** 2 / 12.0 * d2Ec
    t = rs - cut_in
    S = jnp.where(rs <= cut_in, C, A / 3 * t ** 3 + B / 4 * t ** 4 + C)
    cr = pre * _KZBL / 2 * Zi * Zj * (E + S) * cutoff_func_poly(rs, cut_out, dr)
    return jnp.where(r < cut_out, cr, 0.0)


def fexp(x, m):
    """PACE `Fexp`: sign(x)((1-g)|x|^m + lam g |x|), g = exp(-(w|x|)^3), linear
    for |x| <= 1e-10.  No prefactor (w_p is applied by the caller)."""
    w = 1.0e6
    lam = (1.0 / w) ** (m - 1.0)
    a = jnp.abs(x)
    big = a > 1e-10
    as_ = jnp.where(big, a, 1.0)
    w3 = (w * as_) ** 3
    g = jnp.where(w3 > 30.0, 0.0, jnp.exp(-jnp.minimum(w3, 30.0)))
    s = jnp.where(x < 0, -1.0, 1.0)
    return jnp.where(big, s * ((1.0 - g) * as_ ** m + lam * g * as_), lam * x)


def fexp_shifted_scaled(x, m):
    """PACE `FexpShiftedScaled`; identity when |m - 1| < 1e-10."""
    lin = jnp.abs(m - 1.0) < 1e-10
    ms = jnp.where(lin, 2.0, m)
    a = jnp.abs(x)
    e = jnp.exp(-a)
    nu = 1.0 / ms
    xoff = nu ** (nu / (1.0 - nu)) * e
    yoff = nu ** (1.0 / (1.0 - nu)) * e
    s = jnp.where(x < 0, -1.0, 1.0)
    return jnp.where(lin, x, s * ((xoff + a) ** ms - yoff))
