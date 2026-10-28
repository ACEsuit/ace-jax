"""PACE radial basis, core-repulsion and embedding functions for `.yace` models.

Implemented from docs/dev/pace-radial-spec.md, which states each function as an equation and cites
its source: Bochkarev et al., Phys. Rev. Materials 6, 013804 (2022), App. C ([PRM]); Lysogorskiy et
al., npj Comput. Mater. 7, 97 (2021), SI ([SI]); Kocer et al., J. Chem. Phys. 150, 154102 (2019)
(SBessel); Ziegler-Biersack-Littmark (1985) and LAMMPS `pair_style zbl` (ZBL).  Where ML-PACE's
output departs from a published equation, or none is published (ChebLinear, Fexp, the ZBL site
factor, the zbl-regime inner cut), the spec states the behaviour, matched to ML-PACE output: the
reference values are data (fixtures/pace/unit_ref.npz, tests/test_pace_radial.py), not code.
Every untaken branch is evaluated on a safe argument (double where), so values and gradients stay
finite at r >= rcut, r = 0, x = 0 and m = 1.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

PI = math.pi
_POLY_CUT = (15 / 8, -5 / 4, 3 / 8)                  # [PRM] C9: a, b, c


def _cheb(x, n):
    """T_0..T_n stacked on the last axis (three-term recurrence)."""
    T = [jnp.ones_like(x), x]
    for _ in range(n - 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return jnp.stack(T[:n + 1], axis=-1)


def _cos_switch(t):
    """(1 + cos(pi t)) / 2: 1 at t = 0, 0 at t = 1."""
    return 0.5 * (1.0 + jnp.cos(PI * t))


def cutoff_func_poly(r, r_in, delta):
    """[PRM] C8-C10 (spec 2): 1 for r <= r_in - delta, 0 for r >= r_in, (1 + y(x)) / 2 between,
    y = a x + b x^3 + c x^5, x = 1 - 2 (1 + (r - r_in)/delta).  delta = 0 is a step at r_in."""
    d = jnp.where(delta == 0, 1.0, delta)
    x = 1.0 - 2.0 * (1.0 + (r - r_in) / d)
    a, b, c = _POLY_CUT
    x2 = x * x
    f = 0.5 * (1.0 + x * (a + x2 * (b + c * x2)))
    return jnp.where(r <= r_in - delta, 1.0, jnp.where(r >= r_in, 0.0, f))


def _cheb_exp_cos(r, lam, rc, dcut, K):
    """Spec 3.2, [PRM] C5-C6: x = 1 - 2 (e^{lam (1 - r/rc)} - 1)/(e^lam - 1), g_0 = the envelope
    (1 + cos(pi r/rc))/2, g_k = (1 - T_k(x))/2 x envelope; a cosine taper over (rc - dcut, rc)."""
    x = 1.0 - 2.0 * jnp.expm1(lam * (1.0 - r / rc)) / jnp.expm1(lam)
    T = _cheb(x, max(K - 1, 1))[..., :K]
    g = jnp.concatenate([T[..., :1], 0.5 * (1.0 - T[..., 1:])], axis=-1)
    d = jnp.where(dcut == 0, 1.0, dcut)      # dcut = 0: no taper; keep the untaken branch's grad finite
    taper = jnp.where(r > rc - dcut, _cos_switch((r - rc + dcut) / d), 1.0)
    return g * (_cos_switch(r / rc) * taper)[..., None]


def _cheb_pow(r, lam, rc, K):
    """Spec 3.1, [PRM] C3-C4: g_k = (1 - T_k(x))/2, x = 1 - 2 (1 - r/rc)^lam, k = 1..K."""
    T = _cheb(1.0 - 2.0 * (1.0 - r / rc) ** lam, K)
    return 0.5 * (1.0 - T[..., 1:])


def _cheb_linear(r, rc, K):
    """Spec 3.4 (matched to ML-PACE output): g_k = (1 - T_k(1 - r/rc))/2, k = 1..K."""
    T = _cheb(1.0 - r / rc, K)
    return 0.5 * (1.0 - T[..., 1:])


def _sinc(x):
    xs = jnp.where(x == 0, 1.0, x)
    return jnp.where(x == 0, 1.0, jnp.sin(xs) / xs)


def _sbessel(r, rc, K):
    """SBessel g_0..g_{K-1} (spec 3.3: [PRM] C7, [Kocer]).  sin(k x), k = 1..K+1, from one sin and one cos by
    the angle-addition (rotation) recurrence
        sin((k+1)x) = sin(kx) cos(x) + cos(kx) sin(x)
        cos((k+1)x) = cos(kx) cos(x) - sin(kx) sin(x)
    and one reciprocal instead of a division per k: the per-edge transcendental
    cost no longer grows with K (-20% of a PACE force call,
    docs/dev/pace-performance-gap.md #3). Advancing sin and cos together this way
    (rather than the three-term Chebyshev recurrence sin((k+1)x) =
    2 cos(x) sin(kx) - sin((k-1)x), which is otherwise algebraically
    equivalent) keeps float32 error growth close to O(k * eps) instead of the
    three-term form's larger, cancellation-driven growth.
    f(0) always reads terms through index 2, so at least one rotation step
    runs even when K = 0 (K = 0 or 1 each return a single basis function, as
    the pre-recurrence code did)."""
    x = r * PI / rc
    xs = jnp.where(x == 0, 1.0, x)
    s1, c1 = jnp.sin(xs), jnp.cos(xs)
    inv = 1.0 / xs
    s, c = [jnp.zeros_like(xs), s1], [jnp.ones_like(xs), c1]
    for _ in range(max(K, 1)):
        sk, ck = s[-1], c[-1]
        s.append(sk * c1 + ck * s1)
        c.append(ck * c1 - sk * s1)
    sinc = [None] + [jnp.where(x == 0, 1.0, s[k] * inv * (1.0 / k)) for k in range(1, len(s))]
    rc15 = rc ** 1.5

    def f(n):
        pre = ((-1) ** n * math.sqrt(2) * PI * (n + 1) * (n + 2)
               / math.sqrt((n + 1) ** 2 + (n + 2) ** 2))
        return pre / rc15 * (sinc[n + 1] + sinc[n + 2])
    g, d_prev = [f(0)], 1.0
    for n in range(1, K):
        en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = 1 - en / d_prev
        g.append((f(n) + math.sqrt(en / d_prev) * g[-1]) / math.sqrt(dn))
        d_prev = dn
    return jnp.stack(g, axis=-1)


def sbessel_matrix(K):
    """M (K+1, K) with `_sbessel`'s g_n = sum_k S_k M[k, n] / rc^1.5, where
    S_k = sinc((k+1) x): PACE's orthogonalisation is linear with constant
    coefficients, so the whole recurrence folds into one constant matrix.
    K = 0 gives one column, as `_sbessel` does."""
    Kf = max(K, 1)
    P = np.zeros((Kf + 1, Kf))                      # f_n = pre_n (S_n + S_{n+1})
    for n in range(Kf):
        pre = ((-1) ** n * math.sqrt(2) * PI * (n + 1) * (n + 2)
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
    return (P @ L)[:, :max(K, 1)]


def _sbessel_mm(r, rc, K):
    """`_sbessel` as one elementwise sin per (edge, k) and a constant matmul,
    g = (S @ M) / rc^1.5 (`sbessel_matrix`).  Exact to roundoff.  The stacked
    recurrence fuses into one kernel that re-evaluates its chain per column, so
    its cost grows ~K^2 per edge; this form is faster from nradbase ~12 and
    slower below (the (E, K+1) sin array and the matmul cost more), which is
    why `load_yace` chooses it per model (docs/dev/ace-vs-pace-gap.md 4.3)."""
    x = r * PI / rc
    xs = jnp.where(x == 0, 1.0, x)
    k = jnp.arange(1, max(K, 1) + 2, dtype=r.dtype)
    kx = k * xs[..., None]
    S = jnp.where((x == 0)[..., None], 1.0, jnp.sin(kx) / kx)
    M = jnp.asarray(sbessel_matrix(K), r.dtype)
    return (jnp.matmul(S, M, precision=jax.lax.Precision.HIGHEST)
            / jnp.expand_dims(jnp.asarray(rc, r.dtype) ** 1.5, -1))


SBESSEL_FORMS = ("rotation", "matmul")


def hermite_table(r, f, df, h, zi, zj):
    """Cubic-Hermite interpolation of tabulated radials (spec 3.5): the `.yace` `Tabulated` g_k and
    dg_k/dr on r_i = i h.  r, zi, zj (E,); f, df (NZ, NZ, N+1, K); h (NZ, NZ).  Gathers the two grid
    rows each edge needs, never an edge's whole table.  Returns (E, K)."""
    hh = h[zi, zj]
    x = r / hh
    N = f.shape[-2] - 1
    i = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, N - 1)
    t = (x - i)[:, None]
    u = 1.0 - t
    hh = hh[:, None]
    return ((1 + 2 * t) * u * u * f[zi, zj, i] + t * u * u * hh * df[zi, zj, i]
            + t * t * (3 - 2 * t) * f[zi, zj, i + 1] + t * t * (t - 1) * hh * df[zi, zj, i + 1])


def _family(name, rs, lam, rc, dcut, K, sbessel_form, table):
    if name == "ChebExpCos":
        return _cheb_exp_cos(rs, lam, rc, dcut, K)
    if name == "ChebPow":
        return _cheb_pow(rs, lam, rc, K)
    if name == "ChebLinear":
        return _cheb_linear(rs, rc, K)
    if name == "Tabulated":
        return table(rs)
    if name == "SBessel":
        if sbessel_form not in SBESSEL_FORMS:
            raise ValueError(f"sbessel_form must be one of {SBESSEL_FORMS}, got {sbessel_form!r}")
        return (_sbessel_mm if sbessel_form == "matmul" else _sbessel)(rs, rc, K)
    raise NotImplementedError(f"radial basis {name!r}")


def radbase(r, name, inner, lam, rc, dcut, cut_in, dcut_in, K, sbessel_form="rotation", table=None):
    """g_k(r), k < K (spec 3), zero outside (cut_in - dcut_in, rc).  Inner cutoff (spec 4, [PRM] C2):
    "distance" multiplies by 1 - f(r; cut_in, dcut_in); "density" has none (the energy cutoff in
    `pace_model` replaces it); "zbl" cuts as "distance" at an effective cut_in of 1 when
    dcut_in = 0 and 0 otherwise (matched to ML-PACE output).  `sbessel_form` picks `_sbessel`
    ("rotation") or `_sbessel_mm` ("matmul"); "Tabulated": `table(rs)` gives the (E, K) values."""
    if inner == "zbl":
        one = jnp.ones_like(dcut_in)              # typed operands: no f64 buffer
        cut_in = jnp.where(dcut_in == 0, one, 0 * one)
    inside = (r > cut_in - dcut_in) & (r < rc)
    rs = jnp.where(inside, r, 0.5 * rc)      # safe argument for the untaken branch
    g = _family(name, rs, lam, rc, dcut, K, sbessel_form, table)
    if inner in ("distance", "zbl"):
        g = g * (1.0 - cutoff_func_poly(rs, cut_in, dcut_in))[..., None]
    return jnp.where(inside[..., None], g, 0.0)


def radcore(r, pre, lamhc, rc, r_in, delta_in, inner):
    """Core repulsion (spec 5; [PRM] C12, [SI] 19, matched to ML-PACE output):
    |pre| e^{-|lamhc| r^2}/r (1 + cos(pi r/rc))/2 on 0 < r < rc, times f(r; r_in, delta_in) in the
    "distance" regime only."""
    ok = (r > 0) & (r < rc)
    rs = jnp.where(ok, r, 0.5 * rc)
    cr = jnp.abs(pre) * jnp.exp(-jnp.abs(lamhc) * rs * rs) / rs * _cos_switch(rs / rc)
    if inner == "distance":
        cr = cr * cutoff_func_poly(rs, r_in, delta_in)
    return jnp.where(ok, cr, 0.0)


# ZBL universal screening: phi(x) = sum_i c_i e^{-d_i x}, a = 0.8854 a_B / (Zi^0.23 + Zj^0.23)
_ZBL_C = (0.18175, 0.50986, 0.28022, 0.02817)
_ZBL_D = (3.19980, 0.94229, 0.40290, 0.20162)
_ZBL_A0 = 0.46850                                      # 0.8854 a_B in A
_KE = 14.399645351950543                               # e^2 / (4 pi eps0) in eV A


def _zbl_screened(r, a):
    """E = phi(r/a)/r and dE/dr, d2E/dr2 in closed form: with u_i = e^{-d_i r/a} and
    q_i = -d_i/a - 1/r, E' = sum c_i u_i q_i / r and E'' = sum c_i u_i (q_i^2 + 1/r^2) / r."""
    E = dE = d2E = 0.0
    for c, d in zip(_ZBL_C, _ZBL_D):
        cu = c * jnp.exp(-d * r / a) / r
        q = -d / a - 1.0 / r
        E, dE, d2E = E + cu, dE + cu * q, d2E + cu * (q * q + 1.0 / (r * r))
    return E, dE, d2E


def pace_zbl(r, Zi, Zj, rc, dcut, pre):
    """ZBL core (spec 6): pre (k_e/2) Zi Zj (E(r) + S(r)) f(r; rc, dcut) on 0 < r < rc.  S is the
    LAMMPS `pair_style zbl` switch over (rc - dcut, rc), so E + S and two derivatives vanish at rc;
    the factor 1/2 is the site share of the pair energy and f (spec 2) an extra taper."""
    a = _ZBL_A0 / (Zi ** 0.23 + Zj ** 0.23)
    ok = (r > 0) & (r < rc)
    rs = jnp.where(ok, r, 0.5 * rc)
    Ec, dEc, d2Ec = _zbl_screened(rc, a)
    dl = dcut
    A = (-3.0 * dEc + dl * d2Ec) / dl ** 2
    B = (2.0 * dEc - dl * d2Ec) / dl ** 3
    C = -Ec + 0.5 * dl * dEc - dl ** 2 / 12.0 * d2Ec
    t = jnp.maximum(rs - (rc - dl), 0.0)
    S = t ** 3 * (A / 3.0 + B / 4.0 * t) + C
    E = _zbl_screened(rs, a)[0]
    cr = pre * (0.5 * _KE) * Zi * Zj * (E + S) * cutoff_func_poly(rs, rc, dl)
    return jnp.where(ok, cr, 0.0)


def _sign(x):
    one = jnp.ones_like(x)                   # typed operands: literal-only where is f64
    return jnp.where(x < 0, -one, one)


def fexp(x, m):
    """FinnisSinclair embedding (spec 7.1, matched to ML-PACE output):
    sgn(x) ((1 - g)|x|^m + lam g |x|), g = e^{-(w|x|)^3}, w = 1e6, lam = w^{1-m}: |x|^m, linear
    with slope lam near 0.  1 - g by expm1.  No prefactor (the caller applies w_p)."""
    w = 1.0e6
    lam = w ** (1.0 - m)
    a = jnp.abs(x)
    nz = a > 0
    as_ = jnp.where(nz, a, 1.0)
    w3 = (w * as_) ** 3
    one_g = -jnp.expm1(-w3)
    F = _sign(x) * (one_g * as_ ** m + (1.0 - one_g) * lam * as_)
    return jnp.where(nz, F, lam * x)


def fexp_shifted_scaled(x, m):
    """FinnisSinclairShiftedScaled embedding (spec 7.2): sgn(x) ((|x| + c e^{-|x|})^m - d e^{-|x|}),
    c = (1/m)^{1/(m-1)} (where x^m has unit slope), d = c^m (F(0) = 0); [SI] Eq. 22 is m = 1/2.
    The m -> 1 limit, F = x, is used for |m - 1| < 1e-10."""
    lin = jnp.abs(m - 1.0) < 1e-10
    ms = jnp.where(lin, 2.0, m)
    c = (1.0 / ms) ** (1.0 / (ms - 1.0))
    a = jnp.abs(x)
    e = jnp.exp(-a)
    return jnp.where(lin, x, _sign(x) * ((a + c * e) ** ms - c ** ms * e))
