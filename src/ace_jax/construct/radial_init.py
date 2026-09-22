"""Seeded radial-coefficient initialisation for Python model authoring.

Pure numpy: this module produces the coefficient arrays (agnesi transforms,
normalized-Legendre recursions, envelopes, Wnlq weights) that a frozen
`ACEModel` needs, mirroring the defaults of ACEpotentials'
`ace_learnable_Rnlrzz` / `ace_model` (src/models/ace_heuristics.jl):

    tensor basis   agnesi (p, q) = (2, 2), PolyEnvelope2sX(-1, 1, 2, 2),
                   polys = normalized Legendre, Winit = glorot_normal
    pair basis     agnesi (p, q) = (1, 4), PolyEnvelope1sR(rcut, 1),
                   polys = normalized Legendre, Winit = onehot
                   (then splinified in Julia; the analytic contract is kept here)

The fixtures span several ACEpotentials versions whose radial defaults changed
(tensor (2,4) vs (2,2), pair (1,3) vs (1,4), ACE1 vs plain pair envelope); the
defaults here follow the current dev code, and everything is overridable.

Glorot note: Julia draws `glorot_normal(rng, Float64, len_nl, len_q)` per
species pair from Lux's RNG.  We draw Normal(0, sqrt(2/(len_nl+len_q))) from a
seeded numpy Generator instead -- same distribution, different bit stream, so
authored radials are reproducible in Python but not bit-identical to a Julia
run with the same seed.
"""

import math

import numpy as np

# bond_len[z] for 76 elements, from ACEpotentials
# data/length_scales_VASP_auto_length_scales.yaml (VASP auto length scales).
BOND_LEN = {
    3: 3.0, 4: 2.2, 5: 1.8, 6: 1.4, 7: 1.6, 8: 1.7, 9: 2.1,
    11: 3.7, 12: 3.2, 13: 2.9, 14: 2.4, 15: 2.5, 16: 2.3, 17: 2.6,
    19: 4.6, 20: 3.8, 21: 3.2, 22: 2.9, 23: 2.6, 24: 2.5, 25: 2.5,
    26: 2.5, 27: 2.5, 28: 2.5, 29: 2.6, 30: 2.8, 31: 3.0, 32: 2.5,
    33: 2.7, 34: 2.8, 35: 2.8, 37: 4.7, 38: 4.2, 39: 3.6, 40: 3.2,
    41: 2.9, 42: 2.7, 43: 2.7, 44: 2.7, 45: 2.7, 46: 2.8, 47: 2.9,
    48: 3.2, 49: 3.4, 50: 2.9, 51: 3.1, 52: 3.2, 53: 3.2, 56: 4.3,
    57: 3.7, 58: 3.4, 59: 3.7, 60: 3.7, 61: 3.6, 62: 3.6, 64: 3.6,
    65: 3.6, 66: 3.5, 69: 3.5, 70: 3.8, 72: 3.2, 73: 2.9, 74: 2.8,
    75: 2.8, 76: 2.7, 77: 2.7, 78: 2.8, 79: 2.9, 80: 3.3, 81: 3.4,
    82: 3.5, 83: 3.3, 84: 3.3, 89: 4.0, 90: 3.6, 91: 3.3,
}

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu"
).split()


def resolve_elements(elements):
    """Atomic numbers from ints or chemical symbols ("Si" / "si" / 14).

    Order is preserved (it is the species index order, as ACEpotentials'
    `_convert_zlist` keeps it); duplicates are an error."""
    zs = []
    for e in elements:
        if isinstance(e, (int, np.integer)):
            z = int(e)
        elif isinstance(e, str):
            sym = e.strip().capitalize()
            if sym not in _SYMBOLS:
                raise ValueError(f"unknown element symbol {e!r}")
            z = _SYMBOLS.index(sym) + 1
        else:
            raise TypeError(f"element must be int or str, got {type(e)}")
        if not 1 <= z <= len(_SYMBOLS):
            raise ValueError(f"atomic number {z} out of range")
        if z in zs:
            raise ValueError(f"duplicate element {e!r} (Z={z}) in {list(elements)!r}")
        zs.append(z)
    if not zs:
        raise ValueError("elements must be non-empty")
    return zs


def bond_len(z):
    """ACEpotentials `DefaultHypers.bond_len(z)`: the tabulated length, or an
    error (Julia falls back to JuLIP's rnn table, which is not ported)."""
    if z not in BOND_LEN:
        raise ValueError(f"no tabulated bond length for Z={z}; pass `r0` explicitly")
    return BOND_LEN[z]


def r0_table(zs, r0=None):
    """(NZ, NZ) per-pair r0.  Default is ACEpotentials `_default_rin0cuts`:
    r0(zi, zj) = (bond_len(zi) + bond_len(zj)) / 2.  `r0` may be a scalar
    (every pair) or an (NZ, NZ) table."""
    NZ = len(zs)
    if r0 is None:
        bl = [bond_len(z) for z in zs]
        return np.array([[(bl[i] + bl[j]) / 2 for j in range(NZ)] for i in range(NZ)])
    r0 = np.asarray(r0, float)
    if r0.ndim == 0:
        return np.full((NZ, NZ), float(r0))
    if r0.shape != (NZ, NZ):
        raise ValueError(f"r0 must be a scalar or an ({NZ}, {NZ}) table, got shape {r0.shape}")
    return r0.copy()


# ---------------------------------------------------------------------------
#  agnesi transform (radial_transforms.jl GeneralizedAgnesiTransform)
# ---------------------------------------------------------------------------

def agnesi_a(p, q):
    """Default `a`: |x'(r)| maximised at r = r0 (ace_heuristics.jl default)."""
    return (-2 * q + p * (-2 + 4 * q)) / (p + p * p + q + q * q)


def agnesi_raw(r, p, q, a, rin, r0):
    """Raw transform value; 1 at r <= rin."""
    s = np.where(r <= rin, 0.0, (np.asarray(r, float) - rin) / (r0 - rin))
    return 1.0 / (1.0 + a * s**q / (1.0 + s**(q - p)))


def agnesi_transform_params(rin, r0, rcut, p, q):
    """The exported 7-tuple (p, q, a, rin, r0, yin, ycut) with
    yin = trans(rin) = 1 (rin = 0) and ycut = trans(rcut)."""
    a = agnesi_a(p, q)
    if not (a > 0 and 0 < r0 < rcut and p > 0 and q >= p):
        raise ValueError(f"invalid agnesi transform p={p} q={q} a={a} r0={r0} rcut={rcut}")
    yin = float(agnesi_raw(rin, p, q, a, rin, r0))
    ycut = float(agnesi_raw(rcut, p, q, a, rin, r0))
    return (float(p), float(q), float(a), float(rin), float(r0), yin, ycut)


def agnesi_normalized(r, tr):
    """ace-jax's exported eval contract: map r through the 7-tuple transform to
    the clamped [-1, 1] coordinate (eval/radial.py `agnesi_normalized`)."""
    p, q, a, rin, r0, yin, ycut = tr
    y = agnesi_raw(r, p, q, a, rin, r0)
    x = -1.0 + 2.0 * (y - yin) / (ycut - yin)
    return np.clip(x, -1.0, 1.0)


# ---------------------------------------------------------------------------
#  normalized Legendre polynomials (Polynomials4ML legendre_basis)
# ---------------------------------------------------------------------------

def legendre_3term(n_polys):
    """3-term recurrence arrays (A, B, C) for Q_k = sqrt((2k+1)/2) P_k in the
    OrthPolyBasis1D3T convention (Polynomials4ML `orthpolybasis`, and
    `eval/radial.py:poly_recursion`):

        Q_0 = A[0];  Q_1 = A[1] x + B[1];  Q_k = (A[k] x + B[k]) Q_{k-1} + C[k] Q_{k-2}

    Q_1 carries no Q_0 factor, so A[1] = sqrt(3/2) (the P0 normalisation is
    folded in), matching the exporter's polys_A."""
    k = np.arange(n_polys, dtype=float)
    A = np.zeros(n_polys)
    B = np.zeros(n_polys)
    C = np.zeros(n_polys)
    A[0] = 1.0 / math.sqrt(2.0)
    if n_polys > 1:
        A[1:] = ((2 * k[1:] - 1) / k[1:]) * np.sqrt((2 * k[1:] + 1) / (2 * k[1:] - 1))
        A[1] *= A[0]
        C[2:] = -((k[2:] - 1) / k[2:]) * np.sqrt((2 * k[2:] + 1) / (2 * k[2:] - 3))
    return A, B, C


def poly_eval(x, A, B, C):
    """Evaluate the recurrence at x (any shape) -> (..., n_polys); same
    convention as `poly_recursion` on the eval side."""
    x = np.asarray(x, float)
    n = len(A)
    Q = np.zeros(x.shape + (n,))
    Q[..., 0] = A[0]
    if n > 1:
        Q[..., 1] = A[1] * x + B[1]
    for k in range(2, n):
        Q[..., k] = (A[k] * x + B[k]) * Q[..., k - 1] + C[k] * Q[..., k - 2]
    return Q


# ---------------------------------------------------------------------------
#  envelopes (radial_envelopes.jl)
# ---------------------------------------------------------------------------

def envelope2sx_eval(r, x, env):
    """PolyEnvelope2sX(x1, x2, p1, p2) in the transformed coordinate x;
    env = (x1, x2, p1, p2, s), s = 1/(abs(x2-x1)/2)^(p1+p2)."""
    x1, x2, p1, p2, _s = env
    inside = (x1 < x) & (x < x2)
    return np.where(inside, env[4] * (x - x1) ** p1 * (x2 - x) ** p2, 0.0)


def envelope1sr_eval(r, env):
    """PolyEnvelope1sR(rcut, p) in the radial coordinate r; env = (rcut, p)."""
    rcut, p = env[0], env[1]
    t = np.asarray(r, float) / rcut
    return np.where(r < rcut, (t ** (-p) - 1.0) * (1.0 - t), 0.0)


def envelope2sx_params(x1=-1.0, x2=1.0, p1=2, p2=2):
    s = 1.0 / (abs(x2 - x1) / 2.0) ** (p1 + p2)
    return (float(x1), float(x2), float(p1), float(p2), float(s))


def envelope1sr_params(rcut, p=1):
    return (float(rcut), float(p))


# ---------------------------------------------------------------------------
#  Wnlq initialisation (Rnl_learnable.jl)
# ---------------------------------------------------------------------------

def glorot_weights(n_out, n_in, rng):
    """Seeded Lux-style glorot_normal for one (len_nl, len_q) matrix."""
    std = math.sqrt(2.0 / (n_out + n_in))
    return rng.normal(0.0, std, size=(n_out, n_in))


def onehot_weights(spec_n, n_q, NZ):
    """Port of `set_onehot_weights!`: the linear-ACE convention
    R_{n'l}(r, Z1, Z2) = P_{n'}(r) * delta_{z', Z2}, with
    z' = mod1(n, NZ), n' = div(n - 1, NZ) + 1 (n = channel's radial index,
    spec_n = list of n per spec row).  Returns (NZ, NZ, n_rnl, n_q)."""
    n_rnl = len(spec_n)
    W = np.zeros((NZ, NZ, n_rnl, n_q))
    for i_nl, n in enumerate(spec_n):
        z_ = (n - 1) % NZ                       # mod1(n, NZ) - 1
        n_ = (n - 1) // NZ                      # div(n-1, NZ) + 1 - 1
        if n_ < n_q:
            W[:, z_, i_nl, n_] = 1.0
    return W


def _init_Wnlq(spec_n, n_q, NZ, mode, seed):
    rng = np.random.default_rng(seed)
    if mode == "glorot_normal":
        return np.stack([np.stack(
            [glorot_weights(len(spec_n), n_q, rng) for _ in range(NZ)], axis=0)
            for _ in range(NZ)], axis=0)
    if mode == "onehot":
        return onehot_weights(spec_n, n_q, NZ)
    if mode == "zero":
        return np.zeros((NZ, NZ, len(spec_n), n_q))
    raise ValueError(f"unknown radial init mode {mode!r}")


# ---------------------------------------------------------------------------
#  the two basis initialisers
# ---------------------------------------------------------------------------

def transform_table(elements, rcut, r0, rin, p, q):
    """(NZ, NZ, 7) agnesi 7-tuples, one per species pair (per-pair r0; rin,
    rcut, p, q shared)."""
    r0t = r0_table(elements, r0)
    NZ = len(elements)
    return np.array([[agnesi_transform_params(rin, r0t[i, j], rcut, p, q)
                      for j in range(NZ)] for i in range(NZ)])


def tensor_radial_init(elements, Rnl_spec, *, rcut,
                       r0=None, rin=0.0, p=2, q=2, mode="glorot_normal", seed=0):
    """Coefficients for the many-body (tensor) radial basis.

    elements: atomic numbers; Rnl_spec: (n, l) list from `build_spec` (defines
    n_rnl and the onehot convention); r0: None (per-pair bond-length default),
    a scalar or an (NZ, NZ) table.  Returns dict with rnl_transform (NZ,NZ,7),
    rnl_envelope (NZ,NZ,5), rnl_Wnlq (NZ,NZ,n_rnl,n_q) and polys_A/B/C (n_q,)."""
    NZ = len(elements)
    n_rnl = len(Rnl_spec)
    actual_maxn = max(n for n, _ in Rnl_spec)
    n_q = math.ceil(actual_maxn * 1.5)
    env = envelope2sx_params(-1.0, 1.0, 2, 2)
    A, B, C = legendre_3term(n_q)
    return {
        "rnl_transform": transform_table(elements, rcut, r0, rin, p, q),
        "rnl_envelope": np.tile(np.array(env), (NZ, NZ, 1)),
        "rnl_Wnlq": _init_Wnlq([n for n, _ in Rnl_spec], n_q, NZ, mode, seed),
        "polys_A": A, "polys_B": B, "polys_C": C,
    }


def pair_radial_init(elements, pair_maxn, *, rcut, r0=None, rin=0.0,
                     p=1, q=4, mode="onehot", seed=0):
    """Coefficients for the pair radial basis (analytic contract; Julia
    splinifies this before export).  n_pair = pair_maxn channels (n, 0)."""
    NZ = len(elements)
    n_q = math.ceil(pair_maxn * 1.5)
    A, B, C = legendre_3term(n_q)
    return {
        "pair_transform": transform_table(elements, rcut, r0, rin, p, q),
        "pair_envelope": np.tile(np.array(envelope1sr_params(rcut, 1)), (NZ, NZ, 1)),
        "pair_Wnlq": _init_Wnlq(list(range(1, pair_maxn + 1)), n_q, NZ, mode, seed),
        "pair_polys_A": A, "pair_polys_B": B, "pair_polys_C": C,
    }
