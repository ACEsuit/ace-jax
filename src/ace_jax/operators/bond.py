"""Off-site features from the shared site features (design option B).

For a directed bond i -> j with vector r, the features of symmetry (L, p):

  "bond"  phi_n(r) Y_LM(r)                           the two-centre functions; parity (-1)^L
  "i"     phi_n(r) [Y_l(r) (x) B_i^(lam, pi)]^L      bond harmonics coupled with a site feature of i
  "j"     the same with the site features of j       parity (-1)^l pi

with the real CG tensors of `cg`.  The site features are the ones the on-site
model evaluates, once per atom, so a bond costs only these contractions: no
pooling over an environment of its own.  Under r -> -r and i <-> j the groups
map into each other ("bond" and the coupled terms pick up (-1)^L and (-1)^l),
which `symmetric` uses for blocks between equal shells of equal species.

What these three groups cannot represent: terms joint in the environments of i and j
beyond their sum, and an environment centred on the bond.  On aluminium that leaves the
blocks of bonds beyond the first neighbour shell too inaccurate for band structures
(docs/dev/ham-results.md), so there is a fourth group:

  "env"   phi_n(r) [Y_l(r) (x) A^env_(n' lam)]^L     A^env = sum_k w_k R_n'(|x_k|) Y_lam(x_k)

the bond-environment ACE at correlation order 1 (ACEhamiltonians' off-site model): x_k is the
position of environment atom k relative to the bond's MIDPOINT, and w_k = e(d_ik) + e(d_jk)
with e a smooth cutoff at `r_env` around each end (a dumbbell where ACEhamiltonians has a
cylinder; for bonds up to about 2 r_env it covers the same region, and it lets the pooling
run over each end's own neighbour list).  It is the same for a bond and its reverse, so under
reversal the group only takes the sign (-1)^l of the bond harmonic.  This group does pool an
environment per bond.

The environment atoms' species enter through channels: A^env_(c n' lam) carries a further factor
emb[z_k, c], with emb the (elements, channels) table `env_embedding`: the identity by default (one
channel per element of the site basis: a categorical environment; one element, one channel), or
`BondBasis.env_emb`, an element embedding whose width, not the number of elements, then sets the
number of features."""
from typing import NamedTuple

import numpy as np

from ..eval.harmonics import real_spherical_harmonics
from ..eval.nlist import sparse_graph
from ..basis import cg
from .features import SiteBasis


class BondBasis(NamedTuple):
    """site: the shared `SiteBasis`; rcut_bond: longest bond with a block; n_bond: radial
    functions of the two-centre features; n_cpl, l_cpl: radial functions and largest bond
    harmonic of the coupled features (0 radial functions: two-centre only).  Bond environment
    (off when n_env = 0): r_env, the cutoff around each end; n_env, l_env: radial functions and
    largest harmonic of the environment about the midpoint; n_ebond, l_ebond: radial functions
    and largest bond harmonic it is coupled with; env_emb: None (one channel per element) or the
    (elements, channels) embedding of the environment atoms' species as nested tuples, rows in the
    order of `site.elements`.  env_core (angstrom): the harmonics of degree l of an environment
    atom at distance x from the midpoint are scaled by (x^2 / (x^2 + env_core^2))^(l/2), `core`:
    1 away from the midpoint and ~ x^l at it, so that the features are smooth there.  0 switches
    it off: the form of ace-jax 0.7.0, whose features jump when an atom crosses the midpoint (a
    harmonic of l > 0 has no value at x = 0), which made symmetry-equivalent bonds of a perfect
    lattice differ (an atom at the midpoint of a second-neighbour bond; ACEhamiltonians.jl met the
    same and moved its origin to the first atom).  Files written by 0.7.0 load with 0."""

    site: SiteBasis
    rcut_bond: float
    n_bond: int
    n_cpl: int = 0
    l_cpl: int = 0
    r_env: float = 0.0
    n_env: int = 0
    l_env: int = 0
    n_ebond: int = 0
    l_ebond: int = 0
    env_emb: tuple = None
    env_core: float = 1.0


class BondGraph(NamedTuple):
    """Directed bonds: block (R, i, j) couples atom i in the home cell with atom j in cell R;
    rij is the vector from i to j; reverse[k] is the index of the bond (-R, j, i)."""

    i: np.ndarray
    j: np.ndarray
    R: np.ndarray
    rij: np.ndarray
    reverse: np.ndarray


def radial(r, rcut, n):
    """(..., n): Chebyshev polynomials in 2 r / rcut - 1 times (1 - (r / rcut)^2)^2, which
    vanishes with its first derivative at the cutoff."""
    x = np.clip(np.asarray(r, float) / rcut, 0.0, 1.0)
    t = 2 * x - 1
    T = [np.ones_like(t), t]
    for _ in range(2, n):
        T.append(2 * t * T[-1] - T[-2])
    return np.stack(T[:n], axis=-1) * ((1 - x * x) ** 2)[..., None] if n else np.zeros((*np.shape(r), 0))


def bond_graph(basis, positions, cell, numbers, pbc=True):
    positions, cell = np.asarray(positions, float), np.asarray(cell, float)
    g = sparse_graph(positions, cell, pbc, basis.rcut_bond)
    R = np.rint(g.shifts @ np.linalg.inv(cell)).astype(np.int64)
    i, j = g.senders.astype(np.int64), g.receivers.astype(np.int64)
    # reverse[k]: the bond (j, i, -R).  Both key lists hold the same set, so their sort orders pair them up.
    fwd = np.lexsort((R[:, 2], R[:, 1], R[:, 0], j, i))
    bwd = np.lexsort((-R[:, 2], -R[:, 1], -R[:, 0], i, j))
    reverse = np.empty(len(i), np.int64)
    reverse[bwd] = fwd
    if len(i) and not (np.array_equal(i[reverse], j) and np.array_equal(R[reverse], -R)):
        raise ValueError("bond list is not symmetric: a pair within the bond cutoff has no reverse")
    return BondGraph(i, j, R, g.rij, reverse)


def env_embedding(basis):
    """(elements, channels) table of the environment atoms' species: `env_emb`, or the identity."""
    S = len(basis.site.elements)
    if basis.env_emb is None:
        return np.eye(S)
    emb = np.asarray(basis.env_emb, float)
    if emb.ndim != 2 or emb.shape[0] != S:
        raise ValueError(f"env_emb must have one row per element of the site basis ({S}), got shape {emb.shape}")
    return emb


def env_lists(basis, positions, cell, pbc=True, numbers=None):
    """For the bond environment: dense neighbour vectors within `r_env` of every atom (n, K, 3), counts
    (n,) and the neighbours' species indices (n, K) (`site.species_index`; `numbers` may be left out
    for a one-element basis).  None when the basis has no environment group."""
    if not basis.n_env:
        return None
    from ..eval.nlist import dense_graph
    positions, cell = np.asarray(positions, float), np.asarray(cell, float)
    g = sparse_graph(positions, cell, pbc, basis.r_env)
    K = max(int(np.bincount(g.senders, minlength=len(positions)).max()) if len(g.rij) else 1, 1)
    d = dense_graph(positions, cell, pbc, basis.r_env, K)
    if numbers is None:
        if len(basis.site.elements) > 1:
            raise ValueError("env_lists: pass numbers= (the environment group needs the neighbours' species)")
        z = np.zeros(np.asarray(d.idx).shape, np.int32)
    else:
        z = basis.site.species_index(numbers)[np.asarray(d.idx)]
    return np.asarray(d.rij), np.asarray(d.count), z


def env_layout(basis, L, p):
    """[(l, lam)] of the bond-environment groups of symmetry (L, p): bond harmonic l with the
    environment's lam, parity (-1)^(l + lam)."""
    if not basis.n_env:
        return []
    return [(l, lam) for lam in range(basis.l_env + 1) for l in range(basis.l_ebond + 1)
            if (-1) ** (l + lam) == p and abs(l - lam) <= L <= l + lam]


def core(xp, x2, l_env, a):
    """(..., (l_env+1)^2): the factor (x^2 / (x^2 + a^2))^(l/2) of each harmonic (l, m) at squared
    distance x2 from the midpoint (`BondBasis.env_core`); xp: numpy or jax.numpy.  a = 0: ones."""
    ls = np.repeat(np.arange(l_env + 1), 2 * np.arange(l_env + 1) + 1)
    if not a:
        return xp.ones((*x2.shape, len(ls)), x2.dtype)
    return (x2 / (x2 + a * a))[..., None] ** xp.asarray(0.5 * ls, x2.dtype)


def _env_side(basis, env, atoms, r):
    """(n_bonds, channels * n_env, (l_env+1)^2), channel-major: the half of A^env pooled over the
    neighbours of the bond's first atom `atoms`, for bond vectors `r` (the partner atom itself is
    left out)."""
    rij, count, z = env
    d = rij[atoms]                                                     # (e, K, 3) from the first atom
    K = d.shape[1]
    dist = np.linalg.norm(d, axis=-1)
    live = (np.arange(K)[None, :] < count[atoms][:, None]) & (np.linalg.norm(d - r[:, None, :], axis=-1) > 1e-6)
    w = np.where(live & (dist < basis.r_env), (1 - np.clip(dist / basis.r_env, 0, 1) ** 2) ** 2, 0.0)
    x = d - 0.5 * r[:, None, :]                                        # relative to the midpoint
    R = radial(np.linalg.norm(x, axis=-1), basis.r_env + 0.5 * basis.rcut_bond, basis.n_env)
    Y = np.asarray(real_spherical_harmonics(x.reshape(-1, 3), max(basis.l_env, 1))).reshape(*x.shape[:2], -1)
    Y = Y[..., :(basis.l_env + 1) ** 2] * core(np, (x * x).sum(-1), basis.l_env, basis.env_core)
    ch = env_embedding(basis)[z[atoms]]                                # (e, K, channels)
    return np.einsum("ek,ekc,ekn,eky->ecny", w, ch, R, Y).reshape(len(atoms), -1, (basis.l_env + 1) ** 2)


def layout(basis, L, p):
    """What the (L, p) features of a bond are: (has two-centre features, [(l, lam, pi, n_B)])
    for the coupled groups, in the order `features` stacks them."""
    two = (1 if L % 2 == 0 else -1) == p and basis.n_bond > 0
    cpl = []
    if basis.n_cpl:
        for lam, pi in basis.site.lps:
            for l in range(basis.l_cpl + 1):
                if ((-1) ** l) * pi == p and abs(l - lam) <= L <= l + lam:
                    cpl.append((l, lam, pi, basis.site.n(lam, pi)))
    return two, cpl


def n_features(basis, L, p):
    """(two-centre, coupled per site) feature counts of symmetry (L, p)."""
    two, cpl = layout(basis, L, p)
    return (basis.n_bond if two else 0), basis.n_cpl * sum(nB for *_, nB in cpl)


def n_env_features(basis, L, p, same=False):
    """Bond-environment feature count of symmetry (L, p); `same`: only the groups a block
    between equal shells of equal species keeps (bond harmonic of the parity of L)."""
    if not basis.n_env:
        return 0
    return (basis.n_ebond * basis.n_env * env_embedding(basis).shape[1]
            * sum(1 for l, _ in env_layout(basis, L, p) if not same or (l - L) % 2 == 0))


def smoothness(basis, L, p, same):
    """(n_columns,) the smoothness prior gamma of each column of `design` for symmetry (L, p):
    `basis.prior`'s functional sum (l / wl)^p + (n / wn)^p over the factors of a feature.
    Two-centre: (bond radial n, L).  Coupled: (bond radial n, bond l) plus the site feature's
    own gamma.  Bond environment: (bond radial n, bond l) plus (environment radial n', lam),
    the same for every species channel.  Radial indices count from 1 (the polynomial degree
    plus one of the Chebyshev radials here)."""
    from ..basis.prior import P, WL, WN
    term = lambda n, l: (l / WL) ** P + (n / WN) ** P
    two, cpl = layout(basis, L, p)
    out = [np.array([term(n, L) for n in range(1, basis.n_bond + 1)]) if two else np.zeros(0)]
    g = [np.add.outer([term(n, l) for n in range(1, basis.n_cpl + 1)], basis.site.smoothness(lam, pi)).ravel()
         for l, lam, pi, _ in cpl]
    g = np.concatenate(g) if g else np.zeros(0)
    out += [g] if same else [g, g]
    if basis.n_env:
        ch = env_embedding(basis).shape[1]
        for l, lam in env_layout(basis, L, p):
            if same and (l - L) % 2:
                continue
            inner = np.tile([term(n, lam) for n in range(1, basis.n_env + 1)], ch)             # channel-major
            out.append(np.add.outer([term(n, l) for n in range(1, basis.n_ebond + 1)], inner).ravel())
    return np.concatenate(out)


def features(basis, F, graph, sel, lps, env=None):
    """{(L, p): {"bond": (n, k0, 2L+1), "i": (n, k1, 2L+1), "j": (n, k1, 2L+1), "sign": (k1,)}}
    for the bonds `sel` (indices into `graph`) and the symmetries `lps`.  `F`: site features
    {(lam, pi): (n_atoms, n_B, 2 lam + 1)} as numpy arrays.  "sign"[k] = (-1)^l of coupled
    feature k: the factor its group takes under r -> -r.  With a bond environment (`env`:
    `env_lists`) also "env": (n, k2, 2L+1) and "env_sign": (k2,)."""
    sel = np.asarray(sel)
    r = graph.rij[sel]
    dist = np.linalg.norm(r, axis=1)
    lmax = max([L for L, _ in lps] + [basis.l_cpl])
    Y = np.asarray(real_spherical_harmonics(r, max(lmax, 1)))
    y = lambda l: Y[:, l * l:(l + 1) ** 2]
    phi0, phi1 = radial(dist, basis.rcut_bond, basis.n_bond), radial(dist, basis.rcut_bond, basis.n_cpl)
    i, j = graph.i[sel], graph.j[sel]
    if basis.n_env:
        if env is None:
            raise ValueError("this bond basis has a bond environment: pass env=env_lists(...)")
        # the environment about the midpoint: each end's half; the second end's is the reverse bond's first
        Aenv = _env_side(basis, env, i, r) + _env_side(basis, env, j, -r)
        phi2 = radial(dist, basis.rcut_bond, basis.n_ebond)
    out = {}
    for L, p in lps:
        two, cpl = layout(basis, L, p)
        k = 2 * L + 1
        bond = phi0[:, :, None] * y(L)[:, None, :] if two else np.zeros((len(sel), 0, k))
        gi, gj, sign = [], [], []
        for l, lam, pi, nB in cpl:
            T = cg.real_cg(l, lam, L)                                      # (m, mu, M)
            W = np.einsum("em,mnM->enM", y(l), T)                          # (e, mu, M)
            for site, acc in ((i, gi), (j, gj)):
                g = np.einsum("ebn,enM->ebM", F[(lam, pi)][site], W)       # (e, n_B, M)
                acc.append((phi1[:, :, None, None] * g[:, None]).reshape(len(sel), -1, k))
            sign += [(-1) ** l] * (basis.n_cpl * nB)
        empty = np.zeros((len(sel), 0, k))
        out[(L, p)] = {"bond": bond, "i": np.concatenate(gi, axis=1) if gi else empty,
                       "j": np.concatenate(gj, axis=1) if gj else empty, "sign": np.asarray(sign, float)}
        ge, esign = [], []
        for l, lam in env_layout(basis, L, p):
            W = np.einsum("em,mnM->enM", y(l), cg.real_cg(l, lam, L))
            a = np.einsum("ebn,enM->ebM", Aenv[:, :, lam * lam:(lam + 1) ** 2], W)       # (e, channels n_env, M)
            ge.append((phi2[:, :, None, None] * a[:, None]).reshape(len(sel), -1, k))
            esign += [(-1) ** l] * (basis.n_ebond * Aenv.shape[1])
        out[(L, p)]["env"] = np.concatenate(ge, axis=1) if ge else empty
        out[(L, p)]["env_sign"] = np.asarray(esign, float)
    return out


def design(f, L, same):
    """(n, n_feat, 2L+1) feature matrix of a canonical block from `features`' entry `f`.

    Different shells or species: [bond | i | j].  Equal shells of equal species (`same`): the
    block of the reverse bond is this block's transpose, so the coefficients of the i and j
    groups are tied: [bond | (i + (-1)^L sign j) / 2], and two-centre features exist only for
    even L (they do anyway: parity +).
    The bond-environment group is the same for a bond and its reverse up to the sign of its bond
    harmonic, so it needs no tying; an equal-shell block keeps only its features of sign (-1)^L."""
    if not same:
        return np.concatenate([f["bond"], f["i"], f["j"], f["env"]], axis=1)
    s = ((-1) ** L) * f["sign"]
    keep = f["env_sign"] == (-1) ** L
    return np.concatenate([f["bond"], 0.5 * (f["i"] + s[None, :, None] * f["j"]), f["env"][:, keep]], axis=1)
