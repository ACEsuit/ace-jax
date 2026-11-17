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

What this basis cannot represent: terms joint in the environments of i and j
beyond their sum, and an environment centred on the bond.  Its accuracy is an
empirical question (docs/dev/ham-results.md)."""
from typing import NamedTuple

import numpy as np

from ..eval.harmonics import real_spherical_harmonics
from ..eval.nlist import sparse_graph
from . import cg
from .features import SiteBasis


class BondBasis(NamedTuple):
    """site: the shared `SiteBasis`; rcut_bond: longest bond with a block; n_bond: radial
    functions of the two-centre features; n_cpl, l_cpl: radial functions and largest bond
    harmonic of the coupled features (0 radial functions: two-centre only)."""

    site: SiteBasis
    rcut_bond: float
    n_bond: int
    n_cpl: int = 0
    l_cpl: int = 0


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


def features(basis, F, graph, sel, lps):
    """{(L, p): {"bond": (n, k0, 2L+1), "i": (n, k1, 2L+1), "j": (n, k1, 2L+1), "sign": (k1,)}}
    for the bonds `sel` (indices into `graph`) and the symmetries `lps`.  `F`: site features
    {(lam, pi): (n_atoms, n_B, 2 lam + 1)} as numpy arrays.  "sign"[k] = (-1)^l of coupled
    feature k: the factor its group takes under r -> -r."""
    sel = np.asarray(sel)
    r = graph.rij[sel]
    dist = np.linalg.norm(r, axis=1)
    lmax = max([L for L, _ in lps] + [basis.l_cpl])
    Y = np.asarray(real_spherical_harmonics(r, max(lmax, 1)))
    y = lambda l: Y[:, l * l:(l + 1) ** 2]
    phi0, phi1 = radial(dist, basis.rcut_bond, basis.n_bond), radial(dist, basis.rcut_bond, basis.n_cpl)
    i, j = graph.i[sel], graph.j[sel]
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
    return out


def design(f, L, same):
    """(n, n_feat, 2L+1) feature matrix of a canonical block from `features`' entry `f`.

    Different shells or species: [bond | i | j].  Equal shells of equal species (`same`): the
    block of the reverse bond is this block's transpose, so the coefficients of the i and j
    groups are tied: [bond | (i + (-1)^L sign j) / 2], and two-centre features exist only for
    even L (they do anyway: parity +)."""
    if not same:
        return np.concatenate([f["bond"], f["i"], f["j"]], axis=1)
    s = ((-1) ** L) * f["sign"]
    return np.concatenate([f["bond"], 0.5 * (f["i"] + s[None, :, None] * f["j"])], axis=1)
