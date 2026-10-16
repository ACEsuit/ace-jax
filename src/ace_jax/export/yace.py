"""A linear ACEModel as an ML-PACE C-tilde potential (`.yace`), exactly.

`ace_ctilde_tree(model)` builds the `.yace` tree that `pace_io.write_yace` writes for an ACEModel; ML-PACE
(`pair_style pace`, `pace/kk`) then evaluates the model itself.  Exact: the radials are tables of the
model's own values and derivatives, the products a re-expression of its coefficients.

- **Radials** (`radbasename: Tabulated`): each radial column the rank > 1 products read, `R(r) r^l` with
  solid harmonics (`R(r)` with spherical), becomes one PACE `R_nl` per bond, tabulated with its autodiff
  r-derivative on ML-PACE's own look-up grid (nbins = rcut / deltaSplineBins), which its cubic-Hermite
  tables reproduce exactly.  The pair radial and the order-1 terms become one rank-1 `g_1` per bond.
  Below `r_min` the tables continue linearly (the ACE pair envelope is singular at r -> 0).  ML-PACE needs
  the tabulated-radial addition (docs/dev/ml-pace-tabulated-radials.md); `parse_yace` reads it.
- **Products**: ACE's real Y become PACE's complex Y (Y^R = U^-1 Y^PACE, `pace_build.ylm_map`; <= 2 terms
  a factor), merged by sorted (mu, n, l, m), then halved as pacemaker writes them: an invariant has
  sum(m) = 0, so an m-tuple and its negation give the same Re(c P).
- **Species**: ACE pools every neighbour species into one A, its species entering through the radial;
  PACE keeps one A per neighbour species.  So each factor becomes a sum over the neighbour species its
  radial column is live for (nonzero on some bond), and an order-nu product up to NZ^nu PACE products.
  ACE1's species-owned columns are live for one species each, so they add no products; PACE n is
  numbered per (neighbour species, l), so such columns share PACE R_nl slots.
- **summed=True** keeps ACE's pooled A instead: every factor is element 0, each bond's tables hold that bond's
  own values of every column, and the tree sets `summed_neighbour_A`, which tells ML-PACE to accumulate every
  neighbour into the element-0 A (an ML-PACE addition, pace_ref/fast/0004).  No NZ^nu expansion: the right form
  for species-embedded (tensor-reduced) models, whose columns are live for every species; for ACE1's
  species-owned columns it widens nradmax NZ-fold instead.
- **Shape**: lmax and nradmax are what the products read, not the model's (ML-PACE works on the whole
  nradmax x (lmax+1) rectangle per neighbour; o4d20 has lmax 14 but its products read l <= 6).

Only the linear ACEModel (no GP, no wrapper); the full model, not `lean` (it needs the A basis).
"""
import itertools
import math
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from ase.data import chemical_symbols

DELTA = 0.001                    # deltaSplineBins (Å): ML-PACE's default look-up grid
R_MIN = 0.5                      # tables continue linearly below this (Å)
_S4PI = math.sqrt(4 * math.pi)


def _read_columns(model, ctilde):
    """The radial columns rank > 1 products read (with a nonzero coefficient for some centre species):
    {column: l}, and the A-entry maps (radial column, l, m) of every A entry."""
    ar, ay = np.asarray(model.aspec_r), np.asarray(model.aspec_y)
    al = np.floor(np.sqrt(ay) + 1e-9).astype(int)
    read, off = set(), 0
    for g in model.aa_specs:
        g = np.asarray(g)
        live = np.any(ctilde[off:off + len(g)] != 0, axis=1)
        if g.shape[1] > 1:
            read.update(int(a) for a in np.unique(g[live]))
        off += len(g)
    col_l = {}
    for a in sorted(read):
        c, l = int(ar[a]), int(al[a])
        if col_l.setdefault(c, l) != l:
            raise ValueError(f"radial column {c} is read with two l: not an ACE1/ACEpotentials R_nl basis")
    return col_l, ar, al, ay - al * al - al


def _sample(model, z_i, z_j, rcut, n_bins, r_min):
    """(r, Rnl, dRnl, Rpair, dRpair) of bond (z_i, z_j) on r_k = k rcut / n_bins, from the model's own
    radial (forward-mode autodiff for the r-derivatives); below r_min the values are those at r_min."""
    r = np.arange(n_bins + 1) * (rcut / n_bins)
    rs = jnp.asarray(np.maximum(r, r_min))
    zi, zj = jnp.full(len(r), z_i, jnp.int32), jnp.full(len(r), z_j, jnp.int32)
    f = lambda x: model.radial(jnp.stack([x, 0 * x, 0 * x], -1), zi, zj)
    (Rnl, Rp), (dRnl, dRp) = jax.jvp(f, (rs,), (jnp.ones(len(r)),))
    return (r,) + tuple(np.asarray(x) for x in (Rnl, dRnl, Rp, dRp))


def _products(model, ctilde, numbering, ar, al, am, lmax):
    """Per centre species: {sorted ((mu, n, l, m), ...): c} (rank > 1, half basis), each factor summed over
    the neighbour species mu its radial column is live for (numbering[mu]: column -> PACE n), and the
    order-1 weights {radial column: c / sqrt(4 pi)} that join the pair radial in g_1."""
    from ..eval.pace_build import ylm_map
    nz = ctilde.shape[1]
    V = {l: np.linalg.inv(u) for l, u in ylm_map(lmax).items()}           # Y^R = V Y^P
    acc = [defaultdict(complex) for _ in range(nz)]
    order1 = [defaultdict(float) for _ in range(nz)]
    off = 0
    for g in model.aa_specs:
        g = np.asarray(g)
        for row, cz in zip(g, ctilde[off:off + len(g)]):
            off += 1
            if not np.any(cz):
                continue
            if len(row) == 1:
                for z in np.nonzero(cz)[0]:
                    order1[z][int(ar[row[0]])] += cz[z] / _S4PI
                continue
            opts = []                    # per factor: [((mu, n, l, m), coefficient)]
            for a in row:
                c, l, mr = int(ar[a]), int(al[a]), int(am[a])
                vrow = V[l][l + mr]
                opts.append([((mu, numbering[mu][c], l, mp - l), vrow[mp]) for mu in range(nz) if c in numbering[mu]
                             for mp in np.nonzero(np.abs(vrow) > 1e-14)[0]])
            for combo in itertools.product(*opts):
                key = tuple(sorted(t for t, _ in combo))
                v = np.prod([w for _, w in combo])
                for z in np.nonzero(cz)[0]:
                    acc[z][key] += cz[z] * v
    imag = max((abs(v.imag) for a in acc for v in a.values()), default=0.0)
    scale = max((abs(v) for a in acc for v in a.values()), default=1.0)
    half = []
    for a in acc:                    # Re(c P(-m)) = Re(c P(m)): keep the smaller key, summed coefficient
        h = defaultdict(float)
        for key, v in a.items():
            neg = tuple(sorted((mu, n, l, -m) for mu, n, l, m in key))
            h[min(key, neg)] += v.real
        half.append({k: v for k, v in h.items() if abs(v) >= 1e-14 * scale})
    return half, order1, imag / scale


def _table(r, v, dv, rcut, r_min):
    """Values and derivatives as written: zero from rcut on, linear below r_min."""
    v = np.where(r < r_min, v + dv * (r - r_min), v)
    return np.where(r >= rcut, 0.0, v).tolist(), np.where(r >= rcut, 0.0, dv).tolist()


def ace_ctilde_tree(model, *, delta=DELTA, r_min=R_MIN, summed=False, info=None):
    """The `.yace` tree of a linear ACEModel (see the module docstring).  `delta` is deltaSplineBins and
    the table spacing; each bond's rcut must be a whole number of it (as ML-PACE sizes its tables:
    int(rcut / deltaSplineBins)).  `info`, a dict, receives the export's sizes."""
    from ..eval.model import ACEModel, fold_readout
    if type(model) is not ACEModel:
        raise TypeError(f"ace_ctilde_tree takes a linear ACEModel, not {type(model).__name__}")
    model.require_full("the .yace export")
    if not jax.config.jax_enable_x64:
        raise ValueError("the .yace export needs float64 (jax.config.update('jax_enable_x64', True))")
    model = fold_readout(model)
    ctilde = np.asarray(model.ctilde, np.float64)                          # (n_AA, NZ)
    nz = ctilde.shape[1]
    col_l, ar, al, am = _read_columns(model, ctilde)
    if not col_l:
        raise ValueError("the model has no rank > 1 term with a nonzero coefficient")
    rcuts = np.asarray(model.table_cutoffs(), np.float64)
    bins, samples = {}, {}
    for i, j in itertools.product(range(nz), repeat=2):
        rc = float(rcuts[i, j])
        n_bins = int(round(rc / delta))
        if abs(n_bins * delta - rc) > 1e-9 * rc or int(rc / delta) != n_bins:
            raise ValueError(f"bond {i}-{j}: rcut {rc} is not a whole number of deltaSplineBins {delta}")
        bins[i, j] = n_bins
        samples[i, j] = _sample(model, i, j, rc, n_bins, r_min)
    # PACE n per (neighbour species, l): a column live for neighbour species mu (nonzero on some bond
    # (z, mu)) takes the next n of its l there.  ACE1's species-owned columns then share PACE slots, and a
    # factor is summed only over the species its column is live for.
    # summed: one numbering, for element 0, over the columns live on any bond; the other elements own none
    numbering = []
    for mu in range(1 if summed else nz):
        live = [c for c in sorted(col_l)
                if any(np.any(samples[z, nu][1][:, c] != 0) for z in range(nz) for nu in (range(nz) if summed else (mu,)))]
        count = defaultdict(int)
        numbering.append({})
        for c in live:
            count[col_l[c]] += 1
            numbering[mu][c] = count[col_l[c]]
    if summed:
        numbering += [{} for _ in range(nz - 1)]
    lmax = max(col_l.values())
    nradmax = max((n for nm in numbering for n in nm.values()), default=1)
    half, order1, imag = _products(model, ctilde, numbering, ar, al, am, lmax)
    bonds = {}
    for (i, j), (r, Rnl, dRnl, Rp, dRp) in samples.items():
        rc, rs = float(rcuts[i, j]), np.maximum(r, r_min)
        wp = np.asarray(model.Wpair[:, i])
        g = Rp @ wp + sum(w * Rnl[:, c] for c, w in order1[i].items())
        dg = dRp @ wp + sum(w * dRnl[:, c] for c, w in order1[i].items())
        gf, gdf = _table(r, g, dg, rc, r_min)
        rnl = []
        for c, n in sorted(numbering[0 if summed else j].items(), key=lambda t: (col_l[t[0]], t[1])):
            l = col_l[c]
            p = l if model.ysolid else 0                  # solid harmonics: r^l Y_lm, r^l goes in the radial
            v = Rnl[:, c] * rs ** p
            dv = dRnl[:, c] * rs ** p + (p * Rnl[:, c] * rs ** (p - 1) if p else 0.0)
            vf, vdf = _table(r, v, dv, rc, r_min)
            rnl.append({"n": int(n), "l": int(l), "f": vf, "df": vdf})
        bonds[(i, j)] = {"nradmax": int(nradmax), "lmax": int(lmax), "nradbasemax": 1, "radbasename": "Tabulated",
                         "prehc": 0, "lambdahc": 0, "rcut": rc, "dcut": 0.01, "rcut_in": 0, "dcut_in": 0,
                         "inner_cutoff_type": "distance",
                         "radtable": {"nbins": bins[i, j], "gk": [{"k": 1, "f": gf, "df": gdf}], "rnl": rnl}}
    functions = {}
    for z in range(nz):
        fl = [{"mu0": z, "rank": 1, "ndensity": 1, "num_ms_combs": 1, "mus": [mu], "ns": [1], "ls": [0],
               "ms_combs": [0], "ctildes": [1.0]} for mu in range(1 if summed else nz)]   # g_1 per neighbour species
        groups = defaultdict(list)
        for key, c in half[z].items():
            groups[tuple((mu, n, l) for mu, n, l, _ in key)].append(([m for *_, m in key], c))
        for spec in sorted(groups, key=lambda t: (len(t), t)):
            terms = sorted(groups[spec])
            fl.append({"mu0": z, "rank": len(spec), "ndensity": 1, "num_ms_combs": len(terms),
                       "mus": [int(mu) for mu, _, _ in spec], "ns": [int(n) for _, n, _ in spec],
                       "ls": [int(l) for _, _, l in spec],
                       "ms_combs": [int(m) for ms, _ in terms for m in ms], "ctildes": [float(c) for _, c in terms]})
        functions[z] = fl
    tree = {"elements": [chemical_symbols[int(z)] for z in model.elements],
            "E0": [float(e) for e in np.asarray(model.E0)], "deltaSplineBins": float(delta),
            "embeddings": {z: {"ndensity": 1, "FS_parameters": [1.0, 1.0], "npoti": "FinnisSinclairShiftedScaled",
                               "rho_core_cutoff": 100000, "drho_core_cutoff": 250} for z in range(nz)},
            "bonds": bonds, "functions": functions, **({"summed_neighbour_A": True} if summed else {})}
    if info is not None:
        info.update({"nradmax": int(nradmax), "lmax": int(lmax), "radials": len(col_l),
                     "functions": sum(len(f) for f in functions.values()),
                     "terms": sum(f["num_ms_combs"] for fl in functions.values() for f in fl),
                     "max_imag_rel": float(imag), "nbins": {f"{i}-{j}": b["radtable"]["nbins"]
                                                            for (i, j), b in bonds.items()}})
    return tree
