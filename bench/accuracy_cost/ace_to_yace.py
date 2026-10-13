"""Linear ace-jax ACE model (.npz) -> ML-PACE C-tilde potential (.yace), single species.

    python bench/accuracy_cost/ace_to_yace.py MODEL.npz OUT.yace [--radial tabulated|sbessel]

The product basis converts exactly: real-Y products are expanded into PACE's complex Y (Y^R = U^-1
Y^PACE, `pace_build.ylm_map`; <= 2 terms per factor) and merged by sorted (n, l, m) tuple; true
invariants give real coefficients (the imaginary residue is reported).  Each radial column the A basis
reads, R_nl(r) r^l (solid harmonics put r^l in the radial), becomes a PACE R_nl, and the pair radial plus
the order-1 terms become one rank-1 function h(r).  Two radial forms:

- `tabulated` (default): exact, `radbasename: Tabulated` -- ace-jax's `write_yace` (src/ace_jax/export/yace.py),
  which this script's own tabulated form became; it writes the same files the ace_tab rows timed.  Needs
  ML-PACE's tabulated-radial patch (ICAMS/lammps-user-pace#31; docs/dev/ml-pace-tabulated-radials.md).
- `sbessel`: upstream ML-PACE as is: least-squares combinations of PACE's SBessel g_k on [r_fit, rcut]
  (an approximation: 2.3 meV/Å force rms on o4d20 at --nradbase 80).

The point is timing linear ACE in ML-PACE (CPU and GPU), so the model evaluates as itself.
"""
import argparse
import itertools
import math
from collections import defaultdict

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ase.data import chemical_symbols  # noqa: E402

import ace_jax as aj  # noqa: E402
from ace_jax.eval.model import fold_readout  # noqa: E402
from ace_jax.eval.pace_build import ylm_map  # noqa: E402
from ace_jax.eval.pace_io import dump_tree  # noqa: E402
from ace_jax.eval.pace_radial import radbase  # noqa: E402

S4PI = math.sqrt(4 * math.pi)
R_MIN = 0.5                     # tables continue linearly below this (Å)


def gbasis(r, rc, K):
    """PACE g_k (SBessel, inner_cutoff_type distance with rcut_in = dcut_in = 0), (len(r), K)."""
    one = jnp.ones_like(r)
    return np.asarray(radbase(jnp.asarray(r), "SBessel", "distance", 1.0 * one, rc * one, 0.01 * one,
                              0 * one, 0 * one, K))


def radials(m, r):
    """(Rnl, Rpair) and their r-derivatives at r (forward-mode autodiff of the model's own radial)."""
    z = jnp.zeros(len(r), jnp.int32)
    f = lambda rr: m.radial(jnp.stack([rr, 0 * rr, 0 * rr], -1), z, z)
    (Rnl, Rp), (dRnl, dRp) = jax.jvp(f, (jnp.asarray(r),), (jnp.ones(len(r)),))
    return tuple(np.asarray(x) for x in (Rnl, Rp, dRnl, dRp))


def products(m, cols, lmax):
    """The folded readout as PACE complex products {sorted ((n, l, m), ...): c} (rank > 1), and the order-1
    weights {radial column: c / sqrt(4 pi)} that join the pair radial in h(r)."""
    ar, ay = np.asarray(m.aspec_r), np.asarray(m.aspec_y)
    al = np.floor(np.sqrt(ay) + 1e-9).astype(int)
    am = ay - al * al - al
    ct = np.asarray(m.ctilde[:, 0])
    V = {l: np.linalg.inv(u) for l, u in ylm_map(lmax).items()}          # Y^R = V Y^P
    acc, order1 = defaultdict(complex), defaultdict(float)
    off = 0
    for g in m.aa_specs:
        g = np.asarray(g)
        for row, c in zip(g, ct[off:off + len(g)]):
            off += 1
            if c == 0:
                continue
            if len(row) == 1:                                             # c R_c(r) Y^R_00, Y^R_00 = 1/sqrt(4pi)
                order1[int(ar[row[0]])] += c / S4PI
                continue
            opts = []
            for a in row:
                l, mr = al[a], am[a]
                vrow = V[l][l + mr]                                       # Y^R_{l mr} = sum_m vrow[m] Y^P_{l m}
                n = cols[ar[a]][0]
                opts.append([((n, l, mp - l), vrow[mp]) for mp in np.nonzero(np.abs(vrow) > 1e-14)[0]])
            for combo in itertools.product(*opts):
                acc[tuple(sorted(t for t, _ in combo))] += c * np.prod([v for _, v in combo])
    return acc, order1


def convert(path, radial="sbessel", K=80, r_fit=1.6, delta=0.001):
    m, meta, _ = aj.load(path)
    if m.E0.shape[0] != 1:
        raise NotImplementedError("single species only")
    m = fold_readout(m)
    rc, lmax = float(meta["rcut"]), int(m.lmax)
    ar, ay = np.asarray(m.aspec_r), np.asarray(m.aspec_y)
    al = np.floor(np.sqrt(ay) + 1e-9).astype(int)
    # only the A entries rank > 1 products read become PACE R_nl: ML-PACE works on the whole nradmax x
    # (lmax + 1) rectangle per neighbour, and the order-1 terms are folded into the rank-1 h(r) anyway
    # (o4d20: the model's lmax 14 and 20 columns, but its products read l <= 6 and 19 radials at l = 0)
    ct = np.asarray(m.ctilde[:, 0])
    read, off = set(), 0
    for g in m.aa_specs:
        g = np.asarray(g)
        if g.shape[1] > 1:
            read.update(int(a) for a in np.unique(g[ct[off:off + len(g)] != 0]))
        off += len(g)
    cols = {}                                    # radial column -> PACE (n, l): each l's columns 1, 2, ...
    for c, l in sorted({(int(ar[a]), int(al[a])) for a in read}):
        if c in cols:
            raise ValueError(f"radial column {c} used with two l")
        cols[c] = (sum(1 for v in cols.values() if v[1] == l) + 1, l)
    nradmax = max(n for n, _ in cols.values())
    lmax = max(l for _, l in cols.values())
    acc, order1 = products(m, cols, lmax)
    wpair = np.asarray(m.Wpair[:, 0])
    h = lambda Rnl, Rp: Rp @ wpair + sum(w * Rnl[:, c] for c, w in order1.items())   # rank-1 function

    bond = {"nradmax": int(nradmax), "lmax": lmax, "prehc": 0, "lambdahc": 0, "rcut": rc, "dcut": 0.01,
            "rcut_in": 0, "dcut_in": 0, "inner_cutoff_type": "distance"}
    info = {"radial": radial}
    if radial == "sbessel":
        r = np.linspace(r_fit, rc, 4000)[:-1]
        Rnl, Rp, _, _ = radials(m, r)
        G = gbasis(r, rc, K)
        fit = lambda f: np.linalg.lstsq(G, f, rcond=None)[0]
        relerr = lambda f, w: np.max(np.abs(G @ w - f)) / max(np.max(np.abs(f)), 1e-300)
        crad, errs = np.zeros((nradmax, lmax + 1, K)), []
        for c, (n, l) in cols.items():
            f = Rnl[:, c] * r ** l
            crad[n - 1, l] = fit(f)
            errs.append(relerr(f, crad[n - 1, l]))
        w1 = fit(h(Rnl, Rp))
        bond |= {"nradbasemax": K, "radbasename": "SBessel", "radparameters": [1.0], "radcoefficients": crad.tolist()}
        rank1 = [{"mu0": 0, "rank": 1, "ndensity": 1, "num_ms_combs": 1, "mus": [0], "ns": [k + 1], "ls": [0],
                  "ms_combs": [0], "ctildes": [float(w1[k])]} for k in range(K)]
        info |= {"nradbase": K, "r_fit": r_fit, "radial_fit_max_rel": float(max(errs)),
                 "rank1_fit_max_rel": float(relerr(h(Rnl, Rp), w1))}
        delta = 0.001
    else:
        raise ValueError(f"convert writes the sbessel form; tabulated is write_yace's, got {radial!r}")

    # half basis, as pacemaker writes it: an invariant has sum(m) = 0, so P(-m) = conj(P(m)) and the
    # m-tuple and its negation contribute the same Re(c P); keep one of each pair with the summed
    # coefficient (o4d20: 45k -> 29k terms).  Which one is immaterial to ML-PACE; take the smaller.
    half = defaultdict(float)
    for key, v in acc.items():
        neg = tuple(sorted((n, l, -mm) for n, l, mm in key))
        half[min(key, neg)] += v.real
    funcs = defaultdict(list)                    # (n, l) multiset -> PACE function
    scale = max((abs(v) for v in half.values()), default=1.0)
    for key, v in half.items():
        if abs(v) < 1e-14 * scale:
            continue
        funcs[tuple((n, l) for n, l, _ in key)].append(([mm for _, _, mm in key], v))
    fl = rank1 + [{"mu0": 0, "rank": len(nl), "ndensity": 1, "num_ms_combs": len(funcs[nl]),
                   "mus": [0] * len(nl), "ns": [int(n) for n, _ in nl], "ls": [int(l) for _, l in nl],
                   "ms_combs": [int(mm) for ms, _ in sorted(funcs[nl]) for mm in ms],
                   "ctildes": [float(c) for _, c in sorted(funcs[nl])]}
                  for nl in sorted(funcs, key=lambda t: (len(t), t))]
    tree = {
        "elements": [chemical_symbols[int(m.elements[0])]], "E0": [float(m.E0[0])], "deltaSplineBins": delta,
        "embeddings": {0: {"ndensity": 1, "FS_parameters": [1.0, 1.0], "npoti": "FinnisSinclairShiftedScaled",
                           "rho_core_cutoff": 100000, "drho_core_cutoff": 250}},
        "bonds": {(0, 0): bond}, "functions": {0: fl},
    }
    info |= {"nradmax": int(nradmax), "lmax": lmax, "n_functions": len(fl),
             "n_terms": sum(f["num_ms_combs"] for f in fl),
             "max_imag_rel": float(max((abs(v.imag) for v in acc.values()), default=0.0)
                                   / max((abs(v) for v in acc.values()), default=1.0))}
    return tree, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--radial", default="tabulated", choices=("tabulated", "sbessel"))
    ap.add_argument("--delta", type=float, default=0.001, help="tabulated: grid spacing = deltaSplineBins (Å)")
    ap.add_argument("--nradbase", type=int, default=80, help="sbessel: number of g_k")
    ap.add_argument("--r-fit", type=float, default=1.6, help="sbessel: radials are fitted on [r_fit, rcut] (Å)")
    a = ap.parse_args()
    if a.radial == "tabulated":                  # exact: ace-jax's write_yace (identical files to this script's
        from ace_jax.eval import write_yace       # former tabulated form, which timed the ace_tab rows)
        m, _, _ = aj.load(a.model)
        print(write_yace(m, a.out, delta=a.delta))
        return
    tree, info = convert(a.model, a.radial, a.nradbase, a.r_fit, a.delta)
    dump_tree(tree, a.out)
    print(info)


if __name__ == "__main__":
    main()
