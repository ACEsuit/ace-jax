"""Offline (numpy) translation of a PACE C-tilde basis into ace-jax's real
product basis.

PACE forms products of complex A_{mu,n,l,m} (Y_lm with Y00 = 1, Condon-Shortley
phase, negative m by (-1)^m conj) and keeps Re(c . prod).  ace-jax pools real
A with SpheriCart's L2-normalised real Y.  Y^PACE_l = U_l Y^R_l with U_l having
at most 2 non-zeros per row, so each complex product expands into <= 2^rank real
products; their real parts are merged by sorted index tuple into `aa_specs`,
and T maps the complex coefficients onto the merged real ones.  ctilde_complex
stays the model's leaf, so export never inverts this map.
"""
import itertools
import math
from collections import defaultdict

import numpy as np

_S4PI = math.sqrt(4.0 * math.pi)


def ylm_map(lmax):
    """U[l] (2l+1, 2l+1) complex: rows PACE m = -l..l, cols real m' = -l..l.

    Verified against ML-PACE's compute_ylm in tests/test_pace_build.py."""
    U = {}
    r2 = math.sqrt(2.0)
    for l in range(lmax + 1):
        u = np.zeros((2 * l + 1, 2 * l + 1), complex)
        u[l, l] = _S4PI
        for m in range(1, l + 1):
            sg = (-1) ** m
            u[l + m, l + m] = _S4PI * sg / r2
            u[l + m, l - m] = 1j * _S4PI * sg / r2
            u[l - m, l + m] = _S4PI / r2
            u[l - m, l - m] = -1j * _S4PI / r2
        U[l] = u
    return U


def build_basis(arrays):
    funcs, K = arrays["funcs"], int(arrays["nradbase"])
    lmax = int(arrays["lmax"])
    NZ = len(arrays["Z"])
    U = ylm_map(lmax)

    a_index = {}                                   # (rad_col, y_col) -> local a
    def a_of(col, l, mp):
        key = (col, l * l + l + mp)
        if key not in a_index:
            a_index[key] = len(a_index)
        return a_index[key]

    acc = defaultdict(float)                       # (sorted (mu, a) tuple, el, term) -> val
    term = 0
    for f in funcs:
        rank = f["rank"]
        cols = [(f["ns"][0] - 1) if rank == 1
                else K + (f["ns"][k] - 1) * (lmax + 1) + f["ls"][k] for k in range(rank)]
        ls = [0] * rank if rank == 1 else list(f["ls"])
        for ms in f["ms"]:
            opts = []
            for k in range(rank):
                row = U[ls[k]][ms[k] + ls[k]]
                opts.append([(mp - ls[k], row[mp]) for mp in np.nonzero(row)[0]])
            for combo in itertools.product(*opts):
                coef = np.prod([c for _, c in combo]).real
                if abs(coef) < 1e-14:
                    continue
                key = tuple(sorted((f["mus"][k], a_of(cols[k], ls[k], combo[k][0]))
                                   for k in range(rank)))
                acc[(key, f["el"], term)] += coef
            term += 1

    n_a = len(a_index)
    a_rad = np.zeros(n_a, np.int32)
    a_y = np.zeros(n_a, np.int32)
    for (col, y), a in a_index.items():
        a_rad[a], a_y[a] = col, y

    by_order = defaultdict(dict)                   # order -> {flat tuple: idx}
    for (key, _, _), v in acc.items():
        if v != 0.0:
            flat = tuple(sorted(mu * n_a + a for mu, a in key))
            by_order[len(flat)].setdefault(flat, len(by_order[len(flat)]))
    offsets, aa_specs, off = {}, [], 0
    for order in sorted(by_order):
        offsets[order] = off
        tbl = np.zeros((len(by_order[order]), order), np.int32)
        for flat, i in by_order[order].items():
            tbl[i] = flat
        aa_specs.append(tbl)
        off += len(tbl)

    rows, cols, vals = [], [], []
    for (key, el, t), v in acc.items():
        if v == 0.0:
            continue
        flat = tuple(sorted(mu * n_a + a for mu, a in key))
        aa = offsets[len(flat)] + by_order[len(flat)][flat]
        rows.append(aa * NZ + el)
        cols.append(t)
        vals.append(v)
    return {"a_rad": a_rad, "a_y": a_y, "n_a_local": n_a, "aa_specs": tuple(aa_specs),
            "T_rows": np.asarray(rows, np.int32), "T_cols": np.asarray(cols, np.int32),
            "T_vals": np.asarray(vals, float), "n_aa": off}
