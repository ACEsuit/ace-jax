"""D2 of docs/dev/specs/2026-10-05-gp-discrepancy-design.md: within-configuration correlation of force errors
against distance, and the effective number of independent force rows per configuration (the force-row
over-counting argument of #64).

Errors e_i = F_label - F_pred per atom.  Pair statistic, pooled over configurations within a set:
    c(r) = sum_{pairs in bin} e_i . e_j / (n_pairs(bin) * <|e|^2>)        (vector correlation, c(0) = 1)
and its longitudinal / transverse parts c_L = <(e_i.u)(e_j.u)> / <e_a^2>, c_T = (3c - c_L)/2 (u = unit r_ij).
Pairs: unique (i, j), i != j, at the minimum-image distance; in periodic cells only r < half the shortest
perpendicular cell width (beyond it a pair's distance is ambiguous).  Big cells: free atoms only.

Effective rows: per configuration, C = [c(r_ij) I_3] (c from the set's own curve, 0 beyond the last bin with
>= 100 pairs) gives
    n_mean = (3N)^2 / 1^T C 1      (rows' worth of information about a common mean, e.g. an energy-like offset)
    n_pr   = (tr C)^2 / tr C^2     (participation ratio: the number of effectively independent rows).
Forces sum to zero in a periodic cell for labels and model alike, so sum_i e_i = 0 and c(r) carries a
-1/(N-1) floor; the table reports that floor beside c.

    uv run python d2_error_corr.py --out d2.md --small NAME=pred.npz:test.xyz ... --big NAME=err.npz:big.xyz ...
"""
import argparse
import os

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

DR, RMAX = 0.5, 20.0
EDGES = np.arange(0.0, RMAX + DR, DR)


def _half_width(a):
    if not a.pbc.any():
        return np.inf
    c = a.cell.array
    V = abs(np.linalg.det(c))
    w = [V / np.linalg.norm(np.cross(c[(k + 1) % 3], c[(k + 2) % 3])) for k in range(3) if a.pbc[k]]
    return 0.5 * min(w)


def pairs(a, rmax):
    """Unique (i<j) pairs at the minimum-image distance, with the unit vector, r < rmax."""
    i, j, d, D = neighbor_list("ijdD", a, rmax)
    k = i < j
    i, j, d, D = i[k], j[k], d[k], D[k]
    if len(i) == 0:
        return i, j, d, D
    o = np.lexsort((d, j, i))
    i, j, d, D = i[o], j[o], d[o], D[o]
    first = np.r_[True, (i[1:] != i[:-1]) | (j[1:] != j[:-1])]
    return i[first], j[first], d[first], D[first] / d[first, None]


class Acc:
    def __init__(self):
        nb = len(EDGES) - 1
        self.s, self.sl, self.n = np.zeros(nb), np.zeros(nb), np.zeros(nb)
        self.e2, self.na, self.cfgs = 0.0, 0, []

    def add(self, a, e, live):
        rmax = min(RMAX, _half_width(a) - 1e-6)
        i, j, d, u = pairs(a, rmax)
        k = live[i] & live[j]
        i, j, d, u = i[k], j[k], d[k], u[k]
        b = np.searchsorted(EDGES, d, side="right") - 1
        dot = np.einsum("na,na->n", e[i], e[j])
        lon = np.einsum("na,na->n", e[i], u) * np.einsum("na,na->n", e[j], u)
        self.s += np.bincount(b, dot, len(self.s)); self.sl += np.bincount(b, lon, len(self.s))
        self.n += np.bincount(b, minlength=len(self.s))
        self.e2 += float(np.sum(e[live] ** 2)); self.na += int(live.sum())
        self.cfgs.append((a, live))

    def curve(self):
        m2 = self.e2 / max(self.na, 1)
        c = self.s / np.maximum(self.n, 1) / m2
        cl = self.sl / np.maximum(self.n, 1) / (m2 / 3)
        return c, cl, (3 * c - cl) / 2

    def n_eff(self, max_cfgs=300):
        c, _, _ = self.curve()
        ok = self.n >= 100
        last = np.flatnonzero(ok).max() if ok.any() else -1
        cc = np.where(np.arange(len(c)) <= last, np.where(ok, c, 0.0), 0.0)
        out = []
        sel = np.linspace(0, len(self.cfgs) - 1, min(max_cfgs, len(self.cfgs))).round().astype(int)
        for a, live in (self.cfgs[s] for s in np.unique(sel)):
            idx = np.flatnonzero(live)
            N = len(idx)
            if N < 2:
                continue
            rmax = min(RMAX, _half_width(a) - 1e-6)
            i, j, d, _ = pairs(a, rmax)
            k = live[i] & live[j]
            b = np.searchsorted(EDGES, d[k], side="right") - 1
            w = cc[b]
            one = 3 * N + 2 * 3 * w.sum()                      # 1^T C 1 with C = c(r) I_3 blocks
            tr2 = 3 * N + 2 * 3 * np.sum(w ** 2)
            out.append((N, (3 * N) ** 2 / max(one, 1e-12), (3 * N) ** 2 / tr2))
        return np.array(out)


def run_small(name, pred, xyz):
    P = np.load(pred)
    frames = read(xyz, ":")
    nat = P["nat"]
    assert len(frames) == len(nat) and all(len(a) == n for a, n in zip(frames, nat)), "pred/xyz mismatch"
    off = np.r_[0, np.cumsum(nat)]
    A = Acc()
    for c, a in enumerate(frames):
        e = P["F"][off[c]:off[c + 1]] - P["F_mean"][off[c]:off[c + 1]]
        A.add(a, e, np.ones(len(a), bool))
    return A


def run_big(name, err, xyz):
    Z = np.load(err, allow_pickle=True)
    frames = read(xyz, ":")
    accs = {}
    for c in np.unique(Z["cfg"]):
        m = np.flatnonzero(Z["cfg"] == c)
        fam = str(Z["family"][m[0]])
        accs.setdefault(fam, Acc()).add(frames[c], Z["dF"][m].astype(float), ~Z["fixed"][m].astype(bool))
    return accs


def table(name, A):
    c, cl, ct = A.curve()
    ne = A.n_eff()
    lines = [f"\n### {name}\n", f"{len(A.cfgs)} configurations, {A.na} atoms, rms|e| {np.sqrt(A.e2 / A.na):.4f} eV/A; "
             f"mean atoms per config {A.na / len(A.cfgs):.0f} (sum-to-zero floor -1/(N-1) = {-1 / (A.na / len(A.cfgs) - 1):+.4f})\n",
             "| r (A) | pairs | c(r) | c_L | c_T |", "|---|---|---|---|---|"]
    for b in range(len(c)):
        if A.n[b] >= 100:
            lines.append(f"| {EDGES[b]:.1f}-{EDGES[b + 1]:.1f} | {int(A.n[b])} | {c[b]:+.3f} | {cl[b]:+.3f} | {ct[b]:+.3f} |")
    if len(ne):
        N, nm, npr = ne.T
        lines.append(f"\nEffective rows ({len(ne)} configs): 3N median {np.median(3 * N):.0f}; n_mean / 3N median "
                     f"{np.median(nm / (3 * N)):.3f} [p10 {np.quantile(nm / (3 * N), 0.1):.3f}, p90 "
                     f"{np.quantile(nm / (3 * N), 0.9):.3f}]; n_pr / 3N median {np.median(npr / (3 * N)):.3f} "
                     f"[p10 {np.quantile(npr / (3 * N), 0.1):.3f}, p90 {np.quantile(npr / (3 * N), 0.9):.3f}]")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", nargs="*", default=[]); ap.add_argument("--big", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = ["# D2 within-configuration force-error correlation\n"]
    for spec in a.small:
        name, rest = spec.split("=", 1)
        pred, xyz = rest.split(":")
        lines += table(f"{name} ({os.path.basename(pred)})", run_small(name, pred, xyz))
        print("done", name, flush=True)
    for spec in a.big:
        name, rest = spec.split("=", 1)
        err, xyz = rest.split(":")
        for fam, A in run_big(name, err, xyz).items():
            lines += table(f"{name} {fam} ({os.path.basename(err)})", A)
        print("done", name, flush=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
