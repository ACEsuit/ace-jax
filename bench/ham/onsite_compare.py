# ruff: noqa: B023 -- the timing lambdas are called inside their own iteration
"""On-site prediction time of ace_jax.ham on the structure and orbital basis that
bench/ham/aceh_onsite_time.jl times for ACEhamiltonians.jl (silicon, s to f shells,
30 orbitals; random coefficients in both).
    uv run python bench/ham/onsite_compare.py <structure.txt> <order> <totaldegree> <rcut> [wL]
structure.txt: 3 lattice-vector lines, then one "x y z" line per atom."""
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from ace_jax.operators.features import build_site_basis
from ace_jax.operators.onsite import OnsiteModel, OnsitePredictor, _features, _readout, parts_of, shell_pairs, site_graph
from ace_jax.operators.orbitals import OrbitalBasis

ORB = OrbitalBasis.of({14: (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)})


def best(f, reps=5):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)


def main(path, order, totaldegree, rcut, wL=1.0):
    xyz = np.loadtxt(path)
    cell, pos = xyz[:3], xyz[3:]
    numbers = np.full(len(pos), 14)
    N = len(pos)
    basis = build_site_basis([14], order=order, totaldegree=totaldegree, lmax_feat=6, rcut=rcut, wL=wL)
    rng = np.random.default_rng(0)
    for channels in (("H",), ("H", "S")):
        coef = {(14, s1, s2, L): rng.normal(size=(len(channels), basis.n(L, p) + (L == 0)))
                for s1, s2, l1, l2 in shell_pairs(ORB, 14) for L, p in parts_of(l1, l2, s1 == s2)
                if basis.n(L, p) + (L == 0)}
        fast = OnsitePredictor(OnsiteModel(basis=basis, coef=coef, orbitals=ORB, channels=channels))
        rij, zi, zj, seg, n = site_graph(basis, pos, cell, numbers)
        atoms = (np.arange(N, dtype=np.int32),)
        F = _features(basis, rij, zi, zj, seg, n)
        t_nl = best(lambda: site_graph(basis, pos, cell, numbers), reps=3)
        t_f = best(lambda: jax.block_until_ready(_features(basis, rij, zi, zj, seg, n)))
        t_r = best(lambda: jax.block_until_ready(_readout(fast.readout, F, atoms)))
        t_all = best(lambda: jax.block_until_ready(fast.dense(pos, cell, numbers)[1]), reps=3)
        us = lambda t: round(1e6 * t / N, 1)
        if len(channels) == 1:
            n_aa = sum(int(g.shape[0]) for g in basis.aa_specs)
            print(f"{N} atoms, {len(rij) / N:.1f} neighbours, threads {len(os.sched_getaffinity(0))}; "
                  f"order {order}, total degree {totaldegree}, wL {wL}: n_A {len(basis.aspec_r)}, n_AA {n_aa}, "
                  f"{sum(basis.n_B)} features over {len(basis.lps)} (L, parity); "
                  f"{sum(np.asarray(c).shape[1] for c in coef.values())} coefficients per channel in {len(coef)} parts "
                  f"of {len(shell_pairs(ORB, 14))} shell pairs")
        print(f"  {'+'.join(channels)}: neighbour list {us(t_nl)}, features {us(t_f)}, readout {us(t_r)}, "
              f"sum {us(t_nl + t_f + t_r)}; end to end {us(t_all)} us per atom")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), int(a[3]), float(a[4]), float(a[5]) if len(a) > 5 else 1.0)
