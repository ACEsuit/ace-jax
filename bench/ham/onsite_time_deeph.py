# ruff: noqa: B023 -- the timing lambdas are called inside their own iteration
"""On-site prediction time of ace_jax.ham on a DeepH-layout structure (the input
bench/ham/maceh_time.py times MACE-H on), random coefficients, H only.
    uv run --extra ham python bench/ham/onsite_time_deeph.py <deeph dir> <rcut> <n radials at order 1>"""
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from ace_jax.ham import deeph
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.onsite import OnsiteModel, OnsitePredictor, _features, _readout, parts_of, shell_pairs, site_graph


def best(f, reps=5):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)


d = deeph.read(sys.argv[1], operators=())
rcut, nmax = float(sys.argv[2]), int(sys.argv[3])
elements = [int(z) for z in d.basis.species]
lf = 2 * d.basis.lmax
basis = build_site_basis(elements, order=1, totaldegree=40, lmax_feat=lf, rcut=rcut, wL=0.5,
                         nmax_by_order=(nmax,), lmax_by_order=(lf,))
if os.environ.get("HAM_RADIAL_TABLE"):
    import dataclasses

    from ace_jax.eval.model import with_radial_table
    carrier, info = with_radial_table(basis.carrier, True)
    basis = dataclasses.replace(basis, carrier=carrier)
    print("radial table:", info)
rng = np.random.default_rng(0)
coef = {(z, s1, s2, L): rng.normal(size=(1, basis.n(L, p) + (L == 0)))
        for z in elements for s1, s2, l1, l2 in shell_pairs(d.basis, z) for L, p in parts_of(l1, l2, s1 == s2)
        if basis.n(L, p) + (L == 0)}
fast = OnsitePredictor(OnsiteModel(basis=basis, coef=coef, orbitals=d.basis, channels=("H",)))
N = len(d.numbers)
rij, zi, zj, seg, n = site_graph(basis, d.positions, d.cell, d.numbers)
atoms = tuple(np.flatnonzero(d.numbers == z).astype(np.int32) for z in d.basis.species)
F = _features(basis, rij, zi, zj, seg, n)
t_nl = best(lambda: site_graph(basis, d.positions, d.cell, d.numbers))
t_f = best(lambda: jax.block_until_ready(_features(basis, rij, zi, zj, seg, n)))
t_r = best(lambda: jax.block_until_ready(_readout(fast.readout, F, atoms)))
t_all = best(lambda: jax.block_until_ready(fast.dense(d.positions, d.cell, d.numbers)[1]))
us = lambda t: round(1e6 * t / N, 1)
print(f"ace-jax on-site H, {jax.default_backend()} float64, threads {len(os.sched_getaffinity(0))}: {N} atoms, "
      f"{len(rij) / N:.0f} neighbours within {rcut}, {nmax} radials x l <= {lf} at order 1: {sum(basis.n_B)} features, "
      f"{sum(np.asarray(c).size for c in coef.values())} coefficients; per atom: neighbour list {us(t_nl)}, "
      f"features {us(t_f)}, readout {us(t_r)}, end to end {us(t_all)} us")
