"""Prediction time of the full Hamiltonian (on-site and off-site blocks) with ace_jax.ham's
shared-feature model on a DeepH-layout structure, random coefficients: the counterpart of
bench/ham/maceh_time.py.
    uv run --extra ham python bench/ham/offsite_time_deeph.py <deeph dir> <rcut> <site radials> <n_bond> <n_cpl> <l_cpl> [channels=H]
HAM_X64=0 runs in float32."""
import os
import sys
import time

import jax

X64 = os.environ.get("HAM_X64", "1") != "0"
jax.config.update("jax_enable_x64", X64)

import dataclasses

import numpy as np

from ace_jax.eval.model import with_radial_table
from ace_jax.ham import bond as B
from ace_jax.ham import cg, deeph
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.offsite import OffsiteModel, canonical_pairs
from ace_jax.ham.offsite_fast import OffsitePredictor
from ace_jax.ham.onsite import OnsiteModel, OnsitePredictor, parts_of, shell_pairs


def best(f, reps=3):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)


a = sys.argv[1:]
d = deeph.read(a[0], operators=())
rcut, nsite, n_bond, n_cpl, l_cpl = float(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5])
channels = tuple(a[6].split(",")) if len(a) > 6 else ("H",)
elements, orb = [int(z) for z in d.basis.species], d.basis
lf = 2 * orb.lmax
site = build_site_basis(elements, order=1, totaldegree=40, lmax_feat=lf, rcut=rcut, wL=0.5,
                        nmax_by_order=(nsite,), lmax_by_order=(lf,))
site = dataclasses.replace(site, carrier=with_radial_table(site.carrier, True)[0])
basis = B.BondBasis(site=site, rcut_bond=rcut, n_bond=n_bond, n_cpl=n_cpl, l_cpl=l_cpl)
rng = np.random.default_rng(0)
nc = len(channels)
on = {(z, s1, s2, L): rng.normal(size=(nc, site.n(L, p) + (L == 0)))
      for z in elements for s1, s2, l1, l2 in shell_pairs(orb, z) for L, p in parts_of(l1, l2, s1 == s2)
      if site.n(L, p) + (L == 0)}
off = {}
for zi in elements:
    for zj in elements:
        for s1, s2, l1, l2, same in canonical_pairs(orb, zi, zj):
            p = 1 if (l1 + l2) % 2 == 0 else -1
            for L in cg.Ls(l1, l2):
                k0, k1 = B.n_features(basis, L, p)
                if k0 + k1:
                    off[(zi, zj, s1, s2, L)] = rng.normal(size=(nc, k0 + (k1 if same else 2 * k1)))
f_on = OnsitePredictor(OnsiteModel(basis=site, coef=on, orbitals=orb, channels=channels))
f_off = OffsitePredictor(OffsiteModel(bond=basis, coef=off, orbitals=orb, channels=channels))
N = len(d.numbers)
g, dense = f_off.dense(d.positions, d.cell, d.numbers)
n_blocks = len(g.i) + N
t_off = best(lambda: f_off.dense(d.positions, d.cell, d.numbers))
t_on = best(lambda: jax.block_until_ready(f_on.dense(d.positions, d.cell, d.numbers)[1]))
ncoef = sum(np.asarray(c).size for c in off.values()) + sum(np.asarray(c).size for c in on.values())
dev = jax.devices()[0]
print(f"ace-jax full {'+'.join(channels)}, {dev.platform} {'float64' if X64 else 'float32'}, threads {len(os.sched_getaffinity(0))}: "
      f"{N} atoms, {len(g.i) / N:.0f} bonds per atom within {rcut}; site radials {nsite}, n_bond {n_bond}, n_cpl {n_cpl}, "
      f"l_cpl {l_cpl}; {ncoef} coefficients; off-site {1e3 * t_off:.1f} ms, on-site {1e3 * t_on:.1f} ms per structure; "
      f"{1e6 * (t_on + t_off) / N:.0f} us per atom, {1e6 * (t_on + t_off) / n_blocks:.1f} us per block")
