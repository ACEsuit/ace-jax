"""Where the time of a full-Hamiltonian prediction goes (`OffsitePredictor.dense`), stage by stage,
on a DeepH-layout structure with random coefficients.
    HAM_X64=0 uv run --extra ham python bench/ham/offsite_profile.py <deeph dir> <rcut> <site radials> <n_bond> <n_cpl> <l_cpl>"""
import dataclasses
import os
import sys
import time

import jax

X64 = os.environ.get("HAM_X64", "1") != "0"
jax.config.update("jax_enable_x64", X64)

import numpy as np

from ace_jax.eval.model import with_radial_table
from ace_jax.ham import bond as B
from ace_jax.basis import cg
from ace_jax.ham import deeph
from ace_jax.ham import offsite_fast as OF
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.offsite import OffsiteModel, canonical_pairs
from ace_jax.ham.onsite import _features, site_graph


def best(f, reps=5):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = f(); ts.append(time.perf_counter() - t)
    return min(ts), out


a = sys.argv[1:]
d = deeph.read(a[0], operators=())
rcut, nsite, n_bond, n_cpl, l_cpl = float(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5])
elements, orb = [int(z) for z in d.basis.species], d.basis
lf = 2 * orb.lmax
site = build_site_basis(elements, order=1, totaldegree=40, lmax_feat=lf, rcut=rcut, wL=0.5,
                        nmax_by_order=(nsite,), lmax_by_order=(lf,))
site = dataclasses.replace(site, carrier=with_radial_table(site.carrier, True)[0])
basis = B.BondBasis(site=site, rcut_bond=rcut, n_bond=n_bond, n_cpl=n_cpl, l_cpl=l_cpl)
rng = np.random.default_rng(0)
off = {}
for zi in elements:
    for zj in elements:
        for s1, s2, l1, l2, same in canonical_pairs(orb, zi, zj):
            p = 1 if (l1 + l2) % 2 == 0 else -1
            for L in cg.Ls(l1, l2):
                k0, k1 = B.n_features(basis, L, p)
                if k0 + k1:
                    off[(zi, zj, s1, s2, L)] = rng.normal(size=(1, k0 + (k1 if same else 2 * k1)))
fast = OF.OffsitePredictor(OffsiteModel(bond=basis, coef=off, orbitals=orb, channels=("H",)))
pos, cell, numbers = d.positions, d.cell, np.asarray(d.numbers)
N = len(numbers)
rows = []
t, sg = best(lambda: site_graph(site, pos, cell, numbers)); rows.append(("site neighbour list (host)", t))
rij, zi, zj, seg, n = sg
t, F = best(lambda: jax.block_until_ready(_features(site, rij, zi, zj, seg, n))); rows.append(("site features (device)", t))
t, g = best(lambda: B.bond_graph(basis, pos, cell, numbers), reps=3); rows.append(("bond list + reverse index (host)", t))


def layout():
    return OF.PairLayouts(g, numbers, orb.species, basis.rcut_bond)


t, pl = best(layout); rows.append(("dense layout per ordered species pair (host)", t))
K = round(sum(r * k for r, k in pl.shape.values()) / N)
args = fast.static()
t, canon = best(lambda: jax.block_until_ready(OF._blocks(fast.packs, F, pl.lay, basis.rcut_bond, args)))
rows.append(("off-site blocks, every ordered species pair (device)", t))
def copy():
    fresh = OF._blocks(fast.packs, F, pl.lay, basis.rcut_bond, args)      # a new result: the copy is not cached
    jax.block_until_ready(fresh)
    t0 = time.perf_counter()
    h = {k: np.asarray(v) for k, v in fresh.items()}
    return time.perf_counter() - t0, h


t, host = min((copy() for _ in range(3)), key=lambda x: x[0]); rows.append(("copy blocks to the host", t))
nbytes = sum(v.nbytes for v in host.values())


def assemble():
    return {key: host[key].reshape(-1, *host[key].shape[2:])[pl.flatpos[e]] for key, e in pl.sel.items()}


t, _ = best(assemble, reps=3); rows.append(("select the bonds of each species pair (host)", t))
t_all, _ = best(lambda: fast.dense(pos, cell, numbers), reps=3)
total = sum(t for _, t in rows)
dev = jax.devices()[0]
print(f"{dev.platform} {'float64' if X64 else 'float32'}: {N} atoms, {len(g.i) / N:.0f} bonds per atom, {K} slots per atom over the species pairs; "
      f"n_bond {n_bond}, n_cpl {n_cpl}, l_cpl {l_cpl}; "
      f"blocks copied back: {nbytes / 1e6:.0f} MB")
for name, t in rows:
    print(f"  {name:48s} {1e3 * t:8.1f} ms  {100 * t / total:5.1f} %  {1e6 * t / N:8.0f} us per atom")
print(f"  {'sum of stages':48s} {1e3 * total:8.1f} ms          {1e6 * total / N:8.0f} us per atom;  dense() end to end {1e3 * t_all:.1f} ms")
