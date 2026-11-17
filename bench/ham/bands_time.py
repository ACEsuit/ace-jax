"""Time from positions to eigenvalues with everything after the neighbour lists on the device:
blocks (`HamiltonianPredictor.prepare`), Bloch sums of H and S, and the Hermitian eigensolver.
Random coefficients, so the overlap is not positive definite: the eigensolver is timed on H(k)
alone (the generalised problem adds a Cholesky factor and two triangular solves).
    HAM_X64=0 uv run --extra ham python bench/ham/bands_time.py <deeph dir> <rcut> <site radials> <n_bond> <n_cpl> <l_cpl>"""
import dataclasses
import os
import sys
import time

import jax

X64 = os.environ.get("HAM_X64", "1") != "0"
jax.config.update("jax_enable_x64", X64)

import jax.numpy as jnp
import numpy as np

from ace_jax.eval.model import with_radial_table
from ace_jax.ham import bond as B
from ace_jax.ham import cg, deeph
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.offsite import OffsiteModel, canonical_pairs
from ace_jax.ham.onsite import OnsiteModel, parts_of, shell_pairs
from ace_jax.ham.operator import HamiltonianPredictor


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
CH = ("H", "S")
on = {(z, s1, s2, L): rng.normal(size=(2, site.n(L, p) + (L == 0)))
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
                    off[(zi, zj, s1, s2, L)] = rng.normal(size=(2, k0 + (k1 if same else 2 * k1)))
hp = HamiltonianPredictor(OnsiteModel(basis=site, coef=on, orbitals=orb, channels=CH),
                          OffsiteModel(bond=basis, coef=off, orbitals=orb, channels=CH))
pos, cell, numbers = d.positions, d.cell, np.asarray(d.numbers)
N = len(numbers)
ready = lambda p: jax.block_until_ready((p.on, p.off))
DEVICE = os.environ.get("HAM_DEVICE_LISTS", "0") == "1"       # neighbour lists on the GPU (matscipy-neighbours with CUDA)
prepare = hp.prepare_device if DEVICE else hp.prepare
t_prep, prep = best(lambda: (lambda p: (ready(p), p)[1])(prepare(pos, cell, numbers)), reps=3)
if DEVICE:
    with jax.transfer_guard_host_to_device("disallow"):       # nothing but the explicit inputs goes up
        ready(prepare(pos, cell, numbers))
k = (0.13, 0.27, -0.4)
t_h, H = best(lambda: jax.block_until_ready(hp.hk(prep, k, "H")))
t_s, _ = best(lambda: jax.block_until_ready(hp.hk(prep, k, "S")))
t_e, e = best(lambda: np.asarray(jnp.linalg.eigvalsh(0.5 * (H + H.conj().T))), reps=3)
tot = t_prep + t_h + t_s + t_e
dev = jax.devices()[0]
print(f"{dev.platform} {'float64' if X64 else 'float32'}: {N} atoms, {prep.n_orb} orbitals, {sum(v[0].shape[1] for v in prep.lists['off'].values())} bond slots per atom over the species pairs; "
      f"H and S; n_cpl {n_cpl}, l_cpl {l_cpl}; neighbour lists on the {'device' if DEVICE else 'host'}")
for name, t in ((f"blocks of H and S on the device (lists on the {'device' if DEVICE else 'host'})", t_prep), ("Bloch sum H(k)", t_h), ("Bloch sum S(k)", t_s),
                (f"eigenvalues of H(k) ({prep.n_orb} x {prep.n_orb} Hermitian), copied to the host", t_e)):
    print(f"  {name:72s} {1e3 * t:8.1f} ms  {100 * t / tot:5.1f} %")
print(f"  {'positions to eigenvalues at one k-point':72s} {1e3 * tot:8.1f} ms          {1e6 * tot / N:.0f} us per atom")
