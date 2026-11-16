"""Time the on-site path: one A / AA evaluation per atom, then the features of every
(L, parity), then the numpy readout of all shell pairs and channels.
    ACEJAX_COUPLING_CACHE_ONLY=1 uv run python bench/ham/onsite_time.py [n_cells_per_side]
The structure is zincblende SiC rattled by 0.05 angstrom; the model is the test basis of
tests/test_ham_onsite.py (order 2, total degree 7, wL = 1, features to L = 6) with random
coefficients for the fixture's spdf orbital basis (30 orbitals on Si, 23 on C)."""
import pathlib
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from ace_jax.ham.features import build_site_basis
from ace_jax.ham.onsite import OnsiteModel, _features, parts_of, shell_pairs, site_graph
from ace_jax.ham.orbitals import OrbitalBasis

CACHE = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "ham" / "coupling_cache"
ORB = OrbitalBasis.of({6: (0, 0, 0, 0, 1, 1, 1, 2, 2), 14: (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)})


def sic(n, seed=0):
    a = 4.3596
    cell = n * a * np.eye(3)
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    si = (grid[:, None] + fcc[None]).reshape(-1, 3) * a
    pos = np.concatenate([si, si + a / 4])
    pos += 0.05 * np.random.default_rng(seed).normal(size=pos.shape)
    return pos, cell, np.array([14] * len(si) + [6] * len(si))


def best(f, reps=5):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)


def main(n=4):
    basis = build_site_basis([14, 6], order=2, totaldegree=7, lmax_feat=6, rcut=4.0, wL=1.0,
                             coupling_cache_dir=str(CACHE))
    rng = np.random.default_rng(0)
    coef = {(z, s1, s2, L): rng.normal(size=(2, basis.n(L, p) + (L == 0)))
            for z in ORB.species for s1, s2, l1, l2 in shell_pairs(ORB, z) for L, p in parts_of(l1, l2, s1 == s2)}
    model = OnsiteModel(basis=basis, coef=coef, orbitals=ORB, channels=("H", "S"))
    pos, cell, numbers = sic(n)
    N = len(pos)
    rij, zi, zj, seg, _ = site_graph(basis, pos, cell, numbers)
    n_aa = sum(int(g.shape[0]) for g in basis.carrier.aa_specs)
    print(f"{N} atoms, {len(rij) / N:.1f} neighbours, n_A {len(basis.carrier.aspec_r)}, n_AA {n_aa}, "
          f"features {sum(basis.n_B)} over {len(basis.lps)} (L, parity), {len(coef)} coefficient vectors, "
          f"threads {len(__import__('os').sched_getaffinity(0))}")
    aa = jax.jit(lambda *a: basis.aa(*a, N))
    t_aa = best(lambda: aa(rij, zi, zj, seg).block_until_ready())
    t_feat = best(lambda: jax.block_until_ready(_features(basis, rij, zi, zj, seg, N)))
    t_graph = best(lambda: site_graph(basis, pos, cell, numbers), reps=3)
    t_all = best(lambda: model.predict(pos, cell, numbers), reps=3)
    us = lambda t: 1e6 * t / N
    print(f"per atom: neighbour list {us(t_graph):.1f} us, A+AA {us(t_aa):.1f} us, A+AA+features {us(t_feat):.1f} us, "
          f"everything (H and S, numpy readout) {us(t_all):.1f} us")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
