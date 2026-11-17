"""Fit and test ace-jax on a DeepH-layout dataset split into train and test folders (the layout of
the MACE-H paper's bulk gold data, Zenodo 10.5281/zenodo.15223696), with that paper's metrics
(arXiv:2508.15108, Methods, Accuracy Metrics): the mean absolute error of the Hamiltonian matrix
elements of the test set, and the smeared eigenvalue error
    EE(T) = sum_ki |e_ki - e~_ki| f(e_ki, mu, T) / sum_ki f(e_ki, mu, T)
on a k-grid (7 x 7 x 7 for gold), f the Fermi-Dirac function at T = 1000 K with mu from charge
neutrality, each set of bands measured from its own mu as that paper's Band.py does
(`compute_eigenvalue_error`; the absolute difference is printed beside it); the predicted H is diagonalised with the reference overlap,
as MACE-H's (it predicts H only).
    JAX_PLATFORMS=cpu uv run python bench/ham/deeph_fit.py <dataset dir> <n_train> <valence electrons per atom>
        [bonds=<per structure, 0 = all>] [site=<n_cpl>,<l_cpl>] [env=<r_env>,<n_env>,<l_env>,<n_ebond>,<l_ebond>|none]
        [nsite=12] [nbond=12] [rsite=7.0] [nk=7] [threads=8] [ntest=<structures>] [seed=0]
Training structures: the first <n_train> of <dataset dir>/train in name order after a seeded
shuffle (nested: a larger <n_train> contains a smaller one's)."""
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import scipy.linalg
from scipy.optimize import brentq

from ace_jax.ham import deeph
from ace_jax.operators import bond as B
from ace_jax.operators.data import OperatorData
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import fit_offsite, fit_onsite
from ace_jax.operators.offsite_fast import OffsitePredictor
from ace_jax.operators.onsite import OnsitePredictor

KB = 8.617333262e-5


def onsite(k):
    return k[:3] == (0, 0, 0) and k[3] == k[4]


def subset(d, n, seed):
    """`d` with its on-site blocks and a random `n` of its bonds (0: all)."""
    if not n:
        return d
    keys = sorted(k for k in d.blocks["H"] if not onsite(k))
    pick = np.random.default_rng(seed).choice(len(keys), size=min(n, len(keys)), replace=False)
    keep = {keys[i] for i in pick} | {k for k in d.blocks["H"] if onsite(k)}
    return OperatorData(d.cell, d.positions, d.numbers, d.basis, {"H": {k: d.blocks["H"][k] for k in keep}})


def bloch(d, blocks, ks):
    off = np.concatenate([[0], np.cumsum([d.basis.n_orb(int(z)) for z in d.numbers])])
    M = np.zeros((len(ks), off[-1], off[-1]), complex)
    for (Rx, Ry, Rz, i, j), b in blocks.items():
        M[:, off[i]:off[i + 1], off[j]:off[j + 1]] += np.exp(2j * np.pi * (ks @ (Rx, Ry, Rz)))[:, None, None] * b
    return 0.5 * (M + M.conj().transpose(0, 2, 1))


def fermi(e, mu, T):
    return 1.0 / (1.0 + np.exp(np.clip((e - mu) / (KB * T), -500, 500)))


def chemical_potential(e, n_electrons, T):
    """mu that holds `n_electrons` (two per state) in the bands e (n_k, n) at temperature T."""
    return brentq(lambda m: 2.0 * fermi(e, m, T).sum() / len(e) - n_electrons, e.min(), e.max())


def main(folder, n_train, nel, bonds=3000, site=(2, 2), env=(5.0, 6, 4, 4, 4), nsite=12, nbond=12, rsite=7.0, nk=7, threads=8,
         ntest=None, seed=0, T=1000.0):
    names = sorted(os.listdir(os.path.join(folder, "train")))
    np.random.default_rng(seed).shuffle(names)
    train = [deeph.read(os.path.join(folder, "train", n), operators=("H",)) for n in names[:n_train]]
    orb = train[0].basis
    rmax = max(np.linalg.norm(train[0].bond(k)) for k in train[0].blocks["H"])
    rcut = float(np.ceil(rmax * 10) / 10)
    sb = build_site_basis([int(z) for z in orb.species], order=1, totaldegree=40, lmax_feat=2 * orb.lmax, rcut=rsite, wL=0.5,
                          nmax_by_order=(nsite,), lmax_by_order=(2 * orb.lmax,)).with_table()
    basis = B.BondBasis(site=sb, rcut_bond=rcut, n_bond=nbond, n_cpl=site[0], l_cpl=site[1])
    if env:
        basis = basis._replace(r_env=env[0], n_env=int(env[1]), l_env=int(env[2]), n_ebond=int(env[3]), l_ebond=int(env[4]))
    t = time.perf_counter()
    on, _ = fit_onsite(sb, train, channels=("H",))
    off, _ = fit_offsite(basis, [subset(d, bonds, k) for k, d in enumerate(train)], channels=("H",))
    kmax = max(np.asarray(c).shape[1] for c in off.coef.values())
    print(f"{n_train} training structures ({len(train[0].numbers)} atoms, {len(train[0].blocks['H'])} blocks each), "
          f"{bonds or 'all'} bonds of each; site radials {nsite} to {rsite} A, {nbond} bond radials, site group {site}, env {env}, "
          f"bond cutoff {rcut}; largest problem {kmax} columns; fitted in {time.perf_counter() - t:.0f} s", flush=True)
    del train
    f_on, f_off = OnsitePredictor(on), OffsitePredictor(off, threads=threads)
    ks = np.array([[a, b, c] for a in range(nk) for b in range(nk) for c in range(nk)]) / nk
    tests = sorted(os.listdir(os.path.join(folder, "test")))[:ntest]
    err, ee, ee_rel, times = [], [], [], []
    for name in tests:
        d = deeph.read(os.path.join(folder, "test", name))
        f_on(d.positions, d.cell, d.numbers); f_off(d.positions, d.cell, d.numbers)          # compile, lists
        t = time.perf_counter()
        pred = {**f_off(d.positions, d.cell, d.numbers)["H"], **f_on(d.positions, d.cell, d.numbers)["H"]}
        times.append((time.perf_counter() - t) / len(d.numbers))
        diff = np.concatenate([(pred[k] - v).ravel() for k, v in d.blocks["H"].items()])
        missing = sum(v.size for k, v in d.blocks["H"].items() if k not in pred)
        assert not missing
        err.append(diff)
        S = bloch(d, d.blocks["S"], ks)
        e0 = np.array([scipy.linalg.eigh(h, s, eigvals_only=True) for h, s in zip(bloch(d, d.blocks["H"], ks), S)])
        e1 = np.array([scipy.linalg.eigh(h, s, eigvals_only=True) for h, s in zip(bloch(d, pred, ks), S)])
        mu0, mu1 = (chemical_potential(e, nel * len(d.numbers), T) for e in (e0, e1))
        w = fermi(e0, mu0, T)
        ee.append((np.abs(e1 - e0) * w).sum() / w.sum())
        ee_rel.append((np.abs((e1 - mu1) - (e0 - mu0)) * w).sum() / w.sum())
        print(f"  {name}: H MAE {1e3 * np.abs(diff).mean():.3f} meV, EE {1e3 * ee[-1]:.2f} meV "
              f"(relative to each chemical potential {1e3 * ee_rel[-1]:.2f}), {1e3 * times[-1]:.1f} ms per atom", flush=True)
    err = np.concatenate(err)
    print(f"test set ({len(tests)} structures): H MAE {1e3 * np.abs(err).mean():.3f} meV, RMSE {1e3 * np.sqrt((err ** 2).mean()):.3f} meV, "
          f"largest {1e3 * np.abs(err).max():.1f} meV; EE mean {1e3 * np.mean(ee):.2f} meV (relative to mu {1e3 * np.mean(ee_rel):.2f}); "
          f"{1e3 * np.mean(times):.1f} ms per atom on {threads} threads")


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    for x in [x for x in a if "=" in x]:
        k, v = x.split("=")
        kw[k] = (None if v == "none" else float(v) if k == "rsite" else int(v) if k in ("bonds", "nsite", "nbond", "nk", "threads", "ntest", "seed")
                 else tuple(float(t) if k == "env" and i == 0 else int(t) for i, t in enumerate(v.split(","))))
    pos = [x for x in a if "=" not in x]
    main(pos[0], int(pos[1]), float(pos[2]), **kw)
