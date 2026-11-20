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
        [nsite=12] [nbond=12] [rsite=7.0] [nk=7] [threads=8] [ntest=<structures>] [seed=0] [out=<file.jsonl>] [smooth=<sigma_c>]
`smooth=`: the smoothness prior with that sigma_c (default: the plain ridge).
`out=`: append one JSON record of the settings and results.  The reference bands of each test
structure are cached beside the dataset's test folder (`.bands_<nk>/`), so a grid of runs pays
for them once.
With `pred=<dir>` nothing is fitted (<n_train> is ignored): the predictions of another code,
`<dir>/<structure>/hamiltonians_pred.h5` (MACE-H, DeepH-E3: `bench/ham/maceh_eval.py`), are scored.
Training structures: the first <n_train> of <dataset dir>/train in name order after a seeded
shuffle (nested: a larger <n_train> contains a smaller one's)."""
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.optimize import brentq

from ace_jax.ham import deeph
from ace_jax.ham import metrics as M
from ace_jax.ham.operator import BlockSum, block_bands
from ace_jax.operators import bond as B
from ace_jax.operators.data import OperatorData
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import fit_offsite, fit_onsite
from ace_jax.operators.offsite_fast import OffsitePredictor
from ace_jax.operators.onsite import OnsitePredictor

KB = 8.617333262e-5
WINDOW = (-12.0, 5.0)                    # eV from the reference Fermi level: the states `ham.metrics` scores


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


def fermi(e, mu, T):
    return 1.0 / (1.0 + np.exp(np.clip((e - mu) / (KB * T), -500, 500)))


def chemical_potential(e, n_electrons, T):
    """mu that holds `n_electrons` (two per state) in the bands e (n_k, n) at temperature T."""
    return brentq(lambda m: 2.0 * fermi(e, m, T).sum() / len(e) - n_electrons, e.min(), e.max())


def train_names(folder, seed=0):
    """The training structures in the seeded order every code's nested training sets are cut from."""
    names = sorted(os.listdir(os.path.join(folder, "train")))
    np.random.default_rng(seed).shuffle(names)
    return names


def main(folder, n_train, nel, bonds=3000, site=(2, 2), env=(5.0, 6, 4, 4, 4), nsite=12, nbond=12, rsite=7.0, nk=7, threads=8,
         ntest=None, seed=0, T=1000.0, pred=None, out=None, smooth=None, save_model=None):
    record = dict(n_train=n_train, bonds=bonds, site=site, env=env, nsite=nsite, nbond=nbond, rsite=rsite, seed=seed, sigma_c=smooth,
                  backend=jax.default_backend(), threads=threads)
    if pred:
        res = score(folder, nel, lambda d, name: (deeph.read(os.path.join(folder, "test", name), operators=(
            ("H", os.path.abspath(os.path.join(pred, name, "hamiltonians_pred.h5"))),)).blocks["H"], float("nan")),
            nk, ntest, T, f"predictions in {pred}")
        return save(out, dict(pred=pred, **res))
    names = train_names(folder, seed)
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
    on, _ = fit_onsite(sb, train, channels=("H",), sigma_c=smooth)
    off, _ = fit_offsite(basis, [subset(d, bonds, k) for k, d in enumerate(train)], channels=("H",), sigma_c=smooth)
    kmax = max(np.asarray(c).shape[1] for c in off.coef.values())
    record.update(columns=int(kmax), fit_s=round(time.perf_counter() - t, 1), n_coef=int(sum(np.asarray(c).size for c in off.coef.values())))
    print(f"{n_train} training structures ({len(train[0].numbers)} atoms, {len(train[0].blocks['H'])} blocks each), "
          f"{bonds or 'all'} bonds of each; site radials {nsite} to {rsite} A, {nbond} bond radials, site group {site}, env {env}, "
          f"bond cutoff {rcut}; largest problem {kmax} columns; fitted in {time.perf_counter() - t:.0f} s", flush=True)
    del train
    if save_model:                                       # `operators.io`: aj eval and the calculator read it
        from ace_jax.operators import io
        io.save(save_model, on, off, info=dict(record, data=os.path.abspath(folder)))
    f_on, f_off = OnsitePredictor(on), OffsitePredictor(off, threads=threads)

    def predict(d, name):
        f_on(d.positions, d.cell, d.numbers); f_off(d.positions, d.cell, d.numbers)          # compile, lists
        t = time.perf_counter()
        out = {**f_off(d.positions, d.cell, d.numbers)["H"], **f_on(d.positions, d.cell, d.numbers)["H"]}
        return out, (time.perf_counter() - t) / len(d.numbers)

    save(out, {**record, **score(folder, nel, predict, nk, ntest, T, f"{threads} threads")})


def save(out, record):
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(record) + "\n")


def score(folder, nel, predict, nk, ntest, T, label):
    """The test set's H MAE and smeared eigenvalue error of predict(d, name) -> (blocks, s per atom)."""
    ks = np.array([[a, b, c] for a in range(nk) for b in range(nk) for c in range(nk)]) / nk
    tests = sorted(os.listdir(os.path.join(folder, "test")))[:ntest]
    err, ee, ee_abs, times, per = [], [], [], [], []
    for name in tests:
        d = deeph.read(os.path.join(folder, "test", name))
        pred, dt = predict(d, name)
        times.append(dt)
        assert not [k for k in d.blocks["H"] if k not in pred], "the prediction lacks blocks of the reference"
        diff = np.concatenate([(pred[k] - v).ravel() for k, v in d.blocks["H"].items()])
        err.append(diff)
        bs = BlockSum(list(d.blocks["H"]), d.numbers, d.basis)             # Bloch sums and eigenvalues in batches of k, on the device
        cached = os.path.join(folder, f".bands_{nk}", name + ".npy")
        if os.path.exists(cached):
            e0 = np.load(cached)
        else:
            e0 = block_bands(bs, d.blocks["H"], d.blocks["S"], ks)
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            np.save(cached, e0)
        e1 = block_bands(bs, pred, d.blocks["S"], ks)
        mu0, mu1 = (chemical_potential(e, nel * len(d.numbers), T) for e in (e0, e1))
        w = fermi(e0, mu0, T)
        ee.append((np.abs((e1 - mu1) - (e0 - mu0)) * w).sum() / w.sum())
        ee_abs.append((np.abs(e1 - e0) * w).sum() / w.sum())
        per.append(M.spectrum_metrics(e0, e1, nel * len(d.numbers), len(d.numbers), width=KB * T, window=WINDOW))
        print(f"  {name}: H MAE {1e3 * np.abs(diff).mean():.3f} meV, EE {1e3 * ee[-1]:.2f} meV "
              f"(absolute {1e3 * ee_abs[-1]:.2f}), {1e3 * dt:.1f} ms per atom", flush=True)
    err = np.concatenate(err)
    print(f"test set ({len(tests)} structures): H MAE {1e3 * np.abs(err).mean():.3f} meV, RMSE {1e3 * np.sqrt((err ** 2).mean()):.3f} meV, "
          f"largest {1e3 * np.abs(err).max():.1f} meV; EE mean {1e3 * np.mean(ee):.2f} meV (absolute {1e3 * np.mean(ee_abs):.2f}); "
          f"{1e3 * np.mean(times):.1f} ms per atom; {label}")
    print("\n".join(M.format_summary(M.summary(per))), flush=True)
    return dict(n_test=len(tests), h_mae_meV=1e3 * float(np.abs(err).mean()), h_rmse_meV=1e3 * float(np.sqrt((err ** 2).mean())),
                h_max_meV=1e3 * float(np.abs(err).max()), ee_meV=1e3 * float(np.mean(ee)), ee_abs_meV=1e3 * float(np.mean(ee_abs)),
                ms_per_atom=1e3 * float(np.mean(times)), metrics=M.summary(per), window=list(WINDOW))


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    for x in [x for x in a if "=" in x]:
        k, v = x.split("=")
        kw[k] = (v if k in ("pred", "out", "save_model") else None if v == "none" else float(v) if k in ("rsite", "smooth") else int(v) if k in ("bonds", "nsite", "nbond", "nk", "threads", "ntest", "seed")
                 else tuple(float(t) if k == "env" and i == 0 else int(t) for i, t in enumerate(v.split(","))))
    pos = [x for x in a if "=" not in x]
    main(pos[0], int(pos[1]), float(pos[2]), **kw)
