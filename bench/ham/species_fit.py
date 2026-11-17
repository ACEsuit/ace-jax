"""Many elements: categorical against element-embedding (TrACE) features on one dataset.

Dataset: C, Si, Ge and Sn at random on the diamond lattice (16 atoms, rattled; FHI-aims 261002,
PBE, light defaults; `ham-ref/giv_alloy/make_structures.py`), npz caches.  Each model is one
on-site and one off-site model for H and S: a site basis at correlation order 1, the site-coupled
group and the bond-environment group, whose atoms enter through the same species channels as the
site basis:
  categorical   `build_site_basis`, one radial channel and one environment channel per element
  TrACE d       `build_embedding_site_basis` with d channels of a random element embedding (seeded
                Gaussian rows of unit norm), and the environment group on the same embedding rows
The readouts stay per species (on-site) and per species pair (off-site) in every model.
    JAX_PLATFORMS=cpu uv run python bench/ham/species_fit.py <npz dir> <n_test> [bonds per structure=3000]
        [d=1,2,3] [categorical=1] [threads=8] [seed=0] [smooth=<sigma_c>]
(one model per process is safest: the compiled programs of four elements' species pairs are many,
and a second model in the same process ran out of mappings for compiled code)
Errors on the test structures: block RMSE, and eigenvalues from E_F - 12 to E_F + 5 eV on a k-grid
with the reference overlap, in the span of the overlap's eigenvectors above 1e-3 (light bases are
nearly dependent; docs/dev/ham-results.md)."""
import glob
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc

from ace_jax.ham import cache
from ace_jax.operators import bond as B
from ace_jax.operators.data import OperatorData
from ace_jax.operators.features import build_embedding_site_basis, build_site_basis
from ace_jax.operators.fit import fit_offsite, fit_onsite
from ace_jax.operators.offsite_fast import OffsitePredictor
from ace_jax.operators.onsite import OnsitePredictor

CH, WIDTH, NK, STHR = ("H", "S"), 0.1, 3, 1e-3
VALENCE = {6: 6, 14: 14, 32: 32, 50: 50}             # all-electron bases: every electron


def onsite(k):
    return k[:3] == (0, 0, 0) and k[3] == k[4]


def subset(d, n, seed):
    keys = sorted(k for k in d.blocks["S"] if not onsite(k))
    pick = np.random.default_rng(seed).choice(len(keys), size=min(n, len(keys)), replace=False)
    keep = {keys[i] for i in pick}
    keep |= {(-k[0], -k[1], -k[2], k[4], k[3]) for k in keep} | {k for k in d.blocks["S"] if onsite(k)}
    return OperatorData(d.cell, d.positions, d.numbers, d.basis, {c: {k: d.blocks[c][k] for k in keep} for c in CH})


def bloch(d, blocks, ks):
    off = np.concatenate([[0], np.cumsum([d.basis.n_orb(int(z)) for z in d.numbers])])
    M = np.zeros((len(ks), off[-1], off[-1]), complex)
    for (Rx, Ry, Rz, i, j), b in blocks.items():
        M[:, off[i]:off[i + 1], off[j]:off[j + 1]] += np.exp(2j * np.pi * (ks @ (Rx, Ry, Rz)))[:, None, None] * b
    return 0.5 * (M + M.conj().transpose(0, 2, 1))


def spectrum(H, X, nel):
    e = np.array([np.linalg.eigvalsh(x.conj().T @ h @ x) for h, x in zip(H, X)])
    mu = brentq(lambda m: erfc((e - m) / WIDTH).sum() / len(e) - nel, e.min(), e.max())
    return e, mu


def main(folder, n_test, n_bonds=3000, d=(1, 2, 3), categorical=1, threads=8, seed=0, smooth=None, nsite=12, rsite=6.0, env=(5.0, 4, 4, 2, 4), site=(2, 2)):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    data = [cache.load(f)[0] for f in files]
    train, test = data[:-n_test], data[-n_test:]
    orb = data[0].basis
    els = [int(z) for z in orb.species]
    lf = 2 * orb.lmax
    rcut = float(np.ceil(10 * max(np.linalg.norm(x.bond(k)) for x in data for k in x.blocks["S"])) / 10)
    print(f"{len(train)} training and {len(test)} test structures of {len(data[0].numbers)} atoms; elements {els}, "
          f"orbitals per atom {[orb.n_orb(z) for z in els]}; bond cutoff {rcut}; {n_bonds} training bonds of each structure", flush=True)
    models = []
    if categorical:
        models.append(("categorical", build_site_basis(els, 1, 40, lf, rcut=rsite, wL=0.5, nmax_by_order=(nsite,), lmax_by_order=(lf,)).with_table(), None))
    for dd in d:
        sb = build_embedding_site_basis(els, 1, nsite + 0.5 * lf, lf, "random", d_max=dd, rcut=rsite, wL=0.5, seed=seed)
        models.append((f"TrACE d = {dd}", sb, tuple(tuple(float(v) for v in row) for row in np.asarray(sb.radial.rnl_embedding))))
    ks = np.array([[a, b, c] for a in range(NK) for b in range(NK) for c in range(NK)]) / NK
    ref = []
    for t in test:
        S = bloch(t, t.blocks["S"], ks)
        sv = [np.linalg.eigh(s) for s in S]
        keep = min(int((w > STHR).sum()) for w, _ in sv)
        X = [v[:, -keep:] / np.sqrt(w[-keep:]) for w, v in sv]
        e, mu = spectrum(bloch(t, t.blocks["H"], ks), X, sum(VALENCE[int(z)] for z in t.numbers))
        ref.append((X, e, mu))
    for name, sb, emb in models:
        basis = B.BondBasis(site=sb, rcut_bond=rcut, n_bond=12, n_cpl=site[0], l_cpl=site[1], r_env=env[0], n_env=int(env[1]),
                            l_env=int(env[2]), n_ebond=int(env[3]), l_ebond=int(env[4]), env_emb=emb)
        t0 = time.perf_counter()
        on, _ = fit_onsite(sb, train, channels=CH, sigma_c=smooth)
        off, _ = fit_offsite(basis, [subset(x, n_bonds, k) for k, x in enumerate(train)], channels=CH, sigma_c=smooth)
        tfit = time.perf_counter() - t0
        f_on, f_off = OnsitePredictor(on), OffsitePredictor(off, threads=threads)
        errs, band, times = {}, [], []
        for t, (X, e0, mu0) in zip(test, ref):
            f_on(t.positions, t.cell, t.numbers); f_off(t.positions, t.cell, t.numbers)
            t0 = time.perf_counter()
            po, pf = f_on(t.positions, t.cell, t.numbers), f_off(t.positions, t.cell, t.numbers)
            times.append((time.perf_counter() - t0) / len(t.numbers))
            pred = {c: {**pf[c], **po[c]} for c in CH}
            for c in CH:
                for part, sel in (("on", onsite), ("off", lambda k: not onsite(k))):
                    errs.setdefault((c, part), []).append(np.concatenate(
                        [(pred[c][k] - v).ravel() for k, v in t.blocks[c].items() if sel(k)]))
            e1, mu1 = spectrum(bloch(t, {k: pred["H"][k] for k in t.blocks["H"]}, ks), X, sum(VALENCE[int(z)] for z in t.numbers))
            w = (e0 > mu0 - 12.0) & (e0 < mu0 + 5.0)
            band.append(((e1 - mu1) - (e0 - mu0))[w])
        rms = lambda v: 1e3 * np.sqrt((np.concatenate(v) ** 2).mean())
        nfeat = sum(sb.n(*lp) for lp in sb.lps)
        print(f"{name:14s} site features {nfeat:4d}, env channels {B.env_embedding(basis).shape[1]}, largest off-site problem "
              f"{max(np.asarray(c).shape[1] for c in off.coef.values()):4d} columns, fit {tfit:4.0f} s | H on-site {rms(errs[('H', 'on')]):7.2f} "
              f"off-site {rms(errs[('H', 'off')]):6.2f} meV; S off-site {rms(errs[('S', 'off')]):.3f} meV | eigenvalues RMSE "
              f"{rms(band):6.1f} meV, max {1e3 * np.abs(np.concatenate(band)).max():5.0f} | {1e3 * np.mean(times):.2f} ms per atom, {threads} threads",
              flush=True)
        del f_on, f_off
        jax.clear_caches()


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    for x in [x for x in a if "=" in x]:
        k, v = x.split("=")
        kw[k] = tuple(int(t) for t in v.split(",") if t) if k == "d" else float(v) if k == "smooth" else int(v)
    pos = [x for x in a if "=" not in x]
    main(pos[0], int(pos[1]), *(int(x) for x in pos[2:]), **kw)
