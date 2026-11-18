"""Bands and density of states of the four-element alloy model against DFT.

Fits `species_fit.py`'s categorical model (its defaults, no prior), or loads it from
<out>.model.npz if that exists (`operators.io`; the fit writes it), and writes, for every test
structure, the eigenvalues of the DFT blocks and of the predicted H with the DFT overlap (full
overlap, no threshold) on the high-symmetry path of the cell (ASE's) and on a Monkhorst-Pack grid:
    uv run python bench/ham/alloy_bands.py fit <npz dir> <out.pkl> [n_test=16] [projected=1]
    uv run python bench/ham/alloy_bands.py plot <out.pkl> <figure.png> [structure index | median]"""
import glob
import os
import pickle
import sys

import numpy as np

WIN, SMEAR, NPATH, NK = (-13.0, 6.0), 0.1, 240, 8


def fit(folder, out, n_test=16, projected=1, n_bonds=3000, threads=8):
    import jax
    jax.config.update("jax_enable_x64", True)
    from ase import Atoms
    from ase.dft.kpoints import monkhorst_pack
    from scipy.linalg import eigh
    from scipy.optimize import brentq
    from scipy.special import erfc

    from species_fit import CH, WIDTH, bloch, onsite, subset

    from ace_jax.ham import cache
    from ace_jax.operators import bond as B
    from ace_jax.operators import io
    from ace_jax.operators.features import build_site_basis
    from ace_jax.operators.fit import fit_offsite, fit_onsite
    from ace_jax.operators.offsite_fast import OffsitePredictor
    from ace_jax.operators.onsite import OnsitePredictor
    data = [cache.load(f)[0] for f in sorted(glob.glob(os.path.join(folder, "*.npz")))]
    train, test = data[:-n_test], data[-n_test:]
    orb = data[0].basis
    els, lf = [int(z) for z in orb.species], 2 * orb.lmax
    rcut = float(np.ceil(10 * max(np.linalg.norm(x.bond(k)) for x in data for k in x.blocks["S"])) / 10)
    saved = os.path.splitext(out)[0] + ".model.npz"
    if os.path.exists(saved):
        on, off, _ = io.load(saved)
    else:
        sb = build_site_basis(els, 1, 40, lf, rcut=6.0, wL=0.5, nmax_by_order=(12,), lmax_by_order=(lf,)).with_table()
        basis = B.BondBasis(site=sb, rcut_bond=rcut, n_bond=12, n_cpl=2, l_cpl=2, r_env=5.0, n_env=4, l_env=4, n_ebond=2, l_ebond=4, env_emb=None)
        on, _ = fit_onsite(sb, train, channels=CH)
        off, _ = fit_offsite(basis, [subset(x, n_bonds, k) for k, x in enumerate(train)], channels=CH)
        io.save(saved, on, off, info=dict(data=os.path.abspath(folder), n_train=len(train), n_test=n_test, bonds_per_structure=n_bonds, projected=projected))
    f_on, f_off = OnsitePredictor(on), OffsitePredictor(off, threads=threads)
    res = []
    for t in test:
        po, pf = f_on(t.positions, t.cell, t.numbers), f_off(t.positions, t.cell, t.numbers)
        Hp = {k: {**pf["H"], **po["H"]}[k] for k in t.blocks["H"]}
        atoms = Atoms(numbers=t.numbers, positions=t.positions, cell=t.cell, pbc=True)
        path = atoms.cell.bandpath(npoints=NPATH)
        x, ticks, labels = path.get_linear_kpoint_axis()
        n = np.maximum(1, np.rint(NK * np.linalg.norm(atoms.cell.reciprocal(), axis=1) / np.linalg.norm(atoms.cell.reciprocal(), axis=1).max()).astype(int))
        nel = 4 * len(t.numbers) if projected else int(sum(t.numbers))       # core-projected: four valence electrons per atom
        r = dict(formula=atoms.get_chemical_formula(), x=x, ticks=ticks, labels=labels, grid=tuple(int(v) for v in n), nel=nel)
        for name, ks in (("path", path.kpts), ("grid", monkhorst_pack(n))):
            S = bloch(t, t.blocks["S"], ks)
            for tag, blocks in (("dft", t.blocks["H"]), ("model", Hp)):
                r[f"{name}_{tag}"] = np.array([eigh(h, s, eigvals_only=True) for h, s in zip(bloch(t, blocks, ks), S)])
        for tag in ("dft", "model"):
            e = r[f"grid_{tag}"]
            r[f"mu_{tag}"] = brentq(lambda m, e=e, nel=nel: erfc((e - m) / WIDTH).sum() / len(e) - nel, e.min(), e.max())
        w = (r["grid_dft"] > r["mu_dft"] + WIN[0] + 1) & (r["grid_dft"] < r["mu_dft"] + WIN[1] - 1)
        r["rmse"] = 1e3 * np.sqrt(np.mean((((r["grid_model"] - r["mu_model"]) - (r["grid_dft"] - r["mu_dft"]))[w]) ** 2))
        r["h_rmse"] = 1e3 * np.sqrt(np.mean(np.concatenate([(Hp[k] - v).ravel() for k, v in t.blocks["H"].items()]) ** 2))
        r["h_on"] = 1e3 * np.sqrt(np.mean(np.concatenate([(Hp[k] - v).ravel() for k, v in t.blocks["H"].items() if onsite(k)]) ** 2))
        res.append(r)
        print(f"{len(res) - 1:2d} {r['formula']:14s} grid {r['grid']} eigenvalues RMSE {r['rmse']:7.1f} meV (E_F - 12 to E_F + 5 eV, own Fermi levels), "
              f"E_F model - DFT {1e3 * (r['mu_model'] - r['mu_dft']):6.1f} meV, H blocks {r['h_rmse']:.1f} meV", flush=True)
        pickle.dump(res, open(out, "wb"))


def plot(src, fig, which="median"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = pickle.load(open(src, "rb"))
    i = int(np.argsort([r["rmse"] for r in res])[len(res) // 2]) if which == "median" else int(which)
    r = res[i]
    mu = r["mu_dft"]                                    # one energy zero for both: the DFT Fermi level
    INK, MODEL, MUTED = "#0b0b0b", "#2a78d6", "#8a8f98"          # the reference in ink, the model the one hue
    cut = np.r_[0, np.flatnonzero(np.diff(r["x"]) == 0) + 1, len(r["x"])]               # the path's disconnected pieces
    f, (a, b) = plt.subplots(1, 2, figsize=(9.0, 5.2), sharey=True, gridspec_kw=dict(width_ratios=(3.4, 1), wspace=0.04))
    for lo, hi in zip(cut[:-1], cut[1:]):
        a.plot(r["x"][lo:hi], r["path_dft"][lo:hi] - mu, color=INK, lw=1.6, solid_capstyle="round")
        a.plot(r["x"][lo:hi], r["path_model"][lo:hi] - mu, color=MODEL, lw=1.0, ls=(0, (4, 2)))
    for t in r["ticks"]:
        a.axvline(t, color=MUTED, lw=0.5)
    a.axhline(0.0, color=MUTED, lw=0.5, ls=":")
    a.set_xticks(r["ticks"]); a.set_xticklabels(["Γ" if s == "G" else s for s in r["labels"]])
    a.set_xlim(r["x"][0], r["x"][-1]); a.set_ylim(*WIN); a.set_ylabel("E − E$_F$ (DFT) / eV")
    a.plot([], [], color=INK, lw=1.6, label="DFT (FHI-aims)"); a.plot([], [], color=MODEL, lw=1.0, ls=(0, (4, 2)), label="ACE model")
    a.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="none", fontsize=9)
    e = np.linspace(WIN[0], WIN[1], 1200)
    dos = lambda ev: 2.0 * np.exp(-0.5 * ((e[:, None] - (ev.ravel() - mu)[None]) / SMEAR) ** 2).sum(1) / (SMEAR * np.sqrt(2 * np.pi) * len(ev))
    b.fill_betweenx(e, 0.0, dos(r["grid_dft"]), color=INK, alpha=0.18, lw=0)
    b.plot(dos(r["grid_dft"]), e, color=INK, lw=1.2)
    b.plot(dos(r["grid_model"]), e, color=MODEL, lw=1.0, ls=(0, (4, 2)))
    b.axhline(0.0, color=MUTED, lw=0.5, ls=":")
    b.set_xlim(0.0, None); b.set_xlabel("DoS / states eV$^{-1}$ cell$^{-1}$"); b.tick_params(left=False)
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    g = "×".join(str(v) for v in r["grid"])
    f.suptitle(f"{r['formula']} (test structure {i}): eigenvalue RMSE {r['rmse']:.0f} meV, H blocks {r['h_rmse']:.1f} meV; DoS on a {g} grid, {SMEAR} eV Gaussian",
               fontsize=10, x=0.125, ha="left")
    f.savefig(fig, dpi=200, bbox_inches="tight")
    print(f"structure {i} {r['formula']}: RMSE {r['rmse']:.1f} meV; all: " + " ".join(f"{x['rmse']:.0f}" for x in res))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "fit":
        fit(a[1], a[2], **{k: int(v) for k, v in (x.split("=") for x in a[3:])})
    else:
        plot(*a[1:])
