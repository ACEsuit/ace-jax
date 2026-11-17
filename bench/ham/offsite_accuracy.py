"""Accuracy of the shared-feature off-site model on the aluminium data of the ACEhamiltonians
paper (bench/ham/aitb.py): off-site H and S blocks of pairs within a bond cutoff.

Train on bonds sampled from BCC SK-supercell-001 and FCC SK-supercell-002, test on bonds
sampled from FCC SK-supercell-001 and BCC SK-supercell-002.  Blocks are divided by 27 (the
files hold a 3 x 3 x 3 k-sum).  The supercells are 25 angstrom wide, so a pair within the bond
cutoff has one image and its block is the dense matrix's.
    uv run python bench/ham/offsite_accuracy.py <n site radials> <n_bond> <n_cpl> <l_cpl> [rcut_bond=9] [n_train=20000] [n_test=10000]
    uv run python bench/ham/offsite_accuracy.py export <out.h5> [rcut_bond=9] [n_train=20000] [n_test=10000]
The second form writes the same bonds (drawn with the same seeds) with their blocks in FHI-aims'
own phases, for bench/ham/aceh_offsite_accuracy.jl.
"""
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aitb

from ace_jax.ham import bond as B
from ace_jax.ham import cg
from ace_jax.ham.deeph import OperatorData
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.fit import fit_offsite
from ace_jax.ham.offsite import block_lps, canonical_pairs, site_features

DATA = os.environ.get("ACEH_DATA", "/storage/eng/essswb/ACEhamiltonians-data/training_data")
TRAIN, TEST = ["BCC/SK-supercell-001.h5", "FCC/SK-supercell-002.h5"], ["FCC/SK-supercell-001.h5", "BCC/SK-supercell-002.h5"]
KGRID, N = 27.0, 14
CH = ("H", "S")


def dataset(name, basis, n_bonds, seed):
    """OperatorData holding the blocks of `n_bonds` bonds drawn at random within the bond cutoff."""
    d = aitb.load(os.path.join(DATA, name), operators=CH, raw=True)
    g = B.bond_graph(basis, d["positions"], d["cell"], d["numbers"])
    pick = np.random.default_rng(seed).choice(len(g.i), size=min(n_bonds, len(g.i)), replace=False)
    s = aitb.atom_signs(aitb.AL.ls(13))
    ss = s[:, None] * s[None, :] / KGRID
    blocks = {c: {} for c in CH}
    for e in pick:
        i, j = int(g.i[e]), int(g.j[e])
        key = (int(g.R[e, 0]), int(g.R[e, 1]), int(g.R[e, 2]), i, j)
        for c in CH:
            blocks[c][key] = d[c + "_full"][i * N:(i + 1) * N, j * N:(j + 1) * N] * ss
    return OperatorData(d["cell"], d["positions"], d["numbers"], aitb.AL, blocks), len(g.i)


def errors(model, datasets, chunk=4096):
    """Squared error and count per (channel, block type) and per (channel, distance bin), over the
    canonical blocks of the bonds present in `datasets`."""
    basis, orb = model.bond, model.orbitals
    lps = block_lps(orb)
    by_type, by_dist = {}, {}
    for d in datasets:
        F = site_features(basis, d.positions, d.cell, d.numbers)
        g = B.bond_graph(basis, d.positions, d.cell, d.numbers)
        keys = [(int(r[0]), int(r[1]), int(r[2]), int(a), int(b)) for r, a, b in zip(g.R, g.i, g.j)]
        sel_all = np.flatnonzero([k in d.blocks[CH[0]] for k in keys])
        for lo in range(0, len(sel_all), chunk):
            sel = sel_all[lo:lo + chunk]
            f = B.features(basis, F, g, sel, lps)
            ref = np.stack([[d.blocks[c][keys[e]] for c in CH] for e in sel])
            dist = np.linalg.norm(g.rij[sel], axis=1)
            for s1, s2, l1, l2, same in canonical_pairs(orb, 13, 13):
                p = 1 if (l1 + l2) % 2 == 0 else -1
                blk = 0.0
                for L in cg.Ls(l1, l2):
                    c = model.coef.get((13, 13, s1, s2, L))
                    if c is not None:
                        part = np.einsum("ekm,ck->ecm", B.design(f[(L, p)], L, same), np.asarray(c))
                        blk = blk + np.einsum("ecm,ijm->ecij", part, cg.real_cg(l1, l2, L))
                diff2 = (blk - ref[:, :, orb.shell_slice(13, s1), orb.shell_slice(13, s2)]) ** 2
                for ci, c in enumerate(CH):
                    a = by_type.setdefault((c, "spd"[l1] + "spd"[l2]), [0.0, 0])
                    a[0] += diff2[:, ci].sum(); a[1] += diff2[:, ci].size
                    for lo_, hi_ in ((0, 3.5), (3.5, 5), (5, 7), (7, 99)):
                        m = (dist >= lo_) & (dist < hi_)
                        a = by_dist.setdefault((c, f"{lo_}-{hi_}"), [0.0, 0])
                        a[0] += diff2[m, ci].sum(); a[1] += diff2[m, ci].size
    rms = lambda t: {k: round(1e3 * float(np.sqrt(v[0] / max(v[1], 1))), 4) for k, v in t.items()}
    tot = {c: round(1e3 * float(np.sqrt(sum(v[0] for k, v in by_type.items() if k[0] == c)
                                        / sum(v[1] for k, v in by_type.items() if k[0] == c))), 4) for c in CH}
    return tot, rms(by_type), rms(by_dist)


def export(out, rcut_bond=9.0, n_train=20000, n_test=10000):
    import h5py
    basis = B.BondBasis(site=None, rcut_bond=rcut_bond, n_bond=1)
    s = aitb.atom_signs(aitb.AL.ls(13))
    undo = 1.0 / (s[:, None] * s[None, :])                       # back to FHI-aims' phases
    with h5py.File(out, "w") as f:
        f["shells"] = np.asarray(aitb.AL.ls(13))
        for tag, names, n, seed0 in (("train", TRAIN, n_train // 2, 0), ("test", TEST, n_test // 2, 10)):
            for k, name in enumerate(names):
                d, _ = dataset(name, basis, n, seed0 + k)
                keys = sorted(d.blocks["H"])
                g = f.create_group(f"{tag}/{k}")
                g["positions"], g["cell"] = d.positions, d.cell
                g["bonds"] = np.asarray(keys, np.int64)                 # (n, 5): Rx, Ry, Rz, i, j (0-based)
                for c in CH:
                    g[c] = np.stack([d.blocks[c][key] * undo for key in keys])
                print(tag, name, len(keys), "bonds")


def main(nsite, n_bond, n_cpl, l_cpl, rcut_bond=9.0, n_train=20000, n_test=10000):
    site = build_site_basis([13], order=1, totaldegree=40, lmax_feat=4, rcut=9.0, wL=0.5,
                            nmax_by_order=(nsite,), lmax_by_order=(4,))
    basis = B.BondBasis(site=site, rcut_bond=rcut_bond, n_bond=n_bond, n_cpl=n_cpl, l_cpl=l_cpl)
    train = [dataset(n, basis, n_train // 2, k) for k, n in enumerate(TRAIN)]
    test = [dataset(n, basis, n_test // 2, 10 + k)[0] for k, n in enumerate(TEST)]
    t = time.perf_counter()
    model, report = fit_offsite(basis, [d for d, _ in train], channels=CH)
    t_fit = time.perf_counter() - t
    ncoef = sum(np.asarray(c).shape[1] for c in model.coef.values())
    kmax = max(np.asarray(c).shape[1] for c in model.coef.values())
    tot, by_type, by_dist = errors(model, test)
    tot_tr, _, _ = errors(model, [d for d, _ in train])
    print(f"site radials {nsite}, n_bond {n_bond}, n_cpl {n_cpl}, l_cpl {l_cpl}, rcut_bond {rcut_bond}: "
          f"{train[0][1] // 729} bonds per atom, {sum(len(d.blocks['H']) for d, _ in train)} training bonds, "
          f"{ncoef} coefficients per channel (largest problem {kmax}); fit {t_fit:.0f} s")
    print(f"   RMSE over canonical blocks, meV (S x 1e3): train {tot_tr}, test {tot}")
    print("   test by block type:", {f"{c} {t_}": v for (c, t_), v in by_type.items()})
    print("   test by bond length:", {f"{c} {t_}": v for (c, t_), v in by_dist.items()})


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "export":
        export(a[1], *(float(x) if k == 0 else int(x) for k, x in enumerate(a[2:])))
    else:
        main(int(a[0]), int(a[1]), int(a[2]), int(a[3]), *(float(x) if k == 0 else int(x) for k, x in enumerate(a[4:])))
