"""Big-cell calibration of one ARD run on the v2 (thin along the line) and v3 (thick) cells.

v2: <run>/big_err.npz scored with sites.npz's per-atom families; v3: <run>/big3_err.npz, which
carries its own family / r_core / fixed arrays (modal_bench365.big_errors).  Free atoms only.
Per family and r_core band: rms|dF|, median sigma, rms-z (with a 95 % block-bootstrap interval over
cells: atoms in one cell, and the 3 rattled copies relax_label writes of it, are not independent), cov90, Spearman rho(sigma, |dF|).

    python eval_big.py <sites.npz> <ard_run_dir>
"""
import sys

import numpy as np
from scipy.stats import chi2, spearmanr

S = np.load(sys.argv[1], allow_pickle=True)
run = sys.argv[2]
sets = {}
B = np.load(f"{run}/big_err.npz")
free = ~S["fixed_big"]
sets["v2 (thin)"] = (B["err"][free], B["sd"][free], S["fam_big"][free], S["rcore_big"][free],
                     S["cfg_big"][free] // 3)                     # 3 consecutive rattles per cell
try:
    C = np.load(f"{run}/big3_err.npz", allow_pickle=True)
    f3 = ~C["fixed"].astype(bool)
    sets["v3 (thick)"] = (C["err"][f3], C["sd"][f3], C["family"][f3].astype(str), C["r_core"][f3],
                          C["cfg"][f3] // 3)
except FileNotFoundError:
    print("no big3_err.npz")


def row(e, s, cfg, nboot=1000):
    z2 = e ** 2 / (s ** 2 / 3)
    u, inv = np.unique(cfg, return_inverse=True)
    num, den = np.bincount(inv, z2, len(u)), np.bincount(inv, minlength=len(u)).astype(float)
    W = np.random.default_rng(0).multinomial(len(u), np.full(len(u), 1 / len(u)), nboot)
    lo, hi = np.percentile(np.sqrt((W @ num) / (W @ den) / 3), [2.5, 97.5])
    return (f"{len(e):6d} {len(u):3d}  {np.sqrt(np.mean(e ** 2)):.3f}  {np.median(s):.3f}  "
            f"{np.sqrt(np.mean(z2) / 3):5.2f} [{lo:.2f},{hi:.2f}]  "
            f"{np.mean(z2 <= chi2.ppf(0.9, 3)):5.2f}  {spearmanr(s, e)[0]:5.2f}")


print(f"{'set':11s} {'family':6s} {'band':>9s} {'n':>6s} cells rmsF   med s  rms-z [95% CI]     cov90   rho")
for name, (e, s, fam, rc, cfg) in sets.items():
    for x in ("crack", "edge", "screw"):
        m = fam == x
        if not m.any():
            continue
        print(f"{name:11s} {x:6s} {'all':>9s} {row(e[m], s[m], cfg[m])}")
        for lo, hi in ((0, 10), (10, 20), (20, 80)):
            b = m & (rc >= lo) & (rc < hi)
            if b.sum() >= 50:
                print(f"{'':11s} {'':6s} {f'{lo}-{hi} A':>9s} {row(e[b], s[b], cfg[b])}")
