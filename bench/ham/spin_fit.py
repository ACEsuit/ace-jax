"""Collinear-spin fit: bcc Fe (16-atom rattled and strained supercells, FHI-aims 261002, PBE,
light defaults; `ingest.jl fhi-aims-spin`, npz caches).  One on-site and one off-site model with
the channels (H_up, H_dn, S) on shared features; tested on structures not in the fit: block
errors per channel, the eigenvalues of each spin channel on a k-grid, and the moment
N_up - N_dn at the Fermi level of the predicted bands.
    JAX_PLATFORMS=cpu uv run python bench/ham/spin_fit.py <npz dir> <n_test> [bonds per structure=3000]
        [site=<n_cpl>,<l_cpl>] [env=<r_env>,<n_env>,<l_env>,<n_ebond>,<l_ebond>] [nsite=12] [threads=8]
The last <n_test> structures (by name) are the test set."""
import glob
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import scipy.linalg
from scipy.optimize import brentq
from scipy.special import erfc

from ace_jax.ham import bond as B
from ace_jax.ham import cache
from ace_jax.ham.deeph import OperatorData
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.fit import fit_offsite, fit_onsite
from ace_jax.ham.offsite_fast import OffsitePredictor
from ace_jax.ham.onsite import OnsitePredictor

CH, WIDTH, NK, RCUT = ("H_up", "H_dn", "S"), 0.1, 4, 9.0
STHR = (1e-3, 1e-2, 1e-1)


def subset(d, n, seed):
    """`d` with its on-site blocks and a random `n` of its bonds (and their reverses)."""
    keys = sorted(k for k in d.blocks["S"] if k[:3] != (0, 0, 0) or k[3] != k[4])
    pick = np.random.default_rng(seed).choice(len(keys), size=min(n, len(keys)), replace=False)
    keep = {keys[i] for i in pick}
    keep |= {(-k[0], -k[1], -k[2], k[4], k[3]) for k in keep} | {k for k in d.blocks["S"] if k not in set(keys)}
    return OperatorData(d.cell, d.positions, d.numbers, d.basis, {c: {k: d.blocks[c][k] for k in keep} for c in CH})


def dense(d, blocks, ks):
    """{channel: (n_k, n, n)} Bloch matrices of a dict of blocks."""
    off = np.concatenate([[0], np.cumsum([d.basis.n_orb(int(z)) for z in d.numbers])])
    out = {}
    for c in blocks:
        M = np.zeros((len(ks), off[-1], off[-1]), complex)
        for (Rx, Ry, Rz, i, j), b in blocks[c].items():
            M[:, off[i]:off[i + 1], off[j]:off[j + 1]] += np.exp(2j * np.pi * (ks @ (Rx, Ry, Rz)))[:, None, None] * b
        out[c] = 0.5 * (M + M.conj().transpose(0, 2, 1))
    return out


def spectrum(M, S, nel, sthr=0.0):
    """Eigenvalues (2, n_k, n) of the two spin channels, the Fermi level that holds `nel`
    electrons with Gaussian smearing, and the moment there.  `sthr` > 0: solve in the span of
    the eigenvectors of S(k) with eigenvalue above it (canonical orthogonalisation; the same
    number of them at every k, the smallest count), which leaves out the nearly dependent
    combinations of orbitals that a small error in H turns into spurious states."""
    if sthr <= 0:
        e = np.array([[scipy.linalg.eigh(h, s, eigvals_only=True) for h, s in zip(M[c], S)] for c in ("H_up", "H_dn")])
    else:
        sv = [np.linalg.eigh(s) for s in S]
        keep = min(int((w > sthr).sum()) for w, _ in sv)
        X = [v[:, -keep:] / np.sqrt(w[-keep:]) for w, v in sv]
        e = np.array([[np.linalg.eigvalsh(x.conj().T @ h @ x) for h, x in zip(M[c], X)] for c in ("H_up", "H_dn")])
    occ = lambda mu: 0.5 * erfc((e - mu) / WIDTH).sum(axis=(1, 2)) / e.shape[1]
    mu = brentq(lambda m: occ(m).sum() - nel, e.min(), e.max())
    n_up, n_dn = occ(mu)
    return e, mu, n_up - n_dn


def main(folder, n_test, n_bonds=3000, site_cpl=(2, 2), env=(5.0, 6, 4, 4, 4), nsite=12, threads=8):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    data = [cache.load(f)[0] for f in files]
    train, test = data[:-n_test], data[-n_test:]
    orb = data[0].basis
    site = build_site_basis([26], order=1, totaldegree=40, lmax_feat=2 * orb.lmax, rcut=7.0, wL=0.5,
                            nmax_by_order=(nsite,), lmax_by_order=(2 * orb.lmax,))
    basis = B.BondBasis(site=site, rcut_bond=RCUT, n_bond=12, n_cpl=site_cpl[0], l_cpl=site_cpl[1])
    if env:
        basis = basis._replace(r_env=env[0], n_env=int(env[1]), l_env=int(env[2]), n_ebond=int(env[3]), l_ebond=int(env[4]))
    t = time.perf_counter()
    onsite, _ = fit_onsite(site, train, channels=CH)
    offsite, _ = fit_offsite(basis, [subset(d, n_bonds, k) for k, d in enumerate(train)], channels=CH)
    kmax = max(np.asarray(c).shape[1] for c in offsite.coef.values())
    print(f"{len(train)} training structures ({len(train[0].numbers)} atoms, {orb.n_orb(26)} orbitals per atom), {n_bonds} bonds of each; "
          f"site radials {nsite}, site group {site_cpl}, env {env}, bond cutoff {RCUT}; largest problem {kmax} columns; "
          f"fitted in {time.perf_counter() - t:.0f} s")
    f_on, f_off = OnsitePredictor(onsite), OffsitePredictor(offsite, threads=threads)
    ks = np.array([[a, b, c] for a in range(NK) for b in range(NK) for c in range(NK)]) / NK
    acc = {}
    for name, d in zip(files[-n_test:], test):
        t = time.perf_counter()
        on, off = f_on(d.positions, d.cell, d.numbers), f_off(d.positions, d.cell, d.numbers)
        dt = time.perf_counter() - t
        pred = {c: {**off[c], **on[c]} for c in CH}
        for c in CH:
            for part, sel in (("on-site", lambda k: k[:3] == (0, 0, 0) and k[3] == k[4]), ("off-site", lambda k: not (k[:3] == (0, 0, 0) and k[3] == k[4]))):
                diff = np.concatenate([(pred[c][k] - v).ravel() for k, v in d.blocks[c].items() if sel(k) and k in pred[c]])
                acc.setdefault((c, part), []).append(diff)
        nel = float(d.numbers.sum())
        ref, prd = dense(d, d.blocks, ks), dense(d, pred, ks)
        e0, mu0, m0 = spectrum(ref, ref["S"], nel)
        smin = min(np.linalg.eigvalsh(x)[0] for x in ref["S"])
        acc.setdefault(("smin", ""), []).append(smin)
        cases = [("predicted H and S", prd["S"], 0.0), ("predicted H, reference S", ref["S"], 0.0)]
        cases += [(f"predicted H, reference S, S(k) eigenvalues above {t:g}", ref["S"], t) for t in STHR]
        for label, S, sthr in cases:
            if sthr:                                   # the reference in the same subspace, and what leaving the rest out costs
                e0s, mu0s, m0s = spectrum(ref, ref["S"], nel, sthr)
                w = (e0[..., :e0s.shape[-1]] > mu0 - 12.0) & (e0[..., :e0s.shape[-1]] < mu0 + 5.0)
                acc.setdefault(("trunc", label), []).append(((e0s - mu0s) - (e0[..., :e0s.shape[-1]] - mu0))[w])
            else:
                e0s, mu0s, m0s = e0, mu0, m0
            try:
                e1, mu1, m1 = spectrum(prd, S, nel, sthr)
            except np.linalg.LinAlgError:              # the predicted S(k) is not positive definite
                acc.setdefault(("failed", label), []).append(os.path.basename(name))
                continue
            e0_, mu0_, m0_ = e0, mu0, m0
            e0, mu0, m0 = e0s, mu0s, m0s
            val = (e0 > mu0 - 12.0) & (e0 < mu0 + 5.0)
            err = (e1 - mu1) - (e0 - mu0)
            acc.setdefault(("bands", label), []).append(err[val])
            acc.setdefault(("moment", label), []).append((m1 / len(d.numbers), m0 / len(d.numbers)))
            acc.setdefault(("fermi", label), []).append(mu1 - mu0)
            e0, mu0, m0 = e0_, mu0_, m0_
        print(f"  {os.path.basename(name)}: predicted in {dt:.2f} s ({1e3 * dt / len(d.numbers):.1f} ms per atom, {threads} threads); "
              f"reference moment per atom {m0 / len(d.numbers):.3f}, smallest eigenvalue of S(k) {smin:.2e}")
    print("block errors on the test structures, meV (RMSE; largest):")
    for (c, part), v in acc.items():
        if c in CH:
            v = np.concatenate(v)
            print(f"  {c:5s} {part:9s} {1e3 * np.sqrt((v ** 2).mean()):9.3f} {1e3 * np.abs(v).max():10.1f}")
    for label in [k[1] for k in acc if k[0] == "bands"] + [k[1] for k in acc if k[0] == "failed" and ("bands", k[1]) not in acc]:
        if ("trunc", label) in acc:
            v = np.concatenate(acc[("trunc", label)])
            print(f"reference in that subspace against the full reference: RMSE {1e3 * np.sqrt((v ** 2).mean()):.2f} meV, max {1e3 * np.abs(v).max():.1f} meV")
        if ("failed", label) in acc:
            print(f"{label}: S(k) not positive definite for {len(acc[('failed', label)])} of {n_test} test structures")
        if ("bands", label) not in acc:
            continue
        v = np.concatenate(acc[("bands", label)])
        m = np.asarray(acc[("moment", label)])
        print(f"{label}: eigenvalues from E_F - 12 to E_F + 5 eV, both spins, relative to E_F ({v.size} values): "
              f"RMSE {1e3 * np.sqrt((v ** 2).mean()):.1f} meV, max {1e3 * np.abs(v).max():.0f} meV; "
              f"E_F error {1e3 * np.sqrt(np.mean(np.square(acc[('fermi', label)]))):.1f} meV RMS; "
              f"moment per atom: mean error {np.mean(m[:, 0] - m[:, 1]):+.4f}, RMS {np.sqrt(np.mean((m[:, 0] - m[:, 1]) ** 2)):.4f} mu_B "
              f"(reference {m[:, 1].min():.3f} to {m[:, 1].max():.3f})")


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    for x in [x for x in a if "=" in x]:
        k, v = x.split("=")
        kw[{"site": "site_cpl"}.get(k, k)] = (int(v) if k in ("nsite", "threads") else None if v == "none"
                                              else tuple(float(t) if k == "env" and i == 0 else int(t) for i, t in enumerate(v.split(","))))
    pos = [x for x in a if "=" not in x]
    main(pos[0], int(pos[1]), *(int(x) for x in pos[2:]), **kw)
