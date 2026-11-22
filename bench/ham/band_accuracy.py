"""Band-structure error on FCC and BCC aluminium with the metric of the ACEhamiltonians paper
(arXiv:2111.13736, Fig. 6(c); `band_error` in its test/plots.jl), its target being < 0.4 eV for
both phases with its optimised model.

Train as bench/ham/eigen_accuracy.py (500 K MD supercells).  Test on the equilibrium FCC and BCC
crystals (`reference_data/<phase>/<phase>-supercell-000.h5`, 9 x 9 x 9 primitive cells, not in
the training data): the primitive cell's H(R), S(R) are one atom's row of the supercell matrices;
bands along the paper's k-path (`ACEhamiltonians.jl/data/<phase>_tb_bandpath.json`).

  band error = sqrt( sum over k and over bands 2..14 of (e f(e) - e_ref f(e_ref))^2 / (N_k 14) ),
  f the Fermi function at the DFT Fermi level with sigma = 0.086 eV (1000 K); the 1s band is left out.

Also printed: the plain RMSE of the valence bands up to E_F + 10 eV, and the same two numbers for
the reference truncated at the bond cutoff.
    JAX_PLATFORMS=cpu uv run python bench/ham/band_accuracy.py <site radials> <n_bond> <n_cpl> <l_cpl> [rcut_bond=9] [n_train=20000]
        [env=<r_env>,<n_env>,<l_env>,<n_ebond>,<l_ebond>]
`env=`: add the bond-environment feature group (`operators/bond.py`).
"""
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aitb
import offsite_accuracy as OA
import onsite_accuracy as ON

from ace_jax.operators import bond as B
from ace_jax.operators.data import OperatorData
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import fit_offsite, fit_onsite
from ace_jax.operators.offsite_fast import OffsitePredictor
from ace_jax.operators.onsite import OnsitePredictor

ROOT = os.path.dirname(OA.DATA)                      # .../ACEhamiltonians-data
N, KGRID, SIGMA = 14, 27.0, 0.086


def bands(HR, SR, n, kpts):
    """(n_k, 14) eigenvalues of sum_R exp(2 pi i k.n_R) H(R) c = e (same for S) c."""
    out = []
    for k in kpts:
        ph = np.exp(2j * np.pi * (n @ k))
        H, S = np.einsum("r,rij->ij", ph, HR), np.einsum("r,rij->ij", ph, SR)
        out.append(scipy.linalg.eigh(0.5 * (H + H.conj().T), 0.5 * (S + S.conj().T), eigvals_only=True, driver="gvx"))
    return np.asarray(out)


def band_error(e, ref, ef):
    f = lambda x: 1.0 / (1.0 + np.exp((x - ef) / SIGMA))
    d = (e * f(e) - ref * f(ref))[:, 1:]                           # bands 2..14
    return np.sqrt((d ** 2).sum() / e.size)


def report(label, e, ref, ef):
    val = (ref > ef - 15.0) & (ref < ef + 10.0)
    d = (e - ref)[val]
    return (f"{label}: paper's band error {band_error(e, ref, ef):.4f} eV; valence bands up to E_F + 10 eV "
            f"({int(val.sum())} values): RMSE {1e3 * np.sqrt((d ** 2).mean()):.1f} meV, max {1e3 * np.abs(d).max():.0f} meV")


def main(nsite, n_bond, n_cpl, l_cpl, rcut_bond=9.0, n_train=20000, env=None):
    site = build_site_basis([13], order=1, totaldegree=40, lmax_feat=4, rcut=9.0, wL=0.5,
                            nmax_by_order=(nsite,), lmax_by_order=(4,))
    basis = B.BondBasis(site=site, rcut_bond=rcut_bond, n_bond=n_bond, n_cpl=n_cpl, l_cpl=l_cpl)
    if env:
        basis = basis._replace(r_env=env[0], n_env=int(env[1]), l_env=int(env[2]), n_ebond=int(env[3]), l_ebond=int(env[4]))
    t = time.perf_counter()
    on_train = []
    for name, _ in ON.TRAIN:
        d = aitb.load(os.path.join(ON.DATA, name), operators=("H", "S"))
        on_train.append(OperatorData(d["cell"], d["positions"], d["numbers"], aitb.AL,
                                     {c: {(0, 0, 0, i, i): d[c][i] / KGRID for i in range(len(d["positions"]))} for c in ("H", "S")}))
    onsite, _ = fit_onsite(site, on_train, channels=("H", "S"), atoms=[a for _, a in ON.TRAIN])
    del on_train
    offsite, _ = fit_offsite(basis, [OA.dataset(n, basis, n_train // 2, k)[0] for k, n in enumerate(OA.TRAIN)], channels=("H", "S"))
    kmax = max(np.asarray(c).shape[1] for c in offsite.coef.values())
    print(f"site radials {nsite}, n_bond {n_bond}, n_cpl {n_cpl}, l_cpl {l_cpl}, bond cutoff {rcut_bond}, env {env}, {n_train} training bonds; "
          f"largest problem {kmax} columns; fitted in {time.perf_counter() - t:.0f} s")
    f_on = OnsitePredictor(onsite)
    f_off = None if env else OffsitePredictor(offsite)
    s = aitb.atom_signs(aitb.AL.ls(13))
    for phase in ("FCC", "BCC"):
        path = json.load(open(os.path.join(ROOT, "ACEhamiltonians.jl", "data", f"{phase}_tb_bandpath.json")))
        kpts, ef = np.asarray(path["kpoints"], float), float(path["fermi"][0])
        d = aitb.load(os.path.join(ROOT, "reference_data", phase, f"{phase}-supercell-000.h5"), operators=("H", "S"), raw=True)
        pos, cell = d["positions"], d["cell"]
        prim = cell / 9.0                          # 9 x 9 x 9 primitive cells (the path file's cell has another lattice constant)
        na = len(pos)
        # the reference H(R), S(R): atom 0's row, each atom at its minimum image
        frac = (pos - pos[0]) @ np.linalg.inv(cell)
        rmin = (frac - np.rint(frac)) @ cell
        n_ref = np.rint(rmin @ np.linalg.inv(prim))
        assert np.abs(n_ref @ prim - rmin).max() < 1e-6, "the supercell is not a lattice of the primitive cell"
        # these files are not multiplied by a k-grid size, unlike the MD training files (27): the
        # on-site overlap says which (1 here)
        ss = s[:, None] * s[None, :] / np.rint(d["S_full"][0, 0])
        assert abs(d["S_full"][0, 0] / np.rint(d["S_full"][0, 0]) - 1) < 1e-3
        HR = np.stack([d["H_full"][:N, j * N:(j + 1) * N] * ss for j in range(na)])
        SR = np.stack([d["S_full"][:N, j * N:(j + 1) * N] * ss for j in range(na)])
        ref = bands(HR, SR, n_ref, kpts)
        inside = np.linalg.norm(rmin, axis=1) < rcut_bond
        trunc = bands(HR[inside], SR[inside], n_ref[inside], kpts)
        # the prediction: atom 0's on-site block and its bonds
        _, (on_blocks,) = f_on.dense(pos, cell, d["numbers"])
        if f_off is not None:
            g, off = f_off.dense(pos, cell, d["numbers"])
            e, blk = off[(13, 13)]
        else:                                           # reference predictor, atom 0's bonds only
            g = B.bond_graph(basis, pos, cell, d["numbers"])
            out = offsite.predict(pos, cell, d["numbers"], only=lambda gg: np.flatnonzero(gg.i == 0))
            e = np.flatnonzero(g.i == 0)
            keys = [(int(g.R[b, 0]), int(g.R[b, 1]), int(g.R[b, 2]), 0, int(g.j[b])) for b in e]
            blk = np.stack([[out[c][k] for c in ("H", "S")] for k in keys])
        mine = g.i[e] == 0
        n_pred = np.concatenate([np.zeros((1, 3)), np.rint(g.rij[e][mine] @ np.linalg.inv(prim))])
        Hp = np.concatenate([np.asarray(on_blocks)[:1, 0], blk[mine, 0]])
        Sp = np.concatenate([np.asarray(on_blocks)[:1, 1], blk[mine, 1]])
        pred = bands(Hp, Sp, n_pred, kpts)
        pred_refS = bands(Hp, np.concatenate([SR[:1], np.stack([SR[int(j)] for j in g.j[e][mine]])]), n_pred, kpts)
        print(f"  {phase}: {len(kpts)} k-points, E_F {ef:.3f} eV, {int(mine.sum())} bonds per atom inside the cutoff")
        print("    " + report("reference truncated at the cutoff", trunc, ref, ef))
        print("    " + report("predicted H and S", pred, ref, ef))
        print("    " + report("predicted H, reference S", pred_refS, ref, ef))
        # which part of H carries the error: swap in the reference for one part at a time (reference S throughout)
        jj = g.j[e][mine]
        Sr = np.concatenate([SR[:1], SR[jj]])
        H_on_ref = np.concatenate([HR[:1], blk[mine, 0]])
        H_off_ref = np.concatenate([np.asarray(on_blocks)[:1, 0], HR[jj]])
        print("    " + report("reference on-site H, predicted off-site H", bands(H_on_ref, Sr, n_pred, kpts), ref, ef))
        print("    " + report("predicted on-site H, reference off-site H", bands(H_off_ref, Sr, n_pred, kpts), ref, ef))
        dist = np.linalg.norm(g.rij[e][mine], axis=1)
        for lo, hi in ((0, 3.5), (3.5, 5.5), (5.5, 99)):
            Hm = np.concatenate([HR[:1], np.where(((dist >= lo) & (dist < hi))[:, None, None], blk[mine, 0], HR[jj])])
            print("    " + report(f"predicted H only for bonds {lo}-{hi} A", bands(Hm, Sr, n_pred, kpts), ref, ef))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("env=")]
    env = next((tuple(float(v) for v in x[4:].split(",")) for x in sys.argv[1:] if x.startswith("env=")), None)
    main(int(a[0]), int(a[1]), int(a[2]), int(a[3]), *(float(x) if k == 0 else int(x) for k, x in enumerate(a[4:])), env=env)
