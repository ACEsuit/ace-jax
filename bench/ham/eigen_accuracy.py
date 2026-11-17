"""Eigenvalue error of ace_jax.ham's model on a held-out aluminium supercell of the
ACEhamiltonians paper data: fit on-site and off-site H and S on two supercells, predict the full
matrices of a third (729 atoms, 10206 orbitals, Gamma point of the supercell), solve
H c = e S c and compare the spectrum with the reference's.

Reported, against the full reference spectrum (eV -> meV), for the occupied states and for the
valence window (above E_F - 15 eV, below E_F + 3 eV):
  truncation: the REFERENCE matrices with pairs beyond the bond cutoff removed (what any model
              with this cutoff can reach at best);
  predicted H and S; and predicted H with the reference S.
    JAX_PLATFORMS=cpu uv run python bench/ham/eigen_accuracy.py <site radials> <n_bond> <n_cpl> <l_cpl> [rcut_bond=9] [n_train=20000] [test=FCC/SK-supercell-001.h5]
With `dump=<file.npz>` as a last argument the predicted and reference spectra are saved."""
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

from ace_jax.ham import bond as B
from ace_jax.ham.features import build_site_basis
from ace_jax.ham.fit import fit_offsite, fit_onsite
from ace_jax.ham.offsite_fast import OffsitePredictor
from ace_jax.ham.onsite import OnsitePredictor

N, KGRID = 14, 27.0
N_ELECTRONS = 13


def reference(name):
    d = aitb.load(os.path.join(OA.DATA, name), operators=("H", "S"), raw=True)
    s = np.tile(aitb.atom_signs(aitb.AL.ls(13)), len(d["positions"]))
    return d, {c: d[c + "_full"] * (s[:, None] * s[None, :]) / KGRID for c in ("H", "S")}


def spectrum(H, S):
    return scipy.linalg.eigh(0.5 * (H + H.T), 0.5 * (S + S.T), eigvals_only=True, check_finite=False)


def compare(e, ref, n_occ):
    ef = 0.5 * (ref[n_occ - 1] + ref[n_occ])
    d = e - ref
    occ = slice(0, n_occ)
    val = (ref > ef - 15.0) & (ref < ef + 3.0)                   # the 3s/3p band and 3 eV of empty states
    f = lambda x: f"MAE {1e3 * np.abs(x).mean():.3f}, RMSE {1e3 * np.sqrt((x ** 2).mean()):.3f}, max {1e3 * np.abs(x).max():.2f} meV"
    core = ref < ef - 15.0
    return (f"occupied ({n_occ} states): {f(d[occ])}; valence window E_F-15..E_F+3 eV ({int(val.sum())} states): {f(d[val])}; "
            f"core ({int(core.sum())} states): {f(d[core])}")


def main(nsite, n_bond, n_cpl, l_cpl, rcut_bond=9.0, n_train=20000, test="FCC/SK-supercell-001.h5", dump=None):
    site = build_site_basis([13], order=1, totaldegree=40, lmax_feat=4, rcut=9.0, wL=0.5,
                            nmax_by_order=(nsite,), lmax_by_order=(4,))
    basis = B.BondBasis(site=site, rcut_bond=rcut_bond, n_bond=n_bond, n_cpl=n_cpl, l_cpl=l_cpl)
    t = time.perf_counter()
    on_train = []
    for name, _ in ON.TRAIN:
        d = aitb.load(os.path.join(ON.DATA, name), operators=("H", "S"))
        from ace_jax.ham.deeph import OperatorData
        on_train.append(OperatorData(d["cell"], d["positions"], d["numbers"], aitb.AL,
                                     {c: {(0, 0, 0, i, i): d[c][i] / KGRID for i in range(len(d["positions"]))} for c in ("H", "S")}))
    onsite, _ = fit_onsite(site, on_train, channels=("H", "S"), atoms=[a for _, a in ON.TRAIN])
    offsite, _ = fit_offsite(basis, [OA.dataset(n, basis, n_train // 2, k)[0] for k, n in enumerate(OA.TRAIN)], channels=("H", "S"))
    print(f"site radials {nsite}, n_bond {n_bond}, n_cpl {n_cpl}, l_cpl {l_cpl}, bond cutoff {rcut_bond}, "
          f"{n_train} training bonds: fitted in {time.perf_counter() - t:.0f} s; test structure {test}")
    d, ref = reference(test)
    na = len(d["positions"])
    n_occ = (N_ELECTRONS * na + 1) // 2
    e_ref = spectrum(ref["H"], ref["S"])
    # predicted matrices
    t = time.perf_counter()
    _, (on_blocks,) = OnsitePredictor(onsite).dense(d["positions"], d["cell"], d["numbers"])
    g, off = OffsitePredictor(offsite).dense(d["positions"], d["cell"], d["numbers"])
    e, blk = off[(13, 13)]
    t_pred = time.perf_counter() - t
    pred = {c: np.zeros((na * N, na * N)) for c in ("H", "S")}
    keep = np.zeros((na, na), bool)                               # pairs inside the bond cutoff
    for ci, c in enumerate(("H", "S")):
        M = pred[c].reshape(na, N, na, N)
        M[np.arange(na), :, np.arange(na), :] = np.asarray(on_blocks)[:, ci]
        np.add.at(M, (g.i[e], slice(None), g.j[e], slice(None)), blk[:, ci])
    keep[g.i[e], g.j[e]] = True
    keep[np.arange(na), np.arange(na)] = True
    mask = np.repeat(np.repeat(keep, N, 0), N, 1)
    print(f"   prediction of H and S: {t_pred:.1f} s ({len(e) / na:.0f} bonds per atom); block RMSE inside the cutoff: "
          f"H {1e3 * np.sqrt(((pred['H'] - ref['H'])[mask] ** 2).mean()):.3f} meV, S {np.sqrt(((pred['S'] - ref['S'])[mask] ** 2).mean()):.2e}; "
          f"reference outside the cutoff: max |H| {1e3 * np.abs(ref['H'][~mask]).max():.2f} meV, max |S| {np.abs(ref['S'][~mask]).max():.2e}")
    out = {"ref": e_ref}
    for label, H, S in (("truncation only (reference inside the cutoff)", ref["H"] * mask, ref["S"] * mask),
                        ("predicted H and S", pred["H"], pred["S"]),
                        ("predicted H, reference S", pred["H"], ref["S"])):
        try:
            ev = spectrum(H, S)
            print(f"   {label}: {compare(ev, e_ref, n_occ)}")
            out[label] = ev
        except np.linalg.LinAlgError as err:
            print(f"   {label}: eigensolver failed ({err})")
    if dump:
        np.savez(dump, **{k.split(" (")[0].replace(" ", "_").replace(",", ""): v for k, v in out.items()})


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("dump=")]
    dump = next((x[5:] for x in sys.argv[1:] if x.startswith("dump=")), None)
    extra = [float(a[4])] if len(a) > 4 else []
    extra += [int(a[5])] if len(a) > 5 else []
    extra += [a[6]] if len(a) > 6 else []
    main(int(a[0]), int(a[1]), int(a[2]), int(a[3]), *extra, dump=dump)
