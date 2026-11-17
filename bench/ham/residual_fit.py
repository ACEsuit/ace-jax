"""Effective potential by residual: total energy = band free energy of the predicted H and S
+ an ace-jax potential fitted to what is left of the DFT energy, forces and stress.

    stage 1  fit H and S (on-site and off-site models) to the blocks of a few structures;
    stage 2  for every labelled structure, the band free energy, forces and stress of the PREDICTED
             H and S (`HamiltonianCalculator`: the calculation's own k-grid, smearing and electron
             count), and the residual labels DFT - band, written as extxyz;
    stage 3  `aj fit` on the residual file: an ordinary ace-jax model;
    also     `aj fit` with the same basis on the DFT labels themselves, for comparison.

    JAX_PLATFORMS=cpu uv run python bench/ham/residual_fit.py <npz dir> <labels dir> <work dir>
        [nh=2] [order=3] [degree=12] [rcut=6.0] [nk=2] [width=0.1] [nel=13] [limit=<structures>] [core=0]
`core`: lowest bands per atom left out of the band term (5 for Al's 1s 2s 2p: the valence band energy).
<npz dir>: H and S caches (`<name>.npz`); <labels dir>: `<name>/aims.out` with the total energy and,
if computed, forces and stress; names `*_100`.. are the test set, as in ham-ref/al_small.
The residual is defined by the H model and the electronic settings: both are recorded in
<work dir>/residual.json, and a potential fitted here is only valid with them."""
import json
import os
import re
import subprocess
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from ase import Atoms
from ase.io import read

from ace_jax.ham import cache
from ace_jax.ham.calculator import HamiltonianCalculator
from ace_jax.ham.operator import BlockSum, block_bands, smeared_occupations
from ace_jax.operators import bond as B
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import fit_offsite, fit_onsite


def labels(path):
    """(free energy, forces or None, stress 3x3 or None) of an FHI-aims output."""
    text = open(path).read()
    F = float(re.findall(r"\| Electronic free energy\s+:\s+(-?[\d.E+-]+) eV", text)[-1])
    forces = stress = None
    if "Total atomic forces" in text:
        a = read(path, format="aims-output")
        forces = a.get_forces()
        try:
            stress = a.get_stress(voigt=False)
        except Exception:
            stress = None
    return F, forces, stress


def write_xyz(path, frames):
    """extxyz with energy, forces and stress (eV, eV/A, eV/A^3) where given."""
    with open(path, "w") as f:
        for atoms, E, forces, stress in frames:
            info = 'Lattice="' + " ".join(f"{v:.12f}" for v in np.asarray(atoms.cell).ravel()) + '" pbc="T T T" ' + f"energy={E:.10f}"
            if stress is not None:
                info += ' stress="' + " ".join(f"{v:.12f}" for v in np.asarray(stress).ravel()) + '"'
            props = "species:S:1:pos:R:3" + (":forces:R:3" if forces is not None else "")
            f.write(f"{len(atoms)}\n{info} Properties={props}\n")
            for k, (s, p) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions)):
                row = f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}"
                if forces is not None:
                    row += " " + " ".join(f"{v:.10f}" for v in forces[k])
                f.write(row + "\n")


def aj_fit(train, test, out, order, degree, rcut, forces, stress):
    cmd = [sys.executable, "-m", "ace_jax.cli", "fit", "--train", train, "--test", test, "--elements", "Al", "--order", str(order),
           "--max-degree", str(degree), "--rcut", str(rcut), "--e0", "lsq", "--rungs", "map", "--out", out, "--energy-key", "energy"]
    cmd += ["--force-key", "forces"] if forces else []
    cmd += ["--stress-key", "stress"] if stress else []
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(os.path.join(out + ".log"), "w").write(r.stdout + r.stderr)
    return r.returncode, [ln for ln in (r.stdout + r.stderr).splitlines() if re.search(r"rmse|RMSE|mae|MAE", ln)][-12:]


def main(npz, lab, work, nh=2, order=3, degree=12, rcut=6.0, nk=2, width=0.1, nel=13, limit=None, core=0):
    os.makedirs(work, exist_ok=True)
    names = sorted(n for n in os.listdir(lab) if os.path.exists(os.path.join(lab, n, "aims.out")) and os.path.exists(os.path.join(npz, n + ".npz"))
                   and "Have a nice day" in open(os.path.join(lab, n, "aims.out")).read()[-3000:] and re.match(r"(fcc|bcc)_\d+", n))
    names = names[:: max(1, len(names) // limit)] if limit else names
    fit_names = [n for ph in ("fcc", "bcc") for n in sorted(os.listdir(npz)) if n.startswith(ph)][:: 1]
    fit_names = [f"fcc_{k:03d}.npz" for k in range(nh // 2)] + [f"bcc_{k:03d}.npz" for k in range(nh - nh // 2)]
    t = time.perf_counter()
    train = [cache.load(os.path.join(npz, n))[0] for n in fit_names]
    site = build_site_basis([13], order=1, totaldegree=40, lmax_feat=4, rcut=7.0, wL=0.5, nmax_by_order=(12,), lmax_by_order=(4,)).with_table()
    rmax = float(np.ceil(10 * max(np.linalg.norm(train[0].bond(k)) for k in train[0].blocks["S"])) / 10)
    basis = B.BondBasis(site=site, rcut_bond=rmax, n_bond=12, n_cpl=2, l_cpl=2, r_env=5.0, n_env=6, l_env=4, n_ebond=4, l_ebond=4)
    on, _ = fit_onsite(site, train)
    off, _ = fit_offsite(basis, train)
    print(f"stage 1: H and S fitted to {fit_names} in {time.perf_counter() - t:.0f} s (bond cutoff {rmax})", flush=True)
    calc = HamiltonianCalculator((on, off), kpts=(nk, nk, nk), nelectrons={13: nel}, width=width, smearing="gaussian",
                                 core_bands={13: core} if core else None)
    # the settings against the DFT calculation: band energy of the REFERENCE blocks and the model's
    d = train[0]
    bs = BlockSum(list(d.blocks["H"]), d.numbers, d.basis)
    e = block_bands(bs, d.blocks["H"], d.blocks["S"], calc.kpts)
    f, mu, mts = smeared_occupations(jax.numpy.asarray(e), jax.numpy.asarray(calc.weights), nel * len(d.numbers), width, "gaussian")
    text = open(os.path.join(lab, fit_names[0][:-4], "aims.out")).read()
    soe = float(re.findall(r"Sum of eigenvalues\s+:\s+(-?[\d.]+) Ha\s+(-?[\d.]+) eV", text)[-1][1])
    a0 = Atoms(numbers=d.numbers, positions=d.positions, cell=d.cell, pbc=True); a0.calc = calc
    eb = float((np.asarray(f) * e * calc.weights[:, None])[:, core * len(d.numbers):].sum())
    soe = soe if not core else float("nan")                              # FHI-aims prints the sum over all states only
    print(f"check on {fit_names[0][:-4]}: sum of eigenvalues, FHI-aims {soe:.4f} eV; reference blocks {eb:.4f} ({1e3 * (eb - soe) / len(d.numbers):+.3f} meV per atom); "
          f"predicted blocks {a0.get_potential_energy():.4f} ({1e3 * (a0.get_potential_energy() - eb) / len(d.numbers):+.3f} meV per atom); "
          f"Fermi level {float(mu):.4f} eV", flush=True)
    del train
    frames = {k: {"train": [], "test": []} for k in ("res", "resmu", "dft")}
    t = time.perf_counter()
    has_f = has_s = True
    os.makedirs(os.path.join(work, "band"), exist_ok=True)
    for n in names:
        F, forces, stress = labels(os.path.join(lab, n, "aims.out"))
        a = read(os.path.join(lab, n, "geometry.in"), format="aims")
        saved = os.path.join(work, "band", n + ".npz")                   # the band terms of this H model, once
        if os.path.exists(saved):
            z = np.load(saved)
            fb, mu, bf, bs_ = float(z["free"]), float(z["mu"]), z["forces"], z["stress"]
        else:
            a.calc = calc
            fb, mu, bf, bs_ = a.get_potential_energy(force_consistent=True), calc.get_fermi_level(), a.get_forces(), a.get_stress(voigt=False)
            np.savez(saved, free=fb, mu=mu, forces=bf, stress=bs_)
        rf = None if forces is None else forces - bf
        rs = None if stress is None else stress - bs_
        has_f, has_s = has_f and forces is not None, has_s and stress is not None
        part = "test" if int(n.split("_")[1]) >= 100 else "train"
        b = Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=True)
        frames["res"][part].append((b, F - fb, rf, rs))
        # the band energy measured from the Fermi level, sum f (e - mu): unchanged by a rigid shift of
        # the spectrum (the zero of the potential).  Energies only: its forces are not implemented.
        frames["resmu"][part].append((b, F - (fb - nel * len(a) * mu), None, None))
        frames["dft"][part].append((b, F, forces, stress))
    print(f"stage 2: band energy{', forces' if has_f else ''}{', stress' if has_s else ''} of {len(names)} structures in {time.perf_counter() - t:.0f} s "
          f"({len(frames['res']['train'])} train, {len(frames['res']['test'])} test)", flush=True)
    json.dump(dict(h_fit=fit_names, kpts=[nk] * 3, smearing="gaussian", width=width, nelectrons_per_atom=nel, core_bands_per_atom=core, bond_cutoff=rmax,
                   site="order 1, 12 radials, l <= 4, rcut 7", offsite="site (2, 2) + env (5, 6, 4, 4, 4)"),
              open(os.path.join(work, "residual.json"), "w"), indent=1)
    title = {"res": "residual", "resmu": "residual, band energy from the Fermi level (energies only)", "dft": "DFT labels directly"}
    for kind in ("res", "resmu", "dft"):
        for part in ("train", "test"):
            write_xyz(os.path.join(work, f"{kind}_{part}.extxyz"), frames[kind][part])
        E = np.array([fr[1] / len(fr[0]) for fr in frames[kind]["train"]])
        ff = has_f and kind != "resmu"
        print(f"{title[kind]}: energy per atom over the training set: mean {E.mean():.4f} eV, spread {1e3 * E.std():.2f} meV"
              + ("" if not ff else f"; force RMS {np.sqrt(np.mean(np.concatenate([fr[2].ravel() for fr in frames[kind]['train']]) ** 2)):.4f} eV/A"), flush=True)
        rc, lines = aj_fit(os.path.join(work, f"{kind}_train.extxyz"), os.path.join(work, f"{kind}_test.extxyz"), os.path.join(work, f"fit_{kind}"),
                           order, degree, rcut, ff, has_s and kind != "resmu")
        print(f"stage 3, {title[kind]}: aj fit order {order}, degree {degree}, rcut {rcut} (exit {rc})")
        for ln in lines:
            print("   ", ln[:200])


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {k: (float(v) if k in ("rcut", "width") else int(v)) for k, v in (x.split("=") for x in a if "=" in x)}
    main(*[x for x in a if "=" not in x], **kw)
