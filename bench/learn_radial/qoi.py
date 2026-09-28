"""Quantities of interest for comparing interatomic potentials against a reference
(here the MACE-MH-1 teacher): equation of state, cubic elastic constants and
relaxed vacancy removal energies, computed with identical ASE protocols for any
calculator.

Calculator specs (each needs its own environment):
    ace:<model.npz>              ace-jax (fixed-shape jitted calculator)
    pace:<potential.yaml|yace>   pyace (pacemaker output)
    mace:<model>:<head>          mace-torch

Systems:
    sige    diamond Si, diamond Ge, ordered SiGe (8-atom conventional cells);
            vacancies in 64-atom Si and Ge supercells
    cantor  3 random equiatomic CrMnFeCoNi fcc draws (108 atoms); vacancies at
            `--vac-sites` sites per species in draw 0

Protocols (all relaxations FIRE, fmax --fmax eV/A):
    relax      cell + positions (FrechetCellFilter) -> equilibrium structure
    EOS        7 volumes (+-4% in V), ions relaxed at each, Birch-Murnaghan -> a0, B
    elastic    +-0.5% strains, ions relaxed (relaxed-ion), central differences of
               the stress -> C11, C12, C44 (cubic averages for alloys), GPa
    vacancy    E(N-1, relaxed positions at fixed bulk cell) - (N-1)/N E(N, bulk), eV

    python bench/learn_radial/qoi.py --calc ace:model.npz --system sige --out qoi_ace.json
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.units import GPa

try:
    from ase.filters import FrechetCellFilter
except ImportError:                                  # older ASE
    from ase.constraints import ExpCellFilter as FrechetCellFilter

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--calc", required=True); p.add_argument("--system", choices=["sige", "cantor"], required=True)
p.add_argument("--fmax", type=float, default=5e-3); p.add_argument("--vac-sites", type=int, default=2)
p.add_argument("--seed", type=int, default=0); p.add_argument("--out", required=True)
a = p.parse_args()


def make_calc(spec):
    kind, rest = spec.split(":", 1)
    if kind == "ace":
        import jax
        jax.config.update("jax_enable_x64", True)
        sys.path.insert(0, str(pathlib.Path(__file__).parent / "md"))
        from padded_calc import PaddedACECalculator
        return PaddedACECalculator(rest)
    if kind == "pace":
        from pyace import PyACECalculator
        return PyACECalculator(rest)
    if kind == "mace":
        import torch
        from mace.calculators import MACECalculator
        model, head = rest.rsplit(":", 1)
        return MACECalculator(model_paths=model, head=head, default_dtype="float64",
                              device="cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown calculator kind {kind!r}")


calc = make_calc(a.calc)


def attach(at):
    at = at.copy()
    at.calc = calc
    return at


def relax_positions(at, steps=1000):
    FIRE(at, logfile=None).run(fmax=a.fmax, steps=steps)
    return at


def relax_full(at, steps=2000):
    at = attach(at)
    FIRE(FrechetCellFilter(at), logfile=None).run(fmax=a.fmax, steps=steps)
    return at


def eos(at0):
    """Relaxed-ion E(V) about the relaxed cell: V0/N, a0 (cubic-equivalent), B (GPa)."""
    vols, ens = [], []
    for s in np.linspace(-0.04, 0.04, 7):
        at = attach(at0)
        at.set_cell(at0.cell * (1 + s) ** (1 / 3), scale_atoms=True)
        relax_positions(at)
        vols.append(at.get_volume()); ens.append(at.get_potential_energy())
    v0, e0, B = EquationOfState(vols, ens, eos="birchmurnaghan").fit()
    return {"V0_per_atom": v0 / len(at0), "E0_per_atom": e0 / len(at0), "B_GPa": B / GPa,
            "a0_cubic": (v0 * cubic_atoms_per_cell(at0) / len(at0)) ** (1 / 3)}


def cubic_atoms_per_cell(at):
    return at.info.get("atoms_per_conventional", 8)


def voigt_stress(at):
    return at.get_stress(voigt=True)                  # ASE: tensile stress positive


def elastic(at0, d=5e-3):
    """Relaxed-ion cubic C11, C12, C44 (GPa) by central differences of the stress."""
    C = np.zeros((6, 6))
    for j in range(6):
        sig = []
        for sgn in (1, -1):
            eps = np.zeros(6)
            eps[j] = sgn * d
            e = np.array([[eps[0], eps[5] / 2, eps[4] / 2], [eps[5] / 2, eps[1], eps[3] / 2],
                          [eps[4] / 2, eps[3] / 2, eps[2]]])      # Voigt (xx yy zz yz xz xy), engineering shear
            at = attach(at0)
            at.set_cell(at0.cell @ (np.eye(3) + e), scale_atoms=True)
            relax_positions(at)
            sig.append(voigt_stress(at))
        C[:, j] = (sig[0] - sig[1]) / (2 * d)
    C = 0.5 * (C + C.T) / GPa
    return {"C11_GPa": float(np.mean(np.diag(C)[:3])),
            "C12_GPa": float(np.mean([C[0, 1], C[0, 2], C[1, 2]])),
            "C44_GPa": float(np.mean(np.diag(C)[3:]))}


def vacancy(bulk_relaxed, idx):
    """Vacancy formation energy E(N-1, relaxed ions at the bulk cell) - (N-1)/N E(N, bulk),
    i.e. with the chemical potential set to the bulk energy per atom (for alloys, the
    mean over species -- the same definition for every model)."""
    n = len(bulk_relaxed)
    e_bulk = attach(bulk_relaxed).get_potential_energy()
    at = attach(bulk_relaxed)
    del at[idx]
    relax_positions(at)
    return float(at.get_potential_energy() - (n - 1) / n * e_bulk)


res = {"calc": a.calc, "system": a.system, "fmax": a.fmax}
t0 = time.time()
if a.system == "sige":
    phases = {"Si": bulk("Si", "diamond", 5.43, cubic=True), "Ge": bulk("Ge", "diamond", 5.66, cubic=True)}
    sige = bulk("Si", "diamond", 5.55, cubic=True)
    sige.set_chemical_symbols(["Si", "Ge"] * 4)          # ordered (zincblende-like) SiGe
    phases["SiGe"] = sige
    for name, at in phases.items():
        at.info["atoms_per_conventional"] = 8
        r = relax_full(at)
        r.info["atoms_per_conventional"] = 8
        res[name] = {**eos(r), **elastic(r)}
        if name in ("Si", "Ge"):
            sc = r.repeat((2, 2, 2))
            sc.calc = None
            res[name]["vacancy_eV"] = vacancy(sc, 0)
        print(name, json.dumps(res[name]), f"{time.time() - t0:.0f}s", flush=True)
else:
    els = ["Cr", "Mn", "Fe", "Co", "Ni"]
    rng = np.random.default_rng(a.seed)
    draws = []
    for k in range(3):
        at = bulk("Ni", "fcc", 3.6, cubic=True).repeat((3, 3, 3))     # 108 atoms
        sym = np.array(els * (len(at) // len(els)) + els[: len(at) % len(els)])
        at.set_chemical_symbols(list(rng.permutation(sym)))
        at.info["atoms_per_conventional"] = 4
        r = relax_full(at)
        r.info["atoms_per_conventional"] = 4
        d = {**eos(r), **elastic(r)}
        if k == 0:
            r0 = r.copy()
            r0.calc = None
            d["vacancy_eV"] = {}
            for el in els:
                sites = [i for i, s in enumerate(r0.get_chemical_symbols()) if s == el][: a.vac_sites]
                d["vacancy_eV"][el] = [vacancy(r0, i) for i in sites]
        draws.append(d)
        print(f"draw {k}", json.dumps(d), f"{time.time() - t0:.0f}s", flush=True)
    res["draws"] = draws
    res["mean"] = {q: float(np.mean([d[q] for d in draws])) for q in ("a0_cubic", "B_GPa", "C11_GPa", "C12_GPa", "C44_GPa")}
res["seconds"] = time.time() - t0
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
