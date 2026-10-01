"""cantor4k-style bulk set RE-CENTRED on MACE-MH-1's own Cantor lattice constant.

Identical recipe to ACEpotentials acejax/bench/distil/make_structures.py (equiatomic random
occupancy, 2x2x2 / 2x2x3 cubic cells, isotropic strain +-3 %, shear 0.02, rattle 0.02-0.10 A,
min separation 2.0 A; one() is vendored below) -- except a0 = 3.6502 A, the
MACE-MH-1 (head matpes_r2scan) zero-pressure lattice constant of the random alloy (3-realisation
mean, defects/big2_props.json).  cantor4k used a0 = 3.59 A: 1.7 % (5 % in volume) compressed
relative to the teacher's equilibrium.  Labels as label.py: mace_energy, mace_force,
mace_virial = -V sigma (flat 9).

    python make_bulk.py <out.xyz> [-n 4000] [--a0 3.6502] [--seed 0]
"""
import argparse
import os
import time
import warnings

import numpy as np
from ase.io import write

warnings.filterwarnings("ignore")
M = os.environ.get("MACE_MODEL")          # path to mace-mh-1.model (head matpes_r2scan is used)
if M is None:
    raise SystemExit("set MACE_MODEL to the MACE-MH-1 model file")


class MS:
    """The cantor4k recipe, vendored from ACEpotentials acejax/bench/distil/make_structures.py
    (only a0 differs here)."""
    ELEMENTS = [24, 25, 26, 27, 28]
    MIN_SEP, A_STRAIN, SHEAR, RATTLE = 2.0, 0.03, 0.02, (0.02, 0.10)

    @staticmethod
    def one(rng, reps, a, rattle, strain):
        from ase.build import bulk
        at = bulk("Ni", "fcc", a=a, cubic=True).repeat(reps)
        n = len(at)
        z = np.repeat(MS.ELEMENTS, n // len(MS.ELEMENTS))
        if len(z) < n:
            z = np.concatenate([z, rng.choice(MS.ELEMENTS, n - len(z))])
        rng.shuffle(z)
        at.set_atomic_numbers(z)
        F = np.eye(3) * (1 + rng.uniform(-MS.A_STRAIN, MS.A_STRAIN))
        F = F + strain * rng.uniform(-1, 1, (3, 3))
        F = (F + F.T) / 2
        at.set_cell(at.get_cell() @ F, scale_atoms=True)
        at.positions += rattle * rng.standard_normal(at.positions.shape)
        return at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out"); p.add_argument("-n", type=int, default=4000)
    p.add_argument("--a0", type=float, default=3.6502); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out, tries, rejected = [], 0, 0
    while len(out) < a.n and tries < 50 * a.n:
        tries += 1
        at = MS.one(rng, rng.choice([(2, 2, 2), (2, 2, 3)]), a.a0, float(rng.uniform(*MS.RATTLE)), MS.SHEAR)
        d = at.get_all_distances(mic=True); np.fill_diagonal(d, np.inf)
        if d.min() < MS.MIN_SEP:
            rejected += 1; continue
        out.append(at)
    print(f"generated {len(out)} (rejected {rejected}), sizes {sorted({len(x) for x in out})}, a0 {a.a0}", flush=True)
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=M, device="cpu", default_dtype="float64", head="matpes_r2scan")
    t0 = time.time()
    for k, at in enumerate(out):
        at.calc = calc
        e = float(at.get_potential_energy()); f = at.get_forces(); s = at.get_stress(voigt=False)
        at.calc = None
        at.info.update(mace_energy=e, mace_virial=(-at.get_volume() * s).reshape(-1), a0=a.a0)
        at.arrays["mace_force"] = np.asarray(f)
        if k % 250 == 0:
            print(f"  labelled {k + 1}/{len(out)}  {time.time() - t0:.0f}s", flush=True)
    write(a.out, out, format="extxyz")
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
