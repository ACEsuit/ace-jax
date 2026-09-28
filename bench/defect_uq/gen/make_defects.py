"""Cantor defect benchmark (small cells), labelled with MACE-MH-1 (torch, head
matpes_r2scan -- the head that reproduces the stored cantor4k labels).

TRAIN families (added to the training set; also a held-back in-distribution test slice):
  vac      single vacancy in bulk (2x2x3 / 3x3x3 cubic)
  surf100  (100) slab, 6 layers, vacuum
  surf111  (111) slab, 6 layers, vacuum
  sf       intrinsic stacking fault, periodic (111) cell (one fault per cell)
OOD families (never trained on: COMBINATIONS of the train defects, plus strain on a surface):
  divac        two nearest-neighbour vacancies
  vac_surf     vacancy in the surface / subsurface layer of a slab
  vac_sf       vacancy in a stacking-fault plane
  surf_strain  slab under 4-8 % in-plane tension (training strain is +-3 %)

Every cell gets the cantor4k training perturbation (equiatomic random species, a0 = 3.55,
isotropic strain +-3 %, shear 0.02, rattle 0.02-0.10 A, min separation 2.0 A) unless noted.
info: family, split ("train" | "test" | "ood"), defect_pos (flattened xyz of the removed
sites / fault or surface plane marker), label keys mace_energy / mace_force / mace_virial.

    python make_defects.py <out.xyz> [--n-train 150] [--n-ood 100] [--seed 0]
"""
import argparse
import os
import time
import warnings

import numpy as np
from ase.build import bulk, fcc100, fcc111
from ase.io import write

warnings.filterwarnings("ignore")
M = os.environ.get("MACE_MODEL")          # path to mace-mh-1.model (head matpes_r2scan is used)
if M is None:
    raise SystemExit("set MACE_MODEL to the MACE-MH-1 model file")
ELEMENTS = [24, 25, 26, 27, 28]
A0, STRAIN, SHEAR, RATTLE, MIN_SEP = 3.6502, 0.03, 0.02, (0.02, 0.10), 2.0   # a0: MACE-MH-1 Cantor (was 3.55)
VACUUM = 6.0


def species(rng, at):
    z = np.resize(ELEMENTS, len(at)).copy(); rng.shuffle(z); at.set_atomic_numbers(z)
    return at


def perturb(rng, at, strain=STRAIN, inplane_only=False):
    """cantor4k-style homogeneous strain + rattle.  Slabs (inplane_only) strain only
    the periodic in-plane axes so the vacuum is untouched."""
    F = np.eye(3) * (1 + rng.uniform(-strain, strain)) + SHEAR * rng.uniform(-1, 1, (3, 3))
    F = (F + F.T) / 2
    if inplane_only:
        F[2, :] = F[:, 2] = 0.0; F[2, 2] = 1.0
    at.set_cell(at.get_cell() @ F, scale_atoms=True)
    at.positions += rng.uniform(*RATTLE) * rng.standard_normal(at.positions.shape)
    return at


def bulk_cell(rng):
    return bulk("Ni", "fcc", a=A0, cubic=True).repeat(tuple(rng.choice([(2, 2, 3), (3, 3, 3)])))


def slab(rng, face):
    # the orthogonal (111) cell needs an even second dimension
    size = tuple(rng.choice([(2, 2, 6), (3, 3, 6)] if face == 100 else [(2, 2, 6), (3, 4, 6)]))
    at = (fcc100 if face == 100 else fcc111)("Ni", size=size, a=A0, vacuum=VACUUM,
                                              **({} if face == 100 else {"orthogonal": True}))
    at.pbc = True
    return at


def sf_cell(rng):
    """Periodic (111) cell with ONE intrinsic stacking fault: shift the upper half by the
    Shockley partial a/sqrt(6) along [11-2] (the orthogonal cell's y) and tilt c by the
    same vector so the periodic boundary stays perfect."""
    nx, ny = tuple(rng.choice([(2, 2), (3, 2)]))
    at = fcc111("Ni", size=(nx, ny, 9), a=A0, orthogonal=True, periodic=True)
    bp = np.array([0.0, -A0 / np.sqrt(6.0), 0.0])  # this sense gives the ISF; +bp stacks A on A
    zmid = at.cell[2, 2] / 2
    up = at.positions[:, 2] > zmid
    at.positions[up] += bp
    cell = at.get_cell().array.copy(); cell[2] += bp; at.set_cell(cell, scale_atoms=False)
    at.wrap()
    return at, zmid


def remove(at, idx):
    pos = at.positions[list(idx)].copy()
    del at[sorted(idx, reverse=True)]
    return pos


def nn_pair(rng, at):
    i = int(rng.integers(len(at)))
    d = at.get_distances(i, range(len(at)), mic=True); d[i] = np.inf
    return i, int(np.argmin(d))


def make(family, rng):
    at = marker = None
    if family == "vac":
        at = species(rng, bulk_cell(rng)); marker = remove(at, [int(rng.integers(len(at)))])
        perturb(rng, at)
    elif family in ("surf100", "surf111"):
        at = species(rng, slab(rng, 100 if family == "surf100" else 111)); perturb(rng, at, inplane_only=True)
        marker = np.array([[0.0, 0.0, at.positions[:, 2].min()], [0.0, 0.0, at.positions[:, 2].max()]])
    elif family == "sf":
        at, zmid = sf_cell(rng); species(rng, at); perturb(rng, at)
        marker = np.array([[0.0, 0.0, zmid]])
    elif family == "divac":
        at = species(rng, bulk("Ni", "fcc", a=A0, cubic=True).repeat((3, 3, 3)))
        marker = remove(at, list(nn_pair(rng, at))); perturb(rng, at)
    elif family == "vac_surf":
        at = species(rng, slab(rng, int(rng.choice([100, 111]))))
        z = at.positions[:, 2]; top = z > z.max() - 0.5 * A0 / np.sqrt(3) - 0.3   # top two layers
        marker = remove(at, [int(rng.choice(np.flatnonzero(top)))]); perturb(rng, at, inplane_only=True)
    elif family == "vac_sf":
        at, zmid = sf_cell(rng); species(rng, at)
        near = np.flatnonzero(np.abs(at.positions[:, 2] - zmid) < A0 / np.sqrt(3) * 0.6)
        marker = remove(at, [int(rng.choice(near))]); perturb(rng, at)
    elif family == "surf_strain":
        at = species(rng, slab(rng, int(rng.choice([100, 111]))))
        e = rng.uniform(0.04, 0.08)
        F = np.diag([1 + e, 1 + e, 1.0])
        at.set_cell(at.get_cell() @ F, scale_atoms=True)
        at.positions += rng.uniform(*RATTLE) * rng.standard_normal(at.positions.shape)
        marker = np.array([[e, e, 0.0]])
    else:
        raise ValueError(family)
    at.info["family"] = family
    at.info["defect_pos"] = np.asarray(marker, float).reshape(-1)
    return at


def ok(at):
    d = at.get_all_distances(mic=True); np.fill_diagonal(d, np.inf)
    return d.min() >= MIN_SEP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out"); p.add_argument("--n-train", type=int, default=150)
    p.add_argument("--n-ood", type=int, default=100); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--a0", type=float, default=A0)
    a = p.parse_args()
    globals()["A0"] = a.a0
    rng = np.random.default_rng(a.seed)
    plan = [(f, a.n_train) for f in ("vac", "surf100", "surf111", "sf")] + \
           [(f, a.n_ood) for f in ("divac", "vac_surf", "vac_sf", "surf_strain")]
    cells = []
    for fam, n in plan:
        got = []
        while len(got) < n:
            at = make(fam, rng)
            if ok(at):
                got.append(at)
        ntest = int(round(a.test_frac * n)) if fam in ("vac", "surf100", "surf111", "sf") else 0
        for k, at in enumerate(got):
            at.info["split"] = "ood" if ntest == 0 and fam not in ("vac", "surf100", "surf111", "sf") else \
                ("test" if k < ntest else "train")
        cells += got
        print(f"{fam:12s} {n} cells, {len(got[0])}-{max(len(x) for x in got)} atoms", flush=True)
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=M, device="cpu", default_dtype="float64", head="matpes_r2scan")
    t0 = time.time()
    for k, at in enumerate(cells):
        at.calc = calc
        e = float(at.get_potential_energy()); f = at.get_forces(); s = at.get_stress(voigt=False)
        at.calc = None
        at.info["mace_energy"] = e
        at.info["mace_virial"] = (-at.get_volume() * s).reshape(-1)
        at.arrays["mace_force"] = np.asarray(f)
        if k % 50 == 0:
            print(f"  labelled {k + 1}/{len(cells)}  {time.time() - t0:.0f}s", flush=True)
    write(a.out, cells, format="extxyz")
    fam = [c.info["family"] for c in cells]
    print("wrote", a.out, {f: fam.count(f) for f in dict.fromkeys(fam)}, flush=True)


if __name__ == "__main__":
    main()
