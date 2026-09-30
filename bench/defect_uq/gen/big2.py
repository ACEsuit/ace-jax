"""Big-cell OOD set v2 (2026-09-27): MACE-consistent crack loading and larger, dissociated
dislocations.  Needs ase 3.23 + matscipy 1.1.1 + mace-torch (Modal: modal/modal_big2.py).

1. MACE Cantor properties (averaged over random equiatomic realisations):
   a0 (isotropic cell relaxation), C11/C12/C44 (+-0.5 % strains, unrelaxed ions of a
   perfect lattice), gamma_111 = (E_slab - E_bulk) / 2A (unrelaxed, same species).
2. (111)[1-10] crack cylinders R = 60 A at K/K_G in {1.1, 1.2, 1.3}, K_G from (C, gamma).
3. a/2<110> edge and screw dislocations, R = 70 A, built dissociated into Shockley
   partials (partial_distance), fixed boundary shell.
Everything relaxed (FIRE, boundary fixed) and labelled; relaxed + rattled 0.05 / 0.10 A.
"""
import time

import numpy as np

ELEMENTS = [24, 25, 26, 27, 28]
EV_A3_PER_GPA = 1 / 160.2176634
FIX = 8.0


def species(rng, at):
    z = np.resize(ELEMENTS, len(at)).copy(); rng.shuffle(z); at.set_atomic_numbers(z)
    return at


def cantor_properties(calc, rng, n_real=3, log=print):
    from ase.build import bulk, fcc111
    from ase.filters import ExpCellFilter
    from ase.optimize import FIRE
    a0s, Cs, gs = [], [], []
    for k in range(n_real):
        at = species(rng, bulk("Ni", "fcc", a=3.55, cubic=True).repeat((4, 4, 4)))
        at.calc = calc
        FIRE(ExpCellFilter(at, hydrostatic_strain=True), logfile=None).run(fmax=0.005, steps=300)
        a0 = (at.get_volume() / len(at) * 4) ** (1 / 3); a0s.append(a0)
        # cubic constants from +-0.5 % strains of the relaxed-volume perfect lattice
        ref = species(np.random.default_rng(1000 + k), bulk("Ni", "fcc", a=a0, cubic=True).repeat((4, 4, 4)))
        def stress(F, ref=ref):
            b = ref.copy(); b.set_cell(ref.cell @ F, scale_atoms=True); b.calc = calc
            return -b.get_stress(voigt=True) / EV_A3_PER_GPA * -1   # GPa, tension positive
        e = 0.005
        s1p, s1m = stress(np.diag([1 + e, 1, 1])), stress(np.diag([1 - e, 1, 1]))
        C11 = (s1p[0] - s1m[0]) / (2 * e); C12 = (s1p[1] - s1m[1]) / (2 * e)
        Fs = np.eye(3); Fs[1, 2] = Fs[2, 1] = e / 2
        Fm = np.eye(3); Fm[1, 2] = Fm[2, 1] = -e / 2
        C44 = (stress(Fs)[3] - stress(Fm)[3]) / (2 * e)
        Cs.append((C11, C12, C44))
        # gamma_111: same species, slab vs periodic
        sl = species(np.random.default_rng(2000 + k), fcc111("Ni", size=(4, 4, 12), a=a0, orthogonal=True,
                                                             periodic=True))
        per = sl.copy(); per.calc = calc; Eb = per.get_potential_energy()
        vac = sl.copy(); vac.center(vacuum=10.0, axis=2); vac.pbc = True; vac.calc = calc
        A = np.linalg.norm(np.cross(sl.cell[0], sl.cell[1]))
        gs.append((vac.get_potential_energy() - Eb) / (2 * A))
        log(f"  realisation {k}: a0 {a0:.4f}  C11/C12/C44 {C11:.0f}/{C12:.0f}/{C44:.0f} GPa  "
            f"gamma111 {gs[-1]:.4f} eV/A^2 ({gs[-1] * 16021.8:.0f} mJ/m^2)")
    return float(np.mean(a0s)), tuple(np.mean(Cs, 0)), float(np.mean(gs))


def crack(rng, a0, C, gamma, k_rel, R=60.0):
    from ase.lattice.cubic import FaceCenteredCubic
    from matscipy.fracture_mechanics.crack import CubicCrystalCrack
    cc = CubicCrystalCrack([1, 1, 1], [1, -1, 0], *(c * EV_A3_PER_GPA for c in C))
    nx = int(np.ceil(2 * R / (a0 * np.sqrt(6)))) + 1
    ny = int(np.ceil(2 * R / (a0 * np.sqrt(3)))) + 1
    at = FaceCenteredCubic(directions=[[1, 1, -2], [1, 1, 1], [1, -1, 0]], size=(nx, ny, 2),
                           symbol="Ni", latticeconstant=a0, pbc=(False, False, True))
    at.positions[:, :2] -= at.cell.diagonal()[:2] / 2
    at = at[np.linalg.norm(at.positions[:, :2], axis=1) < R]
    ys = np.unique(np.round(at.positions[:, 1], 3))
    tip = np.array([0.0, ys[np.argmin(np.abs(ys))] + 0.5 * a0 / np.sqrt(3)])
    K = k_rel * cc.k1g(gamma)
    x, y = at.positions[:, 0] - tip[0], at.positions[:, 1] - tip[1]
    ux, uy = cc.displacements_from_cylinder_coordinates(np.hypot(x, y), np.arctan2(y, x), K)
    at.positions[:, 0] += ux; at.positions[:, 1] += uy
    r = np.hypot(at.positions[:, 0] - tip[0], at.positions[:, 1] - tip[1])
    at.arrays["fixed"] = r > R - FIX
    at.arrays["r_core"] = r
    at.info.update(family="crack", split="ood", K_over_KG=float(k_rel), K_G=float(cc.k1g(gamma)),
                   tip_x=float(tip[0]), tip_y=float(tip[1]))
    cell = np.diag([2 * R + 30.0, 2 * R + 30.0, at.cell[2, 2]])
    at.set_cell(cell, scale_atoms=False); at.positions[:, :2] += cell.diagonal()[:2] / 2
    at.info["tip_x"] += cell[0, 0] / 2; at.info["tip_y"] += cell[1, 1] / 2
    return species(rng, at)


def dislocation(rng, a0, C, kind, R=None, partial_distance=None):
    # R: crack-sized cells; MACE-MH-1 float64 fails on a 7.2k-atom cell (A100 OOM, B200 illegal
    # memory access -- tensor size, not memory); ~3-4k atoms runs.
    R = R or {"edge": 50.0, "screw": 60.0}[kind]
    # start ~30 A (edge) / ~22 A (screw) apart: low-SFE Cantor dissociates by a few nm
    partial_distance = partial_distance or {"edge": 24, "screw": 10}[kind]
    import matscipy.dislocation as D
    cls = D.FCCEdge110Dislocation if kind == "edge" else D.FCCScrew110Dislocation
    dis = cls(a0, *(c * EV_A3_PER_GPA for c in C), symbol="Ni")
    out = dis.build_cylinder(R, partial_distance=partial_distance, fix_width=FIX, return_fix_mask=True,
                             verbose=False)
    at, fix = out[1], np.asarray(out[-1], bool)
    at.pbc = (False, False, True)
    axis = at.positions[fix, :2].mean(0)
    cell = at.get_cell().array.copy(); cell[0] = [2 * R + 30.0, 0, 0]; cell[1] = [0, 2 * R + 30.0, 0]
    at.set_cell(cell, scale_atoms=False)
    shift = cell.diagonal()[:2] / 2 - axis
    at.positions[:, :2] += shift
    at.arrays["fixed"] = fix
    at.arrays["r_core"] = np.linalg.norm(at.positions[:, :2] - cell.diagonal()[:2] / 2, axis=1)
    at.info.update(family=kind, split="ood", partial_distance=int(partial_distance),
                   glide_distance=float(dis.glide_distance))
    return species(rng, at)


def relax_label(at, calc, rng, fmax=0.05, steps=2000, log=print):
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE
    t = time.time()
    fixed = at.arrays["fixed"].astype(bool)
    at.set_constraint(FixAtoms(mask=fixed)); at.calc = calc
    opt = FIRE(at, logfile=None); conv = opt.run(fmax=fmax, steps=steps)
    info = dict(at.info, relaxed_steps=int(opt.nsteps), relax_converged=bool(conv))
    out = []
    for amp in (0.0, 0.05, 0.10):
        b = at.copy(); b.set_constraint()
        if amp:
            b.positions[~fixed] += amp * rng.standard_normal(((~fixed).sum(), 3))
        b.calc = calc
        e = float(b.get_potential_energy()); f = b.get_forces(); b.calc = None
        b.info.clear(); b.info.update(info, rattle=float(amp), mace_energy=e)
        b.arrays["mace_force"] = np.asarray(f)
        out.append(b)
    log(f"  {info['family']} {info.get('K_over_KG', '')} {len(at)} atoms: FIRE {opt.nsteps} conv={conv} "
        f"{time.time() - t:.0f}s")
    return out
