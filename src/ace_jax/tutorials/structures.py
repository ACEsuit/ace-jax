"""Structures of the MLIP-school-derived tutorials (ACEsuit/MLIP-school-2026, E1 and C).

Every builder is deterministic (seeded rattles, fixed lattice constants), so a
notebook rebuilds exactly the structures whose labels ship with it: the label
cache is keyed on structure content (`tutorials.labels.structure_key`), and the
label generator (docs/user/tutorials/data/school/make_labels.py) calls these same
functions."""
import numpy as np

# E1's sliders snap to this grid, the settings whose labels ship (the school's
# sliders stepped 0.01 / 0.005; its defaults, 0.08 and 0.02, are on the grid)
E1_STRAINS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
E1_RATTLES = (0.0, 0.02, 0.04, 0.06, 0.08)
# MACE-MPA-0's diamond-Si lattice constant (A), from a Birch-Murnaghan fit of
# unrattled cubic cells by make_labels.py; the vacancy pair is built at it, so the
# pair does not depend on a student's own fitted a0 and its labels can ship
E1_A0 = 5.4672


def e1_cells(strain, rattle, n=10):
    """E1 Q1: n cubic 8-atom diamond-Si cells at scales linspace(1 - strain, 1 + strain, n)
    of a = 5.43 A, each rattled with stdev `rattle` (seed 2026 + index), tagged bulk."""
    from ase.build import bulk
    out = []
    for i, s in enumerate(np.linspace(1.0 - strain, 1.0 + strain, n)):
        a = bulk("Si", "diamond", a=5.43 * s, cubic=True)
        a.rattle(stdev=rattle, seed=2026 + i)
        a.info["config_type"] = "bulk"
        out.append(a)
    return out


def e1_vacancy_pair(a0=E1_A0):
    """E1 Q3: a 2x2x2 cubic supercell (64 atoms) and the same cell with atom 0 removed
    (63 atoms), unrelaxed, at lattice constant a0."""
    from ase.build import bulk
    sc = bulk("Si", "diamond", a=a0, cubic=True) * (2, 2, 2)
    sc.info["config_type"] = "supercell"
    vac = sc.copy()
    del vac[0]
    vac.info["config_type"] = "vacancy"
    return sc, vac


def shuffle_crystal(a=5.43):
    """Cubic diamond Si shifted by a/4 along [111], so ase.build.surface cuts (111) on the
    wide "shuffle" plane (one broken bond per surface atom) rather than the glide pair."""
    from ase.build import bulk
    c = bulk("Si", "diamond", a=a, cubic=True)
    c.positions += 0.25 * np.array([a, a, a])
    c.wrap()
    return c


def slab(miller, layers, vacuum=8.0):
    from ase.build import surface
    s = surface(shuffle_crystal(), miller, layers, vacuum=vacuum)
    s.info["config_type"] = "slab"
    s.info["miller"] = "".join(str(i) for i in miller)
    return s


def c_structures():
    """C: the reference bulk and 6-layer (111) slab whose surface energy is compared under
    two labellers, and E2's surface recipe (8 rattled bulk cells, 4-layer (100), (110),
    (111) slabs) that the models are fitted to."""
    from ase.build import bulk
    ref = bulk("Si", "diamond", a=5.43, cubic=True)
    ref.info["config_type"] = "bulk"
    recipe = []
    for i, s in enumerate(np.linspace(0.94, 1.06, 8)):
        a = bulk("Si", "diamond", a=5.43 * s, cubic=True)
        a.rattle(stdev=0.02, seed=3000 + i)
        a.info["config_type"] = "bulk"
        recipe.append(a)
    recipe += [slab(m, 4) for m in ((1, 0, 0), (1, 1, 0), (1, 1, 1))]
    return {"bulk": ref, "slab111": slab((1, 1, 1), 6), "recipe": recipe}


def surface_energy(bulk_energy, n_bulk, slab_energy, slab_atoms):
    """gamma = (E_slab - N * E_bulk/atom) / 2A, eV/A^2 (A: the slab's in-plane area)."""
    cell = np.asarray(slab_atoms.cell.array)
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    return (slab_energy - len(slab_atoms) * bulk_energy / n_bulk) / (2 * area)
