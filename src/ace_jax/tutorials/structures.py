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


# --- E2: surfaces ---------------------------------------------------------------------
# the slider grids whose slabs ship labelled (vacuum per side, A; 4 A gives an 8 A gap,
# below the 12 A the labeller needs -- the failing case the checkpoint reports)
E2_LAYERS = (4, 6, 8, 10, 12)
E2_VACUA = (4.0, 6.0, 8.0, 10.0, 12.0)
E2_MILLER = ((1, 0, 0), (1, 1, 0), (1, 1, 1))


def e2_slabs(layers, vacuum):
    """The three low-index faces of shuffle-plane Si (see slab), `layers` thick."""
    return [slab(m, layers, vacuum) for m in E2_MILLER]


def e2_repair_layers(layers):
    """The repair slabs' thickness: never the test slabs' own (4, or 6 when the test is 4)."""
    return 6 if layers == 4 else 4


def e2_displaced(repair_slabs):
    """Two rattles (0.05, 0.12 A; seed 5000 + 10 i + r) of each repair slab, miller kept."""
    out = []
    for i, s in enumerate(repair_slabs):
        for r, amp in enumerate((0.05, 0.12)):
            c = s.copy()
            c.rattle(stdev=amp, seed=5000 + 10 * i + r)
            c.info["config_type"] = "slab-displaced"
            c.info["miller"] = s.info["miller"]
            out.append(c)
    return out


def min_coordination(structures, cutoff=2.8):
    """The smallest neighbour count within `cutoff` (Si-Si 2.35, second shell 3.84 A) over
    every atom: 1 on a glide-plane (111) cut, at least 2 on the shuffle-plane faces."""
    from ase.neighborlist import neighbor_list
    return int(min(np.bincount(neighbor_list("i", s, cutoff), minlength=len(s)).min() for s in structures))


def min_vacuum_gap(structures):
    """The smallest gap between a slab and its periodic image along z, A."""
    return float(min(s.cell[2][2] - (s.positions[:, 2].max() - s.positions[:, 2].min()) for s in structures))


# --- E3: automating curation ------------------------------------------------------------
def e3_seed():
    """The campaign's 8 labelled bulk cells: C's recipe bulk (its labels already ship)."""
    return c_structures()["recipe"][:8]


def e3_targets():
    """The bulk reference and the 6-layer (111) slab whose surface energy is tracked."""
    c = c_structures()
    return c["bulk"], c["slab111"]


def e3_md_starts():
    """MD starting points, in pool order: bulk, a distractor (110) slab, the target (111)."""
    b, s111 = e3_targets()
    return [b, slab((1, 1, 0), 6), s111]


# --- D: bring your own data -------------------------------------------------------------
D_STRAINS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
D_RATTLES = (0.0, 0.03, 0.06, 0.09)
D_NTRAIN = (10, 20, 40, 64)


def d_system():
    """The demo system: zincblende GaAs, a = 5.65 A, the cubic 8-atom cell."""
    from ase.build import bulk
    return bulk("GaAs", "zincblende", a=5.65, cubic=True)


def d_isolated(species):
    """One isolated atom per species in a 15 A periodic box (its energy is E0)."""
    from ase import Atoms
    out = []
    for sym in species:
        a = Atoms(sym, positions=[[0.0, 0.0, 0.0]], cell=[15.0] * 3, pbc=True)
        a.info["config_type"] = "isolated_atom"
        out.append(a)
    return out


def d_targets(system):
    """The target property's structures: the bulk cell and a 4-layer (100) slab (8 A vacuum)."""
    from ase.build import surface
    b = system.copy(); b.info.update(config_type="bulk", role="bulk")
    s = surface(system, (1, 0, 0), 4, vacuum=8.0); s.info.update(config_type="slab", role="slab")
    return [b, s]


def d_training(system, strain, rattle, n_train):
    """n_train copies of `system` with the cell scaled over linspace(1 - s, 1 + s) and rattled
    (seed 2026 + i), tagged byod-bulk."""
    out = []
    for i, sc in enumerate(np.linspace(1.0 - strain, 1.0 + strain, n_train)):
        a = system.copy()
        a.set_cell(system.cell * sc, scale_atoms=True)
        a.rattle(stdev=rattle, seed=2026 + i)
        a.info["config_type"] = "byod-bulk"
        out.append(a)
    return out


def d_repair(system):
    """One repair round: (100) slabs of 3 and 5 layers, 15 rattles each in linspace(0, 0.12)
    (seed 4096 + 100 i + j)."""
    from ase.build import surface
    out = []
    for i, th in enumerate((3, 5)):
        s = surface(system, (1, 0, 0), th, vacuum=8.0)
        for j, amp in enumerate(np.linspace(0.0, 0.12, 15)):
            d = s.copy()
            d.rattle(stdev=float(amp), seed=4096 + 100 * i + j)
            d.info["config_type"] = f"repair-slab-{th}"
            out.append(d)
    return out


def structure_fingerprint(atoms):
    """(formula, volume, sorted minimum-image distances), to 1e-4: the same for a reordered
    or wrapped copy, different for another cell. D's hold-out guard."""
    d = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
    iu = np.triu_indices(len(atoms), 1)
    return (atoms.get_chemical_formula(), round(float(atoms.get_volume()), 4) if np.any(atoms.pbc) else 0.0,
            tuple(np.round(np.sort(d[iu]), 4)))
