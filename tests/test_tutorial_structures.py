"""The school tutorials' structure builders: deterministic, so a notebook rebuilds
exactly the structures whose labels ship with it (the cache is keyed on content)."""
from ace_jax.tutorials import structures as T
from ace_jax.tutorials.labels import structure_key


def _keys(xs):
    return [structure_key(a) for a in xs]


def test_e1_cells_and_grid():
    cells = T.e1_cells(0.08, 0.02)
    assert len(cells) == 10 and all(len(a) == 8 and a.pbc.all() for a in cells)
    assert all(a.info["config_type"] == "bulk" for a in cells)
    assert 0.08 in T.E1_STRAINS and 0.02 in T.E1_RATTLES          # the school's default settings
    assert len(T.E1_STRAINS) * len(T.E1_RATTLES) == 30
    assert _keys(cells) == _keys(T.e1_cells(0.08, 0.02))
    assert _keys(cells) != _keys(T.e1_cells(0.08, 0.04))


def test_vacancy_pair():
    sc, vac = T.e1_vacancy_pair()
    assert (len(sc), len(vac)) == (64, 63) and vac.info["config_type"] == "vacancy"
    assert _keys([sc, vac]) == _keys(T.e1_vacancy_pair())


def test_c_structures():
    c = T.c_structures()
    assert len(c["bulk"]) == 8 and c["slab111"].info["miller"] == "111"
    assert len(c["recipe"]) == 11
    assert [a.info.get("miller") for a in c["recipe"][8:]] == ["100", "110", "111"]
    assert _keys([c["bulk"], c["slab111"], *c["recipe"]]) == _keys([T.c_structures()["bulk"],
                                                                   T.c_structures()["slab111"], *T.c_structures()["recipe"]])


def test_e2_slabs_grid_and_geometry():
    assert (6 in T.E2_LAYERS) and (8.0 in T.E2_VACUA)
    s = T.e2_slabs(6, 8.0)
    assert [a.info["miller"] for a in s] == ["100", "110", "111"]
    assert all(a.info["config_type"] == "slab" for a in s)
    assert _keys(s) == _keys(T.e2_slabs(6, 8.0))
    assert T.min_vacuum_gap(T.e2_slabs(6, 6.0)) >= 12.0 - 1e-6
    assert T.min_vacuum_gap(T.e2_slabs(6, 4.0)) < 12.0
    assert T.min_coordination(s) >= 2                     # shuffle-plane cut: no 1-coordinated atoms


def test_glide_cut_is_caught_by_min_coordination():
    from ase.build import bulk, surface
    glide = surface(bulk("Si", "diamond", a=5.43, cubic=True), (1, 1, 1), 6, vacuum=8.0)
    assert T.min_coordination([glide]) == 1


def test_e2_repair_layers_never_reuse_the_test_thickness():
    assert T.e2_repair_layers(4) == 6 and all(T.e2_repair_layers(L) == 4 for L in T.E2_LAYERS if L != 4)


def test_e2_displaced():
    rep = T.e2_slabs(T.e2_repair_layers(6), 8.0)
    d = T.e2_displaced(rep)
    assert len(d) == 6 and all(a.info["config_type"] == "slab-displaced" for a in d)
    assert [a.info["miller"] for a in d] == ["100", "100", "110", "110", "111", "111"]
    assert _keys(d) == _keys(T.e2_displaced(rep))


def test_e3_seed_is_the_c_recipe_bulk():
    assert _keys(T.e3_seed()) == _keys(T.c_structures()["recipe"][:8])
    b, s = T.e3_targets()
    assert _keys([b, s]) == _keys([T.c_structures()["bulk"], T.c_structures()["slab111"]])
    starts = T.e3_md_starts()
    assert [a.info.get("miller", "bulk") for a in starts] == ["bulk", "110", "111"]


def test_fingerprint_ignores_order_and_wrapping_not_the_cell():
    a = T.d_system()
    b = a.copy(); b.positions[[0, 1]] = b.positions[[1, 0]]; b.numbers[[0, 1]] = b.numbers[[1, 0]]
    c = a.copy(); c.positions[0] += a.cell[0]
    d = a.copy(); d.set_cell(a.cell * 1.01, scale_atoms=True)
    fp = T.structure_fingerprint
    assert fp(a) == fp(b) == fp(c) and fp(a) != fp(d)


def test_d_builders():
    sysm = T.d_system()
    assert sysm.get_chemical_formula() == "As4Ga4" and len(sysm) == 8
    iso = T.d_isolated(("As", "Ga"))
    assert [len(a) for a in iso] == [1, 1] and all(min(a.cell.lengths()) >= 12 for a in iso)
    tg = T.d_targets(sysm)
    assert [a.info["role"] for a in tg] == ["bulk", "slab"]
    tr = T.d_training(sysm, T.D_STRAINS[0], T.D_RATTLES[0], T.D_NTRAIN[0])
    assert len(tr) == T.D_NTRAIN[0] and not ({T.structure_fingerprint(a) for a in tr}
                                              & {T.structure_fingerprint(a) for a in tg})
    rep = T.d_repair(sysm)
    assert len(rep) == 30 and {a.info["config_type"] for a in rep} == {"repair-slab-3", "repair-slab-5"}
