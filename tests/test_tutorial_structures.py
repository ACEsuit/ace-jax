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
