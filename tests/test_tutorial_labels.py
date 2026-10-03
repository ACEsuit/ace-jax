"""Tutorial labels: shipped caches keyed on structure content, with a live labeller only on a miss."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from ace_jax.tutorials import labels as L


def _cells():
    out = []
    for s in (0.98, 1.0, 1.02):
        a = bulk("Cu", "fcc", a=3.6 * s, cubic=True); a.rattle(0.01, seed=1); a.info["config_type"] = "bulk"
        out.append(a)
    return out


def _cache(tmp_path, structures):
    lab = L.label(structures, model="mpa-0", calculator=EMT())        # EMT stands in for MACE in tests
    p = tmp_path / "labels.xyz"
    L.write_cache(p, lab)
    return L.LabelCache.from_file(p), lab


def test_cache_hit_returns_the_shipped_labels_and_keeps_info(tmp_path):
    cache, lab = _cache(tmp_path, _cells())
    got = L.label(_cells(), model="mpa-0", cache=cache, live=False)
    for g, r in zip(got, lab):
        # cache files store labels at extxyz's 8 decimals
        assert abs(g.info["energy"] - r.info["energy"]) < 1e-7 and g.info["config_type"] == "bulk"
        assert g.info["label_model"] == "mpa-0"
        np.testing.assert_allclose(g.arrays["forces"], r.arrays["forces"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(g.info["virial"], r.info["virial"], rtol=0, atol=1e-7)


def test_key_ignores_float_noise_and_wrapping_but_not_order():
    a = _cells()[0]
    b = a.copy(); b.positions += 1e-8
    c = a.copy(); c.positions[0] += a.cell[0]                          # same structure, atom wrapped
    d = a.copy(); d.positions[[0, 1]] = d.positions[[1, 0]]           # atoms swapped: a different labelling
    assert L.structure_key(a) == L.structure_key(b) == L.structure_key(c) != L.structure_key(d)


def test_miss_without_a_labeller_raises_and_names_the_extra(tmp_path, monkeypatch):
    cache, _ = _cache(tmp_path, _cells()[:1])
    monkeypatch.setattr(L, "_mace_calculator", lambda model: (_ for _ in ()).throw(ImportError("no mace")))
    with pytest.raises(L.LabelsUnavailable, match="2 of 3.*mace-torch"):
        L.label(_cells(), model="mpa-0", cache=cache)


def test_wrong_model_is_a_miss(tmp_path):
    cache, _ = _cache(tmp_path, _cells())
    with pytest.raises(L.LabelsUnavailable):
        L.label(_cells(), model="mp-0b3", cache=cache, live=False)


def test_counter_counts_every_labelled_structure(tmp_path):
    cache, _ = _cache(tmp_path, _cells())
    L.reset_labels_used()
    L.label(_cells(), model="mpa-0", cache=cache, live=False)
    assert L.labels_used() == 3


def test_virial_is_minus_stress_times_volume():
    (a,) = L.label(_cells()[:1], calculator=EMT())
    at = _cells()[0]; at.calc = EMT()
    from ase.stress import voigt_6_to_full_3x3_stress
    np.testing.assert_allclose(a.info["virial"], -voigt_6_to_full_3x3_stress(at.get_stress()) * at.get_volume(),
                               rtol=1e-10, atol=1e-12)


def test_cache_keys_are_the_written_structures_not_their_rounded_rereads(tmp_path):
    """Positions are written at 8 decimals; the key stored at write time is the one looked up,
    so a structure rebuilt bit-identically always hits (a re-hash of the rounded positions
    could flip on a 1e-6 rounding boundary)."""
    cells = []
    for k in range(40):
        a = bulk("Cu", "fcc", a=3.6, cubic=True); a.rattle(0.03, seed=k); cells.append(a)
    cache, _ = _cache(tmp_path, cells)
    rebuilt = []
    for k in range(40):
        a = bulk("Cu", "fcc", a=3.6, cubic=True); a.rattle(0.03, seed=k); rebuilt.append(a)
    assert all(cache.get(a, "mpa-0") is not None for a in rebuilt)


def test_downloaded_caches_with_the_same_file_name_do_not_collide(tmp_path, monkeypatch):
    # e1/labels-mpa-0.xyz and c/labels-mpa-0.xyz share a basename; each URL needs its own copy
    srcs = {}
    for name, s in (("e1", 0.97), ("c", 1.03)):
        a = bulk("Cu", "fcc", a=3.6 * s, cubic=True)
        p = tmp_path / name / "labels-mpa-0.xyz"; p.parent.mkdir()
        L.write_cache(p, L.label([a], model="mpa-0", calculator=EMT()))
        srcs[f"https://example.org/{name}/labels-mpa-0.xyz"] = (p, a)
    monkeypatch.setattr(L.pathlib.Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(L.urllib.request, "urlretrieve", lambda url, dst: dst.write_bytes(srcs[url][0].read_bytes()))
    for url, (_, a) in srcs.items():
        assert L.LabelCache.from_file(url).get(a, "mpa-0") is not None


def test_cache_hits_survive_1e7_noise_on_many_rattled_cells(tmp_path):
    # a hash of rounded coordinates flips whenever noise straddles a rounding boundary
    from ace_jax.tutorials.structures import e1_cells
    cells = e1_cells(0.08, 0.02)
    for a in cells:
        a.numbers[:] = 29                                          # EMT has no Si; the geometry is what is keyed
    cache, _ = _cache(tmp_path, cells)
    rng = np.random.default_rng(0)
    for _ in range(10):
        noisy = [a.copy() for a in cells]
        for a in noisy:
            a.positions += rng.uniform(-1e-7, 1e-7, a.positions.shape)
        assert all(cache.get(a, "mpa-0") is not None for a in noisy)


def test_near_miss_beyond_the_tolerance_is_a_miss(tmp_path):
    cache, _ = _cache(tmp_path, _cells())
    a = _cells()[0]; a.positions[0, 0] += 1e-3
    assert cache.get(a, "mpa-0") is None


def test_non_periodic_axis_is_not_wrapped():
    # a slab atom moved by the vacuum-direction cell vector is a different structure
    from ace_jax.tutorials.structures import slab
    s = slab((1, 1, 1), 4)
    t = s.copy(); t.positions[0] += s.cell[2]
    assert L.structure_key(s) != L.structure_key(t)


SCHOOL = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs/user/tutorials/data/school"


def _hits_all(files, structures):
    caches = [L.LabelCache.from_file(SCHOOL / f) for f in files]
    misses = [i for i, a in enumerate(structures) if not any(c.get(a, "mpa-0") is not None for c in caches)]
    assert not misses, f"{len(misses)} of {len(structures)} structures have no shipped label (first: {misses[0]})"


def test_every_tutorial_6_setting_hits_the_shipped_labels():
    from ace_jax.tutorials import structures as T
    grid = [x for L_ in T.E2_LAYERS for v in T.E2_VACUA if v >= 6.0 for x in T.e2_slabs(L_, v)]
    rep = [x for L_ in T.E2_LAYERS for v in T.E2_VACUA if v >= 6.0
           for x in T.e2_displaced(T.e2_slabs(T.e2_repair_layers(L_), v))]
    _hits_all(["e1/labels-mpa-0.xyz", "e2/labels-mpa-0.xyz", "c/labels-mpa-0.xyz"],
              [*T.e1_cells(0.08, 0.02), T.c_structures()["bulk"], *grid, *rep])
    from ace_jax.fit.xyz import read_extxyz
    relaxed = {str(f.info["from_key"]) for f in read_extxyz(str(SCHOOL / "e2/relaxed-mpa-0.xyz"))}
    assert {L.structure_key(s) for s in grid} <= relaxed


def test_every_tutorial_9_setting_hits_the_shipped_labels():
    from ace_jax.tutorials import structures as T
    s = T.d_system()
    grid = [x for a in T.D_STRAINS for r in T.D_RATTLES for n in T.D_NTRAIN for x in T.d_training(s, a, r, n)]
    _hits_all(["d/labels-mpa-0.xyz"], [*T.d_isolated(("As", "Ga")), *T.d_targets(s), *grid, *T.d_repair(s)])


def test_tutorial_7_pools_are_labelled_and_hold_out_the_targets():
    from ase import Atoms
    from ace_jax.fit.xyz import read_extxyz
    from ace_jax.tutorials import structures as T
    pools = [Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
             for f in read_extxyz(str(SCHOOL / "e3/pools.xyz"))]
    _hits_all(["e3/labels-mpa-0.xyz"], pools)
    _hits_all(["c/labels-mpa-0.xyz", "e1/labels-mpa-0.xyz"],
              [*T.e3_targets(), *T.e3_seed(), *T.e1_cells(0.08, 0.02)])
    targets = {L.structure_key(a) for a in T.e3_targets()}
    assert not targets & {L.structure_key(a) for a in pools}


def test_every_tutorial_4_setting_hits_the_shipped_labels():
    from ace_jax.tutorials import structures as T
    _hits_all(["e1/labels-mpa-0.xyz"],
              [x for s in T.E1_STRAINS for r in T.E1_RATTLES for x in T.e1_cells(s, r)] + list(T.e1_vacancy_pair()))
