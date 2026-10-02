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
