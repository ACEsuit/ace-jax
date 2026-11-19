"""The npz operator cache round-trips `OperatorData` exactly."""
import json
import pathlib

import numpy as np
import pytest
from conftest import require_optional

from ace_jax.ham import cache, deeph
from ace_jax.operators.orbitals import OrbitalBasis

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "ham" / "sic_deeph"


def toy():
    rng = np.random.default_rng(0)
    basis = OrbitalBasis.of({1: (0,), 8: (0, 1)})
    numbers = np.array([8, 1, 1])
    n = [basis.n_orb(z) for z in numbers]
    keys = [(0, 0, 0, 0, 0), (0, 0, 0, 0, 1), (1, 0, -1, 2, 0), (0, 0, 0, 1, 2)]
    blocks = {name: {k: rng.normal(size=(n[k[3]], n[k[4]])) for k in keys} for name in ("H_up", "H_dn", "S")}
    return deeph.OperatorData(rng.normal(size=(3, 3)), rng.normal(size=(3, 3)), numbers, basis, blocks)


def same(a, b):
    assert np.array_equal(a.cell, b.cell) and np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.numbers, b.numbers) and a.basis == b.basis and list(a.blocks) == list(b.blocks)
    for name in a.blocks:
        assert set(a.blocks[name]) == set(b.blocks[name])
        assert all(np.array_equal(v, b.blocks[name][k]) for k, v in a.blocks[name].items())


def test_roundtrip_with_channels_and_meta(tmp_path):
    d = toy()
    cache.save(tmp_path / "a.npz", d, meta={"quoll": "f26de28", "format": "fhi-aims"})
    got, meta = cache.load(tmp_path / "a.npz")
    same(d, got)
    assert meta["source"] == {"quoll": "f26de28", "format": "fhi-aims"} and meta["channels"] == ["H_up", "H_dn", "S"]


def test_roundtrip_of_real_data(tmp_path):
    require_optional("h5py")
    d = deeph.read(FIX)
    cache.save(tmp_path / "sic.npz", d)
    same(d, cache.load(tmp_path / "sic.npz")[0])


def test_other_files_are_refused(tmp_path):
    np.savez(tmp_path / "x.npz", meta_json=np.frombuffer(json.dumps({"schema": 99}).encode(), np.uint8))
    with pytest.raises(ValueError, match="schema"):
        cache.load(tmp_path / "x.npz")


def test_block_size_must_match_the_basis(tmp_path):
    d = toy()
    cache.save(tmp_path / "a.npz", d)
    z = dict(np.load(tmp_path / "a.npz"))
    m = json.loads(bytes(z["meta_json"]).decode())
    m["basis"] = [[1, [0]], [8, [0]]]
    z["meta_json"] = np.frombuffer(json.dumps(m).encode(), np.uint8)
    np.savez(tmp_path / "b.npz", **z)
    with pytest.raises(ValueError, match="does not match"):
        cache.load(tmp_path / "b.npz")


def test_channel_names(tmp_path):
    d = toy()
    with pytest.raises(ValueError, match="__"):
        cache.save(tmp_path / "a.npz", d._replace(blocks={"H__up": d.blocks["S"]}))
