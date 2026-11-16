"""Check ham/julia/ingest.jl against Quoll's own reference conversion (the
ham-ingest CI job; not a pytest file): the DeepH directory the script wrote from
FHI-aims output and Quoll's reference DeepH directory for the same calculation
must read identically through ace_jax.ham.deeph, and survive the npz cache.
    python tests/ham/check_ingest.py <ingested dir> <reference dir>"""
import sys
import tempfile

import numpy as np

from ace_jax.ham import cache, deeph


def main(got_dir, ref_dir, tol=1e-12):
    got, ref = deeph.read(got_dir), deeph.read(ref_dir)
    assert got.basis == ref.basis and np.array_equal(got.numbers, ref.numbers)
    assert np.allclose(got.cell, ref.cell, atol=1e-12) and np.allclose(got.positions, ref.positions, atol=1e-12)
    for name in ("H", "S"):
        assert set(got.blocks[name]) == set(ref.blocks[name]), name
        worst = max(float(np.abs(got.blocks[name][k] - v).max()) for k, v in ref.blocks[name].items())
        print(f"{name}: {len(ref.blocks[name])} blocks, max difference {worst:.2e}")
        assert worst <= tol, (name, worst)
    with tempfile.TemporaryDirectory() as t:
        cache.save(f"{t}/c.npz", got, meta={"format": "fhi-aims"})
        back, _ = cache.load(f"{t}/c.npz")
        assert all(np.array_equal(back.blocks["H"][k], v) for k, v in got.blocks["H"].items())
    print("ingest OK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
