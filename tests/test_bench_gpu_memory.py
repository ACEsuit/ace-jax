import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bench"))
from gpu_memory.attribute import _paint, role_of, stage_of
from gpu_memory.run import cell, repeats


def test_cells_sizes_and_surfaces():
    assert repeats("bulk", 32768) == (16, 16, 16) and len(cell("bulk", 32768)) == 32768
    slab, crack = cell("slab", 4096), cell("crack", 4096)
    assert slab.cell[2, 2] - slab.positions[:, 2].max() > 20       # the vacuum
    assert 0 < 4096 - len(crack) < 4096 // 8                        # the slit removes a few layers
    a, b = cell("crack", 4096), cell("crack", 4096)
    assert np.array_equal(a.positions, b.positions)                 # seeded rattle


def test_paint_counts_aliased_bytes_once_largest_first():
    # (name, size, shape, offset): b aliases the first half of a; c is disjoint
    vals = [("a", 100, "", 0), ("b", 50, "", 0), ("c", 30, "", 200)]
    got = {v[0]: n for v, n in _paint(vals)}
    assert got == {"a": 100, "c": 30}
    vals = [("a", 100, "", 0), ("d", 80, "", 60)]                   # a partial overlap: d's free tail
    assert {v[0]: n for v, n in _paint(vals)} == {"a": 100, "d": 40}


def test_stage_and_role():
    src = "/x/src/ace_jax/eval/"
    chain = [("/venv/jax/_src/numpy.py", "where", 1), (src + "aa_dag.py", "_contract_cols", 194),
             (src + "model.py", "ACEModel._readout_folded", 410)]
    assert stage_of(chain) == ("product basis", "aa_dag.py:_contract_cols:194")
    assert stage_of([(src + "model.py", "ACEModel._blocked_a_onehot", 700)])[0] == "A basis"
    assert stage_of([("/venv/jax/x.py", "f", 1)])[0] == "other"
    assert role_of("jit(f)/transpose(jvp())/while/body/checkpoint/rematted_computation/mul") == "residual"
    assert role_of("jit(f)/transpose(jvp())/mul") == "cotangent" and role_of("jit(f)/jvp()/mul") == "forward"
