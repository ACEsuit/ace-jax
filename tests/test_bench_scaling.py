import collections
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bench"))
from scaling.structures import SYSTEMS, n_ladder, supercell


@pytest.mark.parametrize("system", ["SiGe", "Cantor"])
def test_supercell_is_deterministic_with_exact_composition(system):
    a, b = supercell(system, 512), supercell(system, 512)
    assert len(a) == 512 and np.array_equal(a.numbers, b.numbers)
    counts = collections.Counter(a.get_chemical_symbols())
    assert set(counts) == set(SYSTEMS[system]["elements"])
    assert max(counts.values()) - min(counts.values()) <= 1
    assert a.pbc.all() and np.allclose(a.cell.angles(), 90)


def test_ladder_powers_of_two():
    assert n_ladder("SiGe", 4096) == [256, 512, 1024, 2048, 4096]
