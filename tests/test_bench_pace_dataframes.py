"""PACE input-frame tests with no jax / ace_jax import, so they also run in the pacemaker
venv (python 3.9); tests/test_bench_accuracy_cost.py runs this file there as a subprocess."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bench"))
pytest.importorskip("pandas")
ase = pytest.importorskip("ase")


def _atoms(n, e, seed):
    rng = np.random.default_rng(seed)
    a = ase.Atoms(f"Si{n}", positions=rng.uniform(0, 5, (n, 3)), cell=np.eye(3) * 5.4, pbc=True)
    a.info.update(energy=e, config_type="default"); a.arrays["forces"] = rng.normal(size=(n, 3))
    return a


def test_dataframes_carry_weights_and_e0_by_index():
    from accuracy_cost.common import E0_SI
    from accuracy_cost.fit_pace import dataframes
    ats = [_atoms(2, -300.0, 0), _atoms(3, -450.0, 1)]
    wts = [(1e6, np.full(2, 100.0)), (4e6, np.full(3, 25.0))]
    data, w = dataframes(ats, wts)
    assert list(data.index) == list(w.index) == [0, 1]
    assert np.isclose(w["w_energy"].iloc[1], 4e6) and np.allclose(w["w_forces"].iloc[0], 100.0)
    assert np.isclose(data["energy_corrected"].iloc[0], -300.0 - 2 * E0_SI)
    assert np.isclose(data["energy_corrected"].iloc[1], -450.0 - 3 * E0_SI)
    assert np.allclose(data["forces"].iloc[1], ats[1].arrays["forces"])


def test_reduced_cell_keeps_distances_and_fixes_handedness():
    from ase.neighborlist import neighbor_list

    from accuracy_cost.fit_pace import reduced
    a = _atoms(4, -1.0, 2)
    a.set_cell([[5.4, 0, 0], [0.2, 5.5, 0], [0, 0, -15.0]], scale_atoms=False)   # left-handed, as the decohesion frames
    a.positions[:, 2] = np.abs(a.positions[:, 2])
    b = reduced(a)
    assert np.linalg.det(b.cell.array) > 0 and np.allclose(b.positions, a.positions)
    d = lambda x: np.sort(neighbor_list("d", x, 5.0))
    assert np.allclose(d(a), d(b))
    s = _atoms(4, -1.0, 3)
    s.set_cell([[5.4, 0, 0], [5.4 * 3 + 0.3, 5.4, 0], [0, 0, 5.4]], scale_atoms=False)   # a strongly skewed cell
    assert np.allclose(np.sort(neighbor_list("d", s, 5.0)), np.sort(neighbor_list("d", reduced(s), 5.0)))
