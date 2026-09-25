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


from scaling.models import MACE_SIZES, SIZES, planned_models


def test_planned_models_cover_the_matrix():
    rows = planned_models()
    for code in ("acejax-pace", "mlpace", "acejax-ace", "mace"):
        for system in ("SiGe", "Cantor"):
            got = sorted(r["size"] for r in rows if r["code"] == code and r["system"] == system)
            want = sorted(MACE_SIZES) if code == "mace" else sorted(SIZES)
            assert got == want, (code, system)
    pace = {(r["system"], r["size"]): r["path"] for r in rows if r["code"] == "acejax-pace"}
    ml = {(r["system"], r["size"]): r["path"] for r in rows if r["code"] == "mlpace"}
    assert pace == ml                                     # the same .yace files


def test_standalone_row_for_acejax(tmp_path):
    from scaling.run_standalone import run_case
    row = {"code": "acejax-pace", "system": "SiGe", "size": "small",
           "path": str(pathlib.Path(__file__).parent.parent / "fixtures" / "pace" / "gesi_sbessel.yace"),
           "elements": ["Si", "Ge"], "name": "acejax-pace/SiGe/small"}
    out = run_case(row, 256, "float64", "cpu", reps=2)
    assert out["status"] == "ok" and out["n_atoms"] == 256 and out["layout"] in ("dense", "sparse")
    for k in ("call_s", "force_s", "nlist_s", "compile_s"):
        assert out[k] > 0


from scaling.run_lammps import lammps_input, parse_log

LOG = """Step PotEng
       0   -100.0
Loop time of 1.5 on 4 procs for 50 steps with 512 atoms
Step PotEng
      50   -99.9
Loop time of 6.0 on 4 procs for 200 steps with 512 atoms
"""


def test_parse_log_uses_the_timed_segment():
    p = parse_log(LOG)
    assert p["step_s"] == pytest.approx(0.03) and p["n_atoms_end"] == 512 and p["procs"] == 4
    assert p["nan"] is False


def test_parse_log_flags_nan():
    assert parse_log(LOG.replace("-99.9", "nan"))["nan"] is True


@pytest.mark.parametrize("style,device,expect", [
    ("mlpace", "cpu", "pair_style pace\n"), ("mlpace", "gpu", "pair_style pace product"),
    ("acejax", "gpu", "pair_style jax/kk ${pjrt}"), ("mace", "gpu", "pair_style symmetrix/mace")])
def test_lammps_input_pair_lines(style, device, expect):
    txt = lammps_input(style, "/m/model.yace", ["Ge", "Si"], "/d/x.data", device, 200)
    assert expect in txt
    if style in ("mlpace", "mace"):
        assert "pair_coeff * * " in txt and txt.strip().split("pair_coeff * * ")[1].split("\n")[0].endswith("Ge Si")
    assert "run 50" in txt and "run 200" in txt
