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


from scaling.sweep import HOSTS, cases, run_sweep


def test_cases_cover_host_matrix():
    cs = cases("moriarty-gpu")
    assert {c.dtype for c in cs} == {"float32", "float64"}
    assert all(not (c.code == "mlpace" and c.dtype == "float32") for c in cs)   # ML-PACE is f64 only
    assert {c.mode for c in cs} == {"standalone", "lammps"}


def test_sweep_resumes_and_stops_after_oom(tmp_path):
    calls = []

    def fake(case):
        calls.append(case)
        return {"status": "oom" if case.n_atoms >= 1024 else "ok"}
    res = tmp_path / "r.jsonl"
    sel = lambda c: c.model == "acejax-pace/SiGe/small" and c.mode == "standalone" and c.dtype == "float64"
    run_sweep("moriarty-gpu", fake, res, select=sel)
    ns = [c.n_atoms for c in calls]
    assert ns == [256, 512, 1024]                      # stops after the first OOM
    run_sweep("moriarty-gpu", fake, res, select=sel)   # resume: nothing new to run
    assert [c.n_atoms for c in calls] == ns
    rows = [json.loads(l) for l in res.read_text().splitlines()]
    assert [r["status"] for r in rows] == ["ok", "ok", "oom"]


def test_capacity_covers_ghosts_and_neighbours():
    from scaling.run_lammps import capacity
    at = supercell("SiGe", 256)
    cap = capacity(at, 5.0)
    assert cap["max_atoms"] > 256                      # owned + ghost shell
    assert cap["k_dense"] >= cap["k_max"] + 8
    assert cap["max_edges"] >= 256 * cap["k_max"]


def test_read_pe_and_dump(tmp_path):
    from scaling.run_lammps import read_dump_forces, read_pe
    log = "Step PotEng Atoms\n       0   -12.5    3\nLoop time of 0.1 on 1 procs for 0 steps with 3 atoms\n"
    assert read_pe(log) == -12.5
    d = tmp_path / "f.dump"
    d.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n3\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
                 "ITEM: ATOMS id fx fy fz\n1 0.1 0.2 0.3\n2 0 0 0\n3 -1 -2 -3\n")
    F = read_dump_forces(d)
    assert F.shape == (3, 3) and F[2, 1] == -2.0


def test_parity_compare_and_blocking():
    from scaling.parity import blocked, compare
    F = np.zeros((4, 3))
    ok = compare(-10.0, F, -10.0 + 4e-7, F + 5e-6, 4, (1e-6, 1e-5))
    bad = compare(-10.0, F, -10.0 + 4e-5, F, 4, (1e-6, 1e-5))
    assert ok["status"] == "parity_ok" and bad["status"] == "parity_fail"
    assert bad["dE_per_atom"] == pytest.approx(1e-5)
    rows = [{"code": "mlpace", "status": "parity_fail"}, {"code": "acejax-pace", "status": "parity_ok"},
            {"code": "mace", "status": "unsupported", "model": "mace/SiGe/mh1"}]
    assert blocked(rows) == {("mlpace", "lammps")}


def test_plots_from_synthetic_results(tmp_path):
    from scaling.plot import make_figures
    rows = []
    for code in ("acejax-pace", "mlpace", "mace"):
        for n in (256, 512, 1024):
            rows.append({"code": code, "mode": "lammps", "model": f"{code}/SiGe/small", "size": "small",
                         "system": "SiGe", "n_atoms": n, "device": "gpu", "dtype": "float64",
                         "host": "moriarty-gpu", "status": "ok", "step_s": n * 1e-6,
                         "atom_steps_per_s": 1e6, "peak_bytes": n * 1e5})
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    figs = make_figures(str(tmp_path / "*.jsonl"), tmp_path / "figs")
    assert figs and all(pathlib.Path(f).exists() for f in figs)


def test_mace_rows_pin_the_head():
    """MH-1 is multi-head: standalone and Symmetrix must evaluate the same head,
    so it is pinned explicitly rather than left to each tool's default."""
    heads = {r["size"]: r["head"] for r in planned_models() if r["code"] == "mace"}
    assert heads["mh1"] == "omat_pbe"
    assert heads["small"] is None


def test_symmetrix_cmd_uses_current_cli():
    from scaling.models import symmetrix_cmd
    cmd = symmetrix_cmd("x/mace_mh1.model", [14, 32], "omat_pbe", "x/out.json")
    assert cmd[1:] == ["--model", "x/mace_mh1.model", "--atomic-numbers", "14", "32",
                       "--output", "x/out.json", "--head", "omat_pbe"]
    assert "--head" not in symmetrix_cmd("m", [14], None, "o")


def test_mace_mp_sizes_are_symmetrix_exportable_revision():
    """The original MP-0 checkpoints have a residual first block Symmetrix rejects."""
    from scaling.models import MACE
    assert [MACE[s] for s in ("small", "medium", "large")] == ["small-0b2", "medium-0b2", "large-0b2"]


def test_finished_tolerates_teardown_abort_only():
    """The Symmetrix tree aborts in a static destructor after `Total wall time`
    (double free between liblammps and libkokkoskernels); the run itself is done."""
    from scaling.run_lammps import finished
    assert finished(0, "Loop time ...")
    assert finished(-6, "Loop time ...\nTotal wall time: 0:00:01\n")
    assert not finished(-6, "Loop time ...\n")                   # died mid-run
    assert not finished(1, "ERROR: Unrecognized pair style")


def test_acejax_kokkos_newton_on():
    """The bundle's forces are the energy gradient w.r.t. every position,
    ghosts included: they need reverse comm, i.e. newton on (as lammps-jax examples/in.mlip_al)."""
    from scaling.run_lammps import KOKKOS
    assert KOKKOS["acejax"] == "newton on neigh half"      # KOKKOS: full list needs newton off


def test_parity_dump_and_thermo_print_full_precision():
    """Default formats print ~6 significant digits: a 1e-9 force gate would
    measure print rounding (it did: every dF came out 5.0e-06 / 5.0e-07)."""
    txt = lammps_input("acejax", "/m/b.json", ["Si", "Ge"], "/d/x.data", "gpu", 0,
                       dump="/w/f.dump", warmup=0)
    assert "dump_modify d sort id format float %.17g" in txt
    assert "thermo_modify format float %.17g" in txt


def test_acejax_lammps_is_gpu_only():
    """lammps-jax's pair style is jax/kk on KOKKOS+CUDA only: no CPU LAMMPS rows."""
    from scaling.sweep import cases, lammps_supported
    assert not lammps_supported("acejax-pace", "cpu") and lammps_supported("acejax-ace", "gpu")
    assert lammps_supported("mlpace", "cpu") and lammps_supported("mace", "cpu")
    cpu = cases("local-cpu")
    assert not [c for c in cpu if c.code.startswith("acejax") and c.mode == "lammps"]
    assert [c for c in cpu if c.code.startswith("acejax") and c.mode == "standalone"]
    assert [c for c in cases("moriarty-gpu") if c.code == "acejax-pace" and c.mode == "lammps"]


def test_child_env_keeps_the_login_environment(monkeypatch):
    """The lmp wrappers `module load`: a child without PATH/HOME/MODULEPATH
    fails with 'libcudart.so.12: cannot open shared object file'."""
    from scaling.sweep import child_env
    monkeypatch.setenv("MODULEPATH", "/m")
    e = child_env({"pythonpath": "/p", "os_env": {"X": "1"}})
    assert e["PYTHONPATH"] == "/p" and e["X"] == "1"
    assert e["MODULEPATH"] == "/m" and "PATH" in e


def test_main_reuses_recorded_parity_and_never_gates_in_process(tmp_path, monkeypatch):
    """The parent must not import JAX/torch (it would hold GPU memory for the
    whole sweep), and a resumed sweep must not re-gate or duplicate rows."""
    import json as _json
    from scaling import sweep
    res = tmp_path / "r.jsonl"
    res.write_text(_json.dumps({"mode": "parity", "code": "acejax-pace", "gate": "acejax",
                                "system": "SiGe", "model": "m", "status": "parity_fail",
                                "_key": ["parity"], "_line": ["parity"]}) + "\n")
    monkeypatch.setattr(sweep, "gate_in_subprocess", lambda *a: (_ for _ in ()).throw(AssertionError))
    got = {}
    monkeypatch.setattr(sweep, "run_sweep", lambda host, runner, path, select: got.update(
        blocked=[c.code for c in sweep.cases(host) if not select(c)]))
    envs = tmp_path / "envs"
    monkeypatch.setattr(sweep, "_env_path", lambda host: tmp_path / "env.json")
    (tmp_path / "env.json").write_text(_json.dumps({"lmp": "lmp"}))
    sweep.main(["moriarty-gpu", "--results", str(res)])
    assert len(res.read_text().splitlines()) == 1                  # nothing re-appended
    assert set(got["blocked"]) == {"acejax-pace"}                  # the recorded fail blocks


def test_child_env_passes_the_pjrt_plugin():
    """run_lammps reads PJRT_PLUGIN; without it `pair_style jax/kk ${pjrt}` fails."""
    from scaling.sweep import child_env
    assert child_env({"pythonpath": "/p", "pjrt": "/x/xla_cuda_plugin.so"})["PJRT_PLUGIN"] \
        == "/x/xla_cuda_plugin.so"
