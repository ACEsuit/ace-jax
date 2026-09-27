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
    # a "cpu" row must really run on the CPU (jax[cuda12] defaults to the GPU),
    # and record the threading it used (spec: "the threading actually used")
    assert out["platform"] == "cpu" and out["threads"]["cpus"] >= 1
    assert out["nlist_backend"] and out["force_s"] + out["nlist_s"] <= out["call_s"] * 1.5
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

    def fake(case, prev=None):
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


def test_log_y_axis_uses_compact_labels_and_no_minor_labels():
    """Minor log ticks printed '6x10^3' in mathtext next to the styled majors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    from scaling.plot import _count_fmt, _ylog
    assert [_count_fmt(v, None) for v in (0.5, 10, 1000, 6000, 2e6)] == ["0.5", "10", "1k", "6k", "2M"]
    fig, ax = plt.subplots()
    ax.plot([1, 2], [100, 1e5])
    _ylog(ax)
    assert ax.get_yscale() == "log" and isinstance(ax.yaxis.get_minor_formatter(), NullFormatter)
    ax.set_ylim(800, 6000)                   # under a decade: still several labelled ticks
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()
              and 800 <= t.get_position()[1] <= 6000]
    assert labels == ["1k", "2k", "5k"]
    plt.close(fig)


def test_oom_detected_anywhere_in_output():
    """XLA's allocator summary can push RESOURCE_EXHAUSTED out of the tail."""
    from scaling.run_lammps import failure_status
    long = "RESOURCE_EXHAUSTED: Out of memory\n" + "Current allocation summary\n" * 100
    assert failure_status(long) == "oom"
    assert failure_status("ERROR: Unrecognized pair style") == "error"


def test_error_summary_keeps_head_and_tail():
    """NVRTC errors lead with kernel source; the message is at the end."""
    from scaling.parity import error_summary
    s = error_summary(RuntimeError("#define X\n" * 200 + "nvrtc: error: invalid value"))
    assert s.startswith("RuntimeError(") and "nvrtc: error: invalid value" in s and len(s) < 800


def test_kokkos_allocation_failure_is_oom():
    """Kokkos (Symmetrix, pace/kk) never says 'out of memory'."""
    from scaling.run_lammps import failure_status
    msg = ("Kokkos ERROR: Cuda memory space failed to allocate 3.2 GiB (label=\"x\"). "
           "The Cuda allocation returned the error code \"cudaErrorMemoryAllocation\".")
    assert failure_status(msg) == "oom"


def test_lammps_error_summary_keeps_error_lines():
    """An MPI_ABORT banner filled the whole stored tail; keep the ERROR lines."""
    from scaling.run_lammps import error_text
    txt = "ERROR: Kokkos failed to allocate\n" + "MPI_ABORT banner line\n" * 50
    assert "ERROR: Kokkos failed to allocate" in error_text(txt)


def test_merge_rows_appends_only_unseen_keys():
    """A preempted Modal run returns its whole resumed file: merging must not
    duplicate rows the local results already hold."""
    from scaling.sweep import merge_rows
    a = json.dumps({"_key": ["m", "lammps", 256], "status": "ok"})
    b = json.dumps({"_key": ["m", "lammps", 512], "status": "ok"})
    merged, added = merge_rows(a + "\n", a + "\n" + b + "\n")
    assert merged.splitlines() == [a, b] and added == 1
    assert merge_rows("", "")[1] == 0


def test_seed_for_code_combines_results_and_snapshot(tmp_path):
    from scaling.sweep import seed_for
    p = json.dumps({"_key": ["parity", "acejax", "SiGe", "acejax-pace"], "mode": "parity"})
    r = json.dumps({"_key": ["acejax-pace/SiGe/small", "lammps", 256], "code": "acejax-pace"})
    (tmp_path / "acejax-pace.jsonl").write_text(p + "\n" + r + "\n")
    seed = seed_for(p + "\n", tmp_path, "acejax-pace")
    assert seed.splitlines() == [p, r]
    assert seed_for(p + "\n", tmp_path, "mace").splitlines() == [p]      # no snapshot


def test_run_sweep_retries_an_error_once(tmp_path):
    """A transient failure (GPU state left by the previous case) must not end a
    whole line; a second error does, and oom is never retried."""
    from scaling.sweep import run_sweep
    calls = []

    def runner(c, prev=None):
        calls.append((c.model, c.n_atoms))
        first = calls.count((c.model, c.n_atoms)) == 1
        return {"status": "error" if (first and c.n_atoms == 256) else "ok"}

    only = lambda c: c.model == "mlpace/SiGe/small"
    run_sweep("local-cpu", runner, tmp_path / "r.jsonl", select=only)
    rows = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "ok" and rows[0].get("retried") is True
    assert calls[:2] == [("mlpace/SiGe/small", 256)] * 2
    assert all(r["status"] == "ok" for r in rows) and len(rows) > 1


def test_child_env_threads_per_mode(monkeypatch):
    """moriarty's login env sets OMP_NUM_THREADS=1: standalone (torch, one
    process) must get every core; MPI LAMMPS keeps one thread per rank."""
    from scaling.sweep import child_env
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    env = {"pythonpath": "/p"}
    assert child_env(env, mode="standalone", cpus=32)["OMP_NUM_THREADS"] == "32"
    assert child_env(env, mode="lammps", cpus=32)["OMP_NUM_THREADS"] == "1"
    # torch sizes its intra-op pool from MKL_NUM_THREADS too (login env: 1)
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    assert child_env(env, mode="standalone", cpus=32)["MKL_NUM_THREADS"] == "32"
    assert child_env(env, mode="lammps", cpus=32)["MKL_NUM_THREADS"] == "1"
    assert child_env(env)["OMP_NUM_THREADS"] == "1"                    # unchanged by default


def test_row_from_process_reads_the_json_or_classifies_the_death():
    """A case killed by the kernel OOM killer (SIGKILL) prints no JSON; it is an
    oom, not an error, and the row still carries the case identity."""
    from scaling.sweep import Case, row_from_process
    c = Case("mace", "mace/Cantor/large", "standalone", 16384, "float32", "cpu", 16)
    ok = row_from_process(c, 0, 'noise\n{"status": "ok", "call_s": 1.0}\n', "")
    assert ok == {"status": "ok", "call_s": 1.0}
    killed = row_from_process(c, -9, "", "DeprecationWarning ...")
    assert killed["status"] == "oom" and killed["returncode"] == -9
    assert killed["code"] == "mace" and killed["mode"] == "standalone" and killed["n_atoms"] == 16384
    assert row_from_process(c, 1, "", "Traceback ...")["status"] == "error"


def test_moriarty_cpu_ranks_are_physical_cores():
    """Xeon Silver 4216: 16 cores x 2 hyperthreads.  Open MPI refuses 32 ranks
    ('not enough slots'); one rank per physical core."""
    from scaling.sweep import HOSTS
    assert HOSTS["moriarty-cpu"]["ranks"] == 16


def test_choose_steps_budgets_the_timed_segment():
    """Fixed 200 steps took ~3.5 h per 32k-atom MACE CPU case; scale the step
    count from the previous (smaller) case of the same line to ~60 s of MD."""
    from scaling.run_lammps import choose_steps
    assert choose_steps(None, None, 256) == (200, 50)              # no hint: the default
    assert choose_steps(0.001, 256, 512) == (200, 50)              # fast codes unchanged
    steps, warm = choose_steps(0.8, 512, 1024)                     # ~1.6 s/step expected
    assert steps == 38 and warm == 9
    assert choose_steps(50.0, 16384, 32768) == (10, 3)             # floor


def test_run_sweep_hands_each_case_the_previous_ok_row_of_its_line(tmp_path):
    from scaling.sweep import run_sweep
    seen = []

    def runner(c, prev=None):
        seen.append((c.n_atoms, prev and prev["n_atoms"]))
        return {"status": "ok", "n_atoms": c.n_atoms, "step_s": 1e-3}

    run_sweep("local-cpu", runner, tmp_path / "r.jsonl",
              select=lambda c: c.model == "mlpace/SiGe/small")
    assert seen[0] == (256, None) and seen[1] == (512, 256) and seen[2] == (1024, 512)


def test_mace_lammps_runs_float64_only():
    """Symmetrix evaluates in double whatever the input: f32 rows duplicated f64."""
    from scaling.sweep import cases
    got = {c.dtype for c in cases("moriarty-cpu") if c.code == "mace" and c.mode == "lammps"}
    assert got == {"float64"}
    assert {c.dtype for c in cases("moriarty-cpu") if c.code == "mace" and c.mode == "standalone"} \
        == {"float64", "float32"}


def test_run_capped_kills_a_case_over_its_memory_cap():
    """Past the node's RAM the kernel OOM killer took systemd/dbus before the
    case; the runner kills the case's whole process tree at a cap instead."""
    import sys
    pytest.importorskip("psutil")
    from scaling.sweep import run_capped
    grow = [sys.executable, "-c",
            "import time\nx=[]\nfor _ in range(60):\n    x.append(bytearray(20*2**20)); time.sleep(0.05)\n"
            "print('{\"status\": \"ok\"}')"]
    rc, out, err, peak, capped = run_capped(grow, env=None, timeout=60, cap_bytes=200 * 2**20, poll=0.05)
    assert capped and rc != 0 and "ok" not in out and peak > 200 * 2**20
    rc, out, err, peak, capped = run_capped([sys.executable, "-c", "print('{\"status\": \"ok\"}')"],
                                            env=None, timeout=60, cap_bytes=200 * 2**20)
    assert not capped and rc == 0 and '"ok"' in out


def test_capped_case_is_an_oom_row():
    from scaling.sweep import Case, row_from_process
    c = Case("mace", "mace/Cantor/large", "lammps", 16384, "float64", "cpu", 16)
    r = row_from_process(c, -9, "", "", capped=True, peak_rss=5e10)
    assert r["status"] == "oom" and r["capped_rss"] == 5e10


def test_timeout_is_an_error_not_an_oom():
    import sys
    pytest.importorskip("psutil")
    from scaling.sweep import Case, row_from_process, run_capped
    rc, out, err, peak, capped = run_capped([sys.executable, "-c", "import time; time.sleep(30)"],
                                            env=None, timeout=0.5, cap_bytes=None, poll=0.05)
    c = Case("mlpace", "mlpace/SiGe/small", "lammps", 256, "float64", "cpu", 16)
    r = row_from_process(c, rc, out, err, capped=capped)
    assert rc is None and r["status"] == "error" and "timeout" in r["error"]


def test_acejax_lammps_layout_follows_the_line():
    """A dense bundle that OOMs is retried sparse; once a line has gone sparse,
    its larger sizes export sparse directly."""
    from scaling.run_lammps import bundle_layout
    assert bundle_layout(None) == "auto"
    assert bundle_layout({"layout": "dense"}) == "auto"
    assert bundle_layout({"layout": "sparse"}) == "sparse"


def test_peak_memory_asks_the_framework_the_code_used(monkeypatch):
    """MACE rows reported peak_bytes=0: jax was importable in the process but
    never allocated, so torch's counter was never read."""
    import sys
    import types
    from scaling.run_standalone import _peak
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(max_memory_allocated=lambda: 123))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert _peak("gpu", "mace") == 123


def _two_system_rows():
    rows = []
    for system, base in (("SiGe", 1.0), ("Cantor", 3.0)):
        for code in ("acejax-pace", "mace"):
            for n in (256, 512, 1024):
                for dt, f in (("float64", 1.0), ("float32", 1.5)):
                    rows.append({"code": code, "mode": "standalone", "model": f"{code}/{system}/medium",
                                 "size": "medium", "system": system, "n_atoms": n, "device": "gpu",
                                 "dtype": dt, "host": "moriarty-gpu", "status": "ok",
                                 "call_s": n * 1e-6 * base / f, "peak_bytes": n * 1e6 * base,
                                 "layout": "dense"})
    return rows


@pytest.mark.parametrize("fig", ["fig_memory", "fig_precision"])
def test_per_system_panels_never_zigzag(tmp_path, monkeypatch, fig):
    """Memory and precision joined both systems into one line per host panel."""
    import matplotlib
    matplotlib.use("Agg")
    from scaling import plot
    figs = []
    monkeypatch.setattr(plot.plt, "close", lambda f=None: figs.append(f))
    getattr(plot, fig)(_two_system_rows(), tmp_path)
    f = figs[-1]
    assert len([ax for ax in f.axes if ax.lines]) == 2            # one panel per system
    for ax in f.axes:
        for ln in ax.lines:
            x = list(ln.get_xdata())
            if len(x) > 1:
                assert x == sorted(x), (fig, ax.get_title(), x)
    labels = {t.get_text() for lg in f.legends for t in lg.get_texts()}
    assert "MACE" in labels


def test_export_bundle_runs_in_a_child_process(tmp_path, monkeypatch):
    """Exporting in the runner process initialised JAX on the GPU, whose default
    pool (75% of the card) stayed allocated while LAMMPS ran: lammps-jax got a
    quarter of the GPU.  The export runs in a child that exits first."""
    import subprocess as sp
    from scaling import run_lammps
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return sp.CompletedProcess(cmd, 0, stdout='noise\n{"bundle": "/w/bundle.json", '
                                   '"layout": "dense", "compile_s": 1.5}\n', stderr="")

    monkeypatch.setattr(run_lammps.subprocess, "run", fake_run)
    row = {"name": "acejax-pace/SiGe/small", "system": "SiGe", "path": "m.yace", "elements": ["Si", "Ge"]}
    from scaling.structures import supercell
    got = run_lammps.export_bundle(row, supercell("SiGe", 256), "float64", tmp_path, layout="sparse")
    assert got == ("/w/bundle.json", "dense", 1.5)
    assert seen["cmd"][0] == sys.executable and "--export" in seen["cmd"]
    assert "sparse" in seen["cmd"] and "256" in seen["cmd"]


def test_export_bundle_child_writes_a_bundle(tmp_path):
    pytest.importorskip("lammps_jax")
    from scaling.run_lammps import export_bundle
    from scaling.structures import supercell
    y = str(pathlib.Path(__file__).parent.parent / "fixtures" / "pace" / "gesi_sbessel.yace")
    row = {"name": "x", "system": "SiGe", "path": y, "elements": ["Si", "Ge"]}
    bundle, layout, t = export_bundle(row, supercell("SiGe", 256), "float64", tmp_path)
    assert pathlib.Path(bundle).exists() and layout in ("dense", "sparse") and t > 0
