import collections
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bench"))
from conftest import require_optional
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
    for k in ("call_s", "force_s", "compile_s"):
        assert out[k] > 0
    assert out["nlist_s"] >= 0            # 0 when the MD-like calls reuse the skin list


def test_standalone_is_md_like(tmp_path):
    """Each timed call displaces the atoms (MD-like), so ace-jax reuses its
    skin list: the calls after the first build no neighbour list."""
    from scaling.run_standalone import run_case
    row = {"code": "acejax-pace", "system": "SiGe", "size": "small",
           "path": str(pathlib.Path(__file__).parent.parent / "fixtures" / "pace" / "gesi_sbessel.yace"),
           "elements": ["Si", "Ge"], "name": "acejax-pace/SiGe/small"}
    out = run_case(row, 256, "float64", "cpu", reps=4)
    assert out["md_like"] is True and out["rebuilds"] <= 2               # skin reused


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
    assert cap["k_dense"] == cap["k_max"] + 8
    assert cap["max_edges"] == cap["max_owned"] * cap["k_dense"]
    assert cap["max_owned"] == int(np.ceil(1.1 * 256))


def test_capacity_slots_have_skin_headroom():
    """Slots cover the rcut + skin coordination: the random-weight benchmark
    structures compress during the run, and rcut-only slots overflowed (the
    bundle returns NaN; lammps-jax then reports the full list, 256 x 78, as
    "edge capacity exceeded").  See docs/perf-lammps-large-n.md."""
    from ace_jax.eval import sparse_graph
    from scaling.run_lammps import capacity
    at = supercell("Cantor", 256)
    c = capacity(at, 5.0, skin=1.0)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, 6.0)
    assert c["k_dense"] >= int(np.bincount(g.senders, minlength=len(at)).max())
    assert c["max_edges"] >= 1.5 * len(sparse_graph(at.positions, at.cell.array, at.pbc,
                                                     5.0).senders)


def test_capacity_tight_slots_are_opt_in():
    """--tight-slots sizes model slots (k_dense, max_edges) for rcut pairs only;
    the neighbour-matrix list (max_neighbors) still holds the rcut + skin list,
    which LAMMPS copies whole.  The default stays safe."""
    from ace_jax.export.lammps import neighbour_capacity
    from scaling.run_lammps import capacity
    at = supercell("Cantor", 256)
    safe, tight = capacity(at, 5.0), capacity(at, 5.0, tight=True)
    assert safe == capacity(at, 5.0, tight=False)
    assert tight["k_dense"] == neighbour_capacity(at, 5.0, slots="cutoff")["k_dense"] < safe["k_dense"]
    assert tight["max_neighbors"] == safe["max_neighbors"] >= int(np.ceil(1.5 * safe["k_max"]))
    assert tight["max_edges"] == tight["max_owned"] * tight["k_dense"]
    assert capacity(at, 5.0, list_headroom=0.0)["max_neighbors"] == safe["k_max"] + 8


def test_lammps_retries_sparse_after_a_matrix_oom(tmp_path, monkeypatch):
    """The matrix bundle is auto's dense-family choice: its OOM falls back to
    sparse just as the packed dense bundle's does."""
    from scaling import run_lammps
    layouts = []

    def fake_export(row, at, dtype, work, layout="auto", slots="skin", list_headroom=0.5,
                    info=None):
        layouts.append(layout)
        return "b.json", ("matrix" if layout == "auto" else "sparse"), 1.0

    monkeypatch.setattr(run_lammps, "export_bundle", fake_export)
    monkeypatch.setattr(run_lammps, "_run_lammps", lambda *a, **k: {
        "status": "oom" if a[-1]["layout"] == "matrix" else "ok", **a[-1]})
    row = {"code": "acejax-pace", "system": "SiGe", "elements": ["Si", "Ge"], "name": "x",
           "path": "m.yace", "size": "small"}
    out = run_lammps.run_case(row, 256, "float64", "gpu", "lmp", 1, tmp_path)
    assert layouts == ["auto", "sparse"] and out["status"] == "ok" and out["dense_oom"]


def test_lammps_retries_dense_after_a_matrix_list_overflow(tmp_path, monkeypatch):
    """The matrix list holds rcut + skin pairs and aborts when a row outgrows
    max_neighbors (random-weight SiGe large at 131k: 53 against 51).  The
    packed dense layout holds only pairs within rcut, so the case is retried
    dense, and the line's larger sizes export dense directly."""
    from scaling import run_lammps
    layouts = []

    def fake_export(row, at, dtype, work, layout="auto", slots="skin", list_headroom=0.5,
                    info=None):
        layouts.append(layout)
        return "b.json", ("matrix" if layout == "auto" else layout), 1.0

    def fake_run(*a, **k):
        if a[-1]["layout"] == "matrix":
            return {"status": "error", "error": "ERROR: LAMMPS-JAX neighbor capacity exceeded: "
                    "global max 53 neighbors per atom, capacity 51", **a[-1]}
        return {"status": "ok", **a[-1]}

    monkeypatch.setattr(run_lammps, "export_bundle", fake_export)
    monkeypatch.setattr(run_lammps, "_run_lammps", fake_run)
    row = {"code": "acejax-ace", "system": "SiGe", "elements": ["Si", "Ge"], "name": "x",
           "path": "m.npz", "size": "large"}
    out = run_lammps.run_case(row, 256, "float64", "gpu", "lmp", 1, tmp_path)
    assert layouts == ["auto", "dense"] and out["status"] == "ok" and out["matrix_overflow"]
    layouts.clear()
    nxt = run_lammps.run_case(row, 512, "float64", "gpu", "lmp", 1, tmp_path, prev=out)
    assert layouts == ["dense"] and nxt["status"] == "ok" and nxt["matrix_overflow"]


def test_export_passes_the_matrix_list_size(tmp_path, monkeypatch):
    """The benchmark's export passes max_neighbors from neighbour_capacity, so
    layout="auto" can pick the matrix (it never does without it)."""
    from scaling import run_lammps
    seen = {}

    def fake_export(model, meta, path, **kw):
        seen.update(kw)
        return {"ace_jax": {"layout": "matrix"}}

    import ace_jax.export.lammps as lx
    monkeypatch.setattr(lx, "export_lammps", fake_export)
    import ace_jax.eval as ev
    monkeypatch.setattr(ev, "load", lambda p: (None, {"rcut": 5.0}, None))
    at = supercell("SiGe", 256)
    row = {"path": "m.npz", "elements": ["Si", "Ge"]}
    run_lammps._export_inprocess(row, at, "float64", tmp_path, "auto", "skin", 0.25)
    want = run_lammps.capacity(at, 5.0, list_headroom=0.25)
    assert seen["max_neighbors"] == want["max_neighbors"] and seen["k_dense"] == want["k_dense"]


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
            # b"x" * n writes every page: a zero-filled bytearray can stay
            # non-resident (lazy zero pages), so RSS -- and the cap -- never saw it
            "import time\nx=[]\nfor _ in range(60):\n    x.append(b'x' * (20*2**20)); time.sleep(0.05)\n"
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
    require_optional("lammps_jax")
    from scaling.run_lammps import export_bundle
    from scaling.structures import supercell
    y = str(pathlib.Path(__file__).parent.parent / "fixtures" / "pace" / "gesi_sbessel.yace")
    row = {"name": "x", "system": "SiGe", "path": y, "elements": ["Si", "Ge"]}
    bundle, layout, t = export_bundle(row, supercell("SiGe", 256), "float64", tmp_path)
    assert pathlib.Path(bundle).exists() and layout in ("matrix", "dense", "sparse") and t > 0


def test_float32_figure_omits_symmetrix_lammps(tmp_path, monkeypatch):
    """Symmetrix evaluates in double whatever the input: MACE-LAMMPS float32
    rows repeat float64, so the float32 figure must not draw them."""
    import matplotlib
    matplotlib.use("Agg")
    from scaling import plot
    rows = []
    for code, mode in (("mace", "lammps"), ("mace", "standalone"), ("acejax-pace", "standalone")):
        for n in (256, 512):
            rows.append({"code": code, "mode": mode, "model": f"{code}/SiGe/medium", "size": "medium",
                         "system": "SiGe", "n_atoms": n, "device": "gpu", "dtype": "float32",
                         "host": "moriarty-gpu", "status": "ok", "call_s": 1e-3, "step_s": 1e-3,
                         "atom_steps_per_s": n / 1e-3})
    figs = []
    monkeypatch.setattr(plot.plt, "close", lambda f=None: figs.append(f))
    plot.fig_throughput(rows, tmp_path, dtype="float32")
    dashed = [ln for ax in figs[-1].axes for ln in ax.lines
              if ln.get_linestyle() == "--" and len(ln.get_xdata()) > 1]
    assert dashed == []


def test_doc_has_narrative_captions_and_capacity_table(tmp_path):
    """The page is regenerated on every render: prose lives in
    bench/scaling/benchmarks_intro.md and is included, each figure gets a
    caption, and a table gives the largest system that fits per line."""
    from scaling.plot import largest_fits, write_doc
    rows = []
    for n, st in ((256, "ok"), (512, "ok"), (1024, "oom")):
        rows.append({"code": "acejax-pace", "mode": "standalone", "model": "acejax-pace/SiGe/medium",
                     "size": "medium", "system": "SiGe", "n_atoms": n, "device": "gpu",
                     "dtype": "float64", "host": "moriarty-gpu", "status": st, "call_s": 1e-3})
    assert "| moriarty-gpu | SiGe | ace-jax (PACE model) | standalone | 512 | 1024 |" in largest_fits(rows)
    res = tmp_path / "r.jsonl"
    res.write_text("\n".join(json.dumps(r) for r in rows))
    fig = tmp_path / "scaling_memory_float64_medium.png"
    fig.write_bytes(b"")
    doc = write_doc(str(res), [str(fig)], doc=str(tmp_path / "b.md"))
    text = pathlib.Path(doc).read_text()
    intro = (pathlib.Path(__file__).parent.parent / "bench" / "scaling" / "benchmarks_intro.md").read_text()
    assert intro.strip().splitlines()[0] in text
    assert "*Peak device memory" in text                      # a caption under the figure
    assert "## Largest system that fits" in text


def test_load_fills_system_and_size_from_the_model_name(tmp_path):
    """Rows written for killed cases (row_from_process) carry the model name but
    not system/size; the capacity table must still see their oom."""
    from scaling.plot import largest_fits, load
    ok = {"code": "mace", "mode": "standalone", "model": "mace/Cantor/medium", "size": "medium",
          "system": "Cantor", "n_atoms": 4096, "dtype": "float64", "host": "h", "status": "ok",
          "call_s": 1.0}
    killed = {"code": "mace", "mode": "standalone", "model": "mace/Cantor/medium", "n_atoms": 8192,
              "dtype": "float64", "device": "cpu", "status": "oom", "host": "h"}
    (tmp_path / "r.jsonl").write_text(json.dumps(ok) + "\n" + json.dumps(killed) + "\n")
    rows = load(str(tmp_path / "*.jsonl"))
    assert "| h | Cantor | MACE | standalone | 4096 | 8192 |" in largest_fits(rows)


def test_parity_gate_covers_both_bundle_layouts():
    """run_lammps falls back to a sparse bundle on dense OOM, so the sparse
    layout must be gated on a periodic cell too (ghost atoms), not only dense."""
    from scaling.models import planned_models
    from scaling.parity import gate_checks
    small = {(m["code"], m["system"]): m for m in planned_models() if m["size"] == "small"}
    got = {(g, m["code"], lay) for g, m, lay in gate_checks(small, "SiGe")}
    for code in ("acejax-pace", "acejax-ace"):
        for lay in ("matrix", "dense", "sparse"):
            assert ("acejax", code, lay) in got
    assert ("mlpace", "mlpace", None) in got and ("mace", "mace", None) in got


def test_model_size_figure_plots_only_the_target_size(tmp_path, monkeypatch):
    """A line that ran out of memory below N must not be drawn from a smaller N
    under a title that says N: GPU throughput is lower at small N."""
    import matplotlib
    matplotlib.use("Agg")
    from scaling import plot
    rows = []
    for size, ns in (("small", (4096, 8192)), ("medium", (4096,)), ("large", (2048, 4096, 8192))):
        for n in ns:
            rows.append({"code": "mace", "mode": "standalone", "model": f"mace/SiGe/{size}",
                         "size": size, "system": "SiGe", "n_atoms": n, "device": "gpu",
                         "dtype": "float64", "host": "moriarty-gpu", "status": "ok",
                         "call_s": 1.0})
    figs = []
    monkeypatch.setattr(plot.plt, "close", lambda f=None: figs.append(f))
    plot.fig_model_size(rows, tmp_path)
    xs = sorted(x for ax in figs[-1].axes for ln in ax.lines for x in ln.get_xdata())
    assert xs == [0, 2]                            # medium (only 4096) is absent


def _ba_row(code, mode, n, host="modal-a100", dtype="float64", size="medium", system="Cantor",
            status="ok", t=1e-3):
    return {"code": code, "mode": mode, "model": f"{code}/{system}/{size}", "size": size,
            "system": system, "n_atoms": n, "device": "cpu" if "cpu" in host else "gpu",
            "dtype": dtype, "host": host, "status": status, "call_s": t, "step_s": t,
            "layout": "dense"}


def _ba_rows(host="modal-a100", t=1e-3):
    rows = [_ba_row(c, m, n, host=host, t=t) for c in ("acejax-pace", "acejax-ace")
            for m in ("standalone", "lammps") for n in (4096, 8192)]
    rows += [_ba_row("acejax-pace", "standalone", 8192, host=host, dtype="float32", t=t),   # dtype
             _ba_row("acejax-pace", "standalone", 8192, host=host, size="small", t=t),      # size
             _ba_row("acejax-pace", "standalone", 16384, host=host, status="oom", t=t)]     # not ok
    return rows


def test_before_after_series_selects_acejax_and_mlpace_reference():
    """The before/after figure: float64 medium ace-jax lines, before from
    before-perf/ and after from the live rows, with ML-PACE in LAMMPS drawn
    in every panel of its system as the reference."""
    from scaling.plot import before_after_series
    after = _ba_rows() + [_ba_row("mlpace", "lammps", 8192), _ba_row("mace", "standalone", 8192),
                          _ba_row("acejax-pace", "standalone", 8192, host="moriarty-gpu")]
    before = _ba_rows(t=4e-3)
    s = before_after_series(after, before, "modal-a100")
    assert {k[2] for k in s} == {"acejax-pace", "acejax-ace", "mlpace"}            # no MACE
    assert s[("Cantor", "standalone", "acejax-pace", "after")] == [(4096, 4096e3, "dense", None),
                                                                  (8192, 8192e3, "dense", None)]
    assert s[("Cantor", "standalone", "acejax-pace", "before")][-1] == (8192, 8192 / 4e-3, "dense",
                                                                       None)
    assert s[("Cantor", "lammps", "mlpace", "reference")] == [(8192, 8192e3, "dense", None)]
    assert s[("Cantor", "standalone", "mlpace", "reference")] == [(8192, 8192e3, "dense", None)]
    assert len(s) == 2 * 2 * 2 + 2              # modes x codes x phases, + reference per mode


def test_before_after_without_before_rows_or_host_rows(tmp_path, monkeypatch):
    """Rows not yet re-run (moriarty) or no before-perf/ directory: the figure
    draws what exists, skips a host with no re-run ace-jax rows, and the page
    lists that host as pending instead of failing."""
    import matplotlib
    matplotlib.use("Agg")
    from scaling import plot
    after = _ba_rows() + [_ba_row("mlpace", "lammps", 8192, host="moriarty-gpu")]
    before = _ba_rows(t=4e-3) + _ba_rows(host="moriarty-gpu", t=4e-3)
    # no before rows at all: after + reference only
    s = plot.before_after_series(after, [], "modal-a100")
    assert s and {k[3] for k in s} == {"after"}
    # a host whose ace-jax rows are pending: nothing drawn, listed as pending
    assert plot.before_after_series(after, before, "moriarty-gpu") == {}
    assert plot.fig_before_after(after, before, tmp_path, "moriarty-gpu") is None
    assert plot.before_after_hosts(after, before) == (["modal-a100"], ["moriarty-gpu"])
    # CPU: ace-jax runs standalone only, so the figure has one mode column
    cpu = [r for r in _ba_rows(host="moriarty-cpu") if r["mode"] == "standalone"]
    assert {k[1] for k in plot.before_after_series(cpu, [], "moriarty-cpu")} == {"standalone"}
    # end to end: make_figures + write_doc with results/ and results/before-perf/
    res = tmp_path / "results"
    (res / plot.BEFORE_DIR).mkdir(parents=True)
    (res / "h.jsonl").write_text("\n".join(json.dumps(r) for r in after))
    figs = []
    monkeypatch.setattr(plot.plt, "close", lambda f=None: figs.append(f))
    out = plot.make_figures(str(res / "*.jsonl"), tmp_path / "figs")        # before-perf/ empty
    assert any(pathlib.Path(f).name == "scaling_before_after_float64_medium_modal-a100.png"
               for f in out)
    (res / plot.BEFORE_DIR / "h.jsonl").write_text("\n".join(json.dumps(r) for r in before))
    out = plot.make_figures(str(res / "*.jsonl"), tmp_path / "figs")
    ba = [f for f in out if pathlib.Path(f).name.startswith("scaling_before_after")]
    assert len(ba) == 1 and ba[0].endswith("modal-a100.png")
    dashed = [ln for ax in figs[-1].axes for ln in ax.lines
              if ln.get_linestyle() == "--" and len(ln.get_xdata()) > 1]
    assert dashed                                                            # the before lines
    text = pathlib.Path(plot.write_doc(str(res / "*.jsonl"), out, doc=str(tmp_path / "b.md"))).read_text()
    assert "pending (ace-jax rows being re-run): moriarty-gpu" in text
    assert "{{" not in text                                                  # placeholders filled
    assert "| moriarty-gpu | 8192 | 8.19M | pending |" in text               # findings
    gate = {"code": "mlpace", "mode": "parity", "gate": "mlpace", "host": "modal-a100",
            "status": "parity_ok", "dE_per_atom": 1e-14, "max_dF": 1e-10}
    par = plot.parity_table(after + [gate])
    assert "| modal-a100 | mlpace | ML-PACE | 1/1 | 1.0e-14 | 1.0e-10 |" in par
    assert par.endswith("| moriarty-gpu | — | — | pending | | |")        # gates not re-run yet


def test_perf_results_tables_mark_pending_hosts():
    """The results doc's summary covers every host with re-run rows and marks
    the others pending; the criteria compare the Cantor medium A100 rows."""
    from scaling.perf_results import criteria_table, summaries
    after = _ba_rows(t=8192 / 1.2e6)
    before = _ba_rows(t=8192 / 4e5) + _ba_rows(host="moriarty-gpu")
    text = summaries(after, before)
    assert "### modal-a100" in text and "### moriarty-gpu" in text
    assert "Pending: the ace-jax rows are being re-run." in text
    assert "| Cantor | medium | ace-jax (PACE model) | standalone | 400k | 1.20M | 3.0× | — | 8192 | 8192 |" in text
    crit = criteria_table(after, before)
    assert "| end to end (scaling suite, standalone) | ≥ 1.10M atom-steps/s | 400k | 1.20M | **met** |" in crit
    assert "| LAMMPS throughput | ≥ 1.00M atom-steps/s | 400k | 1.20M | **met** |" in crit
    assert "| LAMMPS runs at 32,768 atoms | runs | did not run | did not run | **missed** |" in crit


def test_basis_size_counts_functions_per_central_element():
    """PACE: .yace functions per element; linear ACE: rows of WB, since every
    B function carries its own weight per central element (WB is n_B x NZ).
    Dividing n_B by NZ undercounted the Cantor ACE models 5x."""
    from scaling.models import ace_functions_per_element, pace_functions_per_element
    fix = pathlib.Path(__file__).parent.parent / "fixtures"
    y = fix / "pace" / "gesi_sbessel.yace"
    import re
    block = y.read_text().split("\nfunctions:\n", 1)[1]
    per_el = [part.count("\n    - ") + part.startswith("    - ")
              for part in re.split(r"^  \d+:\n", block, flags=re.M)[1:]]
    assert len(per_el) == 2 and pace_functions_per_element(y) == max(per_el)
    z = fix / "sige_nofit.npz"
    assert ace_functions_per_element(z) == np.load(z)["WB"].shape[0]


def test_end_labels_carry_basis_size(monkeypatch):
    from scaling import plot
    monkeypatch.setattr(plot, "BASIS", {"Cantor/medium": {"acejax-ace": 3824, "acejax-pace": 496}})
    assert plot._label("acejax-ace", "Cantor", "medium") == "ace-jax ACE · 3824 fn"
    assert plot._label("acejax-pace", "Cantor", "medium") == "ace-jax PACE · 496 fn"
    assert plot._label("mace", "Cantor", "medium") == "MACE"      # no size known: plain


def test_rows_record_the_device_they_ran_on(tmp_path, monkeypatch):
    """Modal hands out A100 variants (SXM4 / PCIe) that time differently, so
    every row names the device: the GPU from nvidia-smi, else the CPU model."""
    import types

    from scaling import sweep
    monkeypatch.setattr(sweep, "_DEVICE_NAMES", {})
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout="NVIDIA A100-SXM4-80GB\n"))
    only = lambda c: c.code == "acejax-pace" and c.mode == "standalone" and c.n_atoms == 256
    sweep.run_sweep("moriarty-gpu", lambda c, prev: {"status": "ok"}, tmp_path / "r.jsonl",
                    select=only)
    rows = [json.loads(l) for l in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert rows and all(r["device_name"] == "NVIDIA A100-SXM4-80GB" for r in rows)
    assert sweep.device_name("cpu")                          # the CPU model, never empty


# ---- repeat runs: median point, min-max bar ---------------------------------

def _rep(t, run=None, status="ok", n=8192, host="modal-a100", code="acejax-pace",
         mode="standalone", **kw):
    r = _ba_row(code, mode, n, host=host, status=status, t=t, **kw)
    r["_key"] = [r["model"], mode, n, r["dtype"], r["device"]]
    if run:
        r["run"] = run
    return r


def test_load_picks_up_repeats_beside_the_live_files(tmp_path):
    """repeats/*.jsonl beside the pattern joins the rows it loads, for the live
    rows and (through before_pattern) for before-perf/repeats/ alike."""
    from scaling.plot import before_pattern, load
    res = tmp_path / "results"
    (res / "repeats").mkdir(parents=True)
    (res / "before-perf" / "repeats").mkdir(parents=True)
    (res / "h.jsonl").write_text(json.dumps(_rep(1e-3)) + "\n")
    (res / "repeats" / "h-run1.jsonl").write_text(json.dumps(_rep(2e-3, run="h-run1")) + "\n")
    (res / "before-perf" / "h.jsonl").write_text(json.dumps(_rep(4e-3)) + "\n")
    (res / "before-perf" / "repeats" / "h-run1.jsonl").write_text(
        json.dumps(_rep(5e-3, run="h-run1")) + "\n")
    pattern = str(res / "*.jsonl")
    assert sorted(r.get("run", "live") for r in load(pattern)) == ["h-run1", "live"]
    before = load(before_pattern(pattern))
    assert sorted(r["call_s"] for r in before) == [4e-3, 5e-3]


def test_aggregate_takes_the_median_and_the_min_max_of_repeats():
    from scaling.plot import aggregate, spread, throughput
    rows = [_rep(1e-3), _rep(2e-3, run="a"), _rep(4e-3, run="b"),     # 3 runs of one case
            _rep(1e-3, n=4096)]                                        # a single run
    agg = aggregate(rows)
    assert len(agg) == 2
    big = next(r for r in agg if r["n_atoms"] == 8192)
    assert throughput(big) == pytest.approx(8192 / 2e-3)                 # the median run
    assert spread(big) == pytest.approx((8192 / 4e-3, 8192 / 1e-3))
    one = next(r for r in agg if r["n_atoms"] == 4096)
    assert spread(one) is None and one == rows[-1]                       # untouched
    assert aggregate(agg) == agg                                         # idempotent


def test_a_case_ok_in_one_run_and_oom_in_another_is_ok():
    from scaling.plot import aggregate, largest_fits, spread, throughput
    rows = [_rep(1e-3, n=4096), _rep(1e-3, n=8192), _rep(None, status="oom", n=8192, run="a"),
            _rep(None, status="oom", n=16384), _rep(None, status="oom", n=16384, run="a")]
    agg = aggregate(rows)
    at = {r["n_atoms"]: r for r in agg}
    assert at[8192]["status"] == "ok" and spread(at[8192]) is None      # one ok run: no bar
    assert throughput(at[8192]) == pytest.approx(8192e3)
    assert at[16384]["status"] == "oom" and len(agg) == 3
    assert "| modal-a100 | Cantor | ace-jax (PACE model) | standalone | 8192 | 16384 |" in largest_fits(rows)


def _bars(fig):
    """(x, y_lo, y_hi) of every error bar drawn in the figure."""
    from matplotlib.collections import LineCollection
    out = []
    for ax in fig.axes:
        for c in ax.collections:
            if isinstance(c, LineCollection):
                for seg in c.get_segments():
                    out.append((seg[0][0], min(seg[0][1], seg[1][1]), max(seg[0][1], seg[1][1])))
    return out


@pytest.mark.parametrize("which", ["throughput", "before_after", "model_size"])
def test_repeats_draw_a_min_max_bar_at_the_median(tmp_path, monkeypatch, which):
    import matplotlib
    matplotlib.use("Agg")
    from scaling import plot
    single = [_rep(1e-3, n=4096), _rep(1e-3, n=8192)]
    repeated = single + [_rep(2e-3, run="a", n=8192), _rep(4e-3, run="b", n=8192)]
    figs = []
    monkeypatch.setattr(plot.plt, "close", lambda f=None: figs.append(f))
    draw = {"throughput": lambda rows: plot.fig_throughput(rows, tmp_path),
            "before_after": lambda rows: plot.fig_before_after(rows, [], tmp_path, "modal-a100"),
            "model_size": lambda rows: plot.fig_model_size(rows, tmp_path)}[which]
    draw(single)
    assert _bars(figs[-1]) == []                                         # one run: no bar
    draw(repeated)
    bars = _bars(figs[-1])
    assert len(bars) == 1
    _, lo, hi = bars[0]
    assert (lo, hi) == pytest.approx((8192 / 4e-3, 8192 / 1e-3))
    ys = [y for ax in figs[-1].axes for ln in ax.lines for y in ln.get_ydata()]
    assert pytest.approx(8192 / 2e-3) in ys                              # the line's point: median


def test_tables_use_the_median_and_show_the_spread():
    from scaling.perf_results import summaries
    from scaling.plot import tables
    rows = [_rep(1e-3, n=8192), _rep(2e-3, run="a", n=8192), _rep(4e-3, run="b", n=8192)]
    t = tables(rows)
    # median 4.1M; half the min-max range over the median: (8.19M - 2.05M) / 2 / 4.1M = 75%
    assert "| 4.1e+06 ±75% |" in t
    assert "| 8.19e+06 |" in tables(rows[:1])                            # no ± for a single run
    s = summaries(rows, [_rep(8e-3, n=8192)])
    assert "| 1.02M | 4.10M ±75% | 4.0× |" in s


# --- learned-radial proxy lines (acejax-ace-learned / acejax-ace-analytic) ---

LEARNED_PAIR = ("acejax-ace-learned", "acejax-ace-analytic")


@pytest.fixture(scope="module")
def sige_learned(tmp_path_factory):
    """The learned-radial proxy recipe applied to the committed sige_nofit
    fixture (the benchmark models are git-ignored)."""
    from scaling.models import learned_proxy
    src = pathlib.Path(__file__).parent.parent / "fixtures" / "sige_nofit.npz"
    dst = tmp_path_factory.mktemp("learned") / "sige_learned.npz"
    learned_proxy(src, dst, n_q=12, scale=0.1, seed=0)
    return src, dst


def _learned_row(code, path):
    from scaling.models import LEARNED_CODES
    return {"code": code, "system": "SiGe", "size": "medium", "path": str(path),
            "elements": ["Si", "Ge"], "name": f"{code}/SiGe/medium",
            "spline_tol": LEARNED_CODES[code]}


def test_learned_codes_are_planned_for_medium_only():
    from scaling.models import LEARNED_CODES, learned_path
    rows = planned_models()
    assert LEARNED_CODES == {"acejax-ace-learned": "auto", "acejax-ace-analytic": None}
    for code in LEARNED_PAIR:
        for system in ("SiGe", "Cantor"):
            got = [r for r in rows if r["code"] == code and r["system"] == system]
            assert [r["size"] for r in got] == ["medium"], (code, system)
            assert got[0]["path"] == str(learned_path(system)) and got[0]["path"].endswith(
                f"ace_{system}_medium_learned.npz")
            assert got[0]["spline_tol"] == LEARNED_CODES[code]
    assert all("spline_tol" not in r for r in rows if r["code"] not in LEARNED_PAIR)


@pytest.mark.parametrize("host", ["modal-a100", "moriarty-gpu", "moriarty-cpu"])
def test_learned_lines_follow_the_ace_ladder(host):
    """Same modes, dtypes and N ladder as the stock ACE medium line; --only works."""
    from scaling.sweep import main
    cs = cases(host)
    def line(code):
        return sorted((c.mode, c.dtype, c.n_atoms) for c in cs
                      if c.code == code and c.model.endswith("/medium"))
    for code in LEARNED_PAIR:
        assert line(code) == line("acejax-ace") and line(code)
    main([host, "--only", "acejax-ace-learned", "--dry-run"])


def test_learned_proxy_is_deterministic_and_marked_learned(sige_learned, tmp_path):
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.eval import load
    from ace_jax.fit.radial_model import row_active, to_analytic
    from scaling.models import learned_proxy
    src, dst = sige_learned
    again = tmp_path / "again.npz"
    learned_proxy(src, again, n_q=12, scale=0.1, seed=0)
    a, b = np.load(dst), np.load(again)
    assert set(a.files) == set(b.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k                     # bit-for-bit, same seed
    m, meta, _ = load(str(dst))
    assert m.radial_learned and meta["radial_learned"] is True and m.radial_kind == "analytic"
    base, _ = to_analytic(load(str(src))[0], 12)
    W0, W = np.asarray(base.rnl_Wnlq), np.asarray(m.rnl_Wnlq)
    act = np.asarray(row_active(W0))
    assert 0 < act.sum() < act.size                              # the per-z_j pattern exists
    assert np.array_equal(np.asarray(row_active(W)), act)        # zero rows stay exactly zero
    rel = np.linalg.norm(W[act] - W0[act]) / np.linalg.norm(W0[act])
    assert 0.05 < rel < 0.2                                      # ~10% of each row's rms
    other = tmp_path / "seed1.npz"
    learned_proxy(src, other, n_q=12, scale=0.1, seed=1)
    assert not np.array_equal(np.load(other)["rnl_Wnlq"], a["rnl_Wnlq"])


def test_standalone_passes_spline_tol_per_code(sige_learned):
    """-learned runs as deployed (auto: splined at 1e-10), -analytic exact."""
    from scaling.run_standalone import run_case
    _, dst = sige_learned
    got = {code: run_case(_learned_row(code, dst), 256, "float64", "cpu", reps=2)
           for code in LEARNED_PAIR}
    lr, an = got["acejax-ace-learned"], got["acejax-ace-analytic"]
    assert lr["status"] == "ok" and an["status"] == "ok"
    assert lr["spline_tol"] == "auto" and lr["splined"]["spline_tol"] == 1e-10
    assert lr["splined"]["radials"] == ["rnl"] and lr["splined"]["n_intervals"]["rnl"] > 100
    assert an["spline_tol"] is None and an["splined"] is None
    json.dumps(lr), json.dumps(an)                               # rows stay JSON
    assert abs(lr["energy"] - an["energy"]) <= 1e-9 * abs(an["energy"])


def test_lammps_rows_pass_and_record_spline_tol(tmp_path, monkeypatch):
    """The row's spline_tol reaches the export child, and the bundle's record of
    what was splined lands in the LAMMPS row."""
    import subprocess as sp
    from scaling import run_lammps
    seen = {}

    def fake_run(cmd, **kw):
        seen["row"] = json.loads(cmd[cmd.index("--export") + 1])
        return sp.CompletedProcess(cmd, 0, stdout='{"bundle": "/w/b.json", "layout": "dense", '
                                   '"compile_s": 1.0, "splined": {"spline_tol": 1e-10, '
                                   '"n_intervals": {"rnl": 2048}}}\n', stderr="")

    monkeypatch.setattr(run_lammps.subprocess, "run", fake_run)
    info = {}
    row = _learned_row("acejax-ace-analytic", "m.npz")
    run_lammps.export_bundle(row, supercell("SiGe", 256), "float64", tmp_path, info=info)
    assert "spline_tol" in seen["row"] and seen["row"]["spline_tol"] is None
    assert info["splined"] == {"spline_tol": 1e-10, "n_intervals": {"rnl": 2048}}
    # run_case: the requested spline_tol and the bundle's record go into the row
    extra = {}
    monkeypatch.setattr(run_lammps, "export_bundle", lambda r, at, dt, w, info=None, **kw: (
        info.update(splined={"spline_tol": 1e-10, "n_intervals": {"rnl": 2048}}), ("b", "dense", 1.0))[1])
    monkeypatch.setattr(run_lammps, "_run_lammps", lambda *a: extra.update(a[-1]) or {"status": "ok"})
    run_lammps.run_case(_learned_row("acejax-ace-learned", "m.npz"), 256, "float64", "gpu", "lmp", 1,
                        tmp_path)
    assert extra["spline_tol"] == "auto" and extra["splined"]["spline_tol"] == 1e-10
    assert run_lammps.splined_meta({"spline_tol": None, "spline_intervals": None}) is None


def test_learned_bundle_records_splining(sige_learned, tmp_path):
    require_optional("lammps_jax")
    from scaling.run_lammps import export_bundle
    _, dst = sige_learned
    for code, want in (("acejax-ace-learned", 1e-10), ("acejax-ace-analytic", None)):
        info = {}
        export_bundle(_learned_row(code, dst), supercell("SiGe", 256), "float64", tmp_path / code,
                      info=info)
        assert (info["splined"] or {}).get("spline_tol") == want, code


def test_parity_gates_the_learned_lines():
    from scaling.parity import TOL, blocked, compare_rel, gate_checks
    small = {(m["code"], m["system"]): m for m in planned_models() if m["size"] == "small"}
    medium = {(m["code"], m["system"]): m for m in planned_models() if m["size"] == "medium"}
    got = [(g, m["code"], m["size"], lay) for g, m, lay in gate_checks(small, "Cantor", medium)]
    for code in LEARNED_PAIR:
        for lay in ("matrix", "dense", "sparse"):
            assert ("acejax", code, "medium", lay) in got
    assert ("spline", "acejax-ace-learned", "medium", None) in got
    assert [g for g in got if g[0] == "acejax" and g[1] == "acejax-ace"][0][2] == "small"
    assert gate_checks(small, "Cantor") == [c for c in gate_checks(small, "Cantor", medium)
                                            if c[1]["code"] not in LEARNED_PAIR]
    # splining accuracy, relative: ~1e-9 in E, ~3e-8 of max|F| (docs/learned-radial-splining.md)
    assert TOL["spline"] == (1e-9, 3e-8)
    F = np.array([[1.0, -2.0, 0.5], [0.0, 2.0, -1.0]])
    ok = compare_rel(-100.0, F, -100.0 + 5e-8, F + 4e-8, 2, TOL["spline"])
    assert ok["status"] == "parity_ok" and ok["dE_rel"] == pytest.approx(5e-10)
    assert ok["dF_rel"] == pytest.approx(2e-8) and ok["max_dF"] == pytest.approx(4e-8)
    assert compare_rel(-100.0, F, -100.0 + 2e-7, F, 2, TOL["spline"])["status"] == "parity_fail"
    assert compare_rel(-100.0, F, -100.0, F + 1e-7, 2, TOL["spline"])["status"] == "parity_fail"
    # a failed spline gate blocks the learned line in both modes; acejax only LAMMPS
    rows = [{"code": "acejax-ace-learned", "gate": "spline", "status": "parity_fail"},
            {"code": "acejax-ace-analytic", "gate": "acejax", "status": "parity_fail"}]
    assert blocked(rows) == {("acejax-ace-learned", "lammps"), ("acejax-ace-learned", "standalone"),
                             ("acejax-ace-analytic", "lammps")}


def test_spline_gate_measures_learned_against_analytic(sige_learned, monkeypatch):
    """The spline gate runs both learned lines standalone (no LAMMPS) and
    passes on a real proxy; checks it cannot run are skipped here."""
    from scaling import parity
    _, dst = sige_learned
    med = {(c, "SiGe"): _learned_row(c, dst) for c in LEARNED_PAIR}
    monkeypatch.setattr(parity, "planned_models", lambda: [{**r, "size": "medium"} for r in med.values()])
    monkeypatch.setattr(parity, "gate_checks", lambda small, system, medium: [
        ("spline", medium[("acejax-ace-learned", system)], None)] if system == "SiGe" else [])
    rows = parity.gate("moriarty-cpu", {"lmp": "lmp"})
    assert len(rows) == 1 and rows[0]["status"] == "parity_ok", rows
    assert rows[0]["gate"] == "spline" and rows[0]["dF_rel"] < 3e-8 and rows[0]["dE_rel"] < 1e-9


def test_only_a_new_line_tops_up_the_recorded_gate(tmp_path, monkeypatch):
    """A results file gated before the learned lines existed: `--only` one of
    them runs the gate and appends only the checks whose keys are new."""
    from scaling import sweep
    old = {"mode": "parity", "code": "acejax-ace", "gate": "acejax", "system": "SiGe", "model": "m",
           "status": "parity_ok", "bundle_layout": "dense",
           "_key": ["parity", "acejax", "SiGe", "acejax-ace", "dense"], "_line": ["parity"]}
    res = tmp_path / "r.jsonl"
    res.write_text(json.dumps(old) + "\n")
    fresh = [{k: v for k, v in old.items() if not k.startswith("_")},
             {"mode": "parity", "code": "acejax-ace-learned", "gate": "spline", "system": "SiGe",
              "model": "x", "status": "parity_fail", "device": "gpu"}]
    monkeypatch.setattr(sweep, "gate_in_subprocess", lambda *a: [dict(r) for r in fresh])
    monkeypatch.setattr(sweep, "device_name", lambda d: "gpu0")
    got = {}
    monkeypatch.setattr(sweep, "run_sweep", lambda host, runner, path, select: got.update(
        run=sorted({(c.code, c.mode) for c in sweep.cases(host) if select(c)})))
    monkeypatch.setattr(sweep, "_env_path", lambda host: tmp_path / "env.json")
    (tmp_path / "env.json").write_text(json.dumps({"lmp": "lmp"}))
    sweep.main(["moriarty-gpu", "--only", "acejax-ace-learned", "--results", str(res)])
    lines = [json.loads(l) for l in res.read_text().splitlines()]
    assert [l["code"] for l in lines] == ["acejax-ace", "acejax-ace-learned"]   # old kept, new added
    assert got["run"] == []                                      # the failed spline gate blocks it
    assert sweep.gate_missing(lines, "acejax-ace-learned") is False
    assert sweep.gate_missing(lines, None) is False


def _learned_rows(host="modal-a100"):
    rows = []
    for code, t in (("acejax-ace", 1e-3), ("acejax-ace-learned", 1.05e-3), ("acejax-ace-analytic", 2e-3)):
        for system in ("SiGe", "Cantor"):
            for mode in ("standalone", "lammps"):
                for n in (256, 512, 1024):
                    rows.append(_ba_row(code, mode, n, host=host, system=system, t=t * n / 256))
    return rows


def test_learned_radial_palette_and_labels():
    from scaling import plot
    assert list(plot.CODES)[-2:] == list(LEARNED_PAIR)          # appended, fixed order
    cols = [c for _, c in plot.CODES.values()]
    assert len(set(cols)) == len(cols)                          # never recycled
    assert plot.CODES["acejax-ace-learned"][1] == "#e87ba4" and plot.CODES["acejax-ace-analytic"][1] == "#008300"
    assert plot.SHORT["acejax-ace-learned"] == "ace-jax ACE, learned (splined)"
    assert plot.SHORT["acejax-ace-analytic"] == "ace-jax ACE, learned (analytic)"
    assert len({plot._marker(c) for c in ("acejax-ace", *LEARNED_PAIR)}) == 3   # not colour alone


def test_learned_radial_figure_and_doc(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from scaling import plot
    rows = _learned_rows()
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    figs = plot.make_figures(str(tmp_path / "*.jsonl"), tmp_path / "figs")
    names = [pathlib.Path(f).name for f in figs]
    assert "scaling_learned_radial_float64_medium_modal-a100.png" in names
    assert "scaling_throughput_float64_medium.png" in names
    doc = pathlib.Path(plot.write_doc(str(tmp_path / "*.jsonl"), figs, doc=str(tmp_path / "b.md"))).read_text()
    assert "*The cost of a learned radial" in doc and "Learned-radial figures pending" not in doc
    assert "ace-jax (linear ACE, learned radial, splined)" in doc     # the tables name the line
    captured = []
    monkeypatch.setattr(plot.plt, "close", lambda f=None: captured.append(f))
    plot.fig_learned(rows, tmp_path, "modal-a100")
    f = captured[-1]
    assert len([ax for ax in f.axes if ax.lines]) == 4               # 2 systems x 2 modes
    labels = {t.get_text() for lg in f.legends for t in lg.get_texts()}
    assert {plot.CODES[c][0] for c in ("acejax-ace", *LEARNED_PAIR)} <= labels
    plot.fig_throughput(rows, tmp_path)
    labels = {t.get_text() for lg in captured[-1].legends for t in lg.get_texts()}
    assert {plot.CODES[c][0] for c in LEARNED_PAIR} <= labels


def test_learned_rows_absent_or_partial_are_pending(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from scaling import plot
    base = [r for r in _learned_rows() if r["code"] == "acejax-ace"]
    assert plot.fig_learned(base, tmp_path, "modal-a100") is None
    assert plot.learned_hosts(base) == []
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in base))
    figs = plot.make_figures(str(tmp_path / "*.jsonl"), tmp_path / "figs")
    assert not any("learned" in pathlib.Path(f).name for f in figs)
    doc = pathlib.Path(plot.write_doc(str(tmp_path / "*.jsonl"), figs, doc=str(tmp_path / "b.md"))).read_text()
    assert "Learned-radial figures pending" in doc
    # only the learned line, standalone only, one system: still draws, no crash
    part = [r for r in _learned_rows() if r["code"] == "acejax-ace-learned"
            and r["mode"] == "standalone" and r["system"] == "SiGe"]
    assert plot.fig_learned(part, tmp_path, "modal-a100") is not None
    # before/after figures ignore the learned lines (they have no before rows)
    series = plot.before_after_series(_learned_rows(), [], "modal-a100")
    assert series and not any(k[2] in LEARNED_PAIR for k in series)
    # the parity table names the spline gate
    t = plot.parity_table([{"mode": "parity", "host": "h", "gate": "spline", "code": "acejax-ace-learned",
                            "status": "parity_ok", "dE_per_atom": 1e-12, "max_dF": 1e-9}])
    assert "| h | spline | ace-jax (linear ACE, learned radial, splined) | 1/1 |" in t


def test_basis_sizes_count_the_learned_lines(monkeypatch, tmp_path, sige_learned):
    from scaling import models
    _, dst = sige_learned
    monkeypatch.setattr(models, "planned_models", lambda: [
        {"code": "acejax-ace-learned", "system": "SiGe", "size": "medium", "path": str(dst)}])
    assert models.basis_sizes() == {"SiGe/medium": {"acejax-ace-learned": np.load(dst)["WB"].shape[0]}}


def test_runner_hint_carries_the_matrix_overflow_flag(monkeypatch):
    """The next size of a line must see that its matrix list overflowed, or it
    goes back to the matrix layout (the hint is all a fresh process gets)."""
    from scaling import sweep
    seen = {}

    def fake_capped(cmd, env, timeout, cap_bytes=None):
        seen.update(json.loads(env["BENCH_PREV"]))
        return 0, '{"status": "ok"}', "", 0, False

    monkeypatch.setattr(sweep, "run_capped", fake_capped)
    run = sweep.subprocess_runner("moriarty-gpu", {"lmp": "lmp", "lmp_jax": "lmp-jax"})
    c = next(c for c in sweep.cases("moriarty-gpu") if c.code == "acejax-ace" and c.mode == "lammps")
    run(c, prev={"step_s": 0.01, "n_atoms": 256, "layout": "dense", "matrix_overflow": True})
    assert seen.get("matrix_overflow") is True
