"""`aj export`: a model as an ML-PACE .yace or a lammps-jax bundle, the format from the --out extension."""
import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.io import write  # noqa: E402

from ace_jax.calc.point import ACECalculator  # noqa: E402
from ace_jax.cli import main  # noqa: E402
from conftest import FIXTURE_DIR, pace_fixture, require_optional  # noqa: E402

MODEL = FIXTURE_DIR / "si_ace_model.npz"
PACE_DIR = FIXTURE_DIR / "pace"


def _energy(path):
    at = bulk("Si", "diamond", 5.43, cubic=True).repeat(2)
    at.rattle(0.05, seed=3)
    at.calc = ACECalculator(path, skin=0)
    return at.get_potential_energy()


def test_export_yace_writes_a_yace_model_back(tmp_path, capsys):
    src = PACE_DIR / "si_chebpow_fs.yace"
    pace_fixture(src)
    main(["export", "--model", str(src), "--out", str(tmp_path / "copy.yace")])
    assert "written back" in capsys.readouterr().out
    assert _energy(tmp_path / "copy.yace") == _energy(src)


def test_export_yace_converts_an_ace_model(tmp_path, capsys):
    main(["export", "--model", str(MODEL), "--out", str(tmp_path / "m.yace"), "--delta", "0.01"])
    assert "tabulated radials" in capsys.readouterr().out
    assert abs(_energy(tmp_path / "m.yace") - _energy(MODEL)) / 64 < 1e-6    # 0.01 A tables: O(h^4)
    main(["export", "--model", str(tmp_path / "m.yace"), "--out", str(tmp_path / "copy.yace")])  # a .yace back
    assert _energy(tmp_path / "copy.yace") == _energy(tmp_path / "m.yace")


@pytest.mark.parametrize("args,msg", [
    (["--out", "m.txt"], "--format yace|lammps"),
    (["--out", "m.json"], "needs --structure"),
    (["--out", "m.yace", "--structure", "s.xyz"], "needs none"),
    (["--out", "m.json", "--structure", "s.xyz", "--procs", "2,2"], "three integers"),
    (["--out", "m.json", "--delta", "0.01"], "--delta is for a .yace"),
    (["--out", "m.json", "--structure", "s.xyz", "--platform", "cpu", "--ffi", "--no-ffi"], "--no-ffi exports the XLA"),
    (["--out", "m.json", "--structure", "s.xyz", "--platform", "cpu", "--ffi", "--dtype", "float32"],
     "float64 on the CPU"),
    (["--out", "m.json", "--structure", "s.xyz", "--platform", "cpu", "--layout", "dense", "--ffi"], "matrix layout"),
])
def test_export_usage_errors(tmp_path, capsys, args, msg):
    args = [a if not a.startswith("m.") and a != "s.xyz" else str(tmp_path / a) for a in args]
    with pytest.raises(SystemExit) as e:
        main(["export", "--model", str(MODEL), *args])
    assert e.value.code == 2 and msg in capsys.readouterr().err


def test_export_lammps_bundle(tmp_path):
    require_optional("lammps_jax")
    at = bulk("Si", "diamond", 5.43, cubic=True).repeat(2)
    write(tmp_path / "start.xyz", at, format="extxyz")
    main(["export", "--model", str(MODEL), "--out", str(tmp_path / "m.json"), "--structure",
          str(tmp_path / "start.xyz"), "--layout", "sparse", "--type-elements", "Si"])
    import json
    b = json.loads((tmp_path / "m.json").read_text())
    assert b["ace_jax"]["layout"] == "sparse" and b["ace_jax"]["type_elements"] == [14]


def test_export_lammps_cpu_bundle_per_rank(tmp_path):
    require_optional("lammps_jax")
    import inspect
    import json

    from lammps_jax.export import export_model
    if "platform" not in inspect.signature(export_model).parameters:
        pytest.skip("lammps-jax without export_model(platform=)")
    at = bulk("Si", "diamond", 5.43, cubic=True).repeat(3)
    write(tmp_path / "start.xyz", at, format="extxyz")
    base = ["export", "--model", str(MODEL), "--structure", str(tmp_path / "start.xyz"), "--layout", "sparse"]
    main([*base, "--out", str(tmp_path / "all.json"), "--platform", "cpu"])
    main([*base, "--out", str(tmp_path / "rank.json"), "--platform", "cpu", "--procs", "1,1,2"])
    full, rank = (json.loads((tmp_path / f).read_text()) for f in ("all.json", "rank.json"))
    assert full["export_info"]["platform"] == rank["export_info"]["platform"] == "cpu"
    assert rank["contract"]["max_atoms"] < full["contract"]["max_atoms"]   # one rank's subdomain + ghosts


def test_export_aa_dag_cueq_refuses_a_two_species_model(tmp_path, capsys):
    at = bulk("Si", "diamond", 5.43, cubic=True)
    write(tmp_path / "start.xyz", at, format="extxyz")
    with pytest.raises(SystemExit) as e:
        main(["export", "--model", str(FIXTURE_DIR / "sige_nofit.npz"), "--out", str(tmp_path / "m.json"),
              "--structure", str(tmp_path / "start.xyz"), "--aa-dag", "cueq"])
    assert e.value.code == 2 and "one species" in capsys.readouterr().err
    assert not (tmp_path / "m.json").exists()


@pytest.mark.parametrize("flag,mode", [(None, "auto"), ("chain", None), ("greedy", "greedy"), ("prefix", "prefix")])
def test_export_aa_dag_flag(tmp_path, monkeypatch, flag, mode):
    require_optional("lammps_jax")
    from ace_jax.export import lammps as lx
    seen = {}
    monkeypatch.setattr(lx, "export_lammps", lambda *a, **kw: seen.update(kw) or {"ace_jax": {"layout": "sparse"}})
    at = bulk("Si", "diamond", 5.43, cubic=True)
    write(tmp_path / "start.xyz", at, format="extxyz")
    main(["export", "--model", str(MODEL), "--out", str(tmp_path / "m.json"), "--structure",
          str(tmp_path / "start.xyz"), *(["--aa-dag", flag] if flag else [])])
    assert seen["aa_dag"] == mode


def test_export_aa_dag_is_for_a_bundle(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["export", "--model", str(MODEL), "--out", str(tmp_path / "m.yace"), "--aa-dag", "greedy"])
    assert e.value.code == 2 and "--aa-dag" in capsys.readouterr().err


def test_export_aa_dag_cueq_leaves_the_cueq_import_to_the_export(tmp_path, monkeypatch, capsys):
    """The CLI must not import cueq before export_lammps sets CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR; a cueq (or
    recorder) missing in the export is a usage error."""
    require_optional("lammps_jax")
    from ace_jax.eval import cueq
    from ace_jax.export import lammps as lx
    def no_import():
        raise AssertionError("aj export imported cueq before export_lammps")
    def export(*a, **kw):
        raise cueq.CueqUnavailable("set LAMMPS_JAX_CONTRIB")
    monkeypatch.setattr(cueq, "require", no_import)
    monkeypatch.setattr(cueq, "backend", lambda: "gpu")
    monkeypatch.setattr(lx, "export_lammps", export)
    write(tmp_path / "start.xyz", bulk("Si", "diamond", 5.43, cubic=True), format="extxyz")
    with pytest.raises(SystemExit) as e:
        main(["export", "--model", str(MODEL), "--out", str(tmp_path / "m.json"), "--structure",
              str(tmp_path / "start.xyz"), "--aa-dag", "cueq"])
    assert e.value.code == 2 and "LAMMPS_JAX_CONTRIB" in capsys.readouterr().err


def test_export_aa_dag_cueq_refuses_a_host_without_a_gpu(tmp_path, monkeypatch, capsys):
    from ace_jax.eval import cueq
    monkeypatch.setattr(cueq, "backend", lambda: "cpu")
    write(tmp_path / "start.xyz", bulk("Si", "diamond", 5.43, cubic=True), format="extxyz")
    with pytest.raises(SystemExit) as e:
        main(["export", "--model", str(MODEL), "--out", str(tmp_path / "m.json"), "--structure",
              str(tmp_path / "start.xyz"), "--aa-dag", "cueq"])
    assert e.value.code == 2 and "CUDA GPU" in capsys.readouterr().err
    assert not (tmp_path / "m.json").exists()


def test_export_print_env_resolves_on_this_machine(tmp_path, monkeypatch, capsys):
    """`aj export --print-env BUNDLE` on the LAMMPS machine: the sidecars next to the bundle where it is now,
    this machine's shim ($LAMMPS_JAX_CONTRIB), never the export host's paths.  Runs on a CPU host: without
    cuequivariance the library paths are placeholders, with a note to install the extra."""
    import json
    from ace_jax.eval import cueq
    d = tmp_path / "moved"
    d.mkdir()
    rec = {"version": "0.13.0", "targets": ["uniform_1d_cuda"], "replay": "si.ffi-replay", "cache": "si.cueq-cache",
           "handlers": "uniform_1d_cuda=/export/host/contrib/ffi-replay/ffi_replay_shim.so:ffi_replay_handler"}
    (d / "si.json").write_text(json.dumps({"ace_jax": {"layout": "dense", "cueq": rec}}))
    contrib = tmp_path / "contrib"
    contrib.mkdir()
    (contrib / "ffi_replay_shim.so").write_bytes(b"")
    monkeypatch.setenv("LAMMPS_JAX_CONTRIB", str(contrib))
    main(["export", "--print-env", str(d / "si.json")])
    env = cueq.run_env(rec, d / "si.json")
    out = capsys.readouterr().out
    assert f'export FFI_REPLAY_FILE="{d / "si.ffi-replay"}"' in out
    assert f'export CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR="{d / "si.cueq-cache"}"' in out
    assert f"uniform_1d_cuda={contrib / 'ffi_replay_shim.so'}:ffi_replay_handler" in out
    assert "/export/host" not in out and "build the replay shim" not in out
    assert ("pip install" in out) == env["FFI_REPLAY_LIB"].startswith("<")


@pytest.mark.parametrize("args,msg", [
    (["--print-env", "b.json", "--model", "m.npz"], "without --model"),
    (["--print-env", "missing.json"], "not a readable"),
    (["--print-env", "plain.json"], "neither a cueq nor a C++ kernel bundle"),
    (["--out", "m.json"], "needs --model and --out"),
])
def test_export_print_env_usage_errors(tmp_path, capsys, args, msg):
    (tmp_path / "plain.json").write_text('{"ace_jax": {"layout": "dense"}}')
    args = [str(tmp_path / x) if x.endswith((".json", ".npz")) else x for x in args]
    with pytest.raises(SystemExit) as e:
        main(["export", *args])
    assert e.value.code == 2 and msg in capsys.readouterr().err


def test_export_lammps_ffi_bundle(tmp_path, monkeypatch, capsys):
    """--ffi: the kernel compiled into the cache (ACEJAX_FFI_CACHE), named in the bundle, and the
    LAMMPS_JAX_FFI_HANDLERS line printed."""
    require_optional("lammps_jax")
    import json
    import shutil
    if shutil.which("c++") is None:
        pytest.skip("no C++ compiler for the FFI kernel")
    monkeypatch.setenv("ACEJAX_FFI_CACHE", str(tmp_path / "cache"))
    at = bulk("Si", "diamond", 5.43, cubic=True).repeat(2)
    write(tmp_path / "start.xyz", at, format="extxyz")
    main(["export", "--model", str(MODEL), "--out", str(tmp_path / "m.json"), "--structure", str(tmp_path / "start.xyz"),
          "--platform", "cpu", "--ffi-flags=-O1"])
    f = json.loads((tmp_path / "m.json").read_text())["ace_jax"]["ffi"]
    assert f["library"].startswith(str(tmp_path / "cache"))
    assert f'LAMMPS_JAX_FFI_HANDLERS="{f["handlers"]}"' in capsys.readouterr().out


def test_export_ffi_and_aa_dag_cueq_are_exclusive(tmp_path, capsys):
    write(tmp_path / "start.xyz", bulk("Si", "diamond", 5.43, cubic=True), format="extxyz")
    with pytest.raises(SystemExit) as e:
        main(["export", "--model", str(MODEL), "--out", str(tmp_path / "out" / "m.json"), "--structure",
              str(tmp_path / "start.xyz"), "--platform", "cpu", "--ffi", "--aa-dag", "cueq"])
    assert e.value.code == 2 and "exclusive" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_export_lammps_cpu_default_takes_the_kernel(tmp_path, monkeypatch, capsys):
    """`aj export --platform cpu`: the C++ kernel by default (ffi="auto"), with the line LAMMPS needs;
    --no-ffi: the XLA program; --print-env on the kernel bundle: the same line, and the kernel compiled
    here when the recorded library is not on this machine."""
    require_optional("lammps_jax")
    import json
    import shutil
    if shutil.which("c++") is None:
        pytest.skip("no C++ compiler for the FFI kernel")
    monkeypatch.setenv("ACEJAX_FFI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ACEJAX_FFI_FLAGS", "-O1")
    at = bulk("Si", "diamond", 5.43, cubic=True).repeat(2)
    write(tmp_path / "start.xyz", at, format="extxyz")
    base = ["export", "--model", str(MODEL), "--structure", str(tmp_path / "start.xyz"), "--platform", "cpu"]
    main([*base, "--out", str(tmp_path / "k.json")])
    k = json.loads((tmp_path / "k.json").read_text())["ace_jax"]
    out = capsys.readouterr().out
    assert k["ffi"] and k["ffi_reason"] is None and f'LAMMPS_JAX_FFI_HANDLERS="{k["ffi"]["handlers"]}"' in out
    main([*base, "--out", str(tmp_path / "x.json"), "--no-ffi"])
    x = json.loads((tmp_path / "x.json").read_text())["ace_jax"]
    assert x["ffi"] is None and x["ffi_reason"] is None and "C++ kernel" not in capsys.readouterr().out
    main([*base, "--out", str(tmp_path / "d.json"), "--layout", "dense"])
    assert "C++ kernel not used (layout 'dense'" in capsys.readouterr().out
    main(["export", "--print-env", str(tmp_path / "k.json")])
    assert f'LAMMPS_JAX_FFI_HANDLERS="{k["ffi"]["handlers"]}"' in capsys.readouterr().out
    shutil.rmtree(tmp_path / "cache")                       # the library gone: --print-env compiles it here
    main(["export", "--print-env", str(tmp_path / "k.json")])
    out = capsys.readouterr().out
    assert "compiled the kernel here" in out and "LAMMPS_JAX_FFI_HANDLERS=" in out
    assert (tmp_path / "cache").exists()
