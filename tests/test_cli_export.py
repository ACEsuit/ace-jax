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
    monkeypatch.setattr(lx, "export_lammps", export)
    write(tmp_path / "start.xyz", bulk("Si", "diamond", 5.43, cubic=True), format="extxyz")
    with pytest.raises(SystemExit) as e:
        main(["export", "--model", str(MODEL), "--out", str(tmp_path / "m.json"), "--structure",
              str(tmp_path / "start.xyz"), "--aa-dag", "cueq"])
    assert e.value.code == 2 and "LAMMPS_JAX_CONTRIB" in capsys.readouterr().err
