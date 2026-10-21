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
    (["--out", "m.json", "--structure", "s.xyz", "--ffi"], "needs --platform cpu"),
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
