"""Every fit writes a resolved out/fit.yaml; `aj fit --config` on it reproduces the run."""
import jax
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from conftest import CLI_FAST, FIXTURE_DIR, small_si_xyz

from ace_jax.cli import main

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"


def test_resolved_yaml_is_explicit(built_cli_run):
    d = yaml.safe_load((built_cli_run.out / "fit.yaml").read_text())
    assert d["basis"]["elements"] == ["Si"] and d["basis"]["order"] == 3 and d["basis"]["max_degree"] == 10
    assert isinstance(d["r0"], float) and "model" not in d and "config" not in d
    assert d["train"] == str(built_cli_run.xyz.resolve()) and d["rungs"] == ["map"]
    assert d["provenance"]["command"].startswith("aj fit --order 3 --max-degree 10")
    assert d["provenance"]["ace_jax"]


def test_resolved_yaml_reproduces_from_other_cwd(built_cli_run, tmp_path, monkeypatch):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    a = built_cli_run.out
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    main(["fit", "--config", str(a / "fit.yaml"), "--out", str(tmp_path / "b")])
    za, zb = np.load(a / "model.npz"), np.load(tmp_path / "b" / "model.npz")
    assert sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)
    ya = yaml.safe_load((a / "fit.yaml").read_text())
    yb = yaml.safe_load((tmp_path / "b" / "fit.yaml").read_text())
    for y in (ya, yb):
        y.pop("provenance"); y.pop("out")
    assert ya == yb


def test_model_file_run_records_model(tmp_path):
    xyz = small_si_xyz(tmp_path / "si12.xyz")
    main(["fit", "--model", str(FIXTURE_DIR / "si_ace_model.npz"), "--train", str(xyz), "--r0", "2.35",
          "--out", str(tmp_path / "m"), *CLI_FAST])
    d = yaml.safe_load((tmp_path / "m" / "fit.yaml").read_text())
    assert d["model"] == str((FIXTURE_DIR / "si_ace_model.npz").resolve()) and "basis" not in d
    assert d["provenance"]["coupling"] is None


def test_resolved_keeps_gp_embedding(tmp_path):
    """The fit's own --embedding (the GP species table) is a top-level key, not
    the basis block's embedding: it must survive into the resolved file."""
    from types import SimpleNamespace

    from ace_jax import runfile
    from ace_jax.cli import _parse, _parser
    emb = tmp_path / "mace.json"; emb.write_text("{}")
    argv = ["fit", "--model", "m.npz", "--r0", "2.4", "--embedding", str(emb), "--train", "t.xyz", "--out", "o"]
    a = _parse(argv)
    fit_dests = runfile._dests(_parser()._subparsers._group_actions[0].choices["fit"])
    d = runfile.resolved(a, SimpleNamespace(r0=None, meta={}), fit_dests=fit_dests, argv=argv[1:])
    assert d["embedding"] == str(emb) and "basis" not in d
