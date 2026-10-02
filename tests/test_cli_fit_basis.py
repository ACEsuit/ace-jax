"""`aj fit` builds the basis from --order/--max-degree: no `aj basis` step."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import CLI_FAST

from ace_jax.cli import main



def test_cli_basis_flags_fit_equals_model_file(built_cli_run, tmp_path, monkeypatch):
    """`aj basis` + `aj fit --model` gives the same model.npz as the inline build."""
    import yaml
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    r0 = yaml.safe_load((built_cli_run.out / "fit.yaml").read_text())["r0"]
    main(["basis", "--elements", "Si", "--order", "3", "--max-degree", "10",
          "--coupling-cache-dir", built_cli_run.cache, "--out", str(tmp_path / "si.npz")])
    main(["fit", "--model", str(tmp_path / "si.npz"), "--train", str(built_cli_run.xyz), "--r0", repr(r0),
          "--out", str(tmp_path / "a"), *CLI_FAST])
    za, zb = np.load(tmp_path / "a" / "model.npz"), np.load(built_cli_run.out / "model.npz")
    assert sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)


def test_cli_r0_defaults_when_building(built_cli_run):
    out = built_cli_run.stdout
    assert "r0 " in out and "mean bond length" in out and "(elements from the data)" in out


@pytest.mark.parametrize("argv,msg", [
    (["fit", "--train", "x.xyz", "--out", "o", "--r0", "2"], "one of --model"),
    (["fit", "--model", "m.npz", "--order", "3", "--max-degree", "5", "--train", "x.xyz", "--out", "o"],
     "--model and"),
    (["fit", "--order", "3", "--train", "x.xyz", "--out", "o"], "--max-degree"),
    (["fit", "--model", "m.npz", "--train", "x.xyz", "--out", "o"], "--r0"),
    (["fit", "--model", "m.npz", "--r0", "2", "--out", "o"], "--train"),
])
def test_cli_basis_source_errors(argv, msg, capsys):
    with pytest.raises(SystemExit):
        main(argv)
    assert msg in capsys.readouterr().err
