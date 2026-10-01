"""`aj fit` builds the basis from --order/--max-degree: no `aj basis` step."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR
from test_basis_build import _primed_cache

from ace_jax.cli import main

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
FAST = ["--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial",
        "--m-per-species", "0", "--rungs", "map", "--map-steps", "20", "--opt", "adam",
        "--configs-per-batch", "4"]


def test_cli_basis_flags_fit_equals_model_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    main(["basis", "--elements", "Si", "--order", "3", "--max-degree", "10",
          "--coupling-cache-dir", cache, "--out", str(tmp_path / "si.npz")])
    main(["fit", "--model", str(tmp_path / "si.npz"), "--train", str(XYZ), "--r0", "2.35",
          "--out", str(tmp_path / "a"), *FAST])
    main(["fit", "--order", "3", "--max-degree", "10", "--coupling-cache-dir", cache,
          "--train", str(XYZ), "--r0", "2.35", "--out", str(tmp_path / "b"), *FAST])
    za, zb = np.load(tmp_path / "a" / "model.npz"), np.load(tmp_path / "b" / "model.npz")
    assert sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)


def test_cli_r0_defaults_when_building(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    main(["fit", "--order", "3", "--max-degree", "10", "--coupling-cache-dir", cache,
          "--train", str(XYZ), "--out", str(tmp_path / "c"), *FAST])
    out = capsys.readouterr().out
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
