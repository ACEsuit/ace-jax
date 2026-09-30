"""Every fit writes a resolved out/fit.yaml; `aj fit --config` on it reproduces the run."""
import jax
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR
from test_basis_build import _primed_cache

from ace_jax.cli import main

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
FAST = ["--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial",
        "--m-per-species", "0", "--rungs", "map", "--map-steps", "20", "--opt", "adam",
        "--configs-per-batch", "4"]


def _build_fit(tmp_path, cache, out):
    main(["fit", "--order", "3", "--max-degree", "10", "--coupling-cache-dir", cache,
          "--train", str(XYZ), "--out", str(out), *FAST])


def test_resolved_yaml_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    _build_fit(tmp_path, cache, tmp_path / "a")
    d = yaml.safe_load((tmp_path / "a" / "fit.yaml").read_text())
    assert d["basis"]["elements"] == ["Si"] and d["basis"]["order"] == 3 and d["basis"]["max_degree"] == 10
    assert isinstance(d["r0"], float) and "model" not in d and "config" not in d
    assert d["train"] == str(XYZ.resolve()) and d["rungs"] == ["map"]
    assert d["provenance"]["command"].startswith("aj fit --order 3 --max-degree 10")
    assert d["provenance"]["ace_jax"]


def test_resolved_yaml_reproduces_from_other_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    _build_fit(tmp_path, cache, tmp_path / "a")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    main(["fit", "--config", str(tmp_path / "a" / "fit.yaml"), "--out", str(tmp_path / "b")])
    za, zb = np.load(tmp_path / "a" / "model.npz"), np.load(tmp_path / "b" / "model.npz")
    assert sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)
    ya = yaml.safe_load((tmp_path / "a" / "fit.yaml").read_text())
    yb = yaml.safe_load((tmp_path / "b" / "fit.yaml").read_text())
    for y in (ya, yb):
        y.pop("provenance"); y.pop("out")
    assert ya == yb


def test_model_file_run_records_model(tmp_path):
    main(["fit", "--model", str(FIXTURE_DIR / "si_ace_model.npz"), "--train", str(XYZ), "--r0", "2.35",
          "--out", str(tmp_path / "m"), *FAST])
    d = yaml.safe_load((tmp_path / "m" / "fit.yaml").read_text())
    assert d["model"] == str((FIXTURE_DIR / "si_ace_model.npz").resolve()) and "basis" not in d
    assert d["provenance"]["coupling"] is None
