"""`aj` is a short alias of the `ace-jax` console script."""
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_aj_and_ace_jax_are_the_same_entry_point():
    import tomllib
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["aj"] == scripts["ace-jax"] == "ace_jax.cli:main"


def test_installed_aj_runs():
    exe = shutil.which("aj", path=str(pathlib.Path(sys.executable).parent))
    if exe is None:
        pytest.skip("aj not installed in this environment (re-sync after adding the script)")
    out = subprocess.run([exe, "--help"], capture_output=True, text=True, check=True).stdout
    assert "fit" in out and "eval" in out and "construct" in out

