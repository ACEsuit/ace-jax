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


def test_successful_command_exits_zero(tmp_path):
    """The console script passes main()'s return value to sys.exit, so returning the
    metrics dict made every successful run exit 1 (and dump the dict to stderr)."""
    from conftest import FIXTURE_DIR
    xyz = FIXTURE_DIR / "si_tiny_train.xyz"
    if not xyz.exists():
        pytest.skip("missing si_tiny_train.xyz")
    code = subprocess.run(
        [sys.executable, "-c", "import sys; from ace_jax.cli import main; sys.exit(main(sys.argv[1:]))",
         "eval", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(xyz),
         "--energy-key", "dft_energy", "--force-key", "dft_force", "--out", str(tmp_path / "p.xyz")],
        capture_output=True, text=True)
    assert code.returncode == 0, code.stderr[-500:]
