"""The installed version is visible to Python and on the command line."""
import importlib.metadata
import subprocess
import sys


def test_version_matches_the_distribution():
    import ace_jax
    assert ace_jax.__version__ == importlib.metadata.version("ace-jax")


def test_aj_version_flag():
    r = subprocess.run([sys.executable, "-m", "ace_jax.cli", "--version"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == f"ace-jax {importlib.metadata.version('ace-jax')}"
