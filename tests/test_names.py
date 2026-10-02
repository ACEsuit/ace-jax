"""No legacy names survive the construct -> basis rename (historical docs excepted)."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
HISTORICAL = ("docs/dev/plans/", "docs/dev/specs/", "docs/dev/tier2-plan.md")
LEGACY = ["ace_jax.construct", "from .construct", "from ..construct", "from ...construct",
          "Authoring", '"authoring"', "ACEJAX_NO_JULIA", "python-authoring", "test_python_authoring",
          "[basis]", "--extra basis", "optional `basis` extra", "src/ace_jax/construct"]


def _grep(pattern):
    r = subprocess.run(["git", "grep", "-n", "-F", "-e", pattern, "--", ".", ":!uv.lock"],
                       cwd=ROOT, capture_output=True, text=True)
    return [l for l in r.stdout.splitlines()
            if not l.startswith(HISTORICAL) and not l.startswith("tests/test_names.py")]


def test_no_legacy_names():
    hits = {p: _grep(p) for p in LEGACY}
    assert not any(hits.values()), {p: h[:5] for p, h in hits.items() if h}
