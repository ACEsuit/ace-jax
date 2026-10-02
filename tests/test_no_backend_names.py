"""Users never meet the Julia backend: not in CLI help, messages or user docs.
(Maintainer material -- coupling/, docs/dev/coupling-etshim-spec.md, CLAUDE.md, the
julia/ reference generators -- may and does talk about Julia.)"""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
USER_DOCS = ["README.md", "skills/ace-jax/SKILL.md"]
# the user documentation site (docs/user); the licence page credits the bundled runtime by name
# The Performance page is the one exception: it benchmarks ACEpotentials.jl itself (evaluated
# in Julia, and as a compiled library in LAMMPS), so it has to name Julia.
BACKEND_NAMED = {"docs/user/licence.md", "docs/user/benchmarks.md"}
USER_DOCS += sorted(str(f.relative_to(ROOT)) for f in (ROOT / "docs/user").rglob("*.md")
                    if str(f.relative_to(ROOT)) not in BACKEND_NAMED)
JULIA = re.compile(r"julia", re.I)
FILE_REF = re.compile(r"julia/|\.jl\b|\.jl`|\.github/workflows/\S+")   # maintainer file paths, not prose


def test_cli_help_has_no_backend_names():
    from ace_jax.cli import _parser
    top = _parser()
    texts = [top.format_help()] + [p.format_help() for p in top._subparsers._group_actions[0].choices.values()]
    assert not any(JULIA.search(t) for t in texts)


def test_user_docs_have_no_backend_names():
    hits = [f"{f}:{i}: {line.strip()}" for f in USER_DOCS
            for i, line in enumerate((ROOT / f).read_text().splitlines(), 1)
            if JULIA.search(FILE_REF.sub("", line))]
    assert not hits, hits


def test_source_messages_have_no_backend_names():
    r = subprocess.run(["git", "grep", "-n", "-i", "-e", "julia", "--", "src/ace_jax"], cwd=ROOT,
                       capture_output=True, text=True)
    offenders = [l for l in r.stdout.splitlines() if re.search(r"raise |print\(|log\(|help=|warn", l)]
    assert not offenders, offenders


def test_backend_name_exemptions_are_only_licence_and_benchmarks():
    """Every other user page is still checked; the exemptions are named pages that exist."""
    assert BACKEND_NAMED == {"docs/user/licence.md", "docs/user/benchmarks.md"}
    assert all((ROOT / f).exists() for f in BACKEND_NAMED)
    assert "docs/user/quickstart.md" in USER_DOCS and "docs/user/benchmarks.md" not in USER_DOCS
