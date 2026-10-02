"""Users never meet the Julia backend: not in CLI help, messages or user docs.
(Maintainer material -- coupling/, docs/dev/coupling-etshim-spec.md, CLAUDE.md, the
julia/ reference generators -- may and does talk about Julia.)"""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
USER_DOCS = ["README.md", "skills/ace-jax/SKILL.md"]
# the user documentation site (docs/user); the licence page credits the bundled runtime by name
USER_DOCS += sorted(str(f.relative_to(ROOT)) for f in (ROOT / "docs/user").rglob("*.md") if f.name != "licence.md")
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
