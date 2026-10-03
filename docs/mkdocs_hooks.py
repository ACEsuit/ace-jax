"""MkDocs hooks: render the tutorial notebooks, and fill the CLI reference from the
argparse parser itself.

Before the build, docs/build_tutorials.py runs each tutorial notebook and writes
its page (only when the notebook changed; ACEJAX_DOCS_NOTEBOOKS=skip writes
placeholders instead).

A line `<!-- aj-help fit -->` in a page is replaced by the output of
`aj fit --help`, taken from `ace_jax.cli._parser()` at build time, so the
reference cannot drift from the code.  `<!-- aj-help -->` gives the top-level
help.  The width is fixed so the build is reproducible.
"""
import os, pathlib, re, sys

_MARK = re.compile(r"^<!-- aj-help ?(\w*) -->$", re.M)


def _help(cmd):
    os.environ["COLUMNS"] = "88"   # argparse wraps to the terminal width
    from ace_jax.cli import _parser
    top = _parser()
    if not cmd:
        return top.format_help()
    sub = next(a for a in top._actions if a.__class__.__name__ == "_SubParsersAction")
    return sub.choices[cmd].format_help()


def on_page_markdown(markdown, page, config, files):
    if "aj-help" not in markdown:
        return markdown
    return _MARK.sub(lambda m: "```text\n" + _help(m.group(1)).rstrip() + "\n```", markdown)


def on_pre_build(config):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import build_tutorials
    build_tutorials.main()
