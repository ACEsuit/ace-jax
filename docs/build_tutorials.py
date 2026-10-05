"""Render the tutorial marimo notebooks as documentation pages.

Each notebook in docs/user/tutorials/notebooks/ is run (`marimo export ipynb
--include-outputs`, in a scratch directory) and converted to a Markdown page,
docs/user/tutorials/<page>.md, with its figures in <page>_files/. The page is
plain MkDocs Markdown, so it takes the site's theme, MathJax and admonitions:

* markdown cells stay Markdown;
* code cells become highlighted code blocks (cells marked hide_code show only
  their output);
* outputs: text/markdown is inlined, figures become PNG files, printed text a
  text block; marimo's own elements are mapped to the site's components:
  callouts to admonitions, accordions to collapsible <details>, and UI controls
  (sliders, dropdowns) to a note of their default value, since the page is a
  static run.

The generated pages and figures are not committed (.gitignore); the docs build
runs this through docs/mkdocs_hooks.py, re-running a notebook only when its
source changed. Set ACEJAX_DOCS_NOTEBOOKS=skip for a quick build with
placeholder pages.

    python docs/build_tutorials.py            # all notebooks
    python docs/build_tutorials.py first_fit_si
"""
import base64
import hashlib
import html
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parents[1]
TUT = ROOT / "docs" / "user" / "tutorials"
NOTEBOOKS = {"first_fit_si": "first-fit", "learned_radials_si": "learned-radials",
             "multi_element": "multi-element", "school_dataset_si": "dataset-and-properties",
             "school_basis_si": "basis-and-evidence", "school_truth_si": "truth-about-the-truth",
             "school_byod": "bring-your-own-data", "school_surfaces_si": "surfaces",
             "school_curation_si": "curation"}   # notebook -> page
ADMONITION = {"success": "success", "warn": "warning", "danger": "danger", "info": "info", "neutral": "note"}


def _attr(raw):
    """A marimo element attribute: an HTML-escaped JSON value."""
    return json.loads(html.unescape(raw))


class _Clean(HTMLParser):
    """marimo-rendered Markdown HTML -> plain HTML: its paragraph spans become <p>,
    its wrapper spans go, everything else passes through."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "") or ""
        if tag == "span" and "paragraph" in cls.split():
            self.out.append("<p>"); self.stack.append("</p>")
        elif tag == "span" and "markdown" in cls.split():
            self.stack.append("")
        else:
            self.out.append(self.get_starttag_text()); self.stack.append(None)

    def handle_endtag(self, tag):
        close = self.stack.pop() if self.stack else None
        self.out.append(f"</{tag}>" if close is None else close)

    def handle_startendtag(self, tag, attrs):
        self.out.append(self.get_starttag_text())

    def handle_data(self, data):
        self.out.append(data)

    def handle_entityref(self, name):
        self.out.append(f"&{name};")

    def handle_charref(self, name):
        self.out.append(f"&#{name};")


def clean(fragment):
    p = _Clean()
    p.feed(fragment)
    p.close()
    return "".join(p.out)


def _text(fragment):
    return re.sub(r"<[^>]+>", "", fragment).strip()


def _controls(fragment):
    """The marimo UI controls in an output, as (kind, label, value) triples."""
    out = []
    for m in re.finditer(r"<marimo-([a-z-]+)\s([^>]*?)>", fragment):
        kind, attrs = m.group(1), dict(re.findall(r"([\w-]+)='([^']*)'", m.group(2)))
        if kind in ("ui-element",) or "data-label" not in attrs:
            continue
        label = _text(_attr(attrs["data-label"]) or "") or kind
        value = _attr(attrs["data-initial-value"]) if "data-initial-value" in attrs else None
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        out.append((kind, label, value))
    return out


def html_output(fragment):
    """One text/html output as page Markdown/HTML."""
    if "<marimo-callout-output" in fragment:
        m = re.search(r"<marimo-callout-output\s([^>]*?)>", fragment)
        attrs = dict(re.findall(r"([\w-]+)='([^']*)'", m.group(1)))
        kind = ADMONITION.get(_attr(attrs.get("data-kind", '"neutral"')), "note")
        return f'<div class="admonition {kind}">\n{clean(_attr(attrs["data-html"]))}\n</div>'
    if "<marimo-accordion" in fragment:
        m = re.search(r"<marimo-accordion\s([^>]*?)>(.*)</marimo-accordion>", fragment, re.S)
        attrs = dict(re.findall(r"([\w-]+)='([^']*)'", m.group(1)))
        labels = [_text(lab) for lab in _attr(attrs["data-labels"])]
        bodies = re.findall(r"<div>(.*?)</div>(?=<div>|$)", m.group(2), re.S)
        return "\n".join(f'<details class="note">\n<summary>{lab}</summary>\n{clean(body)}\n</details>'
                         for lab, body in zip(labels, bodies))
    controls = _controls(fragment)
    if controls:
        items = ", ".join(f"<em>{html.escape(label)}</em> = <code>{html.escape(str(value))}</code>"
                          for _, label, value in controls)
        return (f'<div class="admonition abstract">\n<p>Interactive controls in the notebook '
                f'(this page shows their defaults): {items}</p>\n</div>')
    return clean(fragment)


def _join(v):
    return "".join(v) if isinstance(v, list) else v


def convert(nb, page, files_dir):
    """The page Markdown for an exported notebook (figures written to files_dir)."""
    files_dir.mkdir(parents=True, exist_ok=True)
    parts, nfig = [], 0
    for cell in nb["cells"]:
        src = _join(cell["source"]).rstrip()
        if cell["cell_type"] == "markdown":
            parts.append(src)
            continue
        hidden = cell.get("metadata", {}).get("marimo", {}).get("config", {}).get("hide_code", False)
        if src and not hidden:
            parts.append(f"```python\n{src}\n```")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = _join(out.get("text", "")).rstrip()
                if text and out.get("name") == "stdout":
                    lines = text.splitlines()
                    if len(lines) > 40:
                        lines = lines[:20] + [f"... ({len(lines) - 40} lines omitted) ..."] + lines[-20:]
                    parts.append("```text\n" + "\n".join(lines) + "\n```")
                continue
            if out.get("output_type") == "error":
                raise RuntimeError(f"{page}: a cell raised: {out.get('ename')}: {out.get('evalue')}")
            data = out.get("data", {})
            if "image/png" in data:
                nfig += 1
                name = f"fig-{nfig}.png"
                (files_dir / name).write_bytes(base64.b64decode(_join(data["image/png"])))
                parts.append(f"![Figure {nfig}]({files_dir.name}/{name})")
            elif "text/html" in data:
                parts.append(html_output(_join(data["text/html"])))
            else:
                text = _join(data.get("text/markdown", data.get("text/plain", ""))).rstrip()
                if "marimo-" in text:              # a bare UI element: exported as (escaped) HTML
                    parts.append(html_output(html.unescape(text) if "&lt;marimo-" in text else text))
                elif "text/markdown" in data:
                    parts.append(text)
                elif text:
                    parts.append("```text\n" + text + "\n```")
    # the title cell first: a notebook's imports cell usually precedes it
    first_h1 = next((i for i, p in enumerate(parts) if p.startswith("# ")), None)
    if first_h1:
        parts.insert(0, parts.pop(first_h1))
    return "\n\n".join(p for p in parts if p) + "\n"


def _cell_errors(ipynb):
    """The failing cells of an exported notebook, source and error: marimo itself reports only
    that some cells failed, which leaves a CI failure undiagnosable."""
    out = []
    for cell in json.loads(ipynb.read_text())["cells"]:
        for o in cell.get("outputs", []):
            text = json.dumps(o)
            if o.get("output_type") == "error" or "marimo-error" in text or "Traceback" in text:
                src = _join(cell.get("source", "")).strip().splitlines()
                tb = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(o.get("traceback", [])))
                out.append(f"--- cell: {src[0] if src else '?'} ...\n"
                           f"{o.get('ename', '')}: {o.get('evalue', '')}\n{tb[-2000:] or text[:2000]}")
    return "\n\nfailing cells:\n" + "\n".join(out) if out else "\n(no error output found in the export)"


def source_hash(nb_path):
    """The notebook and this converter: a change to either re-renders the page."""
    return hashlib.sha256(nb_path.read_bytes() + pathlib.Path(__file__).read_bytes()).hexdigest()[:16]


def build(name, force=False):
    """Run notebook `name` and write its page; skip if the page is current."""
    nb_path, page = TUT / "notebooks" / f"{name}.py", NOTEBOOKS[name]
    out_md, files_dir = TUT / f"{page}.md", TUT / f"{page}_files"
    stamp = f"<!-- generated by docs/build_tutorials.py from notebooks/{name}.py ({source_hash(nb_path)}) -->"
    if not force and out_md.exists() and out_md.read_text().startswith(stamp):
        return False
    with tempfile.TemporaryDirectory() as d:          # the notebook writes its outputs to the cwd
        ipynb = pathlib.Path(d) / f"{name}.ipynb"
        r = subprocess.run([sys.executable, "-m", "marimo", "export", "ipynb", "--include-outputs",
                            str(nb_path), "-o", str(ipynb)], cwd=d, capture_output=True, text=True)
        if r.returncode != 0 or not ipynb.exists():
            raise RuntimeError(f"marimo export of {name} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
                               + (_cell_errors(ipynb) if ipynb.exists() else ""))
        nb = json.loads(ipynb.read_text())
    for f in files_dir.glob("fig-*.png"):
        f.unlink()
    out_md.write_text(stamp + "\n\n" + convert(nb, page, files_dir))
    return True


def placeholder(name):
    page = NOTEBOOKS[name]
    out_md = TUT / f"{page}.md"
    if not out_md.exists():
        out_md.write_text(f"# {page}\n\n(ACEJAX_DOCS_NOTEBOOKS=skip: notebook `{name}.py` not rendered.)\n")


def main(names=None, force=False):
    names = names or list(NOTEBOOKS)
    if os.environ.get("ACEJAX_DOCS_NOTEBOOKS") == "skip":
        for n in names:
            placeholder(n)
        return
    for n in names:
        print(f"tutorial {n}: {'rendered' if build(n, force) else 'up to date'}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args or None, force="--force" in sys.argv)
