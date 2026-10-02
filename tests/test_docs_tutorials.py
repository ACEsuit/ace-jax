"""docs/build_tutorials.py turns an exported marimo notebook into a site page."""
import base64
import html
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_tutorials", ROOT / "docs" / "build_tutorials.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def _esc(value):
    """How marimo writes an element attribute: HTML-escaped JSON."""
    return html.escape(json.dumps(value), quote=True).replace("'", "&#x27;")


def _md(text):
    return f'<span class="markdown prose dark:prose-invert contents"><span class="paragraph">{text}</span></span>'


def _cell(src, outputs=(), hidden=False):
    md = {"marimo": {"config": {"hide_code": True}}} if hidden else {}
    return {"cell_type": "code", "source": src, "metadata": md, "outputs": list(outputs)}


PNG = base64.b64encode(b"\x89PNG fake").decode()
NB = {"cells": [
    _cell("import marimo as mo"),                                              # imports before the title
    {"cell_type": "markdown", "source": "# Title\n\nSome $E_0$ maths.", "metadata": {}},
    _cell("x = 1\nprint(x)", [{"output_type": "stream", "name": "stdout", "text": "1\n"}]),
    _cell("plot()", [{"output_type": "display_data", "data": {"image/png": PNG}}]),
    _cell("callout()", [{"output_type": "display_data", "data": {"text/html":
        f"<marimo-callout-output data-html='{_esc(_md('<strong>Checkpoint passed</strong>'))}' "
        f"data-kind='{_esc('success')}'></marimo-callout-output>"}}], hidden=True),
    _cell("hints()", [{"output_type": "display_data", "data": {"text/html":
        f"<marimo-accordion data-labels='{_esc([_md('Hint 1')])}' data-multiple='false'>"
        f"<div>{_md('Look at <code>E0</code>.')}</div></marimo-accordion>"}}], hidden=True),
    _cell("d = mo.ui.dropdown(...)", [{"output_type": "display_data", "data": {"text/html":
        f"<marimo-ui-element object-id='a'><marimo-dropdown data-initial-value='{_esc(['10'])}' "
        f"data-label='{_esc(_md('max degree'))}'></marimo-dropdown></marimo-ui-element>"}}]),
    _cell("steps", [{"output_type": "display_data", "data": {"text/plain": html.escape(     # a bare element
        f"<marimo-slider data-initial-value='40' data-label='{_esc(_md('L-BFGS steps'))}'></marimo-slider>")}}]),
]}


def test_page_maps_marimo_output_to_site_components(tmp_path):
    page = B.convert(NB, "demo", tmp_path / "demo_files")
    assert page.startswith("# Title") and "$E_0$" in page                    # markdown kept for MathJax
    assert "```python\nx = 1\nprint(x)\n```" in page and "```text\n1\n```" in page
    assert "![Figure 1](demo_files/fig-1.png)" in page
    assert (tmp_path / "demo_files" / "fig-1.png").read_bytes() == b"\x89PNG fake"
    assert '<div class="admonition success">\n<p><strong>Checkpoint passed</strong></p>' in page
    assert "callout()" not in page and "hints()" not in page                   # hide_code: output only
    assert '<details class="note">\n<summary>Hint 1</summary>\n<p>Look at <code>E0</code>.</p>' in page
    assert "<em>max degree</em> = <code>10</code>" in page and "<em>L-BFGS steps</em> = <code>40</code>" in page
    assert "marimo-" not in page


def test_a_cell_error_fails_the_page(tmp_path):
    nb = {"cells": [_cell("1/0", [{"output_type": "error", "ename": "ZeroDivisionError", "evalue": "x"}])]}
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        B.convert(nb, "demo", tmp_path / "f")
