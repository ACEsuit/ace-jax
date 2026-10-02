"""The user docs' Performance page (docs/user/benchmarks.md) and its generator,
bench/scaling/user_page.py: figure selection, line styles, the 8192-atom table,
and that the page's images, snippets and nav entry exist."""
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "bench"))

PAGE = ROOT / "docs" / "user" / "benchmarks.md"


def _row(code, mode, n, *, host="modal-a100", dtype="float64", system="SiGe", size="medium",
         tp=None, status="ok", device=None, run=None):
    r = {"code": code, "mode": mode, "model": f"{code}/{system}/{size}", "size": size,
         "system": system, "n_atoms": n, "dtype": dtype, "host": host, "status": status,
         "device": device or ("cpu" if "cpu" in host else "gpu"),
         "atom_steps_per_s": tp if tp is not None else 1e3 * n}
    r["_key"] = [r["model"], mode, n, dtype, r["device"]]
    if run:
        r["run"] = run
    return r


def _matrix(codes=(("acejax-pace", "standalone"), ("acejax-pace", "lammps"), ("mlpace", "lammps"),
                   ("mace", "standalone"))):
    rows = []
    for host, dtype in (("moriarty-cpu", "float64"), ("modal-a100", "float64"),
                        ("modal-a100", "float32"), ("moriarty-gpu", "float64")):
        for system in ("SiGe", "Cantor"):
            for code, mode in codes:
                for n in (4096, 8192, 16384):
                    rows.append(_row(code, mode, n, host=host, dtype=dtype, system=system))
    return rows


@pytest.fixture
def up():
    import matplotlib
    matplotlib.use("Agg")
    from scaling import user_page
    return user_page


def test_three_settings_map_to_the_configured_hosts(up):
    assert [s.name for s in up.SETTINGS] == ["cpu_float64", "gpu_float64", "gpu_float32"]
    hosts = {"cpu": "moriarty-cpu", "gpu": "modal-a100"}
    assert up.USER_HOSTS == hosts
    rows = _matrix()
    for s in up.SETTINGS:
        sel = up.select(rows, s)
        assert sel
        assert {r["host"] for r in sel} == {hosts[s.device]}
        assert {r["dtype"] for r in sel} == {s.dtype}
        assert {r["size"] for r in sel} == {"medium"}


def test_select_follows_a_host_override(up):
    rows = _matrix() + [_row("acejax-pace", "standalone", 8192, host="sulis-a100", dtype="float32")]
    s = up.setting("gpu_float32")
    sel = up.select(rows, s, hosts={"cpu": "lestrade-cpu", "gpu": "sulis-a100"})
    assert {r["host"] for r in sel} == {"sulis-a100"}


def test_select_drops_other_sizes_failed_rows_and_symmetrix_float32(up):
    rows = [_row("acejax-pace", "standalone", 8192, dtype="float32"),
            _row("acejax-pace", "standalone", 8192, dtype="float32", size="large"),
            _row("mace", "standalone", 16384, dtype="float32", status="oom"),
            _row("mace", "lammps", 8192, dtype="float32")]      # Symmetrix runs in double
    sel = up.select(rows, up.setting("gpu_float32"))
    assert [(r["code"], r["mode"], r["size"]) for r in sel] == [("acejax-pace", "standalone", "medium")]


def test_learned_radial_proxy_lines_are_excluded(up):
    rows = _matrix(codes=(("acejax-ace", "standalone"), ("acejax-ace-learned", "standalone"),
                          ("acejax-ace-analytic", "lammps")))
    for s in up.SETTINGS:
        assert {r["code"] for r in up.select(rows, s)} == {"acejax-ace"}
    assert "acejax-ace-learned" not in up.table(rows)


def test_unknown_codes_are_picked_up_after_the_known_ones(up):
    rows = _matrix() + [_row("zz-newcode", "standalone", n) for n in (4096, 8192)]
    sel = up.select(rows, up.setting("gpu_float64"))
    codes = up.codes_in(sel)
    assert codes == ["acejax-pace", "mlpace", "mace", "zz-newcode"]   # plot.CODES order, then new
    assert up.label("zz-newcode") == "zz-newcode"
    assert up.colour("zz-newcode").startswith("#")
    assert "zz-newcode" in up.table(rows)


def test_codes_follow_plot_codes_order_and_colours(up):
    from scaling import plot
    rows = _matrix()
    sel = up.select(rows, up.setting("gpu_float64"))
    known = [c for c in plot.CODES if c in {r["code"] for r in sel}]
    assert up.codes_in(sel) == known
    for c in known:
        assert up.colour(c) == plot.CODES[c][1]


def _figure(up, rows, name, tmp_path, monkeypatch):
    figs = []
    monkeypatch.setattr(up.plt, "close", lambda f=None: figs.append(f))
    path = up.figure(rows, up.setting(name), tmp_path)
    return path, figs[-1]


def test_figure_is_one_row_sige_then_cantor_lammps_solid_standalone_dashed(up, tmp_path, monkeypatch):
    path, fig = _figure(up, _matrix(), "gpu_float64", tmp_path, monkeypatch)
    assert path == tmp_path / "gpu_float64.png" and path.exists()
    axes = [ax for ax in fig.axes if ax.lines]
    assert [ax.get_title() for ax in axes] == ["SiGe", "Cantor"]
    for ax in axes:
        styles = {}
        for ln in ax.lines:
            if len(ln.get_xdata()) > 1:
                styles.setdefault(ln.get_color(), set()).add(ln.get_linestyle())
                assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
        pace = up.colour("acejax-pace")
        assert styles[pace] == {"-", "--"}                         # both modes
        assert styles[up.colour("mlpace")] == {"-"}                # LAMMPS only: solid
        assert styles[up.colour("mace")] == {"--"}                 # standalone only: dashed
    assert len(fig.legends) == 1                                   # one legend per figure
    texts = {t.get_text() for t in fig.legends[0].get_texts()}
    assert {"LAMMPS", "standalone"} <= texts


def test_figure_skips_codes_with_no_rows_in_a_panel(up, tmp_path, monkeypatch):
    rows = [r for r in _matrix() if not (r["code"] == "mace" and r["system"] == "Cantor")]
    _, fig = _figure(up, rows, "gpu_float64", tmp_path, monkeypatch)
    cantor = next(ax for ax in fig.axes if ax.get_title() == "Cantor")
    assert up.colour("mace") not in {ln.get_color() for ln in cantor.lines}


def test_table_takes_8192_atoms_from_the_right_host_and_dtype(up):
    rows = []
    for host, dtype, base in (("moriarty-cpu", "float64", 1), ("modal-a100", "float64", 2),
                              ("modal-a100", "float32", 3), ("moriarty-gpu", "float64", 9),
                              ("moriarty-cpu", "float32", 8)):
        for system, k in (("SiGe", 1), ("Cantor", 4)):
            for n in (4096, 8192, 16384):
                tp = base * k * 1e5 if n == 8192 else 7.77e7
                rows.append(_row("acejax-pace", "standalone", n, host=host, dtype=dtype,
                                 system=system, tp=tp))
    md = up.table(rows)
    line = next(ln for ln in md.splitlines() if "standalone" in ln)
    cells = [c.strip() for c in line.strip("|").split("|")]
    # evaluator, mode, then CPU f64 SiGe/Cantor, GPU f64 SiGe/Cantor, GPU f32 SiGe/Cantor
    assert cells[2:] == ["100k", "400k", "200k", "800k", "300k", "1.20M"]
    head = md.splitlines()[0]
    assert "SiGe" in head and "Cantor" in head


def test_table_marks_missing_cells_and_uses_the_median_of_repeats(up):
    rows = [_row("mlpace", "lammps", 8192, host="moriarty-cpu", tp=5e5)]
    for i, tp in enumerate((1e6, 3e6, 2e6)):
        rows.append(_row("acejax-pace", "lammps", 8192, tp=tp, run=f"r{i}"))
    md = up.table(rows)
    ml = next(ln for ln in md.splitlines() if "ML-PACE" in ln)
    pace = next(ln for ln in md.splitlines() if "PACE model" in ln)
    assert [c.strip() for c in ml.strip("|").split("|")][2:] == ["500k", "—", "—", "—", "—", "—"]
    assert [c.strip() for c in pace.strip("|").split("|")][4] == "2.00M"   # median of 1, 3, 2 M


def test_notes_name_the_hosts_and_flag_the_trim_line_only_when_present(up):
    rows = _matrix()
    notes = up.notes(rows)
    assert "moriarty-cpu" in notes and "modal-a100" in notes
    assert "trim" not in notes
    rows += [_row("acepotentials-trim", "lammps", n) for n in (4096, 8192)]
    assert "exact radial" in up.notes(rows)


def test_main_writes_the_figures_and_snippets(up, tmp_path):
    import json
    res = tmp_path / "results"
    res.mkdir()
    (res / "x.jsonl").write_text("\n".join(json.dumps(r) for r in _matrix()))
    up.main(["--results", str(res / "*.jsonl"), "--figs", str(tmp_path / "figs"),
             "--snippets", str(tmp_path / "snip")])
    for name in ("cpu_float64", "gpu_float64", "gpu_float32"):
        assert (tmp_path / "figs" / f"{name}.png").exists()
    assert "|" in (tmp_path / "snip" / up.TABLE_SNIPPET).read_text()
    assert (tmp_path / "snip" / up.NOTES_SNIPPET).read_text().strip()


def test_page_images_and_snippets_exist():
    text = PAGE.read_text()
    images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", text)
    assert {pathlib.Path(p).name for _, p in images} >= {
        "cpu_float64.png", "gpu_float64.png", "gpu_float32.png"}
    for alt, p in images:
        assert alt.strip(), p                                       # alt text on every image
        assert (PAGE.parent / p).exists(), p
    snippets = re.findall(r'--8<-- "([^"]+)"', text)
    from scaling import user_page
    assert {user_page.TABLE_SNIPPET, user_page.NOTES_SNIPPET} <= set(snippets)
    for s in snippets:
        assert (ROOT / "docs" / "snippets" / s).exists(), s
    assert "https://github.com/ACEsuit/ace-jax/blob/main/docs/benchmarks.md" in text


def test_nav_has_performance_after_how_to_guides():
    nav = (ROOT / "mkdocs.yml").read_text().split("\nnav:\n", 1)[1]
    top = [ln.strip()[2:].split(":")[0] for ln in nav.splitlines() if re.match(r"^  - ", ln)]
    assert top[top.index("How-to guides") + 1] == "Performance"
    assert "  - Performance: benchmarks.md" in nav.splitlines()


def test_atom_ticks_are_compact(up):
    assert [up._atoms_fmt(n, None) for n in (256, 4096, 1 << 20, 1 << 21)] == ["256", "4k", "1M", "2M"]


def test_end_label_falls_back_to_the_code_name(up, monkeypatch):
    """A code with a plot.CODES entry but no plot.SHORT one: its long legend
    label would run off the figure as an end label."""
    from scaling import plot
    monkeypatch.setitem(plot.CODES, "zz-new", ("A very long legend label for zz-new", "#4a3aa7"))
    assert up.short("zz-new") == "zz-new"
    assert up.short("zz-newcode") == "zz-newcode"
    assert up.short("acejax-pace") == plot.SHORT["acejax-pace"]


def test_a_system_with_no_rows_gets_a_labelled_empty_panel(up, tmp_path, monkeypatch):
    rows = [r for r in _matrix() if r["system"] == "SiGe"]
    _, fig = _figure(up, rows, "gpu_float64", tmp_path, monkeypatch)
    cantor = next(ax for ax in fig.axes if ax.get_title() == "Cantor")
    assert not cantor.lines and "no results" in " ".join(t.get_text() for t in cantor.texts)
