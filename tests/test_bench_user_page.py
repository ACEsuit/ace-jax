"""The user docs' Performance page (docs/user/benchmarks.md): its images, snippets and nav entry exist, and
the silicon figure's cost files and families are consistent."""
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "bench"))

PAGE = ROOT / "docs" / "user" / "benchmarks.md"


def test_page_images_and_snippets_exist():
    """Every image has alt text and a file; every snippet exists; the multi-component table is the generated one."""
    text = PAGE.read_text()
    images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", text)
    for alt, p in images:
        assert alt.strip(), p                                       # alt text on every image
        assert (PAGE.parent / p).exists(), p
    snippets = re.findall(r'--8<-- "([^"]+)"', text)
    assert "benchmarks-cantor.md" in snippets
    for s in snippets:
        assert (ROOT / "docs" / "snippets" / s).exists(), s
    from accuracy_cost import multicomponent
    assert (ROOT / "docs" / "snippets" / "benchmarks-cantor.md").read_text() == multicomponent.table()
    assert "—" not in multicomponent.table()                        # every code timed on both devices


def test_nav_has_performance_after_how_to_guides():
    nav = (ROOT / "mkdocs.yml").read_text().split("\nnav:\n", 1)[1]
    top = [ln.strip()[2:].split(":")[0] for ln in nav.splitlines() if re.match(r"^  - ", ln)]
    assert top[top.index("How-to guides") + 1] == "Performance"
    assert "  - Performance: benchmarks.md" in nav.splitlines()


def test_interactive_silicon_figure_is_wired_and_consistent():
    """The silicon figure's Plotly spec (bench/si_qoi/plot_cost_web.py) sits next to its fallback PNG inside a
    `plotly-figure` div, the page loads plotly-figures.js, every trace is a plotted family with one legend entry
    per family, the presets name only those families, and the spec is strict JSON (no NaN)."""
    import json
    import yaml
    sys.path.insert(0, str(ROOT / "bench" / "si_qoi"))
    import plot_cost_web as W
    text = PAGE.read_text()
    m = re.search(r'<div class="plotly-figure" markdown>\s*!\[[^\]]+\]\((assets/benchmarks/[^)]+\.png)\)\s*</div>', text)
    assert m, "the silicon figure is not wrapped for the interactive renderer"
    spec = PAGE.parent / m.group(1).replace(".png", ".json")
    fig = json.loads(spec.read_text(), parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    groups = [t["legendgroup"] for t in fig["data"]]
    assert set(groups) <= set(W.P.FAM)
    assert sorted(t["legendgroup"] for t in fig["data"] if t["showlegend"]) == sorted(set(groups))
    presets = fig["layout"]["meta"]["presets"]
    assert [p["label"] for p in presets] == list(W.PRESETS)
    assert all(set(p["families"] or []) <= set(W.P.FAM) for p in presets)
    first = set(presets[0]["families"])
    assert all((t["visible"] is True) == (t["legendgroup"] in first) for t in fig["data"])
    cfg = yaml.safe_load((ROOT / "mkdocs.yml").read_text().replace("!!python/name:", ""))
    assert "javascripts/plotly-figures.js" in cfg["extra_javascript"]
    assert (ROOT / "docs" / "user" / "javascripts" / "plotly-figures.js").exists()


def test_make_cost_keeps_gap_sizes_and_aliases_seed_zero(tmp_path):
    """make_cost.py: a seeded family's seed-0 cost is also its size's (GRACE_c16l3_s0 -> GRACE_c16l3); GAP's sizes
    (s9000) only look like seed suffixes and get no sizeless alias (they once wrote GAP, GAPE30, GAPnoglue)."""
    import json
    import subprocess
    rows = [{"code": "gap", "family": "gap", "size": "s9000", "dtype": "float64", "mode": "lammps"},
            {"code": "grace", "family": "grace", "size": "c16l3_s0", "dtype": "float64", "mode": "lammps"}]
    f = tmp_path / "t.jsonl"
    f.write_text("".join(json.dumps(r | {"mode": "parity", "status": "parity_ok"}) + "\n"
                         + json.dumps(r | {"status": "ok", "atom_steps_per_s": 1e6}) + "\n" for r in rows))
    out = tmp_path / "cost.json"
    subprocess.run([sys.executable, str(ROOT / "bench" / "si_qoi" / "make_cost.py"), str(f), "--out", str(out)],
                   check=True, capture_output=True)
    assert sorted(json.loads(out.read_text())) == ["GAP_s9000", "GRACE_c16l3", "GRACE_c16l3_s0"]


def test_committed_cost_files_have_no_sizeless_aliases():
    """The committed cost files: every name has its size (the seed-0 aliases are seeded families' sizes)."""
    import json
    for d in ("cpu", "gpu"):
        names = json.loads((ROOT / "bench" / "si_qoi" / "results" / f"cost_sulis_{d}.json").read_text())
        assert not {"GAP", "GAPE30", "GAPnoglue"} & set(names), d


def test_make_cost_maps_cueq_rows_to_their_own_line(tmp_path):
    """make_cost.py: an acejax-ace row with aa_dag "cueq" (mode lammps-cueq) is the cueq line, ACEcueq_<size>,
    gated by its own parity row (a plain acejax gate does not cover it), and is kept past --source (which picks
    the ace-jax DAG rows from another file); its codes note says cueq, not dag (no centre dot)."""
    import json
    import subprocess
    base = {"code": "acejax-ace", "family": "ace", "size": "o4d20", "dtype": "float64"}
    auto, cq = tmp_path / "auto.jsonl", tmp_path / "cq.jsonl"
    auto.write_text(json.dumps(base | {"mode": "parity", "status": "parity_ok"}) + "\n"
                    + json.dumps(base | {"mode": "lammps", "status": "ok", "atom_steps_per_s": 1e6 / 3.12,
                                         "aa_dag": "greedy"}) + "\n")
    row = base | {"mode": "lammps-cueq", "status": "ok", "atom_steps_per_s": 1e6 / 1.6, "aa_dag": "cueq"}
    out = tmp_path / "cost.json"

    def run(*lines):
        cq.write_text("".join(json.dumps(r) + "\n" for r in lines))
        subprocess.run([sys.executable, str(ROOT / "bench" / "si_qoi" / "make_cost.py"), str(auto), str(cq),
                        "--source", "acejax-ace=auto", "--out", str(out)], check=True, capture_output=True)
        return json.loads(out.read_text()), json.loads(out.with_suffix(".codes.json").read_text())
    cost, _ = run(base | {"mode": "parity", "status": "parity_ok"}, row)       # only a plain gate: not covered
    assert sorted(cost) == ["ACE_o4d20"]
    cost, codes = run(base | {"mode": "parity-lammps-cueq", "status": "parity_ok"}, row)
    assert cost == {"ACE_o4d20": pytest.approx(3.12), "ACEcueq_o4d20": pytest.approx(1.6)}
    assert "lammps-cueq" in codes["ACEcueq_o4d20"] and " dag:" not in codes["ACEcueq_o4d20"]


def test_cueq_line_is_a_same_model_other_code_family():
    """The cueq line (removed; its rows stay as data): linear ACE's colour, its own marker and line style among the
    GPU's linear ACE lines; a second code for the ace fits (its cost under ACEcueq_<size>); left out of the figures
    (SKIP) and of every preset.  An FFI row is the kernel line, ace_ffi, on the A100 row as on the CPU row."""
    sys.path.insert(0, str(ROOT / "bench" / "si_qoi"))
    import plot_cost as P
    import plot_cost_web as W
    lab, col, mk, ls, _, pat = P.FAM["ace_cueq"]
    assert col == P.FAM["ace"][1] and pat == P.FAM["ace"][5] and "cueq" in lab
    gpu_ace = ("ace", "ace_tab", "ace_tabfork")                   # the blue lines of the A100 row
    assert (mk, ls) not in {P.FAM[f][2:4] for f in gpu_ace} and mk in W.SYMBOL and ls in W.DASH
    assert P.cost_name("ace_cueq", "o4d20") == "ACEcueq_o4d20"
    assert P.family_of({"family": "ace", "size": "o4d20", "code": "acejax-ace", "aa_dag": "cueq"}) == "ace_cueq"
    assert P.family_of({"family": "ace", "size": "o4d20", "code": "acejax-ace", "aa_dag": "greedy"}) == "ace"
    assert "ace_cueq" in P.SKIP and not any("ace_cueq" in (fams or ()) for fams in W.PRESETS.values())
    assert P.family_of({"family": "ace", "size": "o4d20", "code": "acejax-ace", "device": "gpu", "ffi": True}) == "ace_ffi"


def test_nep_two_stage_experiments_stay_off_the_nep_line():
    """The NEP loss-weight and two-stage refits (docs/dev) are a family no figure plots, so the NEP ladder keeps
    one fit per size; every such size in metrics.jsonl is listed."""
    import json
    sys.path.insert(0, str(ROOT / "bench" / "si_qoi"))
    import plot_cost as P
    assert P.family_of({"family": "nep", "size": "n120b-2s"}) == "nep_2s" and "nep_2s" not in P.FAM
    assert P.family_of({"family": "nep", "size": "n120b"}) == "nep"
    rows = [json.loads(l) for l in open(ROOT / "bench" / "accuracy_cost" / "results" / "metrics.jsonl")]
    assert {r["size"] for r in rows if r["family"] == "nep" and "-" in r["size"]} == P.NEP_TWO_STAGE


def test_make_cost_ffi_rows_are_their_own_line(tmp_path):
    """make_cost.py: a timing row of an FFI kernel bundle (`ffi`) is the ACEffi line, not linear ACE's, and its
    code is labelled ffi, so the interactive figure's hover tells the kernel from the XLA bundle."""
    import json
    import subprocess
    rows = [{"code": "acejax-ace", "family": "ace", "size": "o4d20", "dtype": "float64", "mode": "lammps", "ffi": True},
            {"code": "acejax-ace", "family": "ace", "size": "o4d20", "dtype": "float64", "mode": "lammps"}]
    f = tmp_path / "t.jsonl"
    f.write_text("".join(json.dumps(r | {"mode": "parity", "status": "parity_ok"}) + "\n"
                         + json.dumps(r | {"status": "ok", "atom_steps_per_s": 1e6 * (2 if r.get("ffi") else 1)}) + "\n"
                         for r in rows))
    out = tmp_path / "cost.json"
    subprocess.run([sys.executable, str(ROOT / "bench" / "si_qoi" / "make_cost.py"), str(f), "--out", str(out)],
                   check=True, capture_output=True)
    cost, codes = json.loads(out.read_text()), json.loads(out.with_suffix(".codes.json").read_text())
    assert cost == {"ACE_o4d20": 1.0, "ACEffi_o4d20": 0.5}
    assert codes["ACEffi_o4d20"].startswith("acejax-ace float64 lammps ffi")
    assert " ffi" not in codes["ACE_o4d20"]
