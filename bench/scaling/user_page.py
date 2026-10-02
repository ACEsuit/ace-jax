"""Benchmark results (JSONL) -> the user docs' Performance page assets.

    python bench/scaling/user_page.py            # defaults: committed results, docs paths

Writes three figures, `docs/user/assets/benchmarks/{cpu_float64,gpu_float64,
gpu_float32}.png` (each one row: SiGe | Cantor), and two snippets that
`docs/user/benchmarks.md` includes with pymdownx.snippets: the throughput table
at TABLE_N atoms and the data-dependent notes (hosts, footnotes).  The page
text outside those snippets is hand-written.  The docs build never runs this:
the PNGs and snippets are committed, so re-render after new results land.

Which hosts feed the page is USER_HOSTS, the one place to change.  Every code
with rows on those hosts is drawn, except the learned-radial proxy lines
(plot.LEARNED).  Colour and order follow plot.CODES (so the page matches
docs/benchmarks.md); a code not in plot.CODES is drawn after them in a neutral
ink.  Line style is the mode, on this page: solid = LAMMPS, dashed =
standalone.  Repeated cases use plot.aggregate's median.
"""
import argparse
import pathlib
import sys
from collections import defaultdict
from typing import NamedTuple

if __package__ in (None, ""):                         # run as a script
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scaling import plot

ROOT = pathlib.Path(__file__).resolve().parents[2]

# the hosts behind the page: switch these (e.g. "lestrade-cpu", "sulis-a100") and re-render
USER_HOSTS = {"cpu": "moriarty-cpu", "gpu": "modal-a100"}

# hardware per host, for the notes; rows recording `device_name` take precedence
HARDWARE = {
    "moriarty-cpu": "16-core Intel Xeon Silver 4216 (LAMMPS: 16 MPI ranks; standalone: 32 threads)",
    "moriarty-gpu": "NVIDIA RTX A4500, 20 GB",
    "modal-a100": "NVIDIA A100-SXM4-80GB on Modal",
    "lestrade-cpu": "lestrade CPU, 8 P-cores (CPUs 0-15)",
    "sulis-a100": "NVIDIA A100 on Sulis (one GPU of a shared node)",
}


class Setting(NamedTuple):
    name: str          # also the figure's file stem
    device: str        # key into USER_HOSTS
    dtype: str
    title: str


SETTINGS = (Setting("cpu_float64", "cpu", "float64", "CPU, float64"),
            Setting("gpu_float64", "gpu", "float64", "GPU, float64"),
            Setting("gpu_float32", "gpu", "float32", "GPU, float32"))
SYSTEMS = ("SiGe", "Cantor")
SIZE = "medium"
TABLE_N = 8192
EXCLUDED = tuple(plot.LEARNED)
MODE_STYLE = {"lammps": "-", "standalone": "--"}      # this page: solid = LAMMPS
MODE_LABEL = {"lammps": "LAMMPS", "standalone": "standalone"}
UNKNOWN_COLOUR = plot.INK2                            # never a generated hue

FIGS_DIR = ROOT / "docs" / "user" / "assets" / "benchmarks"
SNIPPETS_DIR = ROOT / "docs" / "snippets"
RESULTS = ROOT / "bench" / "scaling" / "results" / "*.jsonl"
TABLE_SNIPPET = "benchmarks-table.md"
NOTES_SNIPPET = "benchmarks-notes.md"


def setting(name):
    return next(s for s in SETTINGS if s.name == name)


def select(rows, s, hosts=None):
    """The ok, timed, medium-model rows of one setting on its host, one per case
    (repeats folded to their median), learned-radial proxies left out."""
    host = (hosts or USER_HOSTS)[s.device]
    return [r for r in plot.aggregate(rows)
            if r.get("host") == host and r.get("dtype") == s.dtype and r.get("size") == SIZE
            and r.get("status") == "ok" and r.get("mode") in MODE_STYLE
            and r.get("code") not in EXCLUDED and r.get("system") in SYSTEMS
            and plot.throughput(r) and plot._measures_dtype(r)]


def codes_in(sel):
    """Codes present, in plot.CODES order, then any unknown ones by name."""
    present = {r["code"] for r in sel}
    return [c for c in plot.CODES if c in present] + sorted(present - set(plot.CODES))


def label(code):
    return plot.CODES[code][0] if code in plot.CODES else code


def short(code):
    return plot.SHORT.get(code, label(code))


def colour(code):
    return plot.CODES[code][1] if code in plot.CODES else UNKNOWN_COLOUR


def _atoms_fmt(x, _pos):
    """256, 1k, 64k, 1M: compact atom counts (powers of two)."""
    return f"{int(x)}" if x < 1024 else f"{int(x) // 1024}k" if x < 1 << 20 else f"{int(x) >> 20}M"


def _xatoms(ax):
    """Log x in atoms, ticks every factor of 4 (the GPU ladder spans 256 to 2M)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plot._xatoms(ax)
    ax.xaxis.set_major_locator(LogLocator(base=4))
    ax.xaxis.set_major_formatter(FuncFormatter(_atoms_fmt))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _modes(sel):
    return [m for m in MODE_STYLE if any(r["mode"] == m for r in sel)]


def figure(rows, s, out, hosts=None):
    """One setting's figure, SiGe | Cantor, log-log throughput vs atoms; None
    when the setting has no rows."""
    sel = select(rows, s, hosts)
    if not sel:
        return None
    codes = codes_in(sel)
    fig, axes = plot._panels(1, len(SYSTEMS), w=4.6, h=3.4)
    for ax, system in zip(axes[0], SYSTEMS):
        lines = defaultdict(list)
        for r in sel:
            if r["system"] == system:
                lines[(r["code"], r["mode"])].append((r["n_atoms"], plot.throughput(r)))
        for code in codes:
            ends = []
            for mode in MODE_STYLE:
                pts = sorted(lines.get((code, mode), []))
                if not pts:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                c = colour(code)
                ax.plot(xs, ys, color=c, ls=MODE_STYLE[mode], lw=1.8, zorder=2,
                        marker=plot._marker(code), ms=4, mfc=c, mec=plot.SURFACE, mew=0.6)
                ends.append((xs[-1], ys[-1]))
            if ends:                                   # one direct label per code
                plot._end_label(ax, *max(ends), short(code))
        _xatoms(ax)
        plot._ylog(ax)
        ax.set_title(system, fontsize=10, color=plot.INK)
        ax.set_xlabel("atoms", fontsize=8, color=plot.INK2)
    axes[0][0].set_ylabel("atom-steps / s", fontsize=8, color=plot.INK2)
    # two legend rows, filled column by column: the codes, then a column for the modes
    k = -(-len(codes) // 2)
    blank = Line2D([], [], lw=0, label=" ")
    handles = [Line2D([], [], color=colour(c), lw=2, marker=plot._marker(c), ms=5, label=label(c))
               for c in codes] + [blank] * (2 * k - len(codes))
    modes = [Line2D([], [], color=plot.MUTED, lw=2, ls=MODE_STYLE[m], label=MODE_LABEL[m])
             for m in _modes(sel)]
    handles += modes + [blank] * (2 - len(modes))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
               ncol=k + 1, fontsize=8, labelcolor=plot.INK2)
    fig.tight_layout(rect=(0, 0, 0.94, 1), w_pad=6.0)
    plot._place_labels(fig)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{s.name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def table(rows, hosts=None):
    """Markdown: atom-steps/s at exactly TABLE_N atoms, rows = evaluator x mode,
    columns = setting x system; "—" where a line has no row at TABLE_N."""
    cells = {}
    present = []
    for s in SETTINGS:
        sel = select(rows, s, hosts)
        present += sel
        for r in sel:
            if r["n_atoms"] == TABLE_N:
                cells[(r["code"], r["mode"], s.name, r["system"])] = plot.throughput(r)
    cols = [(s, system) for s in SETTINGS for system in SYSTEMS]
    out = ["| evaluator | mode | " + " | ".join(f"{s.title}: {sy}" for s, sy in cols) + " |",
           "|---|---|" + "--:|" * len(cols)]
    for code in codes_in(present):
        for mode in MODE_STYLE:
            if not any(r["code"] == code and r["mode"] == mode for r in present):
                continue
            vals = [cells.get((code, mode, s.name, sy)) for s, sy in cols]
            out.append(f"| {label(code)} | {MODE_LABEL[mode]} | "
                       + " | ".join(plot.si(v) if v else "—" for v in vals) + " |")
    return "\n".join(out)


def _hardware(rows, host):
    names = sorted({r["device_name"] for r in rows if r.get("host") == host and r.get("device_name")})
    return ", ".join(names) or HARDWARE.get(host, host)


def notes(rows, hosts=None):
    """Markdown bullets that depend on the data: the hosts, and footnotes for
    lines present only on some hosts."""
    hosts = hosts or USER_HOSTS
    present = [r for s in SETTINGS for r in select(rows, s, hosts)]
    codes = {r["code"] for r in present}
    out = [f"- **CPU:** `{hosts['cpu']}`, {_hardware(rows, hosts['cpu'])}.",
           f"- **GPU:** `{hosts['gpu']}`, {_hardware(rows, hosts['gpu'])}."]
    trim = sorted(c for c in codes if c.endswith("-trim"))
    if trim:
        out.append(f"- {', '.join(label(c) for c in trim)} evaluates the exact radial basis; "
                   "the other ACE lines evaluate splined radials.")
    if any(r.get("size") == "mh1" for r in present):
        out.append("- MACE MH-1 is multi-head and has no LAMMPS (Symmetrix) rows.")
    if "mlpace" in codes:
        out.append("- ML-PACE runs only in float64, so it is absent from the float32 figure.")
    if "mace" in codes:
        out.append("- MACE in LAMMPS (Symmetrix) evaluates in double whatever the input, so "
                   "it is shown in float64 only. A MACE line that stops early ran out of memory.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(RESULTS), help="results glob (repeats/ beside it too)")
    ap.add_argument("--figs", default=str(FIGS_DIR))
    ap.add_argument("--snippets", default=str(SNIPPETS_DIR))
    ap.add_argument("--cpu-host", default=USER_HOSTS["cpu"])
    ap.add_argument("--gpu-host", default=USER_HOSTS["gpu"])
    a = ap.parse_args(argv)
    hosts = {"cpu": a.cpu_host, "gpu": a.gpu_host}
    rows = plot.load(a.results)
    written = [figure(rows, s, a.figs, hosts) for s in SETTINGS]
    snip = pathlib.Path(a.snippets)
    snip.mkdir(parents=True, exist_ok=True)
    (snip / TABLE_SNIPPET).write_text(table(rows, hosts) + "\n")
    (snip / NOTES_SNIPPET).write_text(notes(rows, hosts) + "\n")
    written += [snip / TABLE_SNIPPET, snip / NOTES_SNIPPET]
    for p in written:
        print(p if p else "(no rows for a setting)")
    return written


if __name__ == "__main__":
    main()
