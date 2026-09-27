"""Benchmark results (JSONL) -> docs/figs/scaling_*.png + docs/benchmarks.md.

    python bench/scaling/plot.py 'bench/scaling/results/*.jsonl' docs/figs

Every figure is generated here from committed results; none is edited by hand.
Colour follows the code family in a fixed categorical order (validated palette,
dataviz skill); line style carries the mode (solid = standalone, dashed =
LAMMPS) and marker fill the ace-jax layout, so identity never rests on colour
alone.  Each figure has a legend and direct end-of-line labels, and every
number is also in the Markdown tables (the palette's contrast relief).
"""
import glob
import json
import pathlib
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots 1-4, fixed order (never cycled); ink and chrome tokens
CODES = {
    "acejax-pace": ("ace-jax (PACE model)", "#2a78d6"),
    "acejax-ace": ("ace-jax (linear ACE)", "#eb6834"),
    "mlpace": ("ML-PACE", "#1baf7a"),
    "mace": ("MACE", "#eda100"),
}
SHORT = {"acejax-pace": "ace-jax PACE", "acejax-ace": "ace-jax ACE", "mlpace": "ML-PACE",
         "mace": "MACE"}          # direct end-of-line labels: distinct, short
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SIZES = ("small", "medium", "large", "mh1")
MODES = {"standalone": "-", "lammps": "--"}


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in pathlib.Path(f).read_text().splitlines() if l.strip()]
    return rows


def throughput(r):
    """atom-steps/s: LAMMPS step time, or the standalone calculator call."""
    if r.get("atom_steps_per_s"):
        return r["atom_steps_per_s"]
    t = r.get("step_s") or r.get("call_s")
    return r["n_atoms"] / t if t else None


def _atoms_fmt(x, _pos):
    return f"{int(x)}" if x < 1024 else f"{int(x) // 1024}k"


def _count_fmt(v, _pos):
    for div, suf in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:g}{suf}"
    return f"{v:g}"


def _ylog(ax):
    """Log y with compact major labels; no minor labels (they default to mathtext)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))  # labelled within a decade
    ax.yaxis.set_major_formatter(FuncFormatter(_count_fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _xatoms(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2))
    ax.xaxis.set_major_formatter(FuncFormatter(_atoms_fmt))
    ax.margins(x=0.18)            # room for the direct end-of-line labels


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def _panels(n_rows, n_cols, w=4.2, h=3.2):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * n_cols, h * n_rows), squeeze=False,
                             facecolor=SURFACE)
    for ax in axes.flat:
        _style(ax)
    return fig, axes


def _legend(fig, codes, modes=True, layout=True):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CODES[c][1], lw=2, marker="o", ms=6, label=CODES[c][0])
               for c in codes]
    if modes:
        handles += [Line2D([], [], color=MUTED, lw=2, ls=ls, label=m) for m, ls in MODES.items()]
    if layout:
        handles += [Line2D([], [], color=MUTED, lw=0, marker="o", ms=6, mfc="none",
                           label="hollow = ace-jax sparse layout")]
    ncol = max(2, min(len(handles), int(fig.get_figwidth() // 2)))   # wrap to the figure width
    # above the figure (bbox_inches="tight" keeps it), so it can never cover a title
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False,
               fontsize=8, labelcolor=INK2)


def _end_label(ax, x, y, text):
    """Queue a direct label at a line end; `_place_labels` lays them out."""
    ax._end_labels = getattr(ax, "_end_labels", []) + [(x, y, text)]


def _place_labels(fig, min_gap_pt=9.0):
    """Draw each axis's queued end labels, pushed apart vertically so none
    overlap (sorted by height, at least min_gap_pt between neighbours)."""
    fig.canvas.draw()
    for ax in fig.axes:
        items = getattr(ax, "_end_labels", [])
        if not items:
            continue
        to_pt = ax.transData.transform
        pts = sorted(((to_pt((x, y))[1] * 72 / fig.dpi, x, y, t) for x, y, t in items))
        placed = []
        for yp, *_ in pts:
            placed.append(max(yp, placed[-1] + min_gap_pt) if placed else yp)
        # centre the stack on the lines it labels, so it cannot drift into the title
        shift = (sum(p[0] for p in pts) - sum(placed)) / len(placed)
        for yp, (y0, x, y, t) in zip(placed, pts):
            ax.annotate(t, (x, y), xytext=(5, yp + shift - y0), textcoords="offset points",
                        fontsize=7, color=INK2, va="center")


def _measures_dtype(r):
    """False for rows whose dtype label does not change what ran: Symmetrix
    (MACE in LAMMPS) evaluates in double whatever the input precision."""
    return not (r.get("code") == "mace" and r.get("mode") == "lammps"
                and r.get("dtype") == "float32")


def fig_throughput(rows, out, dtype="float64", size="medium"):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("mode") in MODES
          and r.get("dtype") == dtype and r.get("size") == size and throughput(r)
          and _measures_dtype(r)]
    if not ok:
        return None
    systems = sorted({r["system"] for r in ok})
    hosts = sorted({r["host"] for r in ok})
    fig, axes = _panels(len(systems), len(hosts))
    for i, system in enumerate(systems):
        for j, host in enumerate(hosts):
            ax = axes[i][j]
            lines = defaultdict(list)
            for r in ok:
                if r["system"] == system and r["host"] == host:
                    lines[(r["code"], r["mode"])].append((r["n_atoms"], throughput(r),
                                                          r.get("layout")))
            for (code, mode), pts in sorted(lines.items()):
                pts.sort()
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, color=CODES[code][1], ls=MODES[mode], lw=1.6, zorder=2)
                for x, y, lay in pts:                  # hollow marker = sparse layout
                    ax.plot(x, y, marker="o", ms=5, color=CODES[code][1], zorder=3,
                            mfc="none" if lay == "sparse" else CODES[code][1])
                if mode == "standalone" or code == "mlpace":
                    _end_label(ax, xs[-1], ys[-1], SHORT[code])
            _xatoms(ax)
            _ylog(ax)
            ax.set_title(f"{system} · {host}", fontsize=9, color=INK)
            if i == len(systems) - 1:
                ax.set_xlabel("atoms", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel("atom-steps / s", fontsize=8, color=INK2)
    _legend(fig, [c for c in CODES if any(r["code"] == c for r in ok)])
    fig.tight_layout(rect=(0, 0, 0.94, 1), w_pad=3.0)
    _place_labels(fig)
    p = out / f"scaling_throughput_{dtype}_{size}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")     # keeps the figure legend
    plt.close(fig)
    return p


def _at_fixed_n(rows, host, n_target):
    """Rows closest to n_target atoms per (code, mode, size, system)."""
    best = {}
    for r in rows:
        if r.get("host") != host:
            continue
        k = (r["code"], r["mode"], r["size"], r["system"])
        if k not in best or abs(r["n_atoms"] - n_target) < abs(best[k]["n_atoms"] - n_target):
            best[k] = r
    return best


def fig_model_size(rows, out, dtype="float64"):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("mode") in MODES
          and r.get("dtype") == dtype and throughput(r)]
    if not ok:
        return None
    hosts = sorted({r["host"] for r in ok})
    systems = sorted({r["system"] for r in ok})
    fig, axes = _panels(len(systems), len(hosts))
    for j, host in enumerate(hosts):
        n_target = 2048 if "cpu" in host else 8192
        best = _at_fixed_n(ok, host, n_target)
        for i, system in enumerate(systems):
            ax = axes[i][j]
            for code in CODES:
                for mode, ls in MODES.items():
                    pts = [(SIZES.index(s), throughput(best[(code, mode, s, system)]))
                           for s in SIZES if (code, mode, s, system) in best]
                    if pts:
                        ax.plot(*zip(*pts), color=CODES[code][1], ls=ls, lw=1.6, marker="o", ms=5)
            _ylog(ax)
            ax.set_xticks(range(len(SIZES)), SIZES)
            ax.set_title(f"{system} · {host} · N≈{n_target}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("atom-steps / s", fontsize=8, color=INK2)
    _legend(fig, [c for c in CODES if any(r["code"] == c for r in ok)], layout=False)
    fig.tight_layout(rect=(0, 0, 1, 1))
    p = out / f"scaling_model_size_{dtype}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")     # keeps the figure legend
    plt.close(fig)
    return p


def fig_memory(rows, out, dtype="float64", size="medium"):
    """Peak device memory vs N (standalone), one panel per system x GPU host;
    dotted verticals mark the first size that did not fit, hollow markers the
    sizes where ace-jax chose the sparse layout."""
    sel = [r for r in rows if r.get("mode") == "standalone" and r.get("dtype") == dtype
           and r.get("size") == size and r.get("peak_bytes") is not None]
    if not sel:
        return None
    hosts = sorted({r["host"] for r in sel if r.get("device") == "gpu"}) or sorted({r["host"] for r in sel})
    systems = sorted({r["system"] for r in sel})
    fig, axes = _panels(len(systems), len(hosts))
    drawn = set()
    for i, system in enumerate(systems):
        for j, host in enumerate(hosts):
            ax = axes[i][j]
            lines = defaultdict(list)
            for r in sel:
                if r["host"] == host and r["system"] == system:
                    lines[r["code"]].append((r["n_atoms"], r["peak_bytes"] / 1e9, r.get("status"),
                                             r.get("layout")))
            for code, pts in sorted(lines.items()):
                pts.sort()
                good = [p for p in pts if p[2] == "ok" and p[1] > 0]
                if good:
                    drawn.add(code)
                    ax.plot([p[0] for p in good], [p[1] for p in good], color=CODES[code][1], lw=1.6)
                    for x, y, _, lay in good:
                        ax.plot(x, y, marker="o", ms=5, color=CODES[code][1],
                                mfc="none" if lay == "sparse" else CODES[code][1])
                    _end_label(ax, good[-1][0], good[-1][1], SHORT[code])
                for x, _, st, _ in pts:
                    if st == "oom":                        # the size that did not fit
                        ax.axvline(x, color=CODES[code][1], ls=":", lw=1)
            _xatoms(ax)
            _ylog(ax)
            ax.set_title(f"{system} · {host}", fontsize=9, color=INK)
            if i == len(systems) - 1:
                ax.set_xlabel("atoms", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel("peak memory (GB)", fontsize=8, color=INK2)
    _legend(fig, [c for c in CODES if c in drawn], modes=False)
    fig.tight_layout(rect=(0, 0, 0.94, 1), w_pad=3.0)
    _place_labels(fig)
    p = out / f"scaling_memory_{dtype}_{size}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")     # keeps the figure legend
    plt.close(fig)
    return p


def fig_precision(rows, out, size="medium"):
    """f32 / f64 throughput ratio vs N (standalone), one panel per system x host."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("mode") == "standalone"
          and r.get("size") == size and throughput(r)]
    by = defaultdict(dict)
    for r in ok:
        by[(r["host"], r["code"], r["system"], r["n_atoms"])][r["dtype"]] = throughput(r)
    ratio = defaultdict(list)
    for (host, code, system, n), d in by.items():
        if "float32" in d and "float64" in d:
            ratio[(host, system, code)].append((n, d["float32"] / d["float64"]))
    if not ratio:
        return None
    hosts = sorted({h for h, _, _ in ratio})
    systems = sorted({s for _, s, _ in ratio})
    fig, axes = _panels(len(systems), len(hosts))
    for i, system in enumerate(systems):
        for j, host in enumerate(hosts):
            ax = axes[i][j]
            for (h, s_, code), pts in sorted(ratio.items()):
                if h != host or s_ != system:
                    continue
                pts.sort()
                ax.plot(*zip(*pts), color=CODES[code][1], lw=1.6, marker="o", ms=5)
            ax.axhline(1.0, color=AXIS, lw=1)
            _xatoms(ax)
            ax.set_title(f"{system} · {host}", fontsize=9, color=INK)
            if i == len(systems) - 1:
                ax.set_xlabel("atoms", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel("f32 / f64 throughput", fontsize=8, color=INK2)
    _legend(fig, sorted({c for _, _, c in ratio}, key=list(CODES).index), modes=False, layout=False)
    fig.tight_layout(rect=(0, 0, 1, 1))
    p = out / f"scaling_precision_{size}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")     # keeps the figure legend
    plt.close(fig)
    return p


def tables(rows):
    """Markdown: throughput at fixed N per host (the table view) and compile times."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("mode") in MODES and throughput(r)]
    out = ["| host | system | code | mode | size | dtype | atoms | atom-steps/s |",
           "|---|---|---|---|---|---|---|---|"]
    for host in sorted({r["host"] for r in ok}):
        best = _at_fixed_n([r for r in ok if r["dtype"] == "float64"], host,
                           2048 if "cpu" in host else 8192)
        for (code, mode, size, system), r in sorted(best.items()):
            out.append(f"| {host} | {system} | {CODES[code][0]} | {mode} | {size} | float64 "
                       f"| {r['n_atoms']} | {throughput(r):.3g} |")
    comp = defaultdict(list)
    for r in ok:
        if r.get("compile_s"):
            comp[(r["host"], r["code"], r["mode"])].append(r["compile_s"])
    out += ["", "| host | code | mode | median compile / export (s) |", "|---|---|---|---|"]
    for (host, code, mode), v in sorted(comp.items()):
        out.append(f"| {host} | {CODES[code][0]} | {mode} | {statistics.median(v):.1f} |")
    return "\n".join(out)


def make_figures(pattern, outdir):
    rows = load(pattern)
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figs = [fig_throughput(rows, out, dt) for dt in ("float64", "float32")]
    figs += [fig_model_size(rows, out), fig_memory(rows, out), fig_precision(rows, out)]
    return [str(f) for f in figs if f]


def write_doc(pattern, figs, doc="docs/benchmarks.md"):
    rows = load(pattern)
    versions = {}
    for r in rows:
        if r.get("versions"):
            versions[r["host"]] = r["versions"]
    body = ["# Benchmarks", "",
            "Generated by `bench/scaling/plot.py` from `bench/scaling/results/*.jsonl`",
            "(spec: `docs/benchmark-scaling-spec.md`). Standalone = one ASE calculator call",
            "(energy + forces + stress, neighbour list included); LAMMPS = MD step time.",
            "Random-coefficient models are used where no fitted model exists; they are",
            "timing-only. Hollow markers: ace-jax chose the sparse layout.", ""]
    body += [f"![{pathlib.Path(f).stem}]({pathlib.Path(f).relative_to(pathlib.Path(doc).parent) if pathlib.Path(f).is_relative_to(pathlib.Path(doc).parent) else f})" for f in figs]
    body += ["", "## Tables", "", tables(rows), "", "## Versions", ""]
    body += [f"- **{h}**: " + ", ".join(f"{k} {v}" for k, v in sorted(v.items())) for h, v in sorted(versions.items())]
    pathlib.Path(doc).write_text("\n".join(body) + "\n")
    return doc


if __name__ == "__main__":
    pattern, outdir = sys.argv[1], sys.argv[2]
    figs = make_figures(pattern, outdir)
    print("\n".join(figs))
    print(write_doc(pattern, figs))
