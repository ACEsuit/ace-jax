"""Benchmark results (JSONL) -> docs/dev/figs/scaling_*.png + docs/dev/benchmarks.md.

    python bench/scaling/plot.py 'bench/scaling/results/*.jsonl' docs/dev/figs

Every figure is generated here from committed results; none is edited by hand.
The ace-jax rows from before the speed-ups are read from the `before-perf/`
directory beside the pattern, for the before/after figures; the findings and
parity numbers in the page's intro are filled in from the rows as well.
Colour follows the code family in a fixed categorical order (validated palette,
dataviz skill); line style carries the mode (solid = standalone, dashed =
LAMMPS) and marker fill the ace-jax layout, so identity never rests on colour
alone.  Each figure has a legend and direct end-of-line labels, and every
number is also in the Markdown tables (the palette's contrast relief).

A case run more than once (repeat runs in `repeats/` beside the rows, e.g.
separate Modal containers) is drawn at the median of its ok runs, with a thin
min-max bar in the line's colour; a single run has no bar.
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

# categorical slots 1-6, fixed order (never cycled; a new code takes the next
# slot); ink and chrome tokens.  Slots 1-6 pass the dataviz validator on
# adjacent pairs (light: CVD dE >= 9.1, normal >= 19.6).  Slots 2, 5 and 6
# (stock ACE and the two learned-radial lines) are drawn together in the
# learned-radial figure and do not clear the all-pairs floors (orange-magenta
# normal-vision dE 12.9), so those lines also differ by marker (MARKERS) and
# carry direct labels.
CODES = {
    "acejax-pace": ("ace-jax (PACE model)", "#2a78d6"),
    "acejax-ace": ("ace-jax (linear ACE)", "#eb6834"),
    "mlpace": ("ML-PACE", "#1baf7a"),
    "mace": ("MACE", "#eda100"),
    "acejax-ace-learned": ("ace-jax (linear ACE, learned radial, splined)", "#e87ba4"),
    "acejax-ace-analytic": ("ace-jax (linear ACE, learned radial, analytic)", "#008300"),
}
SHORT = {"acejax-pace": "ace-jax PACE", "acejax-ace": "ace-jax ACE", "mlpace": "ML-PACE",
         "mace": "MACE", "acejax-ace-learned": "ace-jax ACE, learned (splined)",
         "acejax-ace-analytic": "ace-jax ACE, learned (analytic)"}   # direct end-of-line labels
MARKERS = {"acejax-ace-learned": "s", "acejax-ace-analytic": "^"}    # the rest: "o"
LEARNED = ("acejax-ace-learned", "acejax-ace-analytic")


def _marker(code):
    return MARKERS.get(code, "o")


BASIS_JSON = pathlib.Path(__file__).with_name("model_sizes.json")


def _load_basis(path=BASIS_JSON):
    """Basis functions per central element, by "<system>/<size>" and code
    (`models.py sizes` writes it); empty when absent."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


BASIS = _load_basis()


def _label(code, system, size):
    """End-of-line label, with the model's basis size where known."""
    n = BASIS.get(f"{system}/{size}", {}).get(code)
    return SHORT[code] + (f" · {n} fn" if n else "")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SIZES = ("small", "medium", "large", "mh1")
MODES = {"standalone": "-", "lammps": "--"}


REPEATS_DIR = "repeats"              # repeat runs of cases, beside the rows they repeat


def load(pattern):
    """The rows the pattern matches, plus the repeat runs in `repeats/` beside
    them (the same file-name pattern); `before_pattern` rows get theirs from
    `before-perf/repeats/` the same way."""
    p = pathlib.Path(pattern)
    files = set(glob.glob(pattern)) | set(glob.glob(str(p.parent / REPEATS_DIR / p.name)))
    rows = []
    for f in sorted(files):
        rows += [json.loads(l) for l in pathlib.Path(f).read_text().splitlines() if l.strip()]
    for r in rows:
        # rows written for a case that died (sweep.row_from_process) carry the
        # model name "<code>/<system>/<size>" but not the parsed fields
        parts = str(r.get("model", "")).split("/")
        if len(parts) == 3:
            r.setdefault("system", parts[1])
            r.setdefault("size", parts[2])
    return rows


def throughput(r):
    """atom-steps/s: LAMMPS step time, or the standalone calculator call (the
    median over runs for a case `aggregate` merged)."""
    if "_tp" in r:
        return r["_tp"]
    if r.get("atom_steps_per_s"):
        return r["atom_steps_per_s"]
    t = r.get("step_s") or r.get("call_s")
    return r["n_atoms"] / t if t else None


_IDENTITY = ("host", "code", "mode", "model", "system", "size", "n_atoms", "dtype", "device", "gate")


def case_id(r):
    """What makes two rows the same case: host and `_key` (model, mode, N,
    dtype, device), so runs differing only in timings, versions or `run`
    group together. Rows without `_key` (synthetic ones) use those fields."""
    if r.get("_key") is not None:
        return (r.get("host"), json.dumps(r["_key"]))
    return tuple(json.dumps(r.get(f)) for f in _IDENTITY)


def aggregate(rows):
    """One row per case. A case run more than once keeps its ok runs only (an
    ok run beats an out-of-memory one) and becomes a copy of its median run
    with the median throughput and peak memory and the min-max throughput
    (`spread`). A case with a single run, or no ok run, is left as it was."""
    groups = defaultdict(list)
    for r in rows:
        groups[case_id(r)].append(r)
    out = []
    for rs in groups.values():
        ok = [r for r in rs if r.get("status") == "ok"]
        timed = sorted((r for r in ok if throughput(r)), key=throughput)
        if len(timed) < 2:
            out.append(timed[0] if timed else ok[0] if ok else rs[0])
            continue
        tp = [throughput(r) for r in timed]
        row = dict(timed[(len(timed) - 1) // 2])
        row.update(_tp=statistics.median(tp), _tp_range=(tp[0], tp[-1]), _runs=len(timed))
        mem = [r["peak_bytes"] for r in timed if r.get("peak_bytes") is not None]
        if mem:
            row["peak_bytes"] = statistics.median(mem)
        out.append(row)
    return out


def spread(r):
    """(min, max) throughput over a case's runs, or None for a single run."""
    return r.get("_tp_range")


def spread_pct(r):
    """' ±x%' (half the min-max range over the median) for a repeated case."""
    rng = spread(r)
    return f" ±{(rng[1] - rng[0]) / 2 / throughput(r) * 100:.0f}%" if rng else ""


def _bar(ax, x, y, rng, color):
    """A thin min-max bar in the line's colour; nothing for a single run."""
    if rng:
        ax.errorbar([x], [y], yerr=[[y - rng[0]], [rng[1] - y]], fmt="none", ecolor=color,
                    elinewidth=0.9, capsize=0, alpha=0.85, zorder=2.5)


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
    handles = [Line2D([], [], color=CODES[c][1], lw=2, marker=_marker(c), ms=6, label=CODES[c][0])
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
    ok = [r for r in aggregate(rows) if r.get("status") == "ok" and r.get("mode") in MODES
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
                                                          r.get("layout"), spread(r)))
            for (code, mode), pts in sorted(lines.items()):
                pts.sort()
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, color=CODES[code][1], ls=MODES[mode], lw=1.6, zorder=2)
                for x, y, lay, rng in pts:             # hollow marker = sparse layout
                    _bar(ax, x, y, rng, CODES[code][1])
                    ax.plot(x, y, marker=_marker(code), ms=5, color=CODES[code][1], zorder=3,
                            mfc="none" if lay == "sparse" else CODES[code][1])
                if mode == "standalone" or code == "mlpace":
                    _end_label(ax, xs[-1], ys[-1], _label(code, system, size))
            _xatoms(ax)
            _ylog(ax)
            ax.set_title(f"{system} · {host}", fontsize=9, color=INK)
            if i == len(systems) - 1:
                ax.set_xlabel("atoms", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel("atom-steps / s", fontsize=8, color=INK2)
    _legend(fig, [c for c in CODES if any(r["code"] == c for r in ok)])
    fig.tight_layout(rect=(0, 0, 0.94, 1), w_pad=7.0)
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


def _size_tick(system, size):
    """Model-size tick: the size name, then the basis functions per central
    element of the PACE and linear ACE models where known."""
    b = BASIS.get(f"{system}/{size}", {})
    fn = [f"{k} {b[c]}" for c, k in (("acejax-pace", "PACE"), ("acejax-ace", "ACE")) if c in b]
    return "\n".join([size, *fn])


def fig_model_size(rows, out, dtype="float64"):
    ok = [r for r in aggregate(rows) if r.get("status") == "ok" and r.get("mode") in MODES
          and r.get("dtype") == dtype and throughput(r)]
    if not ok:
        return None
    hosts = sorted({r["host"] for r in ok})
    systems = sorted({r["system"] for r in ok})
    fig, axes = _panels(len(systems), len(hosts))
    for j, host in enumerate(hosts):
        n_target = 2048 if "cpu" in host else 8192
        # exactly N: a line that did not reach N (out of memory) is left out
        # rather than drawn from a smaller N, where GPU throughput is lower
        best = {k: r for k, r in _at_fixed_n(ok, host, n_target).items()
                if r["n_atoms"] == n_target}
        for i, system in enumerate(systems):
            ax = axes[i][j]
            for code in CODES:
                for mode, ls in MODES.items():
                    got = [(SIZES.index(s), best[(code, mode, s, system)])
                           for s in SIZES if (code, mode, s, system) in best]
                    if got:
                        ax.plot([x for x, _ in got], [throughput(r) for _, r in got],
                                color=CODES[code][1], ls=ls, lw=1.6, marker=_marker(code), ms=5)
                        for x, r in got:
                            _bar(ax, x, throughput(r), spread(r), CODES[code][1])
            _ylog(ax)
            ax.set_xticks(range(len(SIZES)), [_size_tick(system, s) for s in SIZES],
                          fontsize=7)
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
    sel = [r for r in aggregate(rows) if r.get("mode") == "standalone" and r.get("dtype") == dtype
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
                    _end_label(ax, good[-1][0], good[-1][1], _label(code, system, size))
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
    fig.tight_layout(rect=(0, 0, 0.94, 1), w_pad=7.0)
    _place_labels(fig)
    p = out / f"scaling_memory_{dtype}_{size}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")     # keeps the figure legend
    plt.close(fig)
    return p


def fig_precision(rows, out, size="medium"):
    """f32 / f64 throughput ratio vs N (standalone), one panel per system x host."""
    ok = [r for r in aggregate(rows) if r.get("status") == "ok" and r.get("mode") == "standalone"
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


BEFORE_DIR = "before-perf"            # ace-jax rows from before the speed-ups
ACEJAX = ("acejax-pace", "acejax-ace")
PHASES = {"after": "-", "before": "--", "reference": "-"}


def before_pattern(pattern):
    """The glob for the pre-speed-up rows kept beside the live results."""
    p = pathlib.Path(pattern)
    return str(p.parent / BEFORE_DIR / p.name)


def before_after_series(after, before, host, dtype="float64", size="medium"):
    """{(system, mode, code, phase): [(n, atom-steps/s, layout, spread)]} for one host:
    ace-jax "before" (from before-perf/) and "after", and ML-PACE in LAMMPS as
    the "reference". Empty when the host has no re-run ace-jax rows yet, so a
    host still being measured is skipped rather than drawn from old rows."""
    def pick(rows, codes, modes):
        return [r for r in aggregate(rows) if r.get("host") == host and r.get("status") == "ok"
                and r.get("code") in codes and r.get("mode") in modes
                and r.get("dtype") == dtype and r.get("size") == size and throughput(r)]
    aft = pick(after, ACEJAX, MODES)
    if not aft:
        return {}
    modes = {r["mode"] for r in aft}          # CPU: standalone only (lammps-jax is GPU-only)
    out = defaultdict(list)
    for phase, sel in (("after", aft), ("before", pick(before, ACEJAX, modes)),
                       ("reference", pick(after, ("mlpace",), ("lammps",)))):
        for r in sel:
            for mode in (modes if phase == "reference" else (r["mode"],)):
                out[(r["system"], mode, r["code"], phase)].append(
                    (r["n_atoms"], throughput(r), r.get("layout"), spread(r)))
    return {k: sorted(v) for k, v in out.items()}


def before_after_hosts(after, before):
    """(hosts with re-run ace-jax rows, hosts whose re-run is pending)."""
    def hosts(rows):
        return {r["host"] for r in rows if r.get("code") in ACEJAX and r.get("mode") in MODES
                and r.get("status") == "ok"}
    done = hosts(after)
    return sorted(done), sorted(hosts(before) - done)


def fig_before_after(after, before, out, host, dtype="float64", size="medium"):
    """ace-jax throughput vs N before (dashed) and after (solid) the speed-ups,
    one panel per system x mode, with ML-PACE in LAMMPS for reference."""
    series = before_after_series(after, before, host, dtype, size)
    if not series:
        return None
    systems = sorted({k[0] for k in series})
    modes = [m for m in MODES if any(k[1] == m for k in series)]
    fig, axes = _panels(len(systems), len(modes))
    for i, system in enumerate(systems):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            for (s, m, code, phase), pts in sorted(series.items()):
                if s != system or m != mode:
                    continue
                c = CODES[code][1]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, color=c, ls=PHASES[phase], lw=1.2 if phase == "reference" else 1.6,
                        zorder=2)
                for x, y, lay, rng in pts:             # hollow marker = sparse layout
                    _bar(ax, x, y, rng, c)
                    ax.plot(x, y, marker="o", ms=4 if phase == "before" else 5, color=c, zorder=3,
                            mfc="none" if lay == "sparse" else c)
                if phase != "before":
                    _end_label(ax, xs[-1], ys[-1], _label(code, system, size))
            _xatoms(ax)
            _ylog(ax)
            ax.set_title(f"{system} · ace-jax {mode} · {host}", fontsize=9, color=INK)
            if i == len(systems) - 1:
                ax.set_xlabel("atoms", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel("atom-steps / s", fontsize=8, color=INK2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=CODES[c][1], lw=2, marker="o", ms=6,
                      label=CODES[c][0] + (" in LAMMPS (reference)" if c == "mlpace" else ""))
               for c in (*ACEJAX, "mlpace") if any(k[2] == c for k in series)]
    handles += [Line2D([], [], color=MUTED, lw=2, ls="-", label="after the speed-ups"),
                Line2D([], [], color=MUTED, lw=2, ls="--", label="before"),
                Line2D([], [], color=MUTED, lw=0, marker="o", ms=6, mfc="none",
                       label="hollow = ace-jax sparse layout")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=max(2, min(len(handles), int(fig.get_figwidth() // 2))), frameon=False,
               fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 0.94, 1), w_pad=7.0)
    _place_labels(fig)
    p = out / f"scaling_before_after_{dtype}_{size}_{host}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")     # keeps the figure legend
    plt.close(fig)
    return p


def learned_hosts(rows, dtype="float64"):
    """Hosts with ok learned-radial rows (the learned-radial figure's hosts)."""
    return sorted({r["host"] for r in rows if r.get("code") in LEARNED and r.get("mode") in MODES
                   and r.get("status") == "ok" and r.get("dtype") == dtype and r.get("host")})


def fig_learned(rows, out, host, dtype="float64", size="medium"):
    """The cost of a learned radial and what splining recovers: throughput vs N
    of the stock (spline) linear ACE model and its learned-radial proxy,
    splined as deployed and kept analytic, one panel per system x mode, on one
    host.  None when the host has no learned-radial rows yet (pending)."""
    codes = ("acejax-ace", *LEARNED)
    ok = [r for r in aggregate(rows) if r.get("host") == host and r.get("code") in codes
          and r.get("status") == "ok" and r.get("mode") in MODES and r.get("dtype") == dtype
          and r.get("size") == size and throughput(r)]
    if not any(r["code"] in LEARNED for r in ok):
        return None
    systems = sorted({r["system"] for r in ok if r["code"] in LEARNED})
    modes = [m for m in MODES if any(r["mode"] == m and r["code"] in LEARNED for r in ok)]
    fig, axes = _panels(len(systems), len(modes))
    for i, system in enumerate(systems):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            for code in codes:
                pts = sorted((r["n_atoms"], throughput(r), r.get("layout"), spread(r)) for r in ok
                             if r["code"] == code and r["system"] == system and r["mode"] == mode)
                if not pts:
                    continue
                c = CODES[code][1]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, color=c, ls=MODES[mode], lw=1.6, zorder=2)
                for x, y, lay, rng in pts:             # hollow marker = sparse layout
                    _bar(ax, x, y, rng, c)
                    ax.plot(x, y, marker=_marker(code), ms=5, color=c, zorder=3,
                            mfc="none" if lay == "sparse" else c)
                _end_label(ax, xs[-1], ys[-1], SHORT[code])
            _xatoms(ax)
            _ylog(ax)
            ax.set_title(f"{system} · ace-jax {mode} · {host}", fontsize=9, color=INK)
            if i == len(systems) - 1:
                ax.set_xlabel("atoms", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel("atom-steps / s", fontsize=8, color=INK2)
    _legend(fig, [c for c in codes if any(r["code"] == c for r in ok)], modes=len(modes) > 1)
    fig.tight_layout(rect=(0, 0, 0.94, 1), w_pad=7.0)
    _place_labels(fig)
    p = out / f"scaling_learned_radial_{dtype}_{size}_{host}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")     # keeps the figure legend
    plt.close(fig)
    return p


def tables(rows):
    """Markdown: throughput at fixed N per host (the table view) and compile times."""
    ok = [r for r in aggregate(rows) if r.get("status") == "ok" and r.get("mode") in MODES
          and throughput(r)]
    out = ["| host | system | code | mode | size | dtype | atoms | atom-steps/s |",
           "|---|---|---|---|---|---|---|---|"]
    for host in sorted({r["host"] for r in ok}):
        best = _at_fixed_n([r for r in ok if r["dtype"] == "float64"], host,
                           2048 if "cpu" in host else 8192)
        for (code, mode, size, system), r in sorted(best.items()):
            out.append(f"| {host} | {system} | {CODES[code][0]} | {mode} | {size} | float64 "
                       f"| {r['n_atoms']} | {throughput(r):.3g}{spread_pct(r)} |")
    comp = defaultdict(list)
    for r in ok:
        if r.get("compile_s"):
            comp[(r["host"], r["code"], r["mode"])].append(r["compile_s"])
    out += ["", "| host | code | mode | median compile / export (s) |", "|---|---|---|---|"]
    for (host, code, mode), v in sorted(comp.items()):
        out.append(f"| {host} | {CODES[code][0]} | {mode} | {statistics.median(v):.1f} |")
    return "\n".join(out)


def largest_fits(rows, size="medium", dtype="float64"):
    """Markdown: per host, system, code and mode, the largest size that ran and
    the first size that did not fit (oom), for one model size and dtype."""
    lines = defaultdict(list)
    for r in aggregate(rows):
        if r.get("mode") in MODES and r.get("size") == size and r.get("dtype") == dtype:
            lines[(r["host"], r["system"], r["code"], r["mode"])].append(r)
    out = ["| host | system | code | mode | largest that ran | first out of memory |",
           "|---|---|---|---|---|---|"]
    for (host, system, code, mode), rs in sorted(lines.items()):
        ok = [r["n_atoms"] for r in rs if r["status"] == "ok"]
        oom = [r["n_atoms"] for r in rs if r["status"] == "oom"]
        if not ok:
            continue
        out.append(f"| {host} | {system} | {CODES[code][0]} | {mode} | {max(ok)} "
                   f"| {min(oom) if oom else '—'} |")
    return "\n".join(out)


def target_n(host):
    """The fixed size compared across codes: 8,192 atoms on GPU, 2,048 on CPU."""
    return 2048 if "cpu" in host else 8192


def si(v):
    """Compact atom-steps/s: 346k, 1.58M."""
    return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.0f}k" if v >= 1e3 else f"{v:.0f}"


def _rng(vals, fmt):
    """'a–b' over the values (one value when they format the same)."""
    lo, hi = fmt(min(vals)), fmt(max(vals))
    return lo if lo == hi else f"{lo}–{hi}"


def line_points(rows, host, code, mode, dtype="float64", size="medium"):
    """{system: {n: atom-steps/s}} for one line's ok rows on a host."""
    out = defaultdict(dict)
    for r in aggregate(rows):
        if (r.get("host") == host and r.get("code") == code and r.get("mode") == mode
                and r.get("dtype") == dtype and r.get("size") == size
                and r.get("status") == "ok" and throughput(r)):
            out[r["system"]][r["n_atoms"]] = throughput(r)
    return out


def _ratio_cell(rows, host, num, den):
    """num / den throughput per system, at the host's target N, or at the
    largest N both ran below it (annotated). num, den: (code, mode, dtype)."""
    a, b, n0 = line_points(rows, host, *num), line_points(rows, host, *den), target_n(host)
    got = []
    for s in sorted(set(a) & set(b)):
        common = [n for n in set(a[s]) & set(b[s]) if n <= n0]
        if common:
            n = max(common)
            got.append((n, a[s][n] / b[s][n]))
    if not got:
        return "—"
    cell = _rng([g[1] for g in got], lambda v: f"{v:.2g}") + "×"
    below = sorted({n for n, _ in got if n != n0})
    return cell + (f" (at {', '.join(map(str, below))})" if below else "")


def _value_cell(rows, host, code, mode):
    pts = line_points(rows, host, code, mode)
    vals = [p[target_n(host)] for p in pts.values() if target_n(host) in p]
    return _rng(vals, si) if vals else "—"


def _largest_cell(rows, host, code):
    pts = line_points(rows, host, code, "standalone")
    return _rng([max(p) for p in pts.values()], str) if pts else "—"


def findings(rows):
    """Markdown: the numbers behind the benchmarks page's findings, recomputed
    from the rows on every render (medium models, ranges over the two systems).
    A host whose ace-jax rows are still being re-run shows "pending"."""
    hosts = sorted({r["host"] for r in rows if r.get("mode") in MODES and r.get("status") == "ok"})
    has_acejax = {r["host"] for r in rows if r.get("code") in ACEJAX and r.get("mode") in MODES}
    out = ["ML-PACE in LAMMPS against ace-jax PACE on the same `.yace` models (float64, "
           "atom-steps/s):", "",
           "| host | N | ML-PACE in LAMMPS | ace-jax standalone | ace-jax in LAMMPS "
           "| ML-PACE ÷ ace-jax standalone | ML-PACE ÷ ace-jax in LAMMPS |",
           "|---|---|---|---|---|---|---|"]
    ml = ("mlpace", "lammps", "float64")
    for h in hosts:
        if h not in has_acejax:
            out.append(f"| {h} | {target_n(h)} | {_value_cell(rows, h, 'mlpace', 'lammps')} "
                       "| pending | pending | pending | pending |")
            continue
        out.append(f"| {h} | {target_n(h)} | {_value_cell(rows, h, 'mlpace', 'lammps')} "
                   f"| {_value_cell(rows, h, 'acejax-pace', 'standalone')} "
                   f"| {_value_cell(rows, h, 'acejax-pace', 'lammps')} "
                   f"| {_ratio_cell(rows, h, ml, ('acejax-pace', 'standalone', 'float64'))} "
                   f"| {_ratio_cell(rows, h, ml, ('acejax-pace', 'lammps', 'float64'))} |")
    out += ["", "ace-jax against MACE, same mode (float64 throughput ratio; where MACE ran out "
            "of memory at N, at the largest size both ran):", "",
            "| host | mode | ace-jax PACE ÷ MACE | ace-jax ACE ÷ MACE |", "|---|---|---|---|"]
    for h in hosts:
        for mode in MODES:
            if mode == "lammps" and "cpu" in h:     # lammps-jax is GPU-only
                continue
            if h not in has_acejax:
                out.append(f"| {h} | {mode} | pending | pending |")
                continue
            cells = [_ratio_cell(rows, h, (c, mode, "float64"), ("mace", mode, "float64"))
                     for c in ACEJAX]
            if any(c != "—" for c in cells):
                out.append(f"| {h} | {mode} | {cells[0]} | {cells[1]} |")
    out += ["", "Largest system that ran standalone (float64, atoms), and the float32 / float64 "
            "throughput ratio at N:", "",
            "| host | largest: ace-jax PACE | ace-jax ACE | MACE "
            "| f32 ÷ f64: ace-jax PACE | ace-jax ACE | MACE |", "|---|---|---|---|---|---|---|"]
    for h in hosts:
        pend = h not in has_acejax
        big = ["pending" if pend and c in ACEJAX else _largest_cell(rows, h, c)
               for c in (*ACEJAX, "mace")]
        f32 = ["pending" if pend and c in ACEJAX else
               _ratio_cell(rows, h, (c, "standalone", "float32"), (c, "standalone", "float64"))
               for c in (*ACEJAX, "mace")]
        out.append(f"| {h} | " + " | ".join(big + f32) + " |")
    return "\n".join(out)


def parity_table(rows):
    """Markdown: the parity gates per host (gate x code): passed / run, and the
    largest energy-per-atom and force differences. No rows: pending."""
    by = defaultdict(list)
    for r in rows:
        if r.get("mode") == "parity" and str(r.get("status", "")).startswith("parity_"):
            by[(r["host"], r.get("gate", r["code"]), r["code"])].append(r)
    if not by:
        return "No parity rows yet (pending)."
    out = ["| host | gate | code | passed | max abs dE / atom (eV) | max abs dF (eV/Å) |",
           "|---|---|---|---|---|---|"]
    for (h, gate, code), rs in sorted(by.items()):
        ok = sum(r["status"] == "parity_ok" for r in rs)
        de = max(abs(r.get("dE_per_atom") or 0.0) for r in rs)
        df = max(abs(r.get("max_dF") or 0.0) for r in rs)
        out.append(f"| {h} | {gate} | {CODES[code][0]} | {ok}/{len(rs)} | {de:.1e} | {df:.1e} |")
    timed = {r["host"] for r in rows if r.get("mode") in MODES}
    for h in sorted(timed - {k[0] for k in by}):
        out.append(f"| {h} | — | — | pending | | |")
    return "\n".join(out)


REPEAT_NOTE = ("Where a case was run more than once (Modal: separate containers), the point is "
               "the median and the bar spans min–max.")

CAPTIONS = {
    "scaling_before_after": "ace-jax throughput before (dashed) and after (solid) the "
                            "speed-ups, float64, medium models; ML-PACE in LAMMPS for "
                            "reference. Before rows: `bench/scaling/results/before-perf/`. "
                            "“fn”: basis functions per central element.",
    "scaling_learned_radial": "The cost of a learned radial, and what splining recovers "
                              "(float64, medium linear ACE): the stock model (spline radial), "
                              "its learned-radial proxy splined as deployed (`spline_tol=\"auto\"`, "
                              "1e-10) and the same proxy kept analytic (`spline_tol=None`, exact). "
                              "Solid = standalone, dashed = LAMMPS; see "
                              "`docs/dev/learned-radial-splining.md`.",
    "scaling_throughput_float64": "Throughput vs system size (float64, medium models): "
                                  "solid = standalone, dashed = LAMMPS. “fn”: basis functions per "
                                  "central element (linear ACE is 2–14× the PACE size).",
    "scaling_throughput_float32": "The same in float32. ML-PACE and Symmetrix (MACE in "
                                  "LAMMPS) evaluate in double, so they are absent.",
    "scaling_model_size": "Throughput vs model size at exactly 8,192 atoms on GPU and "
                          "2,048 on CPU; a line is absent at a size that did not fit. Ticks give "
                          "basis functions per central element.",
    "scaling_memory": "Peak device memory vs system size (standalone); dotted verticals "
                      "mark the first size that did not fit.",
    "scaling_precision": "float32 / float64 throughput ratio (standalone).",
}


def _caption(stem):
    return next((c for k, c in CAPTIONS.items() if stem.startswith(k)), "")


def make_figures(pattern, outdir):
    rows = load(pattern)
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figs = [fig_throughput(rows, out, dt) for dt in ("float64", "float32")]
    figs += [fig_model_size(rows, out), fig_memory(rows, out), fig_precision(rows, out)]
    before = load(before_pattern(pattern))
    figs += [fig_before_after(rows, before, out, h) for h in before_after_hosts(rows, before)[0]]
    figs += [fig_learned(rows, out, h) for h in learned_hosts(rows)]
    return [str(f) for f in figs if f]


def basis_table():
    """Basis functions per central element, PACE (ace-jax and ML-PACE run the
    same .yace) against linear ACE, with their ratio."""
    lines = ["| system | size | PACE | linear ACE | ACE / PACE |", "|---|---|--:|--:|--:|"]
    for key in sorted(BASIS, key=lambda k: (k.split("/")[0], SIZES.index(k.split("/")[1]))):
        b = BASIS[key]
        p, a = b.get("acejax-pace"), b.get("acejax-ace")
        ratio = f"{a / p:.1f}×" if p and a else "-"
        lines.append(f"| {key.split('/')[0]} | {key.split('/')[1]} | {p or '-'} | {a or '-'} | {ratio} |")
    return "\n".join(lines) if BASIS else "(model_sizes.json absent)"


def write_doc(pattern, figs, doc="docs/dev/benchmarks.md"):
    rows = load(pattern)
    versions = {}
    for r in rows:
        if r.get("versions"):
            versions[r["host"]] = r["versions"]
    intro = pathlib.Path(__file__).with_name("benchmarks_intro.md").read_text().strip()
    intro = intro.replace("{{findings}}", findings(rows)).replace("{{parity}}", parity_table(rows))
    body = ["# Benchmarks", "", intro, "", "## Figures", "",
            "Hollow markers: ace-jax chose the sparse layout. " + REPEAT_NOTE, ""]
    pending = before_after_hosts(rows, load(before_pattern(pattern)))[1]
    if pending:
        body += [f"Before/after figures pending (ace-jax rows being re-run): {', '.join(pending)}.",
                 ""]
    if not learned_hosts(rows):
        body += ["Learned-radial figures pending (no `acejax-ace-learned` / `-analytic` rows yet).",
                 ""]
    for f in figs:
        stem = pathlib.Path(f).stem
        rel = (pathlib.Path(f).relative_to(pathlib.Path(doc).parent)
               if pathlib.Path(f).is_relative_to(pathlib.Path(doc).parent) else f)
        body += [f"![{stem}]({rel})", "", f"*{_caption(stem)}*", ""]
    body += ["## Model basis sizes", "", basis_table(), "",
             "## Largest system that fits (medium, float64)", "", largest_fits(rows), "",
             "## Tables", "", "±x%: half the min–max range over the median, where a case "
             "was run more than once.", "", tables(rows), "", "## Versions", ""]
    body += [f"- **{h}**: " + ", ".join(f"{k} {v}" for k, v in sorted(v.items())) for h, v in sorted(versions.items())]
    pathlib.Path(doc).write_text("\n".join(body) + "\n")
    return doc


if __name__ == "__main__":
    pattern, outdir = sys.argv[1], sys.argv[2]
    figs = make_figures(pattern, outdir)
    print("\n".join(figs))
    print(write_doc(pattern, figs))
