"""(RMSE, QoI) vs cost for the accuracy-cost benchmark families: three panels.

  A  test force RMSE vs cost, one line per family through its sizes
  B  QoI error: geometric mean over the silicon QoIs of |error| / tolerance (bench/si_qoi/score.py) vs cost
  C  the cheapest model of each family reaching the target (force RMSE <= GAP-18's AND QoI error <= TARGET_QOI)

The per-QoI breakdown behind B is qoi_heatmap.py.

One row per x axis, top to bottom: each --row's cost (make_cost.py: the A100, then the CPU timings), then the
fitted-parameter count, which compares model size, not cost (MACE ~2e4-2e5, ACE ~1e2-5e3; GAP, untimed on
the GPU, appears there only).

    python bench/si_qoi/plot_cost.py [--row "A100 cost (µs per atom-step, log)|cost.json|cost.codes.json" ...] [--out fig.png]
"""
import argparse, json, pathlib, sys

import matplotlib
import matplotlib.ticker  # noqa: E402,F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compare import qois  # noqa: E402
from score import TOLERANCES, score  # noqa: E402  (TOLERANCES already drops score.EXCLUDED)

ROOT = pathlib.Path(__file__).resolve().parents[2]
METRICS = ROOT / "bench/accuracy_cost/results/metrics.jsonl"
TARGET_QOI = 0.5                  # geometric-mean |error|/tolerance: the typical QoI within half its tolerance
# validated (dataviz validate_palette --pairs all, light): red/green CVD 7.2 -> marker shapes + direct labels
FAM = {  # family: (label, colour, marker, linestyle, filled, qoi model name pattern)
    "ace":        ("linear ACE",               "#2a78d6", "o", "-",  True,  "ACE_{s}"),
    "ace_E10":    ("linear ACE, pacemaker's E:F balance", "#2a78d6", "o", "--", False, "ACEE10_{s}"),
    # the same linear ACE fits run by ML-PACE (write_yace, Tabulated radials: exact): linear ACE's colour
    # (same models), the code told apart by marker and line; its accuracy is ACE's, its cost its own (COST_NAME)
    "ace_tab":    ("linear ACE, run by ML-PACE", "#2a78d6", "*", ":",  True,  "ACE_{s}"),
    # and by ML-PACE with the additions proposed upstream (ICAMS/lammps-user-pace#31: the recursive-evaluator
    # speed-ups on the CPU; pace/kk with the prototype kernels of pace_ref/fast on the GPU): the same files, the open star
    "ace_tabfork": ("linear ACE, run by patched ML-PACE", "#2a78d6", "*", "--", False, "ACE_{s}"),
    # and by ACEpotentials.jl's LAMMPS export (bench/accuracy_cost/build_trim.py: the juliac --trim library, exact)
    "ace_trim":   ("linear ACE, run by ACEpotentials.jl", "#2a78d6", "X", "-.", True, "ACE_{s}"),
    # the order-4 ladder in reduced precision (docs/dev/fp32-results.md): the same fits, their own cost --
    # ace-jax float32 bundles, and patched pace/kk (PACE_KK_RHOW=3) in a KOKKOS_PREC=mixed LAMMPS.  Each pairs with its float64 line:
    # its marker and line style in a lighter blue (the wL ramp's light step; the wL ladder is not in this figure)
    "ace_f32":       ("linear ACE, float32",                         "#86b6ef", "o", "-", True, "ACE_{s}"),
    "ace_tab_mixed": ("linear ACE, run by patched ML-PACE, mixed precision", "#86b6ef", "*", "--", False, "ACE_{s}"),
    "ace_lr":     ("ACE, learned radials",     "#e34948", "s", "-",  True,  "ACElr_{s}"),
    "pace":       ("PACE (pacemaker)",         "#eda100", "^", "-",  True,  "PACE_{s}"),
    # MACE is two ladders: L=1 (equivariant messages) costs ~3x an L=0 model of similar size, so one
    # line through both in parameter order zig-zags in cost
    "mace_L0":    ("MACE, L=0",                "#4a3aa7", "D", "--", False, "MACE_{s}"),
    "mace_L1":    ("MACE, L=1",                "#4a3aa7", "D", "-",  True,  "MACE_{s}"),
    "mace_n1":    ("MACE, L=0, one layer",     "#4a3aa7", "d", ":",  False, "MACE_{s}"),
    # the same MACE fits run by Symmetrix-XL (arXiv 2610.01036): MACE's accuracy, its own cost (COST_NAME);
    # validated (dataviz --pairs all, light): only a contrast WARN, as the yellow and teal
    "mace_xl_L0": ("MACE, L=0, Symmetrix-XL",  "#a197ea", "p", "--", False, "MACE_{s}"),
    "mace_xl_L1": ("MACE, L=1, Symmetrix-XL",  "#a197ea", "p", "-",  True,  "MACE_{s}"),
    # NEP (GPUMD-trained NEP4; float64 NEP_CPU for accuracy and the CPU cost row): validated (dataviz --pairs
    # all, light): CVD WARN 6.4 against linear ACE's blue, so its own marker shape too
    "nep":        ("NEP (GPUMD)",              "#c13ec9", "X", "-",  True,  "NEP_{s}"),
    # the published GAP-18 Si NEP4 recipe (sweep.yaml nlit: full batch, 50 neurons, lambda 0.05): NEP's colour,
    # hollow, so it is not read as a rung of the ladder above
    "nep_lit":    ("NEP, published recipe",    "#c13ec9", "X", ":",  False, "NEP_{s}"),
    "gap":        ("GAP refit",                "#008300", "v", "-",  True,  "GAP_{s}"),
    "gap_noglue": ("GAP refit, no glue",       "#008300", "v", "--", False, "GAPnoglue_{s}"),
    "gap_E30":    ("GAP refit, ACE's E:F balance", "#008300", "v", ":", True, "GAPE30_{s}"),
    # ace-jax on pacemaker's basis shape (order 6, nmax/lmax by order) with PACE's SBessel radials: one size
    "ace_pshape":    ("PACE-shaped ACE",                  "#1baf7a", "P", "-", False, "ACE_pshape_sb_fixed_{s}"),
    "ace_pshape_lr": ("PACE-shaped ACE, learned radials", "#1baf7a", "P", "-", True,  "ACE_pshape_sb_lr200_{s}"),
    "ace_pshape_E10": ("PACE-shaped ACE, pacemaker's E:F balance", "#1baf7a", "P", "--", False, "ACE_pshape_sb_fixed_E10_{s}"),
    # linear ACE at other angular weights (bench/accuracy_cost/wl_ladder): a single-hue ramp light -> dark in wL,
    # validated (dataviz --ordinal) with wL 1.5 at step 400 -- WL_FIGURE recolours the baseline to it
    "ace_wl_1":    ("linear ACE, wL 1.0",          "#86b6ef", "o", "-", False, "ACEwl_{s}"),
    "ace_wl_1.75": ("linear ACE, wL 1.75",         "#256abf", "o", "-", False, "ACEwl_{s}"),
    "ace_wl_2":    ("linear ACE, wL 2.0",          "#184f95", "o", "-", False, "ACEwl_{s}"),
    "ace_wl_2.5":  ("linear ACE, wL 2.5",          "#0d366b", "o", "-", False, "ACEwl_{s}"),
    "ace_wl_o5":   ("linear ACE, order 5, wL 2.0", "#184f95", "h", ":", False, "ACEwl_{s}"),
    # GRACE trained from scratch by gracemaker (bench/accuracy_cost/fit_grace.py): three seeds per size, plotted as
    # the seed mean with a min-max bar (SEEDED); sienna, its own marker; FS open and dashed
    "grace":      ("GRACE-1L (gracemaker)",    "#a0522d", "h", "-",  True,  "GRACE_{s}"),
    "grace_fs":   ("GRACE-FS (gracemaker)",    "#a0522d", "h", "--", False, "GRACE_{s}"),
}
# families fitted with several seeds (sizes "<size>_s<k>"): one point per size, the mean over seeds, with the
# min-max spread as an error bar; the per-seed rows stay in metrics.jsonl.  SEED_EXCLUDE: checks, not ladder points
SEEDED = ("grace", "grace_fs")
SEED_EXCLUDE = ("c32l4_long",)               # the 150k-update training-length check
WL_FAMILIES = ("ace", "ace_wl_1", "ace_wl_1.75", "ace_wl_2", "ace_wl_2.5", "ace_wl_o5", "ace_lr", "pace")
WL_RECOLOUR = {"ace": "#3987e5"}                 # the ramp's wL 1.5 step (the categorical blue is too near 1.75)


def family_of(r):
    """The plotted family of a metrics/timing row: MACE split by layers and L (c16L1 -> mace_L1, c16L0n1 ->
    mace_n1), the wL ladder by wL (o4d15_wL1.75 -> ace_wl_1.75; order 5 -> ace_wl_o5)."""
    fam, size = r["family"], r["size"]
    if fam == "mace":
        return ("mace_xl_" if r.get("code", "").startswith("mace-xl") else "mace_") + size[-2:]
    if fam == "ace" and r.get("code") == "acepotentials-trim":
        return "ace_trim"
    if fam == "ace" and r.get("code") == "acejax-ace" and r.get("dtype") == "float32":
        return "ace_f32"
    if fam == "ace_tab" and r.get("code") == "mlpace-mixed":
        return "ace_tab_mixed"
    if fam == "nep" and size.startswith("nlit"):
        return "nep_lit"
    if fam == "ace_tab" and any(t in r.get("_file", "") for t in PATCHED_MLPACE):   # make_cost's per-row file stem
        return "ace_tabfork"
    if fam == "ace_wl":
        return "ace_wl_o5" if size.startswith("o5") else "ace_wl_" + size.split("_wL")[1]
    return fam
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# refits of a timed basis with other loss weights: the same model form, so the same cost
COST_AS = {"ace_E10": "ace"}
# a second code for the same fits: its cost under its own name (make_cost.py writes it there)
COST_NAME = {"mace_xl_L0": "MACExl_{s}", "mace_xl_L1": "MACExl_{s}", "ace_tab": "ACEtab_{s}", "ace_trim": "ACEtrim_{s}",
             "ace_tabfork": "ACEtabfork_{s}", "ace_f32": "ACEf32_{s}", "ace_tab_mixed": "ACEtabmixed_{s}"}
REDUCED = ("ace_f32", "ace_tab_mixed")              # reduced-precision lines: make_cost --dtype keeps them
# timing files of ML-PACE with the proposed additions (sulis-{cpu,gpu}-pacefork.sbatch): their
# mlpace rows are the ace_tabfork line, not stock ML-PACE's
PATCHED_MLPACE = ("ace_tabfork",)
TAB_SIZES = ("o4d12", "o4d14", "o4d16", "o4d20")    # the ace sizes converted (sweep.yaml ace_tab)


def cost_name(fam, size):
    fam = COST_AS.get(fam, fam)
    return COST_NAME.get(fam, FAM[fam][5]).format(s=size)


def seed_split(size):
    """("c16l3", 0) for "c16l3_s0"; (size, None) without a seed suffix."""
    import re
    m = re.fullmatch(r"(.+)_s(\d+)", size)
    return (m.group(1), int(m.group(2))) if m else (size, None)


def seed_average(pts):
    """Per-seed points of one family -> one point per size: mean rmse_F / rmse_E / qoi (qoi only when every seed
    has every QoI) and their min-max (`*_lo`, `*_hi`), n_seeds; x is the seed-0 cost (all seeds share the
    architecture, so the cost) or the parameter count."""
    groups = {}
    for q in pts:
        base, k = seed_split(q["size"])
        if base in SEED_EXCLUDE:
            continue
        groups.setdefault(base, []).append(q)
    out = []
    for base, qs in groups.items():
        p = {"name": qs[0]["name"].replace(qs[0]["size"], base), "size": base, "n": qs[0]["n"], "n_seeds": len(qs),
             "x": next((q["x"] for q in qs if q["x"] is not None), None), "n_qoi": min(q["n_qoi"] for q in qs)}
        for key in ("rmse_F", "rmse_E", "qoi"):
            v = [q[key] for q in qs]
            if any(x is None for x in v):
                p[key] = None
                continue
            p[key], p[key + "_lo"], p[key + "_hi"] = float(np.mean(v)), float(min(v)), float(max(v))
        out.append(p)
    return out
SKIP = set()                                     # families left out of this figure (main: the wL ladder)


RUNS = pathlib.Path("/storage/eng/essswb/projects/ace-jax/runs/accuracy_cost")


def trainable(fam, size, n, model=None):
    """Every fitted parameter, so the x axis compares like with like.  metrics.jsonl's n_params is the
    linear coefficient count for ACE but the FUNCTION count for PACE: each pacemaker function carries
    ndensity coefficients, and its radial coefficients are fitted too (f800: 761 functions, 1972
    parameters).  Learned-radial ACE adds its radial matrix (the nonzero rnl_Wnlq entries)."""
    try:
        if fam == "pace":
            import yaml
            b = yaml.safe_load(open(RUNS / "pace" / size / "model.yaml"))["species"][0]
            return sum(len(f["c"]) for f in b["nbody"]) + int(np.size(b["radcoefficients"]))
        if fam in ("ace_lr", "ace_pshape_lr"):
            z = np.load(model or RUNS / fam / size / "model.npz")
            return n + int(np.count_nonzero(z["rnl_Wnlq"]))
    except FileNotFoundError:
        pass
    return n


def load_rows():
    rows = [json.loads(l) for l in open(METRICS)]
    for r in rows:
        r["n_params"] = trainable(r["family"], r["size"], r.get("n_params"), r.get("model"))
        r["family"] = family_of(r)
    # the Symmetrix-XL series are the MACE fits again (no own metrics row)
    # and linear ACE run by ML-PACE or ACEpotentials.jl is the ace fits again
    return (rows + [r | {"family": "mace_xl_" + r["family"][-2:]} for r in rows if r["family"] in ("mace_L0", "mace_L1")]
            + [r | {"family": "ace_tab"} for r in rows if r["family"] == "ace" and r["size"] in TAB_SIZES]
            + [r | {"family": f} for r in rows if r["family"] == "ace" and r["size"] in TAB_SIZES for f in REDUCED]
            + [r | {"family": "ace_tabfork"} for r in rows if r["family"] == "ace" and r["size"] in TAB_SIZES]
            + [r | {"family": "ace_trim"} for r in rows if r["family"] == "ace"])


def collect(cost):
    ref = qois("CASTEP")
    rows = load_rows()
    gap18 = next(r for r in rows if r["family"] == "gap18_published")
    out = {}
    for r in rows:
        fam = r["family"]
        if fam not in FAM or fam in SKIP or (fam in COST_NAME and not cost):   # a second code: cost rows only
            continue
        name = FAM[fam][5].format(s=r["size"])
        sc = score(name, ref)
        x = cost.get(cost_name(fam, r["size"])) if cost else r.get("n_params")
        out.setdefault(fam, []).append({"name": name, "size": r["size"], "x": x, "n": r.get("n_params") or 0, "rmse_F": r["rmse_F"],
                                        "rmse_E": r["rmse_E"], "qoi": sc["gm_norm"] if sc["n_qoi"] == sc["n_total"] else None,
                                        "n_qoi": sc["n_qoi"]})
    for fam in SEEDED:
        if fam in out:
            out[fam] = seed_average(out[fam])
    for fam in out:
        out[fam].sort(key=lambda p: p["n"])           # ladder order: a line through sizes, even where cost is not monotone
    out = {f: out[f] for f in FAM if f in out}          # fixed family order: legend and colours never reorder
    g = score("GAP18_py3", ref)
    return out, {"rmse_F": gap18["rmse_F"], "qoi": g["gm_norm"], "x": cost.get("GAP18_py3") if cost else gap18.get("n_params")}


def style(ax, xlabel, ylabel):
    ax.set_xscale("log"); ax.set_xlabel(xlabel, color=INK); ax.set_ylabel(ylabel, color=INK)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())   # a sub-decade range labels its minor ticks
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def series(ax, data, key):
    for fam, pts in data.items():
        lab, col, mk, ls, filled, _ = FAM[fam]
        p = [q for q in pts if q[key] is not None and q["x"] is not None]
        if not p:
            continue
        x, y = [q["x"] for q in p], [q[key] for q in p]
        ax.plot(x, y, ls=ls, lw=2, color=col, marker=mk, ms=8, mfc=col if filled else "white", mec=col,
                mew=1.5, label=lab, zorder=3)
        if all(key + "_lo" in q for q in p):           # SEEDED: the min-max over seeds
            ax.errorbar(x, y, yerr=[[q[key] - q[key + "_lo"] for q in p], [q[key + "_hi"] - q[key] for q in p]],
                        fmt="none", ecolor=col, elinewidth=1.2, capsize=3, zorder=3)


def flag(ax, data, codes, key):
    """Label points whose cost is not a float64 LAMMPS timing: make_cost.py fell back to a standalone run, or
    the code is mixed precision only (NEP in GPUMD; make_cost --allow nep-gpumd:mixed)."""
    said = set()                            # one mixed-precision label per code: both NEP series run in GPUMD
    for fam, pts in data.items():           # one label per family and reason: neighbouring points share it
        why = {}
        for q in pts:
            c = codes.get(q["name"], "")
            if q["x"] is None or q[key] is None:
                continue
            if " mixed " in c:
                head = "NEP (GPUMD): mixed precision" if fam.startswith("nep") else f"{FAM[fam][0]}: mixed precision"
                if head not in said or (head, None) in why:
                    said.add(head); why.setdefault((head, None), []).append(q)
            elif "standalone" in c:
                r = ("Symmetrix exports no one-layer MACE" if q["size"].endswith("n1")
                     else "LAMMPS out of memory")
                why.setdefault((None, r), []).append(q)
        for (head, r), qs in why.items():
            q = max(qs, key=lambda q: q[key])      # above the highest: the label sits clear of the data
            head = head or f"{', '.join(p['size'] for p in qs)}: standalone"
            left = ax.transAxes.inverted().transform(ax.transData.transform((q["x"], q[key])))[0] < 0.5   # a right-aligned label off a cheap point runs off the axes
            ax.annotate(head + (f"\n({r})" if r else ""), (q["x"], q[key]), xytext=(6 if left else -6, 10),
                        textcoords="offset points", ha="left" if left else "right", fontsize=7, color=MUTED)


DAG_LABEL = "centre dot: DAG product basis"


def dag_dots(ax, data, codes, key):
    """A centre dot on the ace-jax points whose timing used the DAG product basis (make_cost's codes note
    "dag:<mode>"): white on filled markers, the family's colour on open ones.  Returns whether any was drawn."""
    drawn = False
    for fam, pts in data.items():
        _, col, _, _, filled, _ = FAM[fam]
        p = [q for q in pts if q["x"] is not None and q[key] is not None
             and " dag:" in codes.get(cost_name(fam, q["size"]), "")]   # by cost name: refits (COST_AS) share the timing
        if p:
            ax.plot([q["x"] for q in p], [q[key] for q in p], ls="none", marker="o", ms=3.5,
                    color="white" if filled else col, mec="none", zorder=4)
            drawn = True
    return drawn


def cheapest(data, ref):
    """Per family (in data's order): (family, x of its cheapest point reaching the target or nan, note) -- the
    note is that point's size, or why there is none.  Shared by the PNG's panel C and the web figure."""
    out = []
    for fam, pts in data.items():
        if fam in REDUCED and all(q["x"] is None for q in pts):
            continue                                  # a reduced-precision line is a cost row only: no bar until timed
        ok = [q for q in pts if q["x"] is not None and q["qoi"] is not None and q["rmse_F"] <= ref["rmse_F"]
              and q["qoi"] <= TARGET_QOI]
        if ok:
            best = min(ok, key=lambda q: q["x"]); out.append((fam, best["x"], best["size"]))
        elif all(q["x"] is None for q in pts):
            out.append((fam, np.nan, "not timed here"))
        elif any(q["qoi"] is None for q in pts):
            out.append((fam, np.nan, "QoIs pending"))
        else:
            out.append((fam, np.nan, "not reached in its ladder"))
    return out


def draw_row(axes, data, ref, xlabel, codes, letters, what):
    """One row of the figure: force error, QoI error and the cheapest model reaching the target, all against
    one x (a cost, or the fitted-parameter count)."""
    a, b, c = letters
    ax = axes[0]; style(ax, xlabel, "test force RMSE (eV/Å)")
    series(ax, data, "rmse_F"); dag_dots(ax, data, codes, "rmse_F")
    if "A100" in xlabel or "GPU" in xlabel:                   # on the CPU every ace-jax row is standalone by design
        flag(ax, data, codes, "rmse_F")
    ax.axhline(ref["rmse_F"], color=MUTED, ls=":", lw=1.2)
    ax.annotate("GAP-18 (trained on these configurations)", (ax.get_xlim()[0], ref["rmse_F"]), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_title(f"{a}  force error", loc="left", color=INK, fontweight="bold")

    ax = axes[1]; style(ax, xlabel, f"QoI error: geometric mean of |error| / tolerance\nover {len(TOLERANCES)} QoIs (lower is better)")
    series(ax, data, "qoi"); dag_dots(ax, data, codes, "qoi"); ax.set_yscale("log")             # (standalone points flagged in the force panel only)
    ax.axhline(ref["qoi"], color=MUTED, ls=":", lw=1.2)
    ax.annotate("GAP-18", (ax.get_xlim()[0], ref["qoi"]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
    ax.axhline(TARGET_QOI, color=MUTED, ls="--", lw=1)
    ax.annotate("target", (ax.get_xlim()[1], TARGET_QOI), xytext=(-4, 4), textcoords="offset points", ha="right",
                fontsize=8, color=MUTED)
    ax.set_yticks([0.3, 0.5, 1, 2, 3], ["0.3", "0.5", "1", "2", "3"]); ax.minorticks_off()
    ax.set_title(f"{b}  silicon QoIs vs DFT", loc="left", color=INK, fontweight="bold")

    ax = axes[2]
    allx = [q["x"] for pts in data.values() for q in pts if q["x"]]
    bars = cheapest(data, ref)
    names, vals, notes = [FAM[f][0] for f, _, _ in bars], [v for _, v, _ in bars], [n for _, _, n in bars]
    cols = [FAM[f][1] for f, _, _ in bars]
    y = np.arange(len(names))
    lo, hi = min(allx) / 2, max(allx) * 2
    style(ax, xlabel.split("  (")[0], ""); ax.grid(False, axis="y"); ax.set_xlim(lo, hi)
    for i, (v, n, col) in enumerate(zip(vals, notes, cols)):
        if np.isfinite(v):
            ax.barh(i, v - lo, left=lo, color=col, height=0.6)
            ax.annotate(n, (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
        else:
            ax.annotate(n, (lo, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y, names); ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_title(f"{c}  {what} to reach the target", loc="left", color=INK, fontweight="bold", pad=16)
    ax.text(0, 1.01, f"force RMSE ≤ GAP-18's and QoI error ≤ {TARGET_QOI:g}", transform=ax.transAxes, fontsize=8,
            color=MUTED, va="bottom")


def _row_arg(s):
    """--row "LABEL|COST.json[|CODES.json]": one cost row (e.g. the A100 timings)."""
    parts = s.split("|")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError('--row takes "LABEL|COST.json[|CODES.json]"')
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", type=_row_arg, default=[],
                    help='"LABEL|COST.json[|CODES.json]", repeatable, top to bottom (GPU, then CPU); the '
                         "fitted-parameter row always comes last.  CODES is make_cost.py's sidecar.")
    ap.add_argument("--out", default=str(ROOT / "bench/si_qoi/results/rmse_qoi_cost.png"))
    ap.add_argument("--no-note", action="store_true",
                    help="no footnote (the docs page states the method in its text)")
    ap.add_argument("--wl", action="store_true", help="the wL-ladder figure (WL_FAMILIES, recoloured) instead of the main one")
    ap.add_argument("--seed-table", action="store_true", help="print the SEEDED families' per-size mean and min-max, then exit")
    a = ap.parse_args()
    if a.seed_table:
        data, ref = collect(None)
        print(f"GAP-18: rmse_F {ref['rmse_F']:.4f}  QoI {ref['qoi']:.2f}")
        for fam in SEEDED:
            for q in data.get(fam, []):
                f = lambda k, fmt, q=q: "-" if q.get(k) is None else f"{q[k]:{fmt}} [{q[k + '_lo']:{fmt}}, {q[k + '_hi']:{fmt}}]"
                print(f"{fam}/{q['size']:10s} n {q['n']:7d} seeds {q['n_seeds']}  F {f('rmse_F', '.4f')}  "
                      f"E {f('rmse_E', '.4f')}  QoI {f('qoi', '.2f')}")
        return
    if a.wl:
        SKIP.update(f for f in FAM if f not in WL_FAMILIES)
        for f, c in WL_RECOLOUR.items():
            FAM[f] = (FAM[f][0].replace("linear ACE", "linear ACE, wL 1.5"),) + (c,) + FAM[f][2:]
    else:
        SKIP.update(f for f in FAM if f.startswith("ace_wl_"))
    rows = [(lab, json.load(open(c)), json.load(open(k[0])) if k else {}, "cost") for lab, c, *k in a.row]
    rows.append(("fitted parameters (log)", None, {}, "size"))
    letters = iter("ABCDEFGHIJKL")
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(17, 5.0 * n + 0.6), gridspec_kw={"width_ratios": [1, 1, 0.8]}, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    handles = {}
    for r, (xlabel, cost, codes, what) in enumerate(rows):
        data, ref = collect(cost)
        draw_row(axes[r], data, ref, xlabel, codes, [next(letters) for _ in range(3)], what)
        for h, l in zip(*axes[r][0].get_legend_handles_labels()):
            handles.setdefault(l, h)                       # GAP has no GPU timing: its line is in the last row only
    if any(cost is None for _, cost, _, _ in rows):
        axes[-1][0].annotate("GAP: the sparse-point coefficients (its kernel also stores\nthe sparse descriptors, "
                             "not counted)", (1, 1), xycoords="axes fraction", xytext=(-4, -4),
                             textcoords="offset points", ha="right", va="top", fontsize=7.5, color=MUTED)
    order = [FAM[f][0] for f in FAM if FAM[f][0] in handles]
    if any(" dag:" in v for _, _, codes, _ in rows for v in codes.values()):
        from matplotlib.lines import Line2D
        handles[DAG_LABEL] = (Line2D([], [], ls="none", marker="o", ms=8, color=MUTED),
                              Line2D([], [], ls="none", marker="o", ms=3.5, color="white", mec="none"))
        order.append(DAG_LABEL)
    ncol = min(6, -(-len(order) // 3))                    # at most 6 columns fit the width; more rows instead
    legend_rows = -(-len(order) // ncol)
    from matplotlib.legend_handler import HandlerTuple
    fig.legend([handles[l] for l in order], order, loc="upper center", ncol=ncol, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0), handler_map={tuple: HandlerTuple(ndivide=1)})
    cpu_note = ("CPU row: a 128-core EPYC 7742 node, all cores, ranks bound to cores; each model's fastest layout\n"
                "(128x1 or 64x2 MPI ranks x threads; GAP: 32 ranks).\n"
                "ace-jax runs in LAMMPS (lammps-jax; aa_dag='auto': the DAG product basis, centre dot, from 4,000 chain-form multiplies per site; the chain form below); MACE runs in LAMMPS via Symmetrix and, separately, Symmetrix-XL.\n"
                "Linear ACE run by ML-PACE: the o4 fits exported exactly to .yace with tabulated radials (ML-PACE with the tabulated-radial patch, ICAMS/lammps-user-pace#31; pace and pace/kk).  "
                "GPU ML-PACE: LAMMPS develop ec02ed0 (with the 2026-08 pace/kk GPU rewrite).\n"
                "Patched ML-PACE: the recursive-evaluator speed-ups proposed upstream (ICAMS/lammps-user-pace#31) on the CPU; "
                "pace/kk with prototype ComputeRho/ComputeWeights kernels (pace_ref/fast) on the GPU.\n"
                "Linear ACE run by ACEpotentials.jl (CPU only): each fit as an ACEpotentials model, exported and compiled "
                "with juliac --trim (ACEpotentials PR 309, v0.10.2), in LAMMPS via its pair_style ace plugin.\n"
                "NEP: trained by GPUMD; on the CPU LAMMPS pair_style nep (NEP_CPU, float64); on the GPU GPUMD's own MD code, mixed precision (labelled).\n"
                if any("CPU" in r[0] for r in rows) else "")
    bottom = 0
    if not a.no_note:
        note = (cpu_note + "QoI error clips each QoI to [0.05, 10] x tolerance; compressed hcp is left out (extrapolation, "
                "outside the training data); only models with every QoI computed.\n"
                "Cost: the fastest parity-checked code per model, float64 only except NEP on the GPU (one-layer MACE and, on the GPU, Symmetrix MACE c64L1 have "
                "float32 timings only and are left out of the cost rows). Single fits per point, except GRACE: the mean of three seeds, bars min-max.\n"
                "Light blue: the same o4 linear ACE fits in reduced precision -- ace-jax float32 lammps-jax bundles (where timed), and "
                "patched ML-PACE pace/kk in a KOKKOS_PREC=mixed LAMMPS (accuracy is the float64 fit's).")
        fig.text(0.5, 0.003, note, ha="center", fontsize=8.5, color=MUTED)
        # the axes start above the note: its lines (8.5 pt, 1.2 spacing) plus room for the bottom row's axis label
        bottom = ((note.count("\n") + 1) * 8.5 * 1.2 / 72 + 0.2) / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1 - (0.2 * legend_rows + 0.15) / fig.get_figheight()))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150); print("wrote", a.out)


if __name__ == "__main__":
    main()
