"""Accuracy against training-set size and against one-core cost on the MACE-H paper's bulk gold
data, for the ace-jax grid (`gold_grid.sh`), MACE-H and DeepH-E3 (`maceh_train.py`,
`maceh_eval.py`, scored by `deeph_fit.py pred=`), all with `ace_jax.ham.metrics` on the same
ten test structures:
    python bench/ham/gold_pareto.py <ace grid.jsonl> <other codes.jsonl> <timings.json> <figure.png> [metric=aligned_rmse]
<timings.json>: {"machine": text, "ace": {model tag: ms per atom}, "MACE-H": ms or {network size: ms},
"DeepH-E3": the same}, of one machine (`model_time.py`, `maceh_time.py`).  Network sizes: "full"
(the authors' gold network; runs `gold_<code>_n<N>`) and the smaller ones of
`sulis/train_gold_size.sh` (runs `gold_<code>_n<N>_<size>`), drawn where a scored run exists."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

grid, other, timings, fig = sys.argv[1:5]
metric = dict(a.split("=") for a in sys.argv[5:]).get("metric", "aligned_rmse")
NAME = {"aligned_rmse": "Band RMSE from own Fermi levels / meV", "band_rmse_median": "Band RMSE at a common zero, median structure / meV",
        "band_energy_rmse": "Band-energy RMSE / meV per atom", "fermi_rmse": "Fermi-level RMSE / meV", "dos": "DoS Wasserstein distance / meV"}
ACE, MACEH, DEEPH, MUTED, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98", "#0b0b0b"
T = json.load(open(timings))
tag = lambda r: f"site_{r['site'][0]}_{r['site'][1]}_env_" + ("none" if not r["env"] else "_".join(str(float(v)) if i == 0 else str(int(v)) for i, v in enumerate(r["env"]))) \
    + (f"_nbond_{r['nbond']}_nsite_{r['nsite']}" if r["nbond"] != 12 else "")
ace = {}
for line in open(grid):
    r = json.loads(line)
    ace.setdefault(tag(r), {})[r["n_train"]] = 1e3 * r["metrics"][metric]
oth, small = {}, {}
cost = lambda code, size="full": T[code][size] if isinstance(T[code], dict) else T[code]
for line in open(other):
    r = json.loads(line)
    code, n, *size = r["pred"].rstrip("/").split("/")[-2].split("_")[1:]
    (oth if not size else small).setdefault(code, {})[int(n[1:]) if not size else (size[0], int(n[1:]))] = 1e3 * r["metrics"][metric]
f, (a, b) = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True, gridspec_kw=dict(wspace=0.06))
order = sorted(ace, key=lambda t: T["ace"][t])
show = {order[0]: "two-centre only", [t for t in order if "env_5" in t][0]: "smallest with bond environment", order[-1]: "largest"}
for t, label in show.items():                           # ace-jax: one hue, three model sizes named on the lines
    n = sorted(ace[t])
    a.plot(n, [ace[t][i] for i in n], color=ACE, lw=2, marker="o", ms=5)
    a.annotate(f"ace-jax, {label}", (n[-1], ace[t][n[-1]]), xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=8, color=INK)
for code, col in (("MACE-H", MACEH), ("DeepH-E3", DEEPH)):
    n = sorted(oth.get(code, {}))
    a.plot(n, [oth[code][i] for i in n], color=col, lw=2, marker="s", ms=5)
    if n:
        a.annotate(code, (n[-1], oth[code][n[-1]]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("Training structures (64 atoms each)"); a.set_ylabel(NAME.get(metric, metric))
a.set_xticks([1, 2, 4, 8, 16, 32, 64, 152]); a.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "152"])
nb = max(n for c in oth.values() for n in c) if oth else 32            # cost panel: every code at its largest common training set
na = nb if all(nb in v for v in ace.values()) else max(next(iter(ace.values())))
xs = [T["ace"][t] for t in order]
b.plot(xs, [ace[t][na] for t in order], color=ACE, lw=2, marker="o", ms=5)
b.annotate(f"ace-jax, {len(order)} model sizes", (xs[-1], ace[order[-1]][na]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
for code, col, side in (("MACE-H", MACEH, -1), ("DeepH-E3", DEEPH, 1)):         # every training-set size at the full network's cost
    if code in T and oth.get(code):
        ns, x = sorted(oth[code]), cost(code)
        b.plot([x] * len(ns), [oth[code][n] for n in ns], color=col, marker="s", ms=6, ls="-", lw=1)
        for n in ns:
            b.annotate(str(n), (x, oth[code][n]), xytext=(7 * side, 0), textcoords="offset points", ha="right" if side < 0 else "left",
                       va="center", fontsize=7, color=INK)
        b.annotate(code, (x, max(oth[code].values())), xytext=(0, 9), textcoords="offset points", ha="right" if side < 0 else "left", fontsize=8, color=INK)
        for n in sorted({n for _, n in small.get(code, {})}):                   # smaller networks: the code's own cost curve at n
            pts = sorted([(cost(code, sz), v) for (sz, m), v in small[code].items() if m == n] + ([(x, oth[code][n])] if n in oth[code] else []))
            b.plot([q[0] for q in pts], [q[1] for q in pts], color=col, marker="s", ms=6, ls="--", lw=1.5, mfc="white")
b.set_xscale("log"); b.set_xlabel(f"Block prediction, {T.get('machine', 'one CPU core')} / ms per atom\n(ace-jax: {na} training structures; others: the number beside each point)")
from matplotlib.ticker import FuncFormatter, NullFormatter
a.set_yticks([1, 2, 5, 10, 20, 50, 100]); a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); a.yaxis.set_minor_formatter(NullFormatter())
b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
lo, hi = a.get_ylim(); a.set_ylim(max(lo, 0.5), hi)
for ax in (a, b):
    ax.grid(True, which="major", color=MUTED, lw=0.3, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
b.tick_params(left=False)
f.suptitle("Bulk gold (MACE-H paper's data), 10 test structures: accuracy against training-set size and against cost", fontsize=10, x=0.125, ha="left")
f.savefig(fig, dpi=200, bbox_inches="tight")
