"""One-panel figure of the gold comparison for a proposal: band error against the number of
training structures for the largest ace-jax model, MACE-H and DeepH-E3, each line labelled with
its one-core prediction cost (no second panel or axis).
    python bench/ham/gold_proposal.py <ace grid.jsonl> <other codes.jsonl> <timings.json> <figure stem> [nmax=32] [metric=aligned_rmse]
Writes <stem>.png and <stem>.pdf.  A run listed in UNCONVERGED is drawn with an open marker."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

grid, other, timings, stem = sys.argv[1:5]
kw = dict(a.split("=") for a in sys.argv[5:])
nmax, metric = int(kw.get("nmax", 32)), kw.get("metric", "aligned_rmse")
ACE, MACEH, DEEPH, MUTED, INK, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98", "#0b0b0b", "#ffffff"
UNCONVERGED = {("MACE-H", 32)}                           # stopped at its time limit: its best model so far
T = json.load(open(timings))
big = {}
for line in open(grid):
    r = json.loads(line)
    if r["nbond"] == 20 and r["n_train"] <= nmax:        # the largest model of the grid
        big[r["n_train"]] = 1e3 * r["metrics"][metric]
oth = {}
for line in open(other):
    r = json.loads(line)
    code, n, *size = r["pred"].rstrip("/").split("/")[-2].split("_")[1:]
    if not size and int(n[1:]) <= nmax:
        oth.setdefault(code, {})[int(n[1:])] = 1e3 * r["metrics"][metric]
cost = lambda c: T[c]["full"] if isinstance(T[c], dict) else T[c]
t_ace = max(T["ace"].values())
f, a = plt.subplots(figsize=(3.6, 2.9))
series = [("ace-jax", ACE, "o", big, t_ace), ("MACE-H", MACEH, "s", oth.get("MACE-H", {}), cost("MACE-H")),
          ("DeepH-E3", DEEPH, "^", oth.get("DeepH-E3", {}), cost("DeepH-E3"))]
for name, col, mk, d, t in series:
    n = sorted(d)
    a.plot(n, [d[i] for i in n], color=col, lw=2, zorder=2)
    for i in n:
        a.plot([i], [d[i]], color=col, marker=mk, ms=5.5, mfc=SURF if (name, i) in UNCONVERGED else col, mew=1.5, zorder=3)
    a.annotate(f"{name}\n{t:.0f} ms/atom", (n[-1], d[n[-1]]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK, linespacing=1.15)
a.set_xscale("log", base=2); a.set_yscale("log")
ns = sorted(big)
a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns]); a.xaxis.set_minor_formatter(NullFormatter())
a.set_yticks([1, 2, 5, 10, 20, 50]); a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); a.yaxis.set_minor_formatter(NullFormatter())
a.set_xlim(ns[0] * 0.8, ns[-1] * 3.6); a.set_ylim(0.9, 60)
a.set_xlabel("Training structures (64 Au atoms each)", fontsize=8.5, loc="left"); a.set_ylabel("Band error (RMSE) / meV", fontsize=8.5)
a.tick_params(labelsize=8)
a.grid(True, which="major", color=MUTED, lw=0.3, alpha=0.5)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
a.spines["bottom"].set_bounds(ns[0] * 0.8, ns[-1] * 1.15)
f.savefig(stem + ".png", dpi=300, bbox_inches="tight"); f.savefig(stem + ".pdf", bbox_inches="tight")
print({name: {i: round(d[i], 2) for i in sorted(d)} for name, _, _, d, _ in series}, {name: round(t, 1) for name, *_, t in series})
