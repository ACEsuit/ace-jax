"""Accuracy against training-set size and against one-core cost on the MACE-H paper's bulk gold
data, for the ace-jax grid (`gold_grid.sh`), MACE-H and DeepH-E3 (`maceh_train.py`,
`maceh_eval.py`, scored by `deeph_fit.py pred=`), all with `ace_jax.ham.metrics` on the same
ten test structures:
    python bench/ham/gold_pareto.py <ace grid.jsonl> <other codes.jsonl> <timings.json> <figure.png> [metric=aligned_rmse]
<timings.json>: {"machine": text, "ace": {model tag: ms per atom}, "MACE-H": ms or {network size: ms},
"DeepH-E3": the same}, of one machine (`model_time.py`, `maceh_time.py`).  Network sizes: "full"
(the authors' gold network; runs `gold_<code>_n<N>`) and the smaller ones of
`sulis/train_gold_size.sh`, or ACEhamiltonians.jl's basis sizes (`aceh_gold.jl`; runs
`gold_<code>_n<N>_<size>`), drawn where a scored run exists.  ace-jax's points are named by
architecture: 2c the two-centre features, site(n, l) the site-coupled group, env(n, l, nb, lb)
the bond environment."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

grid, other, timings, fig = sys.argv[1:5]
metric = dict(a.split("=") for a in sys.argv[5:]).get("metric", "aligned_rmse")
NAME = {"aligned_rmse": "Band RMSE from own Fermi levels / meV", "band_rmse_median": "Band RMSE at a common zero, median structure / meV",
        "band_energy_rmse": "Band-energy RMSE / meV per atom", "fermi_rmse": "Fermi-level RMSE / meV", "dos": "DoS Wasserstein distance / meV"}
ACE, MACEH, DEEPH, ACEH, MUTED, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8f98", "#0b0b0b"
OTHERS = (("MACE-H", MACEH), ("DeepH-E3", DEEPH), ("ACEhamiltonians", ACEH))
T = json.load(open(timings))
tag = lambda r: f"site_{r['site'][0]}_{r['site'][1]}_env_" + ("none" if not r["env"] else "_".join(str(float(v)) if i == 0 else str(int(v)) for i, v in enumerate(r["env"]))) \
    + (f"_nbond_{r['nbond']}_nsite_{r['nsite']}" if r["nbond"] != 12 else "")
ace, arch = {}, {}
for line in open(grid):
    r = json.loads(line)
    ace.setdefault(tag(r), {})[r["n_train"]] = 1e3 * r["metrics"][metric]
    e = r["env"]                                          # the architecture, as the figure names it
    arch[tag(r)] = " + ".join(["2c"] + ([f"site({r['site'][0]},{r['site'][1]})"] if r["site"][0] else [])
                              + ([f"env({int(e[1])},{int(e[2])},{int(e[3])},{int(e[4])})"] if e else [])) + (f", {r['nbond']}/{r['nsite']} radials" if r["nbond"] != 12 else "")
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
nb = 32                                                 # cost panel: every code at 32 training structures
na = nb if all(nb in v for v in ace.values()) else max(next(iter(ace.values())))
xs = [T["ace"][t] for t in order]
b.plot(xs, [ace[t][na] for t in order], color=ACE, lw=2, marker="o", ms=5)
for i, t in enumerate(order):                           # each point's architecture, sides alternating where points crowd
    left = i >= 3 and i % 2 == 1
    b.annotate(arch[t], (T["ace"][t], ace[t][na]), xytext=(-6 if left else 6, 0), textcoords="offset points", ha="right" if left else "left",
               va="center", fontsize=6.5, color=INK)
b.annotate("ace-jax", (xs[0], ace[order[0]][na]), xytext=(0, 9), textcoords="offset points", fontsize=8, color=INK)
for code, col in OTHERS:                                # the other codes: every network or basis size scored at nb, joined in cost order
    pts = [(cost(code), oth[code][nb], "")] if code in T and nb in oth.get(code, {}) else []
    pts += [(cost(code, sz), v, sz) for (sz, m), v in small.get(code, {}).items() if m == nb and code in T and sz in T[code]]
    pts.sort()
    if pts:
        b.plot([q[0] for q in pts], [q[1] for q in pts], color=col, marker="s", ms=6, ls="-", lw=1.5)
        for x, y, sz in pts:
            if sz:
                b.annotate(sz, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.5, color=INK)
        top = max(pts, key=lambda q: q[1])
        b.annotate(code + (".jl" if code == "ACEhamiltonians" else ""), top[:2], xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK)
b.set_xscale("log"); b.set_xlabel(f"Block prediction / ms per atom\n({T.get('machine', 'one CPU core')}; {nb} training structures)")
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
