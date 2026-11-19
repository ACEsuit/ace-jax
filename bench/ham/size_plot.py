"""Time against system size from positions to the bands at one k-point: block prediction by each
code, Bloch sums and the eigensolver (`size_time.py`, `maceh_time.py` on `tile_deeph.py`
supercells).  Top: seconds per structure against atoms, with N and N^3 guides; bottom: the share
of each step, one panel per code.
    python bench/ham/size_plot.py <size_time.jsonl> <figure.png> <title> [<code>=<maceh_time log of "N <line>" rows> ...]"""
import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

src, fig, title = sys.argv[1:4]
COL = {"ace-jax": "#2a78d6", "MACE-H": "#eb6834", "DeepH-E3": "#1baf7a"}
BLOCH, EIG, MUTED, INK, SURF = "#b9bcc2", "#52514e", "#8a8f98", "#0b0b0b", "#ffffff"
rows = {}
for line in open(src):
    r = json.loads(line)
    rows[r["atoms"]] = r                                 # the last run of a size
N = np.array(sorted(rows))
pred = {"ace-jax": np.array([rows[n].get("predict_s", np.nan) for n in N])}
for arg in sys.argv[4:]:
    code, path = arg.split("=")
    t = {}
    for line in open(path):
        m = re.match(r"(\d+) .* ([\d.]+) ms per structure", line)
        if m:
            t[int(m.group(1))] = float(m.group(2)) / 1e3
    pred[code] = np.array([t.get(n, np.nan) for n in N])
bloch, eig = (np.array([rows[n][k] for n in N]) for k in ("bloch_s", "eig_s"))
f = plt.figure(figsize=(10.0, 7.2))
gs = f.add_gridspec(2, len(pred), height_ratios=(1.25, 1), hspace=0.42, wspace=0.12)
a = f.add_subplot(gs[0, :])
for code, t in pred.items():
    a.plot(N, t, color=COL[code], lw=2, marker="o", ms=5)
    ok = np.flatnonzero(np.isfinite(t))
    a.annotate(f"{code}: block prediction", (N[ok[0]], t[ok[0]]), xytext=(4, 7 if code != "MACE-H" else -11), textcoords="offset points", fontsize=8, color=INK)
a.plot(N, eig, color=EIG, lw=2, marker="s", ms=5)
a.annotate("eigensolver, one k-point", (N[-1], eig[-1]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
a.plot(N, bloch, color=BLOCH, lw=2, marker="s", ms=5)
a.annotate("Bloch sums H(k), S(k)", (N[-1], bloch[-1]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
g = np.array([N[0], N[-1]], float)                       # guides through the first points: O(N) and O(N^3)
a.plot(g, pred["ace-jax"][0] * g / g[0] * 0.5, color=MUTED, lw=0.8, ls=":"); a.annotate("∝ N", (g[1], pred["ace-jax"][0] * g[1] / g[0] * 0.5), xytext=(7, -8), textcoords="offset points", fontsize=7, color=MUTED)
a.plot(g, eig[0] * (g / g[0]) ** 3 * 0.5, color=MUTED, lw=0.8, ls=":"); a.annotate("∝ N³", (g[1], eig[0] * (g[1] / g[0]) ** 3 * 0.5), xytext=(7, -10), textcoords="offset points", fontsize=7, color=MUTED)
a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xticks(N); a.set_xticklabels([str(n) for n in N])
a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
a.set_xlim(N[0] * 0.85, N[-1] * 2.0); a.set_xlabel("Atoms (9 orbitals each)"); a.set_ylabel("Time per structure / s")
a.grid(True, which="major", color=MUTED, lw=0.3, alpha=0.5)
x = np.arange(len(N))
for i, (code, t) in enumerate(pred.items()):
    b = f.add_subplot(gs[1, i])
    tot = t + bloch + eig
    base = np.zeros(len(N))
    for part, col in ((t, COL[code]), (bloch, BLOCH), (eig, EIG)):
        share = 100 * part / tot
        b.bar(x, share, bottom=base, color=col, width=0.72, edgecolor=SURF, linewidth=2)
        for xi, (s0, sh) in enumerate(zip(base, share)):
            if np.isfinite(sh) and sh >= 9:
                b.text(xi, s0 + sh / 2, f"{sh:.0f}", ha="center", va="center", fontsize=7, color=SURF if col != BLOCH else INK)
        base = base + np.nan_to_num(share)
    for xi, v in enumerate(tot):
        if np.isfinite(v):
            b.text(xi, 102, f"{v:.3g} s", ha="center", va="bottom", fontsize=7, color=INK)
    b.set_xticks(x); b.set_xticklabels([str(n) for n in N]); b.set_ylim(0, 112); b.set_yticks([0, 25, 50, 75, 100])
    b.set_title(code, fontsize=9, loc="left", color=INK); b.set_xlabel("Atoms")
    if i == 0:
        b.set_ylabel("Share of the time / %")
        for name, col in (("block prediction (the code's colour)", COL[code]), ("Bloch sums", BLOCH), ("eigensolver", EIG)):
            b.bar([0], [0], color=col, label=name)
        b.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=3, frameon=False, fontsize=8)
    else:
        b.tick_params(labelleft=False)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
f.suptitle(title, fontsize=10, x=0.125, y=0.95, ha="left")
f.savefig(fig, dpi=200, bbox_inches="tight")
