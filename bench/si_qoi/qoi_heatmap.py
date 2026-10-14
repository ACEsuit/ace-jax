"""Per-QoI breakdown of the silicon QoI scores: one row per model (families in ladder order, GAP-18 on
top), one column per QoI, each cell the signed error / tolerance against CASTEP (score.py).  The 2-D
counterpart of the GAP-18 paper's 3-D error bars (Bartok et al., PRX 8, 041048 (2018)).

Colour is binned on |error| / tolerance, so "within tolerance" reads at a glance: grey within half
the tolerance, a faint tint up to it, then 1-2, 2-5 and >5 x tolerance.  Hue gives the sign: blue
where the model underestimates DFT, red where it overestimates.  The phonon column is an RMS over
the band structure, so it has no sign (always red).  Each cell prints |error| / tolerance.

    python bench/si_qoi/qoi_heatmap.py [--out fig.png]
"""
import argparse, pathlib, sys

import matplotlib
import matplotlib.transforms  # noqa: E402,F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compare import qois  # noqa: E402
from plot_cost import FAM, INK, MUTED, ROOT, load_rows  # noqa: E402
from score import EXCLUDED, TOLERANCES, score  # noqa: E402

# diverging blue <-> red with the reference palette's neutral midpoint (dataviz palette.md): 4 steps per arm
BLUE = ["#104281", "#256abf", "#6da7ec", "#cde2fb"]          # >5, 2-5, 1-2, 0.5-1 (underestimate)
RED = ["#f9d4d1", "#ef8d87", "#d63a3a", "#8f1d1d"]           # 0.5-1, 1-2, 2-5, >5 (overestimate)
NEUTRAL = "#f0efec"
BOUNDS = [-1e9, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 1e9]
DARK = {BLUE[0], BLUE[1], RED[2], RED[3]}                    # cells that take white text

GROUPS = [("elastic", "diamond_"), ("phase energies vs diamond", "dE_"), ("point defects", "_energy"),
          ("surface & planar", ""), ("phonons", "phonon")]
SHORT = {"diamond_bulk_modulus": "B", "diamond_c11": "C11", "diamond_c12": "C12", "diamond_c44": "C44",
         "vacancy_energy": "vacancy", "dumbbell_interstitial_energy": "dumbbell int.",
         "hexagonal_interstitial_energy": "hexagonal int.", "tetrahedral_interstitial_energy": "tetrahedral int.",
         "surface_100 (J/m2)": "(100) surface", "gamma_us glide (J/m2)": "γ_us glide",
         "gamma_us shuffle (J/m2)": "γ_us shuffle", "decohesion_max_stress": "decohesion σ_max",
         "phonon_f rms vs DFT (THz)": "phonon rms"}


def short(k):
    return SHORT.get(k) or k.replace("dE_", "").replace(" (meV/at)", "").replace("hcp_short_and_fat", "hcp (compr.)")


def group_of(k):
    if k.startswith("diamond_"):
        return 0
    if k.startswith("dE_"):
        return 1
    if k.endswith("_energy"):
        return 2
    return 4 if k.startswith("phonon") else 3


def model_rows():
    """[(family, label, qoi model name)]: GAP-18, then each family's ladder by parameter count."""
    rows = load_rows()                                          # plot_cost's families (MACE split by L)
    out = [("gap18", "published", "GAP18_py3")]
    for fam in FAM:
        rs = sorted((r for r in rows if r["family"] == fam), key=lambda r: r.get("n_params") or 0)
        out += [(fam, r["size"], FAM[fam][5].format(s=r["size"])) for r in rs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "bench/si_qoi/results/qoi_heatmap.png"))
    a = ap.parse_args()
    ref = qois("CASTEP")
    keys = [k for k, _, _ in TOLERANCES]
    keys = sorted(keys, key=lambda k: (group_of(k), keys.index(k)))          # stable within each group
    rows, M, gm = [], [], []
    for fam, lab, name in model_rows():
        sc = score(name, ref)
        if sc["n_qoi"] != sc["n_total"]:
            continue                                                         # QoIs still running
        rows.append((fam, lab)); M.append([sc["signed"].get(k, np.nan) for k in keys]); gm.append(sc["gm_norm"])
    M = np.array(M)

    cmap = ListedColormap(BLUE + [NEUTRAL] + RED)
    norm = BoundaryNorm(BOUNDS, cmap.N)
    nr, nc = M.shape
    fig, ax = plt.subplots(figsize=(0.62 * nc + 4.2, 0.30 * nr + 3.4))
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.pcolormesh(np.arange(nc + 1), np.arange(nr + 1), M, cmap=cmap, norm=norm, edgecolors="#fcfcfb", linewidth=1.5)
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            if np.isfinite(v):
                c = cmap(norm(v)); hexc = matplotlib.colors.to_hex(c)
                ax.text(j + 0.5, i + 0.5, f"{abs(v):.1f}" if abs(v) < 10 else f"{abs(v):.0f}", ha="center",
                        va="center", fontsize=6.5, color="white" if hexc in DARK else INK)
    ax.set_xlim(0, nc); ax.set_ylim(nr, 0)
    ax.set_xticks(np.arange(nc) + 0.5, [short(k) for k in keys], rotation=55, ha="right", fontsize=8, color=INK)
    ax.set_yticks(np.arange(nr) + 0.5, [lab for _, lab in rows], fontsize=8, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # family bands on the left, separators between families and QoI groups
    fams = [f for f, _ in rows]
    starts = [i for i in range(nr) if i == 0 or fams[i] != fams[i - 1]]
    yband = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)   # x in axes, y in rows
    for i in starts[1:]:
        ax.axhline(i, color=INK, lw=0.8)
    for i, i0 in enumerate(starts):
        i1 = starts[i + 1] if i + 1 < len(starts) else nr
        f = fams[i0]
        lab, col = ("GAP-18", MUTED) if f == "gap18" else (FAM[f][0], FAM[f][1])
        ax.text(-0.082, (i0 + i1) / 2, lab, transform=yband, ha="right", va="center", fontsize=8.5, color=INK,
                fontweight="bold", clip_on=False)
        ax.plot([-0.072, -0.072], [i0 + 0.1, i1 - 0.1], transform=yband, color=col, lw=4, clip_on=False,
                solid_capstyle="butt")
    gidx = [group_of(k) for k in keys]
    for j in range(1, nc):
        if gidx[j] != gidx[j - 1]:
            ax.axvline(j, color=INK, lw=0.8)
    for g, (gname, _) in enumerate(GROUPS):
        js = [j for j in range(nc) if gidx[j] == g]
        if js:
            ax.annotate(gname, ((js[0] + js[-1] + 1) / 2, 0), xytext=(0, 6), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8.5, color=INK, fontweight="bold", annotation_clip=False)
    # the per-model summary that panel B plots
    for i, g in enumerate(gm):
        ax.annotate(f"{g:.2f}", (nc, i + 0.5), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK, annotation_clip=False)
    ax.annotate("geo.\nmean", (nc, 0), xytext=(8, 6), textcoords="offset points", va="bottom", fontsize=8,
                color=MUTED, annotation_clip=False)
    # binned legend
    cax = fig.add_axes([0.30, 0.9 / fig.get_figheight(), 0.42, 0.018])
    labels = [">5", "2–5", "1–2", "0.5–1", "<0.5", "0.5–1", "1–2", "2–5", ">5"]
    for k, (c, l) in enumerate(zip(BLUE + [NEUTRAL] + RED, labels)):
        cax.add_patch(plt.Rectangle((k, 0), 0.94, 1, color=c))
        cax.text(k + 0.47, -0.4, l, ha="center", va="top", fontsize=7.5, color=INK)
    cax.set_xlim(0, 9); cax.set_ylim(0, 1); cax.axis("off")
    cax.text(2, 1.5, "model underestimates DFT", ha="center", va="bottom", fontsize=8, color=MUTED)
    cax.text(7, 1.5, "model overestimates DFT", ha="center", va="bottom", fontsize=8, color=MUTED)
    cax.text(4.5, 1.5, "|error| / tolerance", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title("Silicon QoIs vs CASTEP: error / tolerance per model", loc="left", color=INK, fontweight="bold",
                 pad=26, fontsize=11)
    tol = "; ".join(f"{short(k)} {t:g}{' rel' if kind == 'rel' else ''}" for k, t, kind in TOLERANCES
                    if not k.startswith("dE_") or k == "dE_bc8 (meV/at)").replace("bc8 10", "phase energies 10 meV/atom")
    fig.text(0.5, 0.002, "tolerances: " + tol + " THz (phonon rms, unsigned); absolute tolerances in eV unless stated. "
             "Left out: " + "; ".join(f"{short(k)} ({why})" for k, why in EXCLUDED.items()), ha="center", va="bottom", fontsize=7,
             color=MUTED, wrap=True)
    fig.subplots_adjust(left=0.22, right=0.93, top=1 - 1.0 / fig.get_figheight(), bottom=2.4 / fig.get_figheight())
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150); print("wrote", a.out, f"({nr} models x {nc} QoIs)")


if __name__ == "__main__":
    main()
