"""Compare prior configurations of the learned-radial benchmark: held-out
scores, the change in the most-moved radials, and the Legendre spectrum of the
change.  Reads bench/learn_radial/run.py outputs (with per-lambda checkpoints)
for each configuration in CONFIGS; SYS points at the moriarty data.

    JAX_PLATFORMS=cpu uv run --with matplotlib python bench/learn_radial/compare_plots.py
"""
import json
import pathlib

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ace_jax.construct.spec import build_spec
from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, flat_edges, load_configs
from ace_jax.fit.radial_model import normalise, poly_env, radial_gram, row_active, to_analytic

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # reference categorical slots 1-4
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "lines.linewidth": 2})
SYM = {14: "Si", 32: "Ge", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni"}
H = pathlib.Path.home()
SYS = {"SiGe": (H / "acegp-run/sige/sige_base.npz", H / "acegp-run/sige/sige_mh1.xyz", 200),
       "Cantor": (H / "acegp-run/cantor/cantor_d4.npz", H / "ACEpotentials-jax/cantor1k_b_mh1.xyz", 150)}
# (label, run root, n_q, checkpoint dir) -- the learned candidates compared in figs B and C
CONFIGS = [("n_q=30, no prior", "runs", 30, "lam_0"),
           ("n_q=30, spectral 1e-4", "runs/spec", 30, "lam_0_spec=0.0001"),
           ("n_q=12, λ=0.1", "runs/nq12", 12, "lam_0.1")]
ALL_RUNS = [("n_q=30", "runs"), ("n_q=30 + spectral", "runs/spec"), ("n_q=12", "runs/nq12")]
out = pathlib.Path("runs/figs_compare")
out.mkdir(parents=True, exist_ok=True)

# ---- fig A: held-out score of every candidate, per system (bars; dot = scored at common theta)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, name in zip(axs, SYS):
    rows = []
    for fam, root in ALL_RUNS:
        info = json.load(open(f"{root}/{name.lower()}/radial_info.json"))
        for k, v in info["scores"].items():
            if k == "init" and rows:
                continue                                              # init is identical across runs
            lab = "initial" if k == "init" else f"{fam}: {k.replace('learned_', '')}"
            rows.append((lab, v, info["scores_at_a0"][k]))
    y = np.arange(len(rows))[::-1]
    init = rows[0][1]
    for yi, (lab, v, v0) in zip(y, rows):
        ax.barh(yi, v, height=0.55, color=INK2 if lab == "initial" else SERIES[0])
        ax.plot(v0, yi, "o", ms=6, mfc=SURF, mec=INK, mew=1.4, zorder=3)
        ax.text(v, yi, f"  {v:.3g} ({100 * (v / init - 1):+.0f}%)" if lab != "initial" else f"  {v:.3g}",
                va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 1.45 * max(max(r[1], r[2]) for r in rows))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{name}: held-out score (lower is better)", loc="left")
axs[1].plot([], [], "o", mfc=SURF, mec=INK, label="scored at common θ")
axs[1].legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out / "A_scores.png", dpi=150)

# ---- figs B, C: radial changes and their spectra
specs = {}
for name, (mpath, data, ntrain) in SYS.items():
    m0, meta, z = load(mpath)
    cf = load_configs(data, "mace_energy", "mace_force", "mace_virial")
    perm = np.random.default_rng(0).permutation(len(cf))
    ds = build_dataset([cf[i] for i in perm[:ntrain]], meta, np.asarray(z["E0"]), configs_per_batch=4)
    NZ = len(meta["elements"])
    els = [SYM[e] for e in meta["elements"]]
    _, Rnl, _ = build_spec(NZ, meta["order"], meta["totaldegree"], 1.5)
    rcut = float(meta["rcut"])
    rg = np.linspace(0.5, rcut, 800)
    rs = {}
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a: a[i], ds)
        rij, s, r_, mk = flat_edges(bt.rij, bt.nbr, bt.nbr_mask)
        mk = np.asarray(mk)
        r = np.linalg.norm(np.asarray(rij), axis=1)[mk]
        for a, b, rr in zip(np.asarray(bt.node_z[s])[mk], np.asarray(bt.node_z[r_])[mk], r):
            rs.setdefault((int(a), int(b)), []).append(rr)
    cand = []
    for lab, root, n_q, ck in CONFIGS:
        model, _ = to_analytic(m0, n_q)
        act = np.asarray(row_active(model.rnl_Wnlq))
        Q = np.asarray(radial_gram(model, ds))
        Wi = np.asarray(normalise(model.rnl_Wnlq, jnp.asarray(Q), jnp.asarray(act)))
        Wl = np.load(f"{root}/{name.lower()}/{ck}/rnl_Wnlq.npy")
        sgn = np.sign(np.einsum("abnq,abqp,abnp->abn", Wl, Q, Wi))
        sgn[sgn == 0] = 1
        dW = Wl * sgn[..., None] - Wi                                  # sign is a gauge freedom
        rel = np.sqrt(np.einsum("abnq,abqp,abnp->abn", dW, Q, dW))
        cand.append(dict(lab=lab, model=model, dW=dW, rel=rel, act=act,
                         spec=np.mean(dW[act] ** 2, axis=0), spec_init=np.mean(Wi[act] ** 2, axis=0)))
    specs[name] = cand
    # radials that moved most in the capped-span run (the best-scoring configuration)
    ref = cand[-1]
    top = np.argwhere(ref["act"])[np.argsort(ref["rel"][ref["act"]])[::-1][:3]]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), sharey=False)
    for ax, (a, b, n) in zip(axs, top):
        h, e = np.histogram(rs.get((int(a), int(b)), []), bins=60, range=(0.5, rcut))
        ax2 = ax.twinx()
        ax2.fill_between(0.5 * (e[1:] + e[:-1]), h, step="mid", color=GRID, lw=0)
        ax2.set_yticks([])
        ax2.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        ax.axhline(0, color=INK2, lw=0.8)
        for c, col in zip(cand, SERIES[1:]):
            pe = np.asarray(poly_env(c["model"], jnp.asarray(rg), jnp.full(rg.shape, a), jnp.full(rg.shape, b)))
            ax.plot(rg, pe @ c["dW"][a, b, n], color=col, lw=1.6,
                    label=f"{c['lab']} (‖ΔR‖={c['rel'][a, b, n]:.2f})")
        ax.set_title(f"{els[a]}–{els[b]}  n={Rnl[n][0]} l={Rnl[n][1]}", loc="left", fontsize=9)
        ax.set_xlabel("r (Å)")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axs[0].set_ylabel("ΔR = learned − initial")
    fig.suptitle(f"{name}: change in the 3 radials that moved most (n_q=12 run); ‖R_init‖=1; grey = pair distances",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"B_radial_change_{name}.png", dpi=150)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, (name, cand) in zip(axs, specs.items()):
    q = np.arange(cand[0]["spec_init"].size)
    ax.semilogy(q, cand[0]["spec_init"], color=SERIES[0], marker="o", ms=3, label="initial W")
    for c, col in zip(cand, SERIES[1:]):
        ax.semilogy(np.arange(c["spec"].size), c["spec"], color=col, marker="o", ms=3, label=f"ΔW, {c['lab']}")
    ax.set_xlabel("Legendre degree q")
    ax.set_ylabel("mean coefficient²")
    ax.set_title(f"{name}: spectrum of the radial change", loc="left")
    ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(out / "C_change_spectrum.png", dpi=150)

for name, cand in specs.items():
    for c in cand:
        v = c["rel"][c["act"]]
        hi = c["spec"][len(c["spec"]) // 2:].sum() / c["spec"].sum()
        print(f"{name:7s} {c['lab']:24s} median ‖ΔR‖ {np.median(v):.4f}  p90 {np.percentile(v, 90):.3f}"
              f"  max {v.max():.3f}  power in upper half of q: {hi:.3f}")
