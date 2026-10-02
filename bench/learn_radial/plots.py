"""Figures for the learned-radial benchmark (docs/dev/figures/learn-radial/): gate
scores, objective traces, the radials that changed most (initial vs learned,
and the change) over the pair-distance histogram, and the Legendre spectrum of
the change.

Run from a checkout whose runs/sige and runs/cantor hold bench/learn_radial/run.py
outputs (with per-lambda checkpoints); SYS below points at the moriarty data.

    uv run --with matplotlib python bench/learn_radial/plots.py
"""
import json, pathlib
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ace_jax.eval import load
from ace_jax.fit.data import build_dataset, load_configs, flat_edges
from ace_jax.fit.radial_model import (to_analytic, radial_gram, normalise, row_active,
                                      poly_env)
from ace_jax.basis.spec import build_spec

C = dict(init="#2a78d6", l0="#eb6834", l1="#1baf7a")          # reference categorical slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2})
SYM = {14: "Si", 32: "Ge", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni"}
H = pathlib.Path.home()
SYS = {"SiGe": (H/"acegp-run/sige/sige_base.npz", H/"acegp-run/sige/sige_mh1.xyz", 200, "runs/sige"),
       "Cantor": (H/"acegp-run/cantor/cantor_d4.npz", H/"ACEpotentials-jax/cantor1k_b_mh1.xyz", 150, "runs/cantor")}
out = pathlib.Path("runs/figs"); out.mkdir(exist_ok=True)
LAB = {"init": "initial", "learned_lam=0": "learned, λ=0", "learned_lam=0.01": "learned, λ=1e-2"}
COL = {"init": C["init"], "learned_lam=0": C["l0"], "learned_lam=0.01": C["l1"]}

# ---- fig 1: gate scores (bars) + score at common theta (dots), one panel per system
fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
for ax, (name, (_, _, _, run)) in zip(axs, SYS.items()):
    info = json.load(open(f"{run}/radial_info.json"))
    labs = list(info["scores"]); y = np.arange(len(labs))[::-1]
    for yi, k in zip(y, labs):
        ax.barh(yi, info["scores"][k], height=0.5, color=COL[k])
        ax.plot(info["scores_at_a0"][k], yi, "o", ms=7, mfc=SURF, mec=INK, mew=1.5, zorder=3)
        ax.text(info["scores"][k], yi, f"  {info['scores'][k]:.3g}", va="center", color=INK2)
    ax.set_yticks(y, [LAB[k] for k in labs]); ax.set_title(f"{name}: held-out score (lower is better)", loc="left")
    ax.set_xlim(0, 1.25 * max(max(info["scores"].values()), max(info["scores_at_a0"].values())))
    ax.grid(axis="y", visible=False)
axs[1].plot([], [], "o", mfc=SURF, mec=INK, label="scored at common θ (init MAP)"); axs[1].legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(out/"1_gate_scores.png", dpi=150)

# ---- fig 2: objective per accepted L-BFGS step
fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
for ax, (name, (_, _, _, run)) in zip(axs, SYS.items()):
    for lam, key in (("0", "learned_lam=0"), ("0.01", "learned_lam=0.01")):
        inf = json.load(open(f"{run}/lam_{lam}/radial_info.json"))
        tr = [inf["r0"]] + inf["trace"]
        ax.plot(range(len(tr)), tr, color=COL[key], label=LAB[key])
    b = inf["round_lengths"][0]; ax.axvline(b, color=INK2, lw=1)
    ax.text(b, ax.get_ylim()[1], " θ re-profiled", va="top", color=INK2, fontsize=8)
    ax.set_title(f"{name}: VarPro objective", loc="left"); ax.set_xlabel("L-BFGS step")
axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(out/"2_objective.png", dpi=150)

# ---- fig 3/4: radials that moved most, their change, and the Legendre spectrum of the change
stats = {}
for name, (mpath, data, ntrain, run) in SYS.items():
    m0, meta, z = load(mpath)
    model, _ = to_analytic(m0, 30)
    cf = load_configs(data, "mace_energy", "mace_force", "mace_virial")
    perm = np.random.default_rng(0).permutation(len(cf))
    ds = build_dataset([cf[i] for i in perm[:ntrain]], meta, np.asarray(z["E0"]), configs_per_batch=4)
    W0 = model.rnl_Wnlq; act = np.asarray(row_active(W0)); Q = radial_gram(model, ds)
    Ws = {"init": np.asarray(normalise(W0, Q, jnp.asarray(act))),
          "learned_lam=0": np.load(f"{run}/lam_0/rnl_Wnlq.npy"),
          "learned_lam=0.01": np.load(f"{run}/lam_0.01/rnl_Wnlq.npy")}
    Qn = np.asarray(Q)
    for k in ("learned_lam=0", "learned_lam=0.01"):                     # align gauge sign with init
        s_ = np.sign(np.einsum("abnq,abqp,abnp->abn", Ws[k], Qn, Ws["init"])); s_[s_ == 0] = 1
        Ws[k] = Ws[k] * s_[..., None]
    NZ = W0.shape[0]; els = [SYM[e] for e in meta["elements"]]
    _, Rnl, _ = build_spec(NZ, meta["order"], meta["totaldegree"], 1.5)
    dW = Ws["learned_lam=0"] - Ws["init"]
    rel = np.sqrt(np.einsum("abnq,abqp,abnp->abn", dW, Qn, dW))       # ||dR|| under data density (||R|| = 1)
    idx = np.argwhere(act); relv = rel[act]
    stats[name] = dict(rel={k: np.sqrt(np.einsum("abnq,abqp,abnp->abn", Ws[k]-Ws["init"], Qn, Ws[k]-Ws["init"]))[act]
                           for k in ("learned_lam=0", "learned_lam=0.01")},
                       spec_init=np.mean(Ws["init"][act] ** 2, axis=0),
                       spec_d={k: np.mean((Ws[k] - Ws["init"])[act] ** 2, axis=0) for k in ("learned_lam=0", "learned_lam=0.01")})
    top = idx[np.argsort(relv)[::-1][:3]]
    rs = {}
    for i in range(ds.n_batches):
        bt = jax.tree.map(lambda a: a[i], ds)
        rij, s, r_, mk = flat_edges(bt.rij, bt.nbr, bt.nbr_mask); mk = np.asarray(mk)
        r = np.linalg.norm(np.asarray(rij), axis=1)[mk]
        zi, zj = np.asarray(bt.node_z[s])[mk], np.asarray(bt.node_z[r_])[mk]
        for a, bb, rr in zip(zi, zj, r): rs.setdefault((int(a), int(bb)), []).append(rr)
    rcut = float(meta["rcut"]); rg = np.linspace(0.5, rcut, 800)
    def R(W, a, b, n):
        pe = np.asarray(poly_env(model, jnp.asarray(rg), jnp.full(rg.shape, a), jnp.full(rg.shape, b)))
        return pe @ W[a, b, n]
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6), sharex="col")
    for j, (a, b, n) in enumerate(top):
        for row in (0, 1):
            ax = axs[row, j]
            h, e = np.histogram(rs.get((int(a), int(b)), []), bins=60, range=(0.5, rcut))
            ax2 = ax.twinx(); ax2.fill_between(0.5 * (e[1:] + e[:-1]), h, step="mid", color=GRID, lw=0)
            ax2.set_yticks([]); ax2.set_zorder(0); ax.set_zorder(1); ax.patch.set_visible(False)
            ax.axhline(0, color=INK2, lw=0.8)
        r0_ = R(Ws["init"], a, b, n)
        for k in Ws:
            axs[0, j].plot(rg, R(Ws[k], a, b, n), color=COL[k], label=LAB[k], lw=2 if k == "init" else 1.6)
            if k != "init":
                axs[1, j].plot(rg, R(Ws[k], a, b, n) - r0_, color=COL[k], lw=1.6)
        axs[0, j].set_title(f"{els[a]}–{els[b]}  n={Rnl[n][0]} l={Rnl[n][1]}   ‖ΔR‖={rel[a,b,n]:.2f}", loc="left", fontsize=9)
        axs[1, j].set_xlabel("r (Å)")
    axs[0, 0].set_ylabel("R(r)"); axs[1, 0].set_ylabel("ΔR = learned − initial")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{name}: the 3 radials that changed most (λ=0), ‖R‖=1 under the data density; grey = pair distances", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out/f"3_radials_{name}.png", dpi=150)

fig, axs = plt.subplots(2, 2, figsize=(9.5, 5.4))
for c, (name, st) in enumerate(stats.items()):
    ax = axs[0, c]; bins = np.logspace(-4, 0.5, 46)
    for k, v in st["rel"].items():
        ax.hist(np.maximum(v, 1e-4), bins=bins, histtype="step", lw=2, color=COL[k],
                label=f"{LAB[k]}: median {np.median(v):.3f}, max {v.max():.2f}")
    ax.set_xscale("log"); ax.set_xlabel("‖ΔR‖ per radial (‖R_init‖ = 1)"); ax.set_ylabel("radials")
    ax.set_title(f"{name}: how much each radial moved", loc="left"); ax.legend(frameon=False, fontsize=8)
    ax = axs[1, c]; q = np.arange(st["spec_init"].size)
    ax.semilogy(q, st["spec_init"], color=C["init"], marker="o", ms=4, label="initial W (mean W_q²)")
    for k, v in st["spec_d"].items():
        ax.semilogy(q, v, color=COL[k], marker="o", ms=4, label=f"change ΔW, {LAB[k]}")
    ax.set_xlabel("Legendre degree q (in agnesi x)"); ax.set_title(f"{name}: spectrum of the radial change", loc="left")
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(out/"4_change_spectrum.png", dpi=150)
for name, st in stats.items():
    for k, v in st["rel"].items():
        sd = st["spec_d"][k]; hi = sd[15:].sum() / sd.sum()
        print(name, k, "median dR", round(float(np.median(v)), 4), "p90", round(float(np.percentile(v, 90)), 3), "max", round(float(v.max()), 3), "frac change power q>=15:", round(float(hi), 3))
