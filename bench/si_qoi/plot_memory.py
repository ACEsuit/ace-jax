"""Peak memory vs cost for the accuracy-cost families: one panel per device, the Pareto front of the two.

  A  A100: the process's peak GPU memory (NVML, bench/scaling/sweep.run_capped) on the timing cell (32768 atoms).
     ace-jax (lammps-jax) rows come only from --gpu-acejax files: before the export child ran without
     preallocation (scaling/run_lammps.export_bundle) it was in the measured tree, and every row peaked at its
     75% pool, 30.04 GiB
  B  CPU node: the peak resident memory summed over the process tree (all 32 MPI ranks for LAMMPS codes) on the
     CPU timing cell (4096 atoms)

x is the run's own cost (µs per atom-step, 1e6 / atom_steps_per_s of the row that gave the memory), one line per
family and code through its sizes in ladder order (plot_cost.trainable's parameter count); the grey step is the
Pareto front over the points accurate enough to compare (test force RMSE <= MAX_RMSE_F; the rest of each ladder
is faded): no cheaper such point uses less memory.  Each family keeps
plot_cost's colour and marker; a second code for the same model (ML-PACE beside ace-jax for PACE, standalone
float32 MACE beside LAMMPS float64) is drawn dotted and hollow.  Rows are the timing runs' max over repeats.

    python bench/si_qoi/plot_memory.py --gpu sulis-gpu-v2.jsonl [...] --gpu-acejax sulis-gpu-mem.jsonl --cpu sulis-cpu-pace.jsonl [...] [--out fig.png]
"""
import argparse, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plot_cost import FAM, INK, MUTED, family_of, load_rows, style  # noqa: E402

SKIP = {"ace_E10", "ace_pshape_E10"}            # refits of a timed basis: the same model, the same memory
WL = "ace_wl"                                   # the wL ladder: its own figure (plot_cost --wl), not here
CODE = {"acejax-ace": "ace-jax", "acejax-pace": "ace-jax", "mlpace": "ML-PACE", "gap": "QUIP"}
MAX_RMSE_F = 0.10                               # eV/A: the front compares models of similar accuracy
MAIN_CODE = {"pace": "acejax-pace"}             # the family's own line style; the other code is dotted


def code_label(r):
    if r["code"] == "mace":
        return "LAMMPS (Symmetrix), float64" if r["mode"] == "lammps" else "PyTorch, float32"
    return CODE.get(r["code"], r["code"])


def series(files, key, acejax_files=()):
    """{(family, code label): [(n_params, µs per atom-step, bytes, size)]} from the ok timing rows that recorded `key`
    (several runs of one size: the fastest, with its memory)."""
    nparams = {(r["family_raw"], r["size"]): r["n_params"] for r in ROWS}
    rmse = {(r["family_raw"], r["size"]): r["rmse_F"] for r in ROWS}
    out = {}
    for f in [*files, *acejax_files]:
        for line in open(f):
            r = json.loads(line)
            if r.get("status") != "ok" or r.get("mode") not in ("lammps", "standalone") or r["family"] == WL:
                continue
            fam = family_of(r)
            if fam in SKIP or fam not in FAM or r.get(key) is None:
                continue
            if key.startswith("gpu") and r["code"].startswith("acejax") and f not in acejax_files:
                continue         # the export child's 75% pool: 30.04 GiB for every bundle, o3d8 to f3200
            n = nparams.get((r["family"], r["size"]))
            if n and r.get("atom_steps_per_s"):
                k = (fam, code_label(r), r["code"] == MAIN_CODE.get(r["family"], r["code"]) and
                     not (r["code"] == "mace" and r["mode"] == "standalone"))
                pts = out.setdefault(k, {})
                us = 1e6 / r["atom_steps_per_s"]
                if r["size"] not in pts or us < pts[r["size"]][1]:
                    pts[r["size"]] = (n, us, r[key], r["size"], rmse[(r["family"], r["size"])])
    return {k: sorted(v.values()) for k, v in out.items()}


def panel(ax, data, title, xlabel):
    for (fam, code, main), pts in sorted(data.items(), key=lambda kv: (list(FAM).index(kv[0][0]), not kv[0][2])):
        lab, col, mk, ls, filled, _ = FAM[fam]
        x, y = [p[1] for p in pts], [p[2] / 2**30 for p in pts]
        kw = dict(color=col, marker=mk, ls=ls if main else (0, (1, 1.5)), lw=2 if main else 1.6,
                  ms=8 if main else 6, mfc=col if (filled and main) else "white", mec=col, mew=1.6)
        ax.plot(x, y, **{**kw, "alpha": 0.25})                    # the whole ladder, faded
        good = [(p[1], p[2] / 2**30) for p in pts if p[4] <= MAX_RMSE_F]
        ax.plot(*zip(*good) if good else ([], []), **kw, label=f"{lab} ({code})",
                gid=(list(FAM).index(fam), not main))
    front, best = [], float("inf")            # Pareto: by cost, keep each point using less memory than all cheaper
    for us, b in sorted((p[1], p[2] / 2**30) for pts in data.values() for p in pts if p[4] <= MAX_RMSE_F):
        if b < best:
            front.append((us, b)); best = b
    if front:
        xmax = max(p[1] for pts in data.values() for p in pts)
        xs, ys = zip(*front, (xmax, front[-1][1]))         # to the right edge: a one-point front still shows
        ax.step(xs, ys, where="post", color="#9a9893", lw=6, alpha=0.35, zorder=0, label=f"Pareto front, force RMSE ≤ {MAX_RMSE_F} eV/Å")
    style(ax, xlabel, "peak memory (GiB, log)"); ax.set_yscale("log")
    ax.set_title(title, loc="left", fontweight="bold", color=INK)


def main():
    global ROWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", nargs="+", default=[]); ap.add_argument("--cpu", nargs="+", default=[])
    ap.add_argument("--gpu-acejax", nargs="+", default=[], help="GPU rows whose ace-jax memory is real")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "results/memory.png"))
    a = ap.parse_args()
    ROWS = _rows_with_raw()
    gpu, cpu = series(a.gpu, "gpu_peak_process_bytes", a.gpu_acejax), series(a.cpu, "peak_rss_bytes")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    panel(axes[0], gpu, "A  A100: peak GPU memory, 32768 atoms", "A100 cost (µs per atom-step, log)")
    panel(axes[1], cpu, "B  CPU node: peak host memory (all ranks), 4096 atoms", "CPU node cost (µs per atom-step, log)")
    seen = {}
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    seen = dict(sorted(seen.items(), key=lambda kv: kv[1].get_gid() or (-1, False)))   # the front first     # FAM order, the main code first
    fig.legend(list(seen.values()), list(seen), loc="upper center", ncol=4, frameon=False, fontsize=8.5)
    fig.text(0.5, 0.005, "GPU: NVML per-process peak, the CUDA context included; lammps-jax grows its XLA pool on demand "
             "(no preallocation), so ace-jax points are what the model used.\n"
             "A dotted line is a second code for the same "
             f"model; faded points have force RMSE > {MAX_RMSE_F} eV/Å.\nCPU: LAMMPS codes run 32 MPI ranks, each holding the model; "
             "ace-jax runs standalone as one process.  MACE L=1 c64 ran out of GPU memory in LAMMPS float64 (no point).",
             ha="center", va="bottom", fontsize=8, color=MUTED)
    n_leg = len(seen) // 4 + 1
    fig.tight_layout(rect=(0, 0.08, 1, 1 - 0.28 * n_leg / fig.get_figheight()))
    fig.savefig(a.out, dpi=150, facecolor="#fcfcfb"); print(f"wrote {a.out}")


def _rows_with_raw():
    """load_rows, keeping the metrics family (load_rows replaces it with the plotted one)."""
    from plot_cost import METRICS
    raw = [json.loads(line) for line in open(METRICS)]
    return [{**r, "family_raw": q["family"]} for r, q in zip(load_rows(), raw)]


if __name__ == "__main__":
    main()
