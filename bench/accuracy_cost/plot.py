"""Accuracy-cost figures: Pareto (test force RMSE vs throughput, one line per
family and code, CPU/GPU panels), smallest model at target, per-config-type heatmap.

    python -m accuracy_cost.plot [target ...]       # default targets 0.1 0.15 eV/A

Reads results/metrics.jsonl and results/lestrade-{cpu,gpu}.jsonl; writes figs/.
Reference lines: gap18_published from metrics.jsonl, plus refs.json (a list of
[name, rmse_F] pairs, e.g. the refit GAP s9000; a name there overrides)."""
import json
import sys

import numpy as np

REFERENCE = {"gap18_published": "GAP-18 published"}      # metrics families drawn as lines, never swept
# categorical slots in fixed order (dataviz reference palette, light); code and
# dtype are a second encoding (marker and line style), so identity is never colour alone
COLORS = {"ace": "#2a78d6", "ace_lr": "#eb6834", "ace_gp": "#1baf7a", "pace": "#eda100",
          "mace": "#e87ba4", "gap": "#008300"}
MARKERS = {"acejax-ace": "o", "acejax-gp": "D", "acejax-pace": "o", "mlpace": "s", "mace": "o", "gap": "^"}


def plateau_markers(rows):
    """Per row: "clean" (N on the plateau), "below" (N/8 throughput differs by
    > 20%) or "unknown" (the N/8 check failed, or no flag)."""
    return ["below" if r.get("not_plateau") is True else "clean" if r.get("not_plateau") is False else "unknown"
            for r in rows]


def _timed(timings, device):
    """(family, size) -> ok timing rows on `device` with a throughput."""
    out = {}
    for t in timings:
        if t.get("device") == device and t.get("status", "ok") == "ok" and t.get("atom_steps_per_s"):
            out.setdefault((t["family"], t["size"]), []).append(t)
    return out


def _join(metrics, timings, device):
    """One row per (model, code, dtype, mode) timed on `device`: metrics + timing."""
    by = {(m["family"], m["size"]): m for m in metrics}
    out = []
    for k, ts in _timed(timings, device).items():
        if k in by:
            for t in ts:
                out.append({**by[k], **{f: t.get(f) for f in ("atom_steps_per_s", "not_plateau", "code",
                                                              "dtype", "mode")}})
    return out


def smallest_at_target(metrics, timings, target, device):
    """Per swept family: the model with the fewest n_params whose test rmse_F
    meets `target` (ties: the faster on `device`), with its best throughput
    there.  Throughput is reported, never used to choose.  A family whose models
    all miss is "not reached" (never extrapolated); a chosen model without a
    timing on `device` has atom_steps_per_s None and note "not timed"."""
    timed = _timed(timings, device)
    best_t = lambda m: max(timed.get((m["family"], m["size"]), []), key=lambda t: t["atom_steps_per_s"],
                           default=None)
    rows = []
    for f in sorted({m["family"] for m in metrics} - set(REFERENCE)):
        ok = [m for m in metrics if m["family"] == f and m.get("rmse_F") is not None and m["rmse_F"] <= target]
        if not ok:
            rows.append({"family": f, "size": None, "note": "not reached"})
            continue
        speed = lambda m: (best_t(m) or {}).get("atom_steps_per_s") or 0.0
        m = min(ok, key=lambda m: (m["n_params"], -speed(m)))
        t = best_t(m)
        r = {"family": f, "size": m["size"], "rmse_F": m["rmse_F"], "n_params": m["n_params"],
             "atom_steps_per_s": t and t["atom_steps_per_s"], "code": t and t.get("code"),
             "dtype": t and t.get("dtype"), "not_plateau": t and t.get("not_plateau")}
        if t is None:
            r["note"] = "not timed"
        rows.append(r)
    return rows


def reference_lines(metrics, refs):
    """[(name, rmse_F)]: the REFERENCE families' metrics rows, then refs.json's
    pairs; a refs.json name replaces a metrics line of the same name."""
    out = {REFERENCE[m["family"]]: m["rmse_F"] for m in metrics if m["family"] in REFERENCE}
    out.update({name: float(v) for name, v in refs})
    return list(out.items())


def pareto(metrics, timings, targets, out, refs=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = reference_lines(metrics, refs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, dev in zip(axes, ("cpu", "gpu")):
        j = _join(metrics, timings, dev)
        for f, code, dt, mode in sorted({(r["family"], r["code"], r["dtype"], r["mode"]) for r in j},
                                        key=lambda k: tuple(map(str, k))):
            sel = sorted((r for r in j if (r["family"], r["code"], r["dtype"], r["mode"]) == (f, code, dt, mode)),
                         key=lambda r: r["rmse_F"])
            x, y = [r["rmse_F"] for r in sel], [r["atom_steps_per_s"] for r in sel]
            ls = "--" if code == "mlpace" else ":" if dt == "float32" else "-"
            col = COLORS.get(f, "0.3")
            ax.plot(x, y, ls=ls, lw=2, marker=MARKERS.get(code, "o"), ms=6, color=col,
                    label=f"{f}: {code} {mode}" + (" f32" if dt == "float32" else ""))
            for xi, yi, kind in zip(x, y, plateau_markers(sel)):
                if kind == "below":                              # N not on the throughput plateau
                    ax.plot(xi, yi, "x", color="k", ms=8)
                elif kind == "unknown":                          # the N/8 check failed: open marker
                    ax.plot(xi, yi, marker=MARKERS.get(code, "o"), ms=6, color=col, mfc="white", ls="none")
        for t in targets:
            ax.axvline(t, color="0.75", lw=1)
        for name, v in lines:
            ax.axvline(v, color="0.35", ls=(0, (1, 2)), lw=1)
            ax.annotate(name, (v, 1), xycoords=("data", "axes fraction"), rotation=90, va="top", ha="right",
                        fontsize=7, color="0.35")
        ax.set(xscale="log", yscale="log", xlabel="test force RMSE (eV/Å)", title=f"lestrade {dev}")
        ax.grid(True, which="major", color="0.92", lw=0.8)
        if j:
            from matplotlib.lines import Line2D
            h, _ = ax.get_legend_handles_labels()
            h += [Line2D([], [], marker="x", color="k", ls="none", label="x: N below plateau"),
                  Line2D([], [], marker="o", color="0.4", mfc="white", ls="none", label="open: plateau unknown")]
            ax.legend(handles=h, fontsize=6, loc="best")
    axes[0].set_ylabel("atom-steps / s")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def heatmap(metrics, sizes, out):
    """Test force RMSE per config type for the chosen (family, size) models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(m["family"], m["size"]): m for m in metrics}
    sel = [by[tuple(k)] for k in sizes if tuple(k) in by]
    types = sorted({t for m in sel for t in m.get("per_type", {})})
    A = np.array([[(m["per_type"].get(t) or {}).get("rmse_F", np.nan) for t in types] for m in sel], float)
    fig, ax = plt.subplots(figsize=(2.5 + 0.35 * len(types), 1.2 + 0.4 * len(sel)))
    im = ax.imshow(A, aspect="auto", cmap="Blues")               # sequential: one hue, light -> dark
    ax.set_xticks(range(len(types)), types, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sel)), [f"{m['family']}/{m['size']}" for m in sel], fontsize=7)
    fig.colorbar(im, label="test force RMSE (eV/Å)"); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _load(p):
    return [json.loads(ln) for ln in open(p) if ln.strip()]


def table(rows, target, device):
    out = [f"\n## target {target} eV/Å, {device}\n", "| family | size | n_params | rmse_F | atom-steps/s | code |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        if r["size"] is None:
            out.append(f"| {r['family']} | not reached | | | | |")
            continue
        sp = "not timed" if r["atom_steps_per_s"] is None else \
            f"{r['atom_steps_per_s']:.3g}" + {"below": " (below plateau)", "unknown": " (plateau unknown)",
                                              "clean": ""}[plateau_markers([r])[0]]
        out.append(f"| {r['family']} | {r['size']} | {r['n_params']} | {r['rmse_F']:.3f} | {sp} | "
                   f"{r.get('code') or ''} {r.get('dtype') or ''} |")
    return "\n".join(out)


def main(argv=None):
    from .common import HERE, RESULTS
    argv = sys.argv[1:] if argv is None else argv
    targets = [float(x) for x in argv] or [0.1, 0.15]
    m = _load(RESULTS / "metrics.jsonl")
    t = sum((_load(RESULTS / f"lestrade-{d}.jsonl") for d in ("cpu", "gpu")
             if (RESULTS / f"lestrade-{d}.jsonl").exists()), [])
    t = [r for r in t if r.get("mode") != "parity"]
    figs = HERE / "figs"; figs.mkdir(exist_ok=True)
    refs = json.loads((HERE / "refs.json").read_text()) if (HERE / "refs.json").exists() else []
    pareto(m, t, targets, figs / "pareto.png", refs)
    keep = {(r["family"], r["size"]) for r in m if r["family"] in REFERENCE}
    for tg in targets:
        for d in ("cpu", "gpu"):
            rows = smallest_at_target(m, t, tg, d)
            print(table(rows, tg, d))
            keep |= {(r["family"], r["size"]) for r in rows if r["size"] is not None}
    heatmap(m, sorted(keep), figs / "per_type.png")


if __name__ == "__main__":
    main()
