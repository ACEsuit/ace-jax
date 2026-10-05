"""Shape evaluation paths and rank truncation (plan: docs/dev/force-uq-vs-calm.md, Step 5).

Variants, each in its own subprocess (peak RSS is per process): `rows` (the whole cell's force design rows,
the default path), `committee` (ACECalculator(shape_path="committee"): the r-output linear ACE), and
`committee:k` (R truncated to rank k).  On labelled cells (free atoms only; frames without the force key are
skipped): wall time per cell (the first
cell, which compiles, reported apart), peak RSS, max relative difference of forces_std from `rows`, Spearman
rho(forces_std, forces_std of rows) (5 significant figures: ties kept), and coverage P(|dF| <= forces_q) with dF = label - model force.

    python shape_eval.py --model M --posterior P --cells big3_mh1.xyz:3 --force-key mace_force \
        --variants rows,committee,committee:800,committee:200,committee:50 --out shape_eval.md
"""
import argparse
import json
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np


def worker(a):
    import jax
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from _frames import read_atoms

    from ace_jax.calc.point import ACECalculator
    v = a.variant
    kw = {} if v == "rows" else {"shape_path": "committee"}
    if ":" in v:
        kw["shape_rank"] = int(v.split(":")[1])
    calc = ACECalculator(a.model, posterior=a.posterior, **kw)
    path, n = (a.cells.split(":") + [None])[:2]
    frames = [f for f in read_atoms(path) if a.force_key in f.arrays]     # unlabelled frames cannot be scored
    if n:
        frames = frames[::max(1, len(frames) // int(n))][:int(n)]
    out = {"t": [], "std": [], "q": [], "err": [], "rank": int(calc.posterior.R.shape[1])}
    for at in frames:
        free = ~np.asarray(at.arrays.get("fixed", np.zeros(len(at), bool))).astype(bool)
        lab = np.asarray(at.arrays[a.force_key])
        at.calc = calc
        F = at.get_forces()
        t = time.perf_counter()
        std = np.asarray(calc.get_property("forces_std", at))
        out["t"].append(time.perf_counter() - t)
        out["std"].append(std[free]); out["q"].append(np.asarray(calc.get_property("forces_q", at))[free])
        out["err"].append(np.linalg.norm(lab - F, axis=1)[free])
    try:                                      # device peak (GPU); None on CPU backends without memory stats
        dev_gb = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2 ** 30
    except Exception:
        dev_gb = float("nan")
    np.savez(a.save, dev_peak_gb=dev_gb, device=str(jax.devices()[0].device_kind), t=out["t"], std=np.concatenate(out["std"]), q=np.concatenate(out["q"]),
             err=np.concatenate(out["err"]), rank=out["rank"],
             maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, n_atoms=[len(f) for f in frames])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--posterior", required=True)
    p.add_argument("--cells", required=True, help="FILE.xyz[:N]")
    p.add_argument("--force-key", default="mace_force")
    p.add_argument("--variants", default="rows,committee")
    p.add_argument("--out")
    p.add_argument("--variant", help=argparse.SUPPRESS)
    p.add_argument("--save", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.variant:
        return worker(a)
    from scipy.stats import spearmanr
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from _frames import sig_round
    res, wd = {}, pathlib.Path(a.out).with_suffix("")
    wd.mkdir(parents=True, exist_ok=True)
    for v in a.variants.split(","):
        f = wd / f"{v.replace(':', '_')}.npz"
        cmd = [sys.executable, __file__, "--model", a.model, "--posterior", a.posterior, "--cells", a.cells,
               "--force-key", a.force_key, "--variant", v, "--save", str(f)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(f"{v}: FAILED\n{r.stderr[-3000:]}")
            continue
        res[v] = dict(np.load(f))
        print(f"{v}: done ({np.sum(res[v]['t']):.1f} s)")
    if not res:
        raise SystemExit("every variant failed")
    ref = res.get("rows")
    dev = str(next(iter(res.values())).get("device", "")) if res else ""
    lines = [f"Device {dev}. Cells {a.cells}: {', '.join(map(str, next(iter(res.values()))['n_atoms']))} atoms.", "",
             "| variant | rank | first cell (s) | later cells, mean (s) | peak RSS (GB) | max rel diff vs rows | "
             "rho(std, std_rows) | coverage P(dF <= forces_q) | median forces_q | device peak (GB) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for v, d in res.items():
        t = np.asarray(d["t"])
        diff = rho = ""
        if ref is not None:
            diff = f"{np.max(np.abs(d['std'] / ref['std'] - 1)):.1e}"
            rho = f"{spearmanr(sig_round(d['std']), sig_round(ref['std']))[0]:.4f}"
        lines.append(f"| {v} | {int(d['rank'])} | {t[0]:.1f} | {t[1:].mean() if len(t) > 1 else float('nan'):.1f} | "
                     f"{float(d['maxrss_gb']):.1f} | {diff} | {rho} | {np.mean(d['err'] <= d['q']):.4f} | "
                     f"{np.median(d['q']):.3f} | {float(d.get('dev_peak_gb', np.nan)):.1f} |")
    md = "\n".join(lines) + "\n"
    pathlib.Path(a.out).write_text(md)
    (wd / "summary.json").write_text(json.dumps({v: {"t": list(map(float, d["t"]))} for v, d in res.items()}))
    print(md)


if __name__ == "__main__":
    main()
