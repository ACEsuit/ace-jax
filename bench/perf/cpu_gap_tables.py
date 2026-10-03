"""Markdown tables for docs/dev/cpu-gap-profile.md from bench/perf/results/cpu_gap/.

    python bench/perf/cpu_gap_tables.py [single|threads|procs|lammps|stages|trace|knobs|probes|all]
"""
import collections
import json
import pathlib
import sys

RES = pathlib.Path(__file__).resolve().parent / "results" / "cpu_gap"
MODELS = ["ace_SiGe_medium", "ace_Cantor_medium", "pace_SiGe_medium", "pace_Cantor_medium"]


def rows(name):
    p = RES / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def acejax(series=None, tag=""):
    out = [r for r in rows("acejax.jsonl") if "call_s" in r and (r.get("tag") or "") == tag]
    return [r for r in out if series is None or r.get("series") == series]


def single():
    print("| model | us/atom-step (call) | step only | K |")
    print("|---|--:|--:|--:|")
    for r in acejax("single"):
        print(f"| {r['model']} | {r['us_per_atom']:.1f} | {r['step_s'] / r['n'] * 1e6:.1f} | {r.get('K')} |")


def threads():
    by = collections.defaultdict(dict)
    for r in acejax():
        if r.get("series") in ("single", "threads"):
            by[r["model"]][r["taskset"]] = r["us_per_atom"]
    cols = ["0", "0-1", "0-3", "0-7", "0-15", "0-31"]
    print("| model | " + " | ".join(f"{c} ({len(range(int(c.split('-')[0]), int(c.split('-')[-1]) + 1))})" for c in cols) + " | speed-up 16 cores |")
    print("|---|" + "--:|" * (len(cols) + 1))
    for m, d in by.items():
        sp = d["0"] / d["0-15"] if "0-15" in d else float("nan")
        print(f"| {m} | " + " | ".join(f"{d.get(c, float('nan')):.1f}" for c in cols) + f" | {sp:.2f}x |")


def procs():
    single_ = {r["model"]: r["us_per_atom"] for r in acejax("single")}
    print("| model | P | atom-steps/s | us/atom-step | speed-up vs 1 core | efficiency |")
    print("|---|--:|--:|--:|--:|--:|")
    for r in rows("acejax.jsonl"):
        if r.get("series") != "procs":
            continue
        s1 = single_.get(r["model"])
        sp = s1 / r["us_per_atom_step"] if s1 else float("nan")
        print(f"| {r['model']} | {r['P']} | {r['atom_steps_per_s']:.0f} | {r['us_per_atom_step']:.2f} | {sp:.1f}x | {sp / r['P']:.0%} |")


def lammps():
    print("| code | system | ranks | us/atom-step | atom-steps/s | speed-up | efficiency | Pair % | Comm % |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    one = {}
    for r in rows("lammps.jsonl"):
        if r.get("status") != "ok":
            continue
        k = (r["code"], r["system"], r.get("no_bind", False))
        if r["ranks"] == 1:
            one[(r["code"], r["system"])] = r["us_per_atom_step"]
        s1 = one.get((r["code"], r["system"]))
        sp = s1 / r["us_per_atom_step"] if s1 else float("nan")
        b = r.get("breakdown_pct", {})
        print(f"| {r['code']}{' (no bind)' if k[2] else ''} | {r['system']} | {r['ranks']} | {r['us_per_atom_step']:.2f} | {r['atom_steps_per_s']:.0f} | {sp:.1f}x | {sp / r['ranks']:.0%} | {b.get('Pair', 0):.1f} | {b.get('Comm', 0):.1f} |")


def stages():
    for m in MODELS:
        p = RES / f"stages_{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        n = d["n"]
        print(f"\n**{m}** (n={n}, K={d['K']}, fill {d['fill']:.2f}; dims {json.dumps(d['dims'])})\n")
        print("| stage | fwd ms | fwd+VJP ms | VJP/fwd | fwd MFLOP | fwd MB | fwd GFLOP/s | fwd GB/s |")
        print("|---|--:|--:|--:|--:|--:|--:|--:|")
        for k, v in d["stages"].items():
            fl, by = v["fwd"].get("flops", 0) / 1e6, v["fwd"].get("bytes", 0) / 1e6
            fb = v["fwd_vjp_s"]
            print(f"| {k} | {v['fwd_s'] * 1e3:.2f} | {fb * 1e3 if fb else float('nan'):.2f} | "
                  f"{(fb / v['fwd_s']) if fb else float('nan'):.1f} | {fl:.1f} | {by:.1f} | "
                  f"{fl / 1e3 / v['fwd_s']:.2f} | {by / 1e3 / v['fwd_s']:.2f} |")
        print(f"| sum of stages | {d['stage_sum_fwd_s'] * 1e3:.1f} | {d['stage_sum_fwd_vjp_s'] * 1e3:.1f} | | | | | |")
        ec, en = d["efv_cost"], d["energy_cost"]
        print(f"| whole energy (fwd) | {d['energy_s'] * 1e3:.1f} | | | {en['flops'] / 1e6:.0f} | {en['bytes'] / 1e6:.0f} | {en['flops'] / 1e9 / d['energy_s']:.2f} | {en['bytes'] / 1e9 / d['energy_s']:.2f} |")
        print(f"| whole efv (E, dE/drij, V) | | {d['efv_s'] * 1e3:.1f} | {d['efv_s'] / d['energy_s']:.1f} | {ec['flops'] / 1e6:.0f} | {ec['bytes'] / 1e6:.0f} | {ec['flops'] / 1e9 / d['efv_s']:.2f} | {ec['bytes'] / 1e9 / d['efv_s']:.2f} |")
        print(f"| calculator step | | {d['step_s'] * 1e3:.1f} | | | | | |")
        print(f"\nper atom: efv {ec['flops'] / n / 1e3:.0f} kflop, {ec['bytes'] / n / 1e3:.0f} kB; "
              f"thunks: efv {ec.get('thunks')}, energy {en.get('thunks')}")


def trace():
    for m in MODELS:
        p = RES / f"trace_{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        print(f"\n**{m}**: {d['thunk_ms_per_call']:.1f} ms of thunks per call "
              f"({d['thunks_per_call']:.0f} thunks), fwd {d['fwd_ms']:.1f} / bwd {d['bwd_ms']:.1f} ms; "
              f"time by SIMD: {json.dumps({k: round(v, 1) for k, v in d['time_by_simd'].items()})}\n")
        print("| stage | fwd ms | bwd ms | share |")
        print("|---|--:|--:|--:|")
        tot = d["thunk_ms_per_call"]
        for k, v in d["by_stage_ms"].items():
            print(f"| {k} | {v['fwd_ms']:.1f} | {v['bwd_ms']:.1f} | {(v['fwd_ms'] + v['bwd_ms']) / tot:.0%} |")
        print("\n| thunk | ms | stage | bwd | MB moved | GB/s | FP instrs (zmm/ymm/xmm/scalar) |")
        print("|---|--:|---|---|--:|--:|---|")
        for r in d["top"][:12]:
            s = r["simd"] or {}
            simd = "lib" if r["simd"] is None else f"{s.get('zmm', 0)}/{s.get('ymm', 0)}/{s.get('xmm', 0)}/{s.get('scalar', 0)}"
            st = ", ".join(f"{k} {v:.0%}" if v < 1 else k for k, v in r["stages"].items())
            print(f"| `{r['thunk']}` | {r['ms_per_call']:.1f} | {st} | {'bwd' if r['bwd'] else 'fwd'} | {r['MB']} | {r['GBps']} | {simd} |")


def knobs():
    base = {r["model"]: r["us_per_atom"] for r in acejax("single")}
    print("| model | knob | us/atom-step | vs default |")
    print("|---|---|--:|--:|")
    for r in rows("acejax.jsonl"):
        if r.get("series") not in ("knobs", "xla") or "call_s" not in r:
            continue
        b = base.get(r["model"])
        print(f"| {r['model']} | {r['tag']} ({r.get('layout')}) | {r['us_per_atom']:.1f} | {b / r['us_per_atom']:.2f}x |")


def probes():
    print("| model | aa | spline | step base ms | variant ms | speed-up | max dF |")
    print("|---|---|---|--:|--:|--:|--:|")
    for r in rows("probes.jsonl"):
        print(f"| {r['model']} | {r['aa']} | {r['spline']} | {r['step_s_base'] * 1e3:.1f} | {r['step_s_variant'] * 1e3:.1f} | {r['speedup']:.2f}x | {r['dF_max']:.1e} |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for f in (single, threads, procs, lammps, stages, trace, knobs, probes):
        if what in ("all", f.__name__):
            print(f"\n### {f.__name__}\n")
            f()
