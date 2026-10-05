"""Per-thunk speed-up of one compiled program from 1 to T threads: two `cpu_trace.py`
outputs (run with `--top 300`, so every thunk is listed) of the same model.

    python bench/perf/cpu_thread_scaling.py <trace_1thread.json> <trace_Tthread.json>

Thunks are matched by name (the HLO is the same: XLA's CPU pipeline does not
depend on the thread count).  Buckets every thunk by its speed-up and reports the
time each bucket costs at 1 and at T threads, by kind (fusion / library dot /
scatter / copy ...) and stage: where the parallel efficiency is lost.
"""
import collections
import json
import sys


def kind(r):
    t = r["thunk"]
    if t.startswith("ynn") or t.startswith("dot"):
        return "library dot (XNN/YNN, Eigen)"
    if "scatter" in t:
        return "scatter"
    if t.startswith(("copy", "transpose")) or "copy_fusion" in t or "transpose" in t:
        return "copy / transpose"
    if "concatenate" in t:
        return "concatenate"
    if "gather" in t:
        return "gather"
    return "elementwise / reduce fusion"


def main(p1, pT):
    a, b = json.load(open(p1)), json.load(open(pT))
    t1 = {r["thunk"]: r for r in a["top"]}
    tT = {r["thunk"]: r for r in b["top"]}
    common = sorted(set(t1) & set(tT), key=lambda k: -t1[k]["ms_per_call"])
    buckets = collections.defaultdict(lambda: [0.0, 0.0, 0])
    kinds = collections.defaultdict(lambda: [0.0, 0.0])
    for k in common:
        x, y = t1[k]["ms_per_call"], tT[k]["ms_per_call"]
        s = x / y if y else float("inf")
        bk = ("<1.5x (serial)" if s < 1.5 else "1.5-4x" if s < 4 else "4-8x" if s < 8 else ">=8x")
        buckets[bk][0] += x
        buckets[bk][1] += y
        buckets[bk][2] += 1
        kk = kind(t1[k])
        kinds[kk][0] += x
        kinds[kk][1] += y
    tot1, totT = sum(v[0] for v in buckets.values()), sum(v[1] for v in buckets.values())
    print(f"matched {len(common)} thunks: {tot1:.1f} ms at 1 thread, {totT:.1f} ms at T "
          f"({tot1 / totT:.2f}x); wall {a['wall_ms_per_call']:.1f} -> {b['wall_ms_per_call']:.1f} ms")
    print("\n| thunk speed-up | thunks | ms at 1 thread | ms at 16 | share of 16-thread time |")
    print("|---|--:|--:|--:|--:|")
    for bk in ("<1.5x (serial)", "1.5-4x", "4-8x", ">=8x"):
        if bk in buckets:
            x, y, n = buckets[bk]
            print(f"| {bk} | {n} | {x:.1f} | {y:.1f} | {y / totT:.0%} |")
    print("\n| kind | ms at 1 | ms at 16 | speed-up | share at 16 |")
    print("|---|--:|--:|--:|--:|")
    for kk, (x, y) in sorted(kinds.items(), key=lambda kv: -kv[1][1]):
        print(f"| {kk} | {x:.1f} | {y:.1f} | {x / y:.1f}x | {y / totT:.0%} |")
    print("\n| top thunks at 16 threads | ms at 1 | ms at 16 | speed-up | stage |")
    print("|---|--:|--:|--:|---|")
    for k in sorted(common, key=lambda k: -tT[k]["ms_per_call"])[:12]:
        x, y = t1[k]["ms_per_call"], tT[k]["ms_per_call"]
        print(f"| `{k}` | {x:.1f} | {y:.1f} | {x / y:.1f}x | {', '.join(t1[k]['stages'])} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
