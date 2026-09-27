"""Per-kernel GPU time from a jax.profiler perfetto trace, joined to the
optimised HLO dump (--xla_dump_to) so each kernel gets its JAX op_name, source
lines, forward/backward role, and the bytes its operands + result occupy.

    python bench/perf/hlo_trace.py <trace_dir> <dump_dir> <n_calls>
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

_DT = {"f64": 8, "f32": 4, "s32": 4, "u32": 4, "pred": 1, "s64": 8, "u64": 8, "bf16": 2,
       "f16": 2, "s8": 1, "u8": 1, "c128": 16, "c64": 8}
_SHAPE = re.compile(r"\b(f64|f32|s32|u32|pred|s64|u64|bf16|f16|s8|u8|c128|c64)\[([\d,]*)\]")
_INSTR = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$")


def shape_bytes(txt):
    tot = 0
    for dt, dims in _SHAPE.findall(txt):
        k = 1
        for d in (dims.split(",") if dims else []):
            k *= int(d)
        tot += k * _DT[dt]
    return tot


def load_trace(trace_dir):
    f = sorted(glob.glob(os.path.join(trace_dir, "**", "*perfetto_trace.json.gz"), recursive=True))
    if not f:
        f = sorted(glob.glob(os.path.join(trace_dir, "**", "*.trace.json.gz"), recursive=True))
    with gzip.open(f[-1], "rt") as fh:
        tr = json.load(fh)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    pname = {e["pid"]: e["args"]["name"] for e in ev
             if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu = {p for p, nm in pname.items() if "GPU" in nm or "gpu" in nm or "/device:" in nm}
    tname = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev
             if e.get("ph") == "M" and e.get("name") == "thread_name"}
    return ev, pname, gpu, tname


def parse_hlo(dump_dir, kernel_names):
    """Pick the after-optimisations module that defines most of `kernel_names`;
    return {instr_name: info}."""
    best, best_hits = None, -1
    for p in glob.glob(os.path.join(dump_dir, "*after_optimizations.txt")):
        txt = open(p).read()
        hits = sum(1 for k in kernel_names if f"%{k} = " in txt or f" {k} = " in txt)
        if hits > best_hits:
            best, best_hits = (p, txt), hits
    if best is None:
        return {}, None
    path, txt = best
    comps, cur, entry = {}, None, None
    for line in txt.splitlines():
        m = re.match(r"^(ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$", line)
        if m and not line.startswith(" "):
            cur = m.group(2)
            comps[cur] = []
            if m.group(1):
                entry = cur
            continue
        if cur is not None and line.strip() and line.strip() != "}":
            comps[cur].append(line)
    shapes = {}
    for c, lines in comps.items():
        for ln in lines:
            m = _INSTR.match(ln)
            if m:
                rhs = m.group(2)
                shapes[m.group(1)] = rhs.split(" ", 1)[0] if not rhs.startswith("(") else \
                    rhs[:rhs.index(")") + 1]
    info = {}
    for c, lines in comps.items():
        for ln in lines:
            m = _INSTR.match(ln)
            if not m:
                continue
            name, rhs = m.groups()
            op = re.search(r'op_name="([^"]*)"', rhs)
            calls = re.search(r"calls=%?([\w.\-]+)", rhs)
            ops_in = re.search(r"\w+\((.*?)\)(?:,|$)", rhs)
            operands = re.findall(r"%([\w.\-]+)", ops_in.group(1)) if ops_in else []
            out_b = shape_bytes(shapes.get(name, ""))
            in_b = sum(shape_bytes(shapes.get(o, "")) for o in operands)
            srcs, names = set(), set()
            if calls and calls.group(1) in comps:
                for l2 in comps[calls.group(1)]:
                    o2 = re.search(r'op_name="([^"]*)"', l2)
                    s2 = re.search(r'source_line=(\d+)', l2)
                    f2 = re.search(r'source_file="([^"]*)"', l2)
                    if o2:
                        names.add(o2.group(1))
                    if s2 and f2:
                        srcs.add(f"{os.path.basename(f2.group(1))}:{s2.group(1)}")
            if op:
                names.add(op.group(1))
            s1 = re.search(r'source_line=(\d+)', rhs)
            f1 = re.search(r'source_file="([^"]*)"', rhs)
            if s1 and f1:
                srcs.add(f"{os.path.basename(f1.group(1))}:{s1.group(1)}")
            km = re.match(r"^(?:\([^)]*\)|\S+)\s+([\w\-]+)\(", rhs)
            kind = km.group(1) if km else ""
            info[name] = {"hlo_op": kind, "out_shape": shapes.get(name, ""), "bytes_in": in_b,
                          "bytes_out": out_b, "op_names": sorted(names)[:6],
                          "srcs": sorted(srcs)[:8],
                          "bwd": any("transpose" in s for s in names)}
    return info, path


def kernel_table(trace_dir, dump_dir, n_calls, top=40):
    ev, pname, gpu, tname = load_trace(trace_dir)
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        if e.get("ph") != "X" or e.get("pid") not in gpu:
            continue
        tn = tname.get((e["pid"], e["tid"]), "")
        if "Stream" not in tn and "stream" not in tn:
            continue
        a = agg[e["name"]]
        a[0] += float(e.get("dur", 0.0))
        a[1] += 1
    total = sum(v[0] for v in agg.values())
    info, path = parse_hlo(dump_dir, list(agg)) if dump_dir else ({}, None)
    rows = []
    for k, (us, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        base = k.split("(")[0].strip()
        # kernel "input_scatter_fusion_5" is HLO instruction "%input_scatter_fusion.5"
        dotted = re.sub(r"_(\d+)$", r".\1", base)
        inf = info.get(base) or info.get(dotted) or {}
        per_call_us = us / n_calls
        b = inf.get("bytes_in", 0) + inf.get("bytes_out", 0)
        rows.append({"kernel": k[:90], "us_per_call": round(per_call_us, 2),
                     "frac": round(us / total, 4) if total else 0, "launches_per_call": cnt / n_calls,
                     "GBps": round(b / (per_call_us * 1e-6) / 1e9, 1) if per_call_us and b else None,
                     **{kk: inf.get(kk) for kk in ("hlo_op", "out_shape", "bytes_in", "bytes_out",
                                                   "bwd", "srcs", "op_names")}})
    return {"total_gpu_us_per_call": total / n_calls, "n_kernels": len(agg),
            "launches_per_call": sum(v[1] for v in agg.values()) / n_calls,
            "hlo_module": path, "rows": rows[:top]}


if __name__ == "__main__":
    print(json.dumps(kernel_table(sys.argv[1], sys.argv[2], int(sys.argv[3])), indent=1))
