"""Per-thunk CPU time of one compiled program, from a jax.profiler trace joined to
the optimised HLO dump, attributed to model stages, with a SIMD check of each
kernel's object code (docs/dev/cpu-gap-profile.md).

    python bench/perf/cpu_trace.py <trace_dir> <dump_dir> <n_calls> [--top 25]

XLA:CPU (jaxlib 0.11, thunk runtime) emits one trace event per executed thunk,
named after its HLO instruction (`multiply_bitcast_fusion.1`, `dot.19`,
`wrapped_scatter`, `ynn_fusion.2` = a dot/fusion handed to XNNPACK's YNN).
Each is matched to the after-optimisations HLO (`hlo_trace.parse_hlo`: operand
and result bytes, op_names, forward vs `transpose(...)` = backward) and to a
stage by the call chains of its fused instructions (`profile_ace.kernel_stages`
with this file's function -> stage map, covering the lean ACE and the PACE
paths).  The kernel's object file (`--xla_dump_to` writes one per kernel) is
disassembled with objdump and its floating-point instructions counted:
packed zmm (512-bit), packed ymm, packed xmm, and scalar (`...sd`/`...ss`).
"""
import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hlo_trace  # noqa: E402
import profile_ace  # noqa: E402

FN_STAGE = {
    # lean ACE (ACEModel._site_energies_blocked)
    "ACEModel._aa": "AA", "ACEModel._readout_folded": "readout", "pool_dense": "readout",
    "ACEModel.pair_radial": "pair", "ACEModel.angular": "angular",
    "ACEModel._radial_one": "radial", "ACEModel.radial": "radial",
    "ACEModel._site_energies_blocked": "A",
    # PACE
    "PACEModel._node_energies_t": "AA+rho", "PACEModel.ctilde_real": "AA+rho",
    "PACEModel._energy_tail": "tail", "PACEModel._embedding": "tail",
    "PACEModel._edge_core": "core", "PACEModel.edge_basis_factors": "radial",
    "PACEModel._geometry": "radial", "EdgeSiteModel.pool_first_dense": "A",
    "PACEModel.pool_first_weights": "A", "PACEModel.site_energies_dense": "other",
    # shared
    "EdgeSiteModel.energy_forces_virial_dense.<locals>.total": "strain",
    "EdgeSiteModel.energy_forces_virial_dense": "force",
    "step": "step",
}
FILE_STAGE = {"radial.py": "radial", "pace_radial.py": "radial", "harmonics.py": "angular",
              "skin.py": "step"}

_FP = re.compile(r"\bv?(add|sub|mul|div|fmadd\w*|fnmadd\w*|fmsub\w*|fnmsub\w*|sqrt|max|min)"
                 r"(p|s)(d|s)\b")


def simd(obj):
    """Counts of FP arithmetic instructions in an object file, by register width."""
    try:
        asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", obj], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:                                             # noqa: BLE001
        return None
    c = collections.Counter()
    for ln in asm.splitlines():
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        ins = parts[-1].strip()
        m = _FP.match(ins)
        if not m:
            continue
        if m.group(2) == "s":
            c["scalar"] += 1
        elif "zmm" in ins:
            c["zmm"] += 1
        elif "ymm" in ins:
            c["ymm"] += 1
        else:
            c["xmm"] += 1
    return dict(c)


def obj_for(dump, name):
    for pat in (f"*obj-file.{name}_kernel_module.o", f"*obj-file.{name}.o"):
        f = glob.glob(os.path.join(dump, pat))
        if f:
            return f[0]
    return None


def table(trace_dir, dump_dir, n_calls, top=25):
    profile_ace._FN_STAGE.clear()
    profile_ace._FN_STAGE.update(FN_STAGE)
    profile_ace._FILE_STAGE.clear()
    profile_ace._FILE_STAGE.update(FILE_STAGE)
    ev, _, _, _ = hlo_trace.load_trace(trace_dir)
    skip = ("ThunkExecutor", "$", "PjitFunction", "Pjit")
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        if e.get("ph") != "X" or any(e["name"].startswith(s) for s in skip):
            continue
        a = agg[e["name"]]
        a[0] += float(e.get("dur", 0.0))
        a[1] += 1
    info, path = hlo_trace.parse_hlo(dump_dir, list(agg))
    kst = profile_ace.kernel_stages(path) if path else {}
    # the program's own wall time per call (ThunkExecutor::Execute, the outermost)
    wall = [float(e["dur"]) for e in ev if e.get("ph") == "X"
            and e["name"] == "ThunkExecutor::Execute (wait for completion)"]
    rows, by_stage = [], collections.defaultdict(lambda: {"fwd_ms": 0.0, "bwd_ms": 0.0})
    for k, (us, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        inf = info.get(k, {})
        w = profile_ace._weights(kst.get(k, {"other": 1}))
        per = us / n_calls / 1e3
        side = "bwd_ms" if inf.get("bwd") else "fwd_ms"
        for s_, f in w.items():
            by_stage[s_][side] += per * f
        b = inf.get("bytes_in", 0) + inf.get("bytes_out", 0)
        ob = obj_for(dump_dir, k)
        rows.append({"thunk": k, "ms_per_call": round(per, 3), "calls": cnt / n_calls,
                     "stages": {s_: round(f, 2) for s_, f in w.items()}, "bwd": inf.get("bwd"),
                     "hlo_op": inf.get("hlo_op"), "out_shape": (inf.get("out_shape") or "")[:60],
                     "MB": round(b / 1e6, 1), "GBps": round(b / (per * 1e-3) / 1e9, 2) if per and b else None,
                     "simd": simd(ob) if ob else None})
    tot = sum(r["ms_per_call"] for r in rows)
    simd_tot = collections.Counter()
    simd_time = collections.Counter()
    for r in rows:
        s = r["simd"] or {}
        n = sum(s.values())
        simd_tot.update(s)
        if n:
            vec = (s.get("zmm", 0) + s.get("ymm", 0) + s.get("xmm", 0)) / n
            simd_time["vector_dominant_ms" if vec >= 0.5 else "scalar_dominant_ms"] += r["ms_per_call"]
        else:
            simd_time["no_fp_or_library_ms"] += r["ms_per_call"]
    return {"hlo_module": path, "n_thunk_kinds": len(rows),
            "thunks_per_call": sum(r["calls"] for r in rows),
            "thunk_ms_per_call": tot, "wall_ms_per_call": sum(wall) / max(len(wall), 1) / 1e3,
            "by_stage_ms": {k: {kk: round(vv, 3) for kk, vv in v.items()}
                            for k, v in sorted(by_stage.items(), key=lambda kv: -sum(kv[1].values()))},
            "fwd_ms": sum(v["fwd_ms"] for v in by_stage.values()),
            "bwd_ms": sum(v["bwd_ms"] for v in by_stage.values()),
            "simd_instructions": dict(simd_tot), "time_by_simd": dict(simd_time),
            "top": rows[:top]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("dump")
    ap.add_argument("n_calls", type=int)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    print(json.dumps(table(a.trace, a.dump, a.n_calls, a.top), indent=1))
