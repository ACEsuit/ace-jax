"""Series of `cpu_gap.py` cases for the CPU-gap study (docs/dev/cpu-gap-profile.md),
one subprocess per case, pinned with taskset, appended to a JSONL.

    PYTHONPATH=bench:src python bench/perf/cpu_gap_sweep.py <series> [--out FILE] [--reps N]

Series:
  single    the four medium models (ace/pace x SiGe/Cantor), 2048 atoms, 1 XLA
            thread on CPU 0
  threads   the same, XLA threads T = 1, 2, 4, 8, 16 on physical cores 0..T-1,
            and 32 on all hardware threads (0-31, as the published moriarty rows)
  knobs     what-if probes on 1 thread (layout, skin, lean, chunk, float32)
  xla       XLA:CPU flags on 1 thread (fast math, vector width, XNNPACK, thunk
            runtime, oneDNN) and the concurrency-optimised scheduler on 16 cores
  nsize     1 thread, atom counts 256..8192 (per-atom cost vs n)
  stages    per-stage timings + profiler trace + HLO dump per model (cpu_stages.py,
            cpu_trace.py), 1 thread on CPU 0
  procs     the rank-parallel emulation: P = 2, 4, 8, 16 concurrent 1-thread
            processes, process p on CPU p, each on a periodic 2048/P-atom cell
            (owned rows only, as a domain-decomposed rank evaluates; no ghost
            exchange).  Throughput = sum over processes of n_p / call_s_p, all
            running at once, so shared L3 / DRAM contention is included.

moriarty: physical cores are CPUs 0-15, their hyperthread siblings 16-31.  The
load average is recorded before and after every case (`loadavg_before/after`);
a case whose 1-min load before exceeds `--max-load` waits up to 10 min for it
to fall, and is marked `noisy` if it does not.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ROOT / "bench" / "scaling" / "models"
MED = [("ace_SiGe_medium.npz", "SiGe"), ("ace_Cantor_medium.npz", "Cantor"),
       ("pace_SiGe_medium.yace", "SiGe"), ("pace_Cantor_medium.yace", "Cantor")]


def cpus(t):
    return "0-31" if t == 32 else ("0" if t == 1 else f"0-{t - 1}")


def series(name):
    """[(cpu list, cpu_gap.py args)]."""
    out = []
    if name == "single":
        for m, s in MED:
            out.append(("0", [str(MODELS / m), s, "2048", "--threads", "1"]))
    elif name == "threads":
        for t in (2, 4, 8, 16, 32):
            for m, s in MED:
                out.append((cpus(t), [str(MODELS / m), s, "2048", "--threads", str(t)]))
    elif name == "nsize":
        for n in (256, 512, 1024, 4096, 8192):
            for m, s in MED:
                out.append(("0", [str(MODELS / m), s, str(n), "--threads", "1"]))
    elif name == "xla":
        flags = {"fast_math": "--xla_cpu_enable_fast_math=true",
                 "vec512": "--xla_cpu_prefer_vector_width=512",
                 "no_xnnpack": "--xla_cpu_use_xnnpack=false",
                 "no_thunks": "--xla_cpu_use_thunk_runtime=false",
                 "onednn": "--xla_cpu_use_onednn=true"}
        for m, s in MED:
            for tag, f in flags.items():
                out.append(("0", [str(MODELS / m), s, "2048", "--threads", "1",
                                  f"--xla-extra={f}", "--tag", tag]))
        for m, s in MED:                     # the scheduler only matters with threads
            out.append(("0-15", [str(MODELS / m), s, "2048", "--xla-extra="
                                 "--xla_cpu_enable_concurrency_optimized_scheduler=true",
                                 "--tag", "conc_sched16"]))
    elif name == "knobs":
        one = ["--threads", "1"]
        for m, s in MED:
            base = [str(MODELS / m), s, "2048"] + one
            out += [("0", base + ["--dtype", "float32", "--tag", "float32"]),
                    ("0", base + ["--skin", "0", "--tag", "skin0"]),
                    ("0", base + ["--layout", "sparse", "--tag", "sparse"]),
                    ("0", base + ["--chunk", "32", "--tag", "chunk32"]),
                    ("0", base + ["--chunk", "128", "--tag", "chunk128"]),
                    ("0", base + ["--chunk", "256", "--tag", "chunk256"]),
                    ("0", base + ["--chunk", "1024", "--tag", "chunk1024"]),
                    ]
            if m.startswith("ace"):
                out += [("0", base + ["--no-lean", "--tag", "nolean"]),
                        ("0", base + ["--layout", "sparse", "--edge-a-kind", "matmul",
                                      "--tag", "sparse_matmul"]),
                        ("0", base + ["--layout", "sparse", "--edge-a-kind", "gather",
                                      "--tag", "sparse_gather"])]
    else:
        raise SystemExit(f"unknown series {name}")
    return out


def procs(a, env):
    """The `procs` series (see the module docstring)."""
    for P in (a.procs_p or (2, 4, 8, 16)):
        for m, s in MED:
            if a.only and a.only not in m:
                continue
            t0 = time.time()
            while load1() > a.max_load and time.time() - t0 < 600:
                time.sleep(15)
            n = 2048 // P
            ps = [subprocess.Popen(["taskset", "-c", str(c), sys.executable, str(HERE / "cpu_gap.py"),
                                    str(MODELS / m), s, str(n), "--threads", "1",
                                    "--reps", str(a.reps * 4)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                  for c in range(P)]
            rows = []
            for p in ps:
                o, e = p.communicate(timeout=3600)
                ln = [l for l in o.splitlines() if l.startswith("{")]
                rows.append(json.loads(ln[-1]) if ln else {"status": "error", "error": e[-500:]})
            ok = [r for r in rows if "call_s" in r]
            row = {"series": "procs", "model": m, "system": s, "P": P, "n_per_proc": n,
                   "n_total": n * P, "ok": len(ok),
                   "call_s_per_proc": [r.get("call_s") for r in rows],
                   "step_s_per_proc": [r.get("step_s") for r in rows],
                   "atom_steps_per_s": sum(r["n"] / r["call_s"] for r in ok),
                   "atom_steps_per_s_step": sum(r["n"] / r["step_s"] for r in ok if r.get("step_s")),
                   "loadavg_before": rows[0].get("loadavg_before")}
            row["us_per_atom_step"] = 1e6 / row["atom_steps_per_s"]
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("model", "P", "ok", "atom_steps_per_s",
                                                  "us_per_atom_step")}), flush=True)


def stages(a, env):
    """`cpu_stages.py` (with --trace / --dump into `--scratch`) then `cpu_trace.py`
    on 1 thread, CPU 0, for every medium model: stages_<model>.json and
    trace_<model>.json next to `--out`."""
    res = pathlib.Path(a.out).parent
    for m, s in MED:
        if a.only and a.only not in m:
            continue
        stem = m.split(".")[0]
        tr, dump = pathlib.Path(a.scratch) / f"tr_{stem}", pathlib.Path(a.scratch) / f"dump_{stem}"
        subprocess.run(["rm", "-rf", str(tr), str(dump)])
        while load1() > a.max_load:
            time.sleep(15)
        p = subprocess.run(["taskset", "-c", "0", sys.executable, str(HERE / "cpu_stages.py"),
                            str(MODELS / m), s, "2048", "--reps", "10", "--trace", str(tr),
                            "--dump", str(dump)], capture_output=True, text=True, env=env)
        (res / f"stages_{stem}.json").write_text(p.stdout or p.stderr[-3000:])
        q = subprocess.run([sys.executable, str(HERE / "cpu_trace.py"), str(tr), str(dump), "5",
                            "--top", "40"], capture_output=True, text=True, env=env)
        (res / f"trace_{stem}.json").write_text(q.stdout or q.stderr[-3000:])
        print(stem, "done", flush=True)


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("--out", default=str(HERE / "results" / "cpu_gap" / "acejax.jsonl"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--max-load", type=float, default=2.0)
    ap.add_argument("--only", default="", help="substring filter on the model name")
    ap.add_argument("--procs-p", type=int, nargs="*", default=None, help="procs: the P values")
    ap.add_argument("--scratch", default="/tmp", help="stages: where traces and HLO dumps go")
    a = ap.parse_args()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'bench'}:{ROOT / 'src'}", "JAX_PLATFORMS": "cpu"}
    if a.series == "procs":
        return procs(a, env)
    if a.series == "stages":
        return stages(a, env)
    for cpu, args in series(a.series):
        if a.only and a.only not in args[0]:
            continue
        noisy, t0 = False, time.time()
        while load1() > a.max_load:
            if time.time() - t0 > 600:
                noisy = True
                break
            time.sleep(15)
        cmd = ["taskset", "-c", cpu, sys.executable, str(HERE / "cpu_gap.py"), *args,
               "--reps", str(a.reps)]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        row = json.loads(lines[-1]) if lines else {"status": "error", "args": args,
                                                   "error": (p.stderr or "")[-800:]}
        row.update({"series": a.series, "taskset": cpu, "noisy": noisy})
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps({k: row.get(k) for k in ("model", "n", "taskset", "tag", "us_per_atom",
                                                  "step_s", "call_s", "layout", "loadavg_before",
                                                  "status")}), flush=True)


if __name__ == "__main__":
    main()
