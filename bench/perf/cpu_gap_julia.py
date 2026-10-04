"""ACEpotentials.jl lines for the CPU-gap study (docs/dev/cpu-gap-profile.md).

    PYTHONPATH=bench:src python bench/perf/cpu_gap_julia.py direct <system> <n> <threads...>
    PYTHONPATH=bench:src python bench/perf/cpu_gap_julia.py build-trim <system> [--out DIR]

direct      `bench/scaling/run_standalone.py`'s `acepotentials` case (julia/run_standalone.jl:
            AtomsCalculators.energy_forces_virial on the npz's splined ace1_model, neighbour
            list built every call), medium model, JULIA_NUM_THREADS = T under
            `taskset -c 0..T-1`; appended to results/cpu_gap/julia.jsonl.
build-trim  `julia/build_trim.jl` for the medium model into --out (default
            /storage/eng/essswb/cpu-gap/trim): juliac targets this host's CPU, so
            moriarty needs its own build (the lestrade libraries target an i9-14900K).
            Then run them with `cpu_gap_lammps.py trim ... --trim-lib <so>`.
Julia: juliaup's 1.12.6 with the pinned depot /storage/eng/essswb/cache/julia-pr309 and
this checkout's bench/scaling/julia project (ACEPOT_* override).
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "bench"))
os.environ.setdefault("ACEPOT_JULIA", os.path.expanduser("~/.juliaup/bin/julia") + " +1.12.6")
os.environ.setdefault("ACEPOT_JULIA_DEPOT", "/storage/eng/essswb/cache/julia-pr309")
os.environ.setdefault("ACEPOT_JULIA_PROJECT", str(ROOT / "bench" / "scaling" / "julia"))


def row_for(code, system):
    from scaling.models import planned_models
    return next(r for r in planned_models() if r["code"] == code and r["system"] == system
                and r["size"] == "medium")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["direct", "build-trim", "_case"])
    ap.add_argument("system")
    ap.add_argument("n", nargs="?", type=int, default=2048)
    ap.add_argument("threads", nargs="*", type=int)
    ap.add_argument("--out", default="/storage/eng/essswb/cpu-gap/trim")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    if a.what == "build-trim":
        from scaling import acepot
        from scaling.structures import supercell
        r = row_for("acepotentials-trim", a.system)
        out = pathlib.Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        so = out / f"libace_{a.system}_medium.so"
        spec = so.with_suffix(".spec.json")
        spec.write_text(json.dumps({**r["ace1"], "npz": r["path"]}))
        xyz = out / f"gate_{a.system}_256.extxyz"
        acepot.roundtrip(supercell(a.system, 256), xyz)
        res = acepot.run_julia(acepot.julia_config(), "build_trim.jl", [str(spec), str(xyz), str(so)],
                               threads=1, timeout=7200)
        print(json.dumps({"so": str(so), **{k: res.get(k) for k in ("gates", "spline_error", "juliac_s",
                                                                     "so_bytes", "versions")}}))
        return
    if a.what == "_case":                 # one direct case in this (pinned) process
        from scaling.run_standalone import run_case
        r = run_case(row_for("acepotentials", a.system), a.n, "float64", "cpu", reps=a.reps)
        print(json.dumps(r))
        return
    res = HERE / "results" / "cpu_gap" / "julia.jsonl"
    res.parent.mkdir(parents=True, exist_ok=True)
    for T in a.threads:
        cpus = "0" if T == 1 else f"0-{T - 1}"
        load = open("/proc/loadavg").read().split()[:3]
        p = subprocess.run(["taskset", "-c", cpus, sys.executable, __file__, "_case", a.system,
                            str(a.n), "--reps", str(a.reps)], capture_output=True, text=True,
                           env={**os.environ, "JULIA_NUM_THREADS": str(T),
                                "PYTHONPATH": f"{ROOT / 'bench'}:{ROOT / 'src'}"})
        ln = [l for l in p.stdout.splitlines() if l.startswith("{")]
        r = json.loads(ln[-1]) if ln else {"status": "error", "error": p.stderr[-1500:]}
        r.update({"taskset": cpus, "julia_threads_req": T, "loadavg_before": load})
        if r.get("call_s"):
            r["us_per_atom"] = r["call_s"] / a.n * 1e6
        with open(res, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(json.dumps({k: r.get(k) for k in ("system", "taskset", "us_per_atom", "gc_frac",
                                                "status", "error")}), flush=True)


if __name__ == "__main__":
    main()
