"""LAMMPS CPU cases for the CPU-gap study (docs/dev/cpu-gap-profile.md): ML-PACE
(`pair_style pace`, the recursive evaluator) and the ACEpotentials.jl trim
library (`pair_style ace`) at R MPI ranks bound to physical cores 0..R-1.

    PYTHONPATH=bench:src python bench/perf/cpu_gap_lammps.py <code> <system> <n> <ranks...> \
        [--size medium] [--steps 200] [--out FILE] [--trim-lib PATH] [--threads-per-rank 1]

code: mlpace | trim.  Uses `bench/scaling/run_lammps.py` (same input deck: 50
warm-up steps, then a timed segment, `neighbor 1.0 bin`, check yes), with the
host's LAMMPS wrappers from envs/moriarty-cpu.json (`lmp` for ML-PACE, `lmp_ace`
for the trim plugin).  One rank runs under `taskset -c 0` with no mpirun; R > 1
runs `mpirun -np R --bind-to core --cpu-set 0-(R-1)` (`--no-bind`: `--bind-to none`,
as the published moriarty-cpu rows ran).  The row
adds LAMMPS's MPI task-timing breakdown (Pair / Neigh / Comm / Modify / Other,
percent of the loop) and us per atom-step.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "bench"))
OMPI_BIN = "/software/easybuild/software/OpenMPI/4.1.5-GCC-12.3.0/bin"


def breakdown(log):
    """{section: percent of loop time} from LAMMPS's 'MPI task timing breakdown'."""
    out = {}
    i = log.rfind("MPI task timing breakdown")
    for ln in log[i:].splitlines()[3:12] if i >= 0 else []:
        m = re.match(r"(\w+)\s*\|.*\|\s*([\d.]+)\s*$", ln)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code", choices=["mlpace", "trim"])
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("ranks", type=int, nargs="+")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--trim-lib", default=None)
    ap.add_argument("--ace-plugin", default="/storage/eng/essswb/bench-scaling-lestrade/ace-plugin/aceplugin.so")
    ap.add_argument("--out", default=str(HERE / "results" / "cpu_gap" / "lammps.jsonl"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--lmp", default=None, help="LAMMPS wrapper (moriarty has no lmp-ace.sh: "
                    "/storage/eng/essswb/bench-scaling-lestrade/lmp-ace.sh runs moriarty's lammps-dev)")
    ap.add_argument("--no-bind", action="store_true", help="mpirun --bind-to none (the published rows)")
    a = ap.parse_args()
    from scaling import run_lammps
    from scaling.models import planned_models
    from scaling.structures import supercell
    env_json = json.loads((ROOT / "bench" / "scaling" / "envs" / "moriarty-cpu.json").read_text())
    code = "mlpace" if a.code == "mlpace" else "acepotentials-trim"
    row = next(r for r in planned_models() if r["code"] == code and r["system"] == a.system
               and r["size"] == a.size)
    model = row["path"] if a.code == "mlpace" else (a.trim_lib or row["trim_lib"])
    lmp = a.lmp or (env_json["lmp"] if a.code == "mlpace" else env_json["lmp_ace"])
    style = "mlpace" if a.code == "mlpace" else run_lammps.TRIM
    os.environ["PATH"] = OMPI_BIN + ":" + os.environ["PATH"]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for R in a.ranks:
        work = pathlib.Path(tempfile.mkdtemp(prefix=f"cpugap_{a.code}_{R}_"))
        at = supercell(a.system, a.n)
        from ase.io import write
        write(work / "x.data", at, format="lammps-data", specorder=row["elements"], masses=True)
        (work / "in.bench").write_text(run_lammps.lammps_input(
            style, model, row["elements"], work / "x.data", "cpu", a.steps, warmup=a.warmup))
        cmd = [lmp, "-in", "in.bench", "-log", "log.lammps", "-nocite"]
        if R == 1:
            cmd = ["taskset", "-c", "0"] + cmd
        else:
            bind = (["--bind-to", "none"] if a.no_bind else
                    ["--bind-to", "core", "--cpu-set", f"0-{R - 1}"])
            cmd = ["mpirun", "-np", str(R), *bind] + cmd
        if style == run_lammps.TRIM:
            cmd += ["-var", "aceplugin", a.ace_plugin]
        load = open("/proc/loadavg").read().split()[:3]
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=7200,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        log = (work / "log.lammps").read_text() if (work / "log.lammps").exists() else p.stdout
        out = {"code": a.code, "system": a.system, "size": a.size, "n": a.n, "ranks": R,
               "model": pathlib.Path(model).name, "no_bind": a.no_bind, "loadavg_before": load,
               "loadavg_after": open("/proc/loadavg").read().split()[:3], "cmd": cmd}
        if "Loop time" not in log:
            out.update(status="error", error=((p.stderr or "") + (p.stdout or ""))[-1500:])
        else:
            out.update(run_lammps.parse_log(log))
            out["status"] = "ok"
            out["atom_steps_per_s"] = a.n / out["step_s"]
            out["us_per_atom_step"] = out["step_s"] / a.n * 1e6
            out["core_us_per_atom_step"] = out["us_per_atom_step"] * R    # cores x time
            out["breakdown_pct"] = breakdown(log)
            m = re.search(r"Neighbor list builds = (\d+)", log)
            out["neigh_builds"] = int(m.group(1)) if m else None
            m = re.search(r"Total # of neighbors = (\d+)", log)
            out["neighbors_total"] = int(m.group(1)) if m else None
        with open(a.out, "a") as fh:
            fh.write(json.dumps(out) + "\n")
        print(json.dumps({k: out.get(k) for k in ("code", "system", "ranks", "us_per_atom_step",
                                                  "core_us_per_atom_step", "atom_steps_per_s",
                                                  "breakdown_pct", "neigh_builds", "status",
                                                  "error")}), flush=True)
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
