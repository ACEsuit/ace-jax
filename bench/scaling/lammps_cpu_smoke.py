"""lammps-jax on the CPU (host Kokkos + XLA's CPU PJRT plugin): parity against
ACECalculator and the MD step time at 1..N MPI ranks, for one ace-jax model.

Per rank count it exports a bundle sized to one rank's subdomain
(neighbour_capacity(procs=...)) and lowered for the CPU (export_lammps(platform=
"cpu")), forces the matching `processors` grid, runs `run 0` with a force dump
(parity) and then a timed NVE segment.  Ranks are bound to cores with one Kokkos
thread each: XLA sizes its pool from the affinity mask.

    python bench/scaling/lammps_cpu_smoke.py MODEL --lmp LMP --plugin-dir DIR --pjrt SO \
        [--rep 8] [--ranks 1 2 4 8 16 32] [--layout matrix] [--out DIR]
"""
import argparse, json, os, pathlib, re, subprocess, time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from ase.build import bulk  # noqa: E402
from ase.io import write  # noqa: E402

from ace_jax.calc.point import ACECalculator  # noqa: E402
from ace_jax.eval import load  # noqa: E402
from ace_jax.export.lammps import export_lammps, neighbour_capacity  # noqa: E402

GRIDS = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2), 16: (4, 2, 2), 32: (4, 4, 2),
         64: (4, 4, 4), 128: (8, 4, 4)}


def deck(data, bundle, procs, dump, warmup, steps):
    return (f"units metal\natom_style atomic\nboundary p p p\nprocessors {' '.join(map(str, procs))}\n"
            f"read_data {data}\nmass 1 28.0855\npair_style jax/kk ${{pjrt}}\npair_coeff * * {bundle}\n"
            "neighbor 1.0 bin\nneigh_modify every 1 delay 0 check yes\n"
            "thermo_style custom step pe\nthermo_modify format float %.17g\n"
            f"dump d all custom 1 {dump} id fx fy fz\ndump_modify d sort id format float %.17g\nrun 0\nundump d\n"
            "velocity all create 600 4928459 mom yes rot yes\ntimestep 0.001\nfix 1 all nve\nthermo 50\n"
            f"run {warmup}\nrun {steps}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--lmp", required=True); ap.add_argument("--plugin-dir", required=True)
    ap.add_argument("--pjrt", required=True); ap.add_argument("--rep", type=int, default=8)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--layout", default="matrix"); ap.add_argument("--spline-tol", type=lambda v: None if v == "none" else float(v), default=1e-10)
    ap.add_argument("--steps", type=int, default=100); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--mem-per-rank", type=float, default=2.0, help="GiB, the export's dense budget")
    ap.add_argument("--mpirun", default="mpirun"); ap.add_argument("--out", default="lammps_cpu_smoke")
    ap.add_argument("--standalone", type=int, default=0, help="also time N ACECalculator calls (one process, all threads)")
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    model, meta, _ = load(a.model)
    rcut = float(meta["rcut"])
    at = bulk("Si", "diamond", 5.43, cubic=True).repeat((a.rep,) * 3)
    at.rattle(0.05, seed=1)
    data = out / "x.data"; write(data, at, format="lammps-data", atom_style="atomic", specorder=["Si"])
    calc = ACECalculator(a.model, spline_tol=a.spline_tol); at.calc = calc
    e_ref, f_ref = at.get_potential_energy(), at.get_forces()
    pos0 = at.positions.copy()
    rows = []
    if a.standalone:      # run_standalone's MD-like calls: every atom walks N(0, 1e-3 A) per call
        rng, ts = np.random.default_rng(0), []
        for _ in range(a.standalone):
            at.positions += rng.normal(0, 1e-3, at.positions.shape); t0 = time.perf_counter()
            at.get_forces(); ts.append(time.perf_counter() - t0)
        row = {"ranks": "standalone", "us_per_atom_step": 1e6 * float(np.median(ts[2:])) / len(at),
               "threads": len(os.sched_getaffinity(0))}
        print(json.dumps(row), flush=True)
        (out / "rows.jsonl").open("a").write(json.dumps({**row, "model": a.model, "n_atoms": len(at)}) + "\n")
        at.positions = pos0; e_ref, f_ref = at.get_potential_energy(), at.get_forces()   # the data file's positions
    for nr in a.ranks:
        procs = GRIDS[nr]; d = out / f"r{nr}"; d.mkdir(exist_ok=True)
        cap = neighbour_capacity(at, rcut, procs=procs)
        graph = ({"max_neighbors": cap["max_neighbors"]} if a.layout == "matrix" else {"max_edges": cap["max_edges"]})
        t0 = time.time()
        b = export_lammps(model, meta, d / "bundle.json", layout=a.layout, max_atoms=cap["max_atoms"],
                          max_owned=cap["max_owned"], k_dense=cap["k_dense"], spline_tol=a.spline_tol,
                          platform="cpu", device_memory=a.mem_per_rank * 2**30, **graph)
        t_export = time.time() - t0
        (d / "in.lmp").write_text(deck(data, d / "bundle.json", procs, d / "f.dump", a.warmup, a.steps))
        env = {**os.environ, "OMP_NUM_THREADS": "1", "LAMMPS_PLUGIN_PATH": a.plugin_dir}
        cmd = [a.mpirun, "-np", str(nr), "--bind-to", "core", "--map-by", "core", a.lmp, "-k", "on", "t", "1",
               "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half", "-var", "pjrt", a.pjrt,
               "-in", "in.lmp", "-log", "log.lammps"]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True)
        wall = time.time() - t0
        log = (d / "log.lammps").read_text() if (d / "log.lammps").exists() else ""
        row = {"ranks": nr, "procs": procs, "layout": b["ace_jax"]["layout"], "cap": cap,
               "export_s": round(t_export, 2), "wall_s": round(wall, 1), "rc": p.returncode}
        if p.returncode == 0:
            pe = float(re.search(r"Step\s+PotEng\s*\n\s*0\s+(\S+)", log).group(1))
            f = np.loadtxt(d / "f.dump", skiprows=9)[:, 1:]
            loops = re.findall(r"Loop time of (\S+) on \d+ procs for (\d+) steps with (\d+) atoms", log)
            t, s, n = loops[-1]
            row.update(dE_per_atom=abs(pe - e_ref) / len(at), max_dF=float(np.abs(f - f_ref).max()),
                       us_per_atom_step=1e6 * float(t) / (int(s) * int(n)))
        else:
            row["error"] = (p.stderr or p.stdout)[-1500:]
        rows.append(row); print(json.dumps(row), flush=True)
        (out / "rows.jsonl").open("a").write(json.dumps({**row, "model": a.model, "n_atoms": len(at)}) + "\n")


if __name__ == "__main__":
    main()
