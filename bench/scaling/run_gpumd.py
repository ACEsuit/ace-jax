"""NEP run by GPUMD's own MD code (`gpumd`), on the GPU: the mixed-precision line beside LAMMPS `pair_style nep`.

GPUMD's NEP kernels (src/force/nep.cu) compute descriptors, the network and pair forces in float and accumulate
in double, so its rows carry dtype "mixed", never float64.  The input mirrors run_lammps.lammps_input: the same
rattled supercell at rest, NVE at 0.1 fs, GPUMD's default 1 Å neighbour skin (rebuilt when an atom moves half of
it), a warmup run, then the timed run; step_s is the timed run's own "Time used for this run" / steps.
`gpumd` is a wrapper (env json `gpumd`) that loads the CUDA modules.
"""
import pathlib
import re
import subprocess

import numpy as np

from .run_lammps import choose_steps
from .structures import supercell

_TIME = re.compile(r"Time used for this run = ([\d.eE+-]+) second")


def write_model_xyz(path, at):
    """GPUMD's model.xyz: zero velocities, so the run starts at rest as LAMMPS's does (no velocity command)."""
    c = " ".join(f"{x:.16g}" for x in at.cell.array.ravel())
    pbc = " ".join("T" if p else "F" for p in at.pbc)
    with open(path, "w") as f:
        f.write(f'{len(at)}\nlattice="{c}" pbc="{pbc}" properties=species:S:1:pos:R:3:vel:R:3\n')
        for s, p in zip(at.get_chemical_symbols(), at.positions):
            f.write(f"{s} {p[0]:.16g} {p[1]:.16g} {p[2]:.16g} 0 0 0\n")


def _run(gpumd, work, timeout=3600):
    p = subprocess.run([gpumd], cwd=work, capture_output=True, text=True, timeout=timeout)
    log = p.stdout + p.stderr
    (pathlib.Path(work) / "gpumd.log").write_text(log)
    return p.returncode, log


def gpumd_ef(model, at, gpumd, work):
    """(E, F) from one NVE step at 1e-8 fs from rest (atoms move ~1e-16 Å): GPUMD's energy and forces at
    these positions.  Energy from thermo.out (full precision), forces from dump_xyz."""
    work = pathlib.Path(work); work.mkdir(parents=True, exist_ok=True)
    write_model_xyz(work / "model.xyz", at)
    (work / "run.in").write_text(f"potential {model}\nensemble nve\ntime_step 1e-8\ndump_thermo 1\n"
                                 "dump_xyz 1 out.xyz force\nrun 1\n")
    for f in ("thermo.out", "out.xyz"):
        (work / f).unlink(missing_ok=True)
    rc, log = _run(gpumd, work, timeout=600)
    if rc != 0:
        raise RuntimeError(f"gpumd exited {rc}: {log[-600:]}")
    rows = [l.split() for l in (work / "thermo.out").read_text().splitlines() if l.strip() and not l.startswith("#")]
    E = float(rows[-1][2])                                          # columns: T KE PE ...
    lines = (work / "out.xyz").read_text().splitlines()
    F = np.array([[float(x) for x in l.split()[4:7]] for l in lines[2:2 + len(at)]])   # species pos(3) force(3)
    return E, F


def run_case(row, n_atoms, device, gpumd, workdir, prev=None):
    """One timed GPUMD case (GPU only)."""
    work = pathlib.Path(workdir); work.mkdir(parents=True, exist_ok=True)
    out = {"code": row["code"], "mode": "gpumd", "model": row["name"], "size": row["size"], "system": row["system"],
           "n_atoms": n_atoms, "device": device, "dtype": "mixed", "status": "ok"}
    if device != "gpu":
        return {**out, "status": "unsupported", "error": "GPUMD runs on the GPU only"}
    steps, warmup = choose_steps((prev or {}).get("step_s"), (prev or {}).get("n_atoms"), n_atoms)
    write_model_xyz(work / "model.xyz", supercell(row["system"], n_atoms, rattle=row.get("rattle", 0.0)))
    (work / "run.in").write_text(f"potential {row['path']}\nensemble nve\ntime_step 0.1\ndump_thermo {max(1, steps)}\n"
                                 f"run {warmup}\nensemble nve\nrun {steps}\n")
    rc, log = _run(gpumd, work)
    t = [float(x) for x in _TIME.findall(log)]
    if rc != 0 or len(t) < 2:
        oom = "out of memory" in log.lower() or "cudaErrorMemoryAllocation" in log
        return {**out, "status": "oom" if oom else "error", "error": log[-1200:]}
    pe = [float(l.split()[2]) for l in (work / "thermo.out").read_text().splitlines()
          if l.strip() and not l.startswith("#")]
    step_s = t[-1] / steps
    return {**out, "steps": steps, "warmup": warmup, "loop_s": t[-1], "step_s": step_s,
            "atom_steps_per_s": n_atoms / step_s, "nan": not np.all(np.isfinite(pe)), "n_atoms_end": n_atoms,
            **({"status": "unstable"} if not np.all(np.isfinite(pe)) else {})}
