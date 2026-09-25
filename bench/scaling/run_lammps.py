"""One LAMMPS benchmark case -> one JSON row.

Writes the data file with species in the model's element order (lammps-jax
maps type t -> species t-1; ML-PACE / Symmetrix map by pair_coeff names), runs a
50-step warm-up and a timed segment, and parses the timed "Loop time".  Runs
that lose atoms or produce NaN are failures, never timings.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
from ase.io import write

from scaling.structures import supercell

KOKKOS = {"acejax": "newton on neigh full", "mlpace": "newton on neigh half",
          "mace": "newton on neigh half"}


def _pair(style, model_path, elements, device):
    els = " ".join(elements)
    if style == "acejax":
        return f"pair_style jax/kk ${{pjrt}}\npair_coeff * * {model_path}\n"
    if style == "mlpace":
        return (f"pair_style pace product\n" if device == "gpu" else "pair_style pace\n") + \
               f"pair_coeff * * {model_path} {els}\n"
    if style == "mace":                     # Symmetrix: element-specific .json
        return f"pair_style symmetrix/mace\npair_coeff * * {model_path} {els}\n"
    raise ValueError(style)


def finished(returncode, log):
    """A run completed if LAMMPS exited cleanly or got as far as its final
    `Total wall time` line: the Symmetrix tree aborts afterwards, in a static
    destructor (a std::map freed twice across liblammps / libkokkoskernels)."""
    return returncode == 0 or "Total wall time" in log


def lammps_input(style, model_path, elements, data_path, device, run_steps, dump=None,
                 warmup=50):
    txt = ("units metal\natom_style atomic\nboundary p p p\natom_modify map yes\n"
           f"read_data {data_path}\n" + _pair(style, model_path, elements, device) +
           "neighbor 1.0 bin\nneigh_modify every 1 delay 0 check yes\n"
           "timestep 0.0001\nfix 1 all nve\nthermo_style custom step pe atoms\nthermo 50\n")
    if dump:
        txt += f"dump d all custom 1 {dump} id fx fy fz\ndump_modify d sort id\nrun 0\nundump d\n"
    return txt + (f"run {warmup}\n" if warmup else "") + f"run {run_steps}\n"


_LOOP = re.compile(r"Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


def parse_log(text):
    loops = _LOOP.findall(text)
    loop_s, procs, steps, natoms = loops[-1]
    return {"loop_s": float(loop_s), "step_s": float(loop_s) / int(steps), "procs": int(procs),
            "n_atoms_end": int(natoms), "nan": bool(re.search(r"\bnan\b", text, re.I))}


def read_pe(text):
    """Potential energy of the first thermo row after 'Step PotEng'."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.split()[:2] == ["Step", "PotEng"])
    return float(lines[i + 1].split()[1])


def read_dump_forces(path):
    """(n, 3) forces from a `dump custom ... id fx fy fz` (sorted by id)."""
    lines = pathlib.Path(path).read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("ITEM: ATOMS"))
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[i + 1:] if ln.strip()])
    return rows[np.argsort(rows[:, 0]), 1:4]


def capacity(at, rcut, skin=1.0):
    """lammps-jax buffer sizes for this structure: owned + ghost atoms (the
    periodic shell within rcut + skin of each face), neighbours per atom."""
    from ace_jax.eval import sparse_graph
    L = np.linalg.norm(at.cell.array, axis=1)
    ghost = float(np.prod((L + 2 * (rcut + skin)) / L))
    g = sparse_graph(at.positions, at.cell.array, at.pbc, rcut + skin)
    k_max = int(np.bincount(g.senders, minlength=len(at)).max())
    return {"max_atoms": int(np.ceil(len(at) * ghost * 1.1)), "k_max": k_max,
            "k_dense": k_max + 8, "max_edges": int(len(at) * (k_max + 8))}


def export_bundle(row, at, dtype, workdir):
    """Export the ace-jax model as a lammps-jax bundle sized for `at`."""
    import time
    import jax
    jax.config.update("jax_enable_x64", dtype == "float64")
    from ace_jax.eval import load
    from ace_jax.export.lammps import export_lammps
    model, meta, _ = load(row["path"])
    cap = capacity(at, float(meta["rcut"]))
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    b = export_lammps(model, meta, pathlib.Path(workdir) / "bundle.json", max_atoms=cap["max_atoms"],
                      max_edges=cap["max_edges"], k_dense=cap["k_dense"], dtype=dtype)
    return str(pathlib.Path(workdir) / "bundle.json"), b["ace_jax"]["layout"], time.perf_counter() - t0


def run_case(row, n_atoms, dtype, device, lmp, ranks, workdir, pjrt=None):
    work = pathlib.Path(workdir); work.mkdir(parents=True, exist_ok=True)
    at = supercell(row["system"], n_atoms)
    data = work / "x.data"
    write(data, at, format="lammps-data", specorder=row["elements"], masses=True)
    style = {"acejax-pace": "acejax", "acejax-ace": "acejax", "mlpace": "mlpace", "mace": "mace"}[row["code"]]
    export = {}
    if style == "acejax" and not row.get("bundle"):              # sized for this structure
        row = dict(row)
        row["bundle"], export["layout"], export["compile_s"] = export_bundle(row, at, dtype, work)
    model = {"acejax": row.get("bundle"), "mace": row.get("symmetrix")}.get(style) or row["path"]
    if style == "mace" and not pathlib.Path(model).exists():     # e.g. MH-1 not exportable
        return {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
                "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
                "status": "unsupported", "error": f"no Symmetrix model {model}"}
    (work / "in.bench").write_text(lammps_input(style, model, row["elements"], data, device, 200))
    cmd = [lmp, "-in", "in.bench", "-log", "log.lammps", "-nocite"]
    if device == "gpu":
        cmd += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", *KOKKOS[style].split()]
    elif ranks > 1 and style != "acejax":
        cmd = ["mpirun", "-np", str(ranks)] + cmd
    if pjrt:
        cmd += ["-var", "pjrt", pjrt]
    out = {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
           "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
           "ranks": ranks, "status": "ok", **export}
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=3600)
    log = (work / "log.lammps").read_text() if (work / "log.lammps").exists() else p.stdout
    if not finished(p.returncode, log) or "Loop time" not in log:
        tail = (p.stderr or p.stdout)[-300:]
        out["status"] = "oom" if "out of memory" in tail.lower() or "RESOURCE_EXHAUSTED" in tail else "error"
        out["error"] = tail
        return out
    parsed = parse_log(log)
    out.update(parsed)
    if parsed["nan"] or parsed["n_atoms_end"] != n_atoms:
        out["status"] = "unstable"
    out["atom_steps_per_s"] = n_atoms / parsed["step_s"]
    return out


if __name__ == "__main__":
    from scaling.models import planned_models
    name, n, dtype, device, lmp, ranks, workdir = sys.argv[1:8]
    row = next(r for r in planned_models() if r["name"] == name)
    row["bundle"] = os.environ.get("ACEJAX_BUNDLE", "")
    print(json.dumps(run_case(row, int(n), dtype, device, lmp, int(ranks), workdir,
                              os.environ.get("PJRT_PLUGIN"))))
