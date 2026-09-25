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

from ase.io import write

from scaling.structures import supercell

KOKKOS = {"acejax": "newton off neigh full", "mlpace": "newton on neigh half",
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


def lammps_input(style, model_path, elements, data_path, device, run_steps, dump=None):
    txt = ("units metal\natom_style atomic\nboundary p p p\natom_modify map yes\n"
           f"read_data {data_path}\n" + _pair(style, model_path, elements, device) +
           "neighbor 1.0 bin\nneigh_modify every 1 delay 0 check yes\n"
           "timestep 0.0001\nfix 1 all nve\nthermo_style custom step pe atoms\nthermo 50\n")
    if dump:
        txt += f"dump d all custom 1 {dump} id fx fy fz\ndump_modify d sort id\nrun 0\nundump d\n"
    return txt + f"run 50\nrun {run_steps}\n"


_LOOP = re.compile(r"Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


def parse_log(text):
    loops = _LOOP.findall(text)
    loop_s, procs, steps, natoms = loops[-1]
    return {"loop_s": float(loop_s), "step_s": float(loop_s) / int(steps), "procs": int(procs),
            "n_atoms_end": int(natoms), "nan": bool(re.search(r"\bnan\b", text, re.I))}


def run_case(row, n_atoms, dtype, device, lmp, ranks, workdir, pjrt=None):
    work = pathlib.Path(workdir); work.mkdir(parents=True, exist_ok=True)
    at = supercell(row["system"], n_atoms)
    data = work / "x.data"
    write(data, at, format="lammps-data", specorder=row["elements"], masses=True)
    style = {"acejax-pace": "acejax", "acejax-ace": "acejax", "mlpace": "mlpace", "mace": "mace"}[row["code"]]
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
           "ranks": ranks, "status": "ok"}
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=3600)
    log = (work / "log.lammps").read_text() if (work / "log.lammps").exists() else p.stdout
    if p.returncode != 0 or "Loop time" not in log:
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
