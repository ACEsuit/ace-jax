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

KOKKOS = {"acejax": "newton on neigh half", "mlpace": "newton on neigh half",
          "mace": "newton on neigh half"}


def _pair(style, model_path, elements, device):
    els = " ".join(elements)
    if style == "acejax":
        return f"pair_style jax/kk ${{pjrt}}\npair_coeff * * {model_path}\n"
    if style == "mlpace":
        return ("pair_style pace product\n" if device == "gpu" else "pair_style pace\n") + \
               f"pair_coeff * * {model_path} {els}\n"
    if style == "mace":                     # Symmetrix: element-specific .json
        return f"pair_style symmetrix/mace\npair_coeff * * {model_path} {els}\n"
    raise ValueError(style)


def failure_status(text):
    """'oom' if the output anywhere reports running out of memory, else 'error'."""
    low = text.lower()
    oom = ("out of memory" in low or "RESOURCE_EXHAUSTED" in text
           or "cudaErrorMemoryAllocation" in text or "failed to allocate" in low)  # Kokkos
    return "oom" if oom else "error"


def error_text(text, tail=300):
    """The ERROR/exception lines (an MPI_ABORT banner buries them), then the tail."""
    keys = ("error", "exception", "failed", "abort")
    lines = [l for l in text.splitlines() if any(k in l.lower() for k in keys)
             and "MPI_ABORT" not in l][:8]
    return "\n".join(lines)[:1200] + "\n...\n" + text[-tail:]


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
           "timestep 0.0001\nfix 1 all nve\nthermo_style custom step pe atoms\nthermo 50\n"
           "thermo_modify format float %.17g\n")       # full precision: parity reads pe
    if dump:
        txt += f"dump d all custom 1 {dump} id fx fy fz\ndump_modify d sort id format float %.17g\nrun 0\nundump d\n"
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
    periodic shell within rcut + skin of each face) size the LAMMPS position
    buffer (max_atoms); neighbour slots (k_dense, max_edges) are sized for
    rcut + skin.  lammps-jax packs only pairs within rcut, but the benchmark's
    random-weight structures compress during the run (Cantor: the largest
    coordination within rcut climbs from 42 to 50 in 250 steps), and between
    list rebuilds no atom can gain more neighbours within rcut than its
    rcut + skin list holds -- so that count is the safe bound
    (docs/perf-lammps-large-n.md).  The dense energy function evaluates owned
    rows only (max_owned), and senders are always owned, so max_edges counts
    owned rows."""
    from ace_jax.eval import sparse_graph
    L = np.linalg.norm(at.cell.array, axis=1)
    ghost = float(np.prod((L + 2 * (rcut + skin)) / L))
    g = sparse_graph(at.positions, at.cell.array, at.pbc, rcut + skin)
    k_max = int(np.bincount(g.senders, minlength=len(at)).max())
    max_owned = int(np.ceil(1.1 * len(at)))
    k_dense = k_max + 8                     # overflow is loud (NaN), never a truncation
    return {"max_atoms": int(np.ceil(len(at) * ghost * 1.1)), "k_max": k_max,
            "k_dense": k_dense, "max_edges": max_owned * k_dense, "max_owned": max_owned}


def _export_inprocess(row, at, dtype, workdir, layout="auto"):
    """Export the ace-jax model as a lammps-jax bundle sized for `at` (this
    process: it initialises JAX, and on a GPU host keeps JAX's memory pool)."""
    import time
    import jax
    jax.config.update("jax_enable_x64", dtype == "float64")
    from ase.data import atomic_numbers
    from ace_jax.eval import load
    from ace_jax.export.lammps import export_lammps
    model, meta, _ = load(row["path"])
    cap = capacity(at, float(meta["rcut"]))
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    b = export_lammps(model, meta, pathlib.Path(workdir) / "bundle.json", max_atoms=cap["max_atoms"],
                      max_edges=cap["max_edges"], k_dense=cap["k_dense"], dtype=dtype, layout=layout,
                      type_elements=[atomic_numbers[e] for e in row["elements"]],  # data-file order
                      max_owned=cap["max_owned"],
                      **({"spline_tol": row["spline_tol"]} if "spline_tol" in row else {}))
    return (str(pathlib.Path(workdir) / "bundle.json"), b["ace_jax"]["layout"], time.perf_counter() - t0,
            splined_meta(b["ace_jax"]))


def splined_meta(aj):
    """What the bundle's lean form splined, from its `ace_jax` metadata:
    {"spline_tol", "n_intervals"}, or None when every radial stayed exact."""
    if aj.get("spline_tol") is None:
        return None
    return {"spline_tol": aj["spline_tol"], "n_intervals": aj.get("spline_intervals")}


def export_bundle(row, at, dtype, workdir, layout="auto", info=None):
    """Export the ace-jax model as a lammps-jax bundle sized for `at`, in a
    child process: exporting here would initialise JAX on the GPU, whose default
    pool (75% of the card) stays allocated while LAMMPS runs, leaving lammps-jax
    a quarter of it.  Returns (bundle path, layout used, export seconds); a
    row's `spline_tol` is passed to export_lammps, and `info` (a dict), when
    given, gets "splined" (`splined_meta` of the bundle)."""
    keep = {k: row[k] for k in ("name", "system", "path", "elements", "spline_tol") if k in row}
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--export", json.dumps(keep),
           str(len(at)), dtype, str(workdir), layout]
    bench = str(pathlib.Path(__file__).resolve().parents[1])     # so the child imports scaling
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [bench] + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x])}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"bundle export failed ({p.returncode}): {(p.stderr or p.stdout)[-2000:]}")
    out = json.loads(lines[-1])
    if info is not None:
        info["splined"] = out.get("splined")
    return out["bundle"], out["layout"], out["compile_s"]




STEP_BUDGET_S = 60.0          # target wall time of the timed segment


def choose_steps(prev_step_s, prev_n, n, target_s=STEP_BUDGET_S, max_steps=200, min_steps=10):
    """(timed steps, warmup steps) for an n-atom case, from the step time of the
    previous (smaller) case of the same line scaled linearly in n: about
    target_s of timed MD, 10..200 steps, warmup a quarter of that (3..50).  No
    hint (the first size of a line): 200 / 50."""
    if not prev_step_s or not prev_n:
        return max_steps, 50
    est = prev_step_s * n / prev_n
    steps = int(min(max_steps, max(min_steps, round(target_s / est))))
    return steps, min(50, max(3, steps // 4))


def bundle_layout(prev):
    """The ace-jax bundle layout for this case: once a line has fallen back to
    sparse (dense out of memory), its larger sizes export sparse directly."""
    return "sparse" if (prev or {}).get("layout") == "sparse" else "auto"


def run_case(row, n_atoms, dtype, device, lmp, ranks, workdir, pjrt=None, prev=None):
    work = pathlib.Path(workdir); work.mkdir(parents=True, exist_ok=True)
    at = supercell(row["system"], n_atoms)
    data = work / "x.data"
    write(data, at, format="lammps-data", specorder=row["elements"], masses=True)
    style = "acejax" if row["code"].startswith("acejax") else row["code"]    # mlpace, mace
    if style == "mace" and not pathlib.Path(row.get("symmetrix") or "").exists():  # e.g. MH-1
        return {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
                "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
                "status": "unsupported", "error": f"no Symmetrix model {row.get('symmetrix')}"}
    steps, warmup = choose_steps((prev or {}).get("step_s"), (prev or {}).get("n_atoms"), n_atoms)
    if style != "acejax":
        model = row.get("symmetrix") if style == "mace" else row["path"]
        return _run_lammps(row, style, model, data, work, n_atoms, dtype, device, lmp, ranks,
                           pjrt, steps, warmup, {})
    # ace-jax: the export's auto layout judges dense against the exporting
    # process's budget, not LAMMPS's (PJRT pool + Kokkos + ghosts): a dense
    # bundle that runs out of memory is retried sparse
    layout = bundle_layout(prev)
    while True:
        info = {}
        bundle, used, compile_s = export_bundle(row, at, dtype, work, layout=layout, info=info)
        out = _run_lammps(row, style, bundle, data, work, n_atoms, dtype, device, lmp, ranks, pjrt,
                          steps, warmup, {"layout": used, "compile_s": compile_s,
                                          "spline_tol": row.get("spline_tol", "auto"),
                                          "splined": info.get("splined")})
        if out["status"] == "oom" and used == "dense":
            layout = "sparse"
            continue
        if layout == "sparse" and bundle_layout(prev) == "auto":
            out["dense_oom"] = True                     # this size is where the line went sparse
        return out


def _run_lammps(row, style, model, data, work, n_atoms, dtype, device, lmp, ranks, pjrt, steps,
                warmup, extra):
    (work / "in.bench").write_text(lammps_input(style, model, row["elements"], data, device, steps,
                                                warmup=warmup))
    cmd = [lmp, "-in", "in.bench", "-log", "log.lammps", "-nocite"]
    if device == "gpu":
        cmd += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", *KOKKOS[style].split()]
    elif ranks > 1 and style != "acejax":
        cmd = ["mpirun", "-np", str(ranks)] + cmd
    if pjrt:
        cmd += ["-var", "pjrt", pjrt]
    out = {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
           "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
           "ranks": ranks, "status": "ok", **extra}
    (work / "log.lammps").unlink(missing_ok=True)
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=3600)
    log = (work / "log.lammps").read_text() if (work / "log.lammps").exists() else p.stdout
    if not finished(p.returncode, log) or "Loop time" not in log:
        tail = error_text((p.stderr or "") + (p.stdout or "") + log)
        out["status"] = failure_status((p.stderr or "") + (p.stdout or "") + log)
        out["error"] = tail
        return out
    parsed = parse_log(log)
    out.update(parsed)
    if parsed["nan"] or parsed["n_atoms_end"] != n_atoms:
        out["status"] = "unstable"
    out["atom_steps_per_s"] = n_atoms / parsed["step_s"]
    return out


if __name__ == "__main__" and sys.argv[1:2] == ["--export"]:
    _row, _n, _dtype, _work, _layout = json.loads(sys.argv[2]), int(sys.argv[3]), *sys.argv[4:7]
    _b, _l, _t, _s = _export_inprocess(_row, supercell(_row["system"], _n), _dtype, _work, _layout)
    print(json.dumps({"bundle": _b, "layout": _l, "compile_s": _t, "splined": _s}))
elif __name__ == "__main__":
    from scaling.models import planned_models
    name, n, dtype, device, lmp, ranks, workdir = sys.argv[1:8]
    row = next(r for r in planned_models() if r["name"] == name)
    row["bundle"] = os.environ.get("ACEJAX_BUNDLE", "")
    prev = json.loads(os.environ["BENCH_PREV"]) if os.environ.get("BENCH_PREV") else None
    print(json.dumps(run_case(row, int(n), dtype, device, lmp, int(ranks), workdir,
                              os.environ.get("PJRT_PLUGIN"), prev)))
