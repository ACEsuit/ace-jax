"""One LAMMPS benchmark case -> one JSON row.

Writes the data file with species in the model's element order (lammps-jax
maps type t -> species t-1; ML-PACE / Symmetrix / the PR 309 `pair_style ace`
map by pair_coeff names), runs a
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
    if style == TRIM:                       # PR 309 plugin, then the juliac --trim library
        return f"plugin load ${{aceplugin}}\npair_style ace\npair_coeff * * {model_path} {els}\n"
    raise ValueError(style)


TRIM = "acepotentials-trim"


def lammps_vars(style, pjrt=None, aceplugin=None):
    """`-var` arguments the input reads: ${pjrt} (lammps-jax), ${aceplugin} (the
    PR 309 plugin, from the host env json's `ace_plugin` via ACE_PLUGIN)."""
    out = ["-var", "pjrt", pjrt] if pjrt else []
    if style == TRIM:
        if not aceplugin:
            raise ValueError("acepotentials-trim needs the PR 309 LAMMPS plugin: the host env json's "
                             "`ace_plugin` (ACE_PLUGIN)")
        out += ["-var", "aceplugin", aceplugin]
    return out


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


def capacity(at, rcut, skin=1.0, tight=False, list_headroom=0.5):
    """lammps-jax buffer sizes for this structure (ace_jax.export.lammps.
    neighbour_capacity, margin 8): owned + ghost atoms (the periodic shell
    within rcut + skin of each face) size the LAMMPS position buffer
    (max_atoms); model slots (k_dense, max_edges) are sized for rcut + skin.
    lammps-jax packs only pairs within rcut, but the benchmark's random-weight
    structures compress during the run (Cantor: the largest coordination within
    rcut climbs from 42 to 50 in 250 steps), and between list rebuilds no atom
    can gain more neighbours within rcut than its rcut + skin list holds -- so
    that count is the safe bound (docs/dev/perf-lammps-large-n.md).  The dense
    energy function evaluates owned rows only (max_owned), and senders are
    always owned, so max_edges counts owned rows.  The neighbour-matrix list
    (max_neighbors) holds the rcut + skin list whatever the model slots.

    tight=True (bench --tight-slots; opt-in, never the main suite): model slots
    for rcut pairs only, 1.2-1.4x faster on Cantor, safe only on a deck whose coordination
    stays within 8 of the start (stable MD); an overflow is a NaN step.

    list_headroom (bench --list-headroom): the matrix list's headroom over
    k(rcut + skin); 0.5 from one observed overflow (neighbour_capacity)."""
    from ace_jax.export.lammps import neighbour_capacity
    c = neighbour_capacity(at, rcut, skin=skin, slots="cutoff" if tight else "skin", margin=8,
                           list_headroom=list_headroom)
    return {"max_atoms": c["max_atoms"], "k_max": c["k_list"], "k_dense": c["k_dense"],
            "max_edges": c["max_edges"], "max_owned": c["max_owned"],
            "max_neighbors": c["max_neighbors"]}


def _export_inprocess(row, at, dtype, workdir, layout="auto", slots="skin", list_headroom=0.5):
    """Export the ace-jax model as a lammps-jax bundle sized for `at` (this
    process: it initialises JAX, and on a GPU host keeps JAX's memory pool)."""
    import time
    import jax
    jax.config.update("jax_enable_x64", dtype == "float64")
    from ase.data import atomic_numbers
    from ace_jax.eval import load
    from ace_jax.export.lammps import export_lammps
    model, meta, _ = load(row["path"])
    cap = capacity(at, float(meta["rcut"]), tight=slots == "cutoff", list_headroom=list_headroom)
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    b = export_lammps(model, meta, pathlib.Path(workdir) / "bundle.json", max_atoms=cap["max_atoms"],
                      max_edges=cap["max_edges"], k_dense=cap["k_dense"], dtype=dtype, layout=layout,
                      type_elements=[atomic_numbers[e] for e in row["elements"]],  # data-file order
                      max_owned=cap["max_owned"], max_neighbors=cap["max_neighbors"],
                      **({"spline_tol": row["spline_tol"]} if "spline_tol" in row else {}))
    return (str(pathlib.Path(workdir) / "bundle.json"), b["ace_jax"]["layout"], time.perf_counter() - t0,
            splined_meta(b["ace_jax"]))


def splined_meta(aj):
    """What the bundle's lean form splined, from its `ace_jax` metadata:
    {"spline_tol", "n_intervals"}, or None when every radial stayed exact."""
    if aj.get("spline_tol") is None:
        return None
    return {"spline_tol": aj["spline_tol"], "n_intervals": aj.get("spline_intervals")}


def export_bundle(row, at, dtype, workdir, layout="auto", slots="skin", list_headroom=0.5,
                  info=None):
    """Export the ace-jax model as a lammps-jax bundle sized for `at`, in a
    child process: exporting here would initialise JAX on the GPU, whose default
    pool (75% of the card) stays allocated while LAMMPS runs, leaving lammps-jax
    a quarter of it.  Returns (bundle path, layout used, export seconds); a
    row's `spline_tol` is passed to export_lammps, and `info` (a dict), when
    given, gets "splined" (`splined_meta` of the bundle)."""
    keep = {k: row[k] for k in ("name", "system", "path", "elements", "spline_tol") if k in row}
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--export", json.dumps(keep),
           str(len(at)), dtype, str(workdir), layout, slots, str(list_headroom)]
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
    sparse (dense out of memory), its larger sizes export sparse directly; once
    its matrix list overflowed, they export packed dense directly."""
    prev = prev or {}
    if prev.get("layout") == "sparse":
        return "sparse"
    return "dense" if prev.get("matrix_overflow") else "auto"


def _trim_status(row):
    """None when the row's trim library exists, else (status, why): `unsupported`
    when the builder recorded a refused export, `error` when it was never built."""
    lib = row.get("trim_lib") or ""
    if pathlib.Path(lib).exists():
        return None
    from scaling import models
    try:
        entry = models.load_manifest().get(lib, {})
    except (OSError, ValueError):
        entry = {}
    if entry.get("unsupported"):
        return "unsupported", f"trim export failed for {lib}: {entry['unsupported']}"
    return "error", f"no trim library {lib}: run `python bench/scaling/models.py acepotentials-trim`"


def _trim_build(row):
    """The manifest's record of the row's trim library (provenance for the row)."""
    from scaling import models
    try:
        e = models.load_manifest().get(row["trim_lib"], {})
    except (OSError, ValueError):
        return None
    return {k: e[k] for k in ("build_id", "sha256", "build_cpu", "cpu_target", "versions", "identity")
            if k in e} or None


def run_case(row, n_atoms, dtype, device, lmp, ranks, workdir, pjrt=None, prev=None,
             slots="skin", list_headroom=0.5, aceplugin=None):
    work = pathlib.Path(workdir); work.mkdir(parents=True, exist_ok=True)
    at = supercell(row["system"], n_atoms)
    data = work / "x.data"
    write(data, at, format="lammps-data", specorder=row["elements"], masses=True)
    style = "acejax" if row["code"].startswith("acejax") else row["code"]    # mlpace, mace
    if style == "mace" and not pathlib.Path(row.get("symmetrix") or "").exists():  # e.g. MH-1
        return {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
                "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
                "status": "unsupported", "error": f"no Symmetrix model {row.get('symmetrix')}"}
    if style == TRIM and _trim_status(row):
        st, why = _trim_status(row)
        return {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
                "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
                "status": st, "error": why}
    steps, warmup = choose_steps((prev or {}).get("step_s"), (prev or {}).get("n_atoms"), n_atoms)
    if style == TRIM:
        return _run_lammps(row, style, row["trim_lib"], data, work, n_atoms, dtype, device, lmp,
                           ranks, pjrt, steps, warmup,
                           {"trim_lib": row["trim_lib"], "trim_build": _trim_build(row)},
                           aceplugin=aceplugin)
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
        bundle, used, compile_s = export_bundle(row, at, dtype, work, layout=layout, slots=slots,
                                                list_headroom=list_headroom, info=info)
        extra = {"layout": used, "compile_s": compile_s,
                 "spline_tol": row.get("spline_tol", "auto"), "splined": info.get("splined")}
        if slots != "skin":
            extra["slots"] = slots
        if list_headroom != 0.5:
            extra["list_headroom"] = list_headroom
        out = _run_lammps(row, style, bundle, data, work, n_atoms, dtype, device, lmp, ranks, pjrt,
                          steps, warmup, extra)
        if out["status"] == "oom" and used in ("dense", "matrix"):
            layout = "sparse"
            continue
        # the matrix list holds rcut + skin pairs and aborts when a row outgrows
        # max_neighbors (random-weight structures compress); packed dense holds
        # only pairs within rcut, so retry the case there
        if (out["status"] == "error" and used == "matrix"
                and "neighbor capacity exceeded" in str(out.get("error", ""))):
            layout = "dense"
            continue
        if layout == "dense" and used == "dense" and bundle_layout(prev) != "sparse":
            out["matrix_overflow"] = True               # this line left the matrix layout
        if layout == "sparse" and bundle_layout(prev) == "auto":
            out["dense_oom"] = True                     # this size is where the line went sparse
        return out


def mpirun_args():
    """The host's extra mpirun arguments (sweep.HOSTS `mpirun_args`, handed over
    as BENCH_MPIRUN_ARGS, a JSON list: lestrade binds one rank per P-core)."""
    return json.loads(os.environ.get("BENCH_MPIRUN_ARGS") or "[]")


def _run_lammps(row, style, model, data, work, n_atoms, dtype, device, lmp, ranks, pjrt, steps,
                warmup, extra, aceplugin=None):
    (work / "in.bench").write_text(lammps_input(style, model, row["elements"], data, device, steps,
                                                warmup=warmup))
    cmd = [lmp, "-in", "in.bench", "-log", "log.lammps", "-nocite"]
    if device == "gpu":
        cmd += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", *KOKKOS[style].split()]
    elif ranks > 1 and style != "acejax":
        cmd = ["mpirun", "-np", str(ranks), *mpirun_args()] + cmd
    cmd += lammps_vars(style, pjrt, aceplugin)
    out = {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
           "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
           "ranks": ranks, "status": "ok", **extra}
    if cmd[0] == "mpirun" and mpirun_args():
        out["mpirun_args"] = mpirun_args()
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
    _slots = sys.argv[7] if len(sys.argv) > 7 else "skin"
    _head = float(sys.argv[8]) if len(sys.argv) > 8 else 0.5
    _b, _l, _t, _s = _export_inprocess(_row, supercell(_row["system"], _n), _dtype, _work, _layout,
                                       _slots, _head)
    print(json.dumps({"bundle": _b, "layout": _l, "compile_s": _t, "splined": _s}))
elif __name__ == "__main__":
    from scaling.models import planned_models
    # --tight-slots: model slots for rcut pairs (capacity(tight=True)); opt-in,
    # never the main suite, whose random-weight structures compress
    # --list-headroom H: the matrix list's headroom over k(rcut + skin) (default 0.5)
    tight = "--tight-slots" in sys.argv
    args = [x for x in sys.argv[1:] if x != "--tight-slots"]
    head = 0.5
    if "--list-headroom" in args:
        i = args.index("--list-headroom")
        head = float(args[i + 1])
        del args[i:i + 2]
    name, n, dtype, device, lmp, ranks, workdir = args[:7]
    row = next(r for r in planned_models() if r["name"] == name)
    row["bundle"] = os.environ.get("ACEJAX_BUNDLE", "")
    prev = json.loads(os.environ["BENCH_PREV"]) if os.environ.get("BENCH_PREV") else None
    print(json.dumps(run_case(row, int(n), dtype, device, lmp, int(ranks), workdir,
                              os.environ.get("PJRT_PLUGIN"), prev,
                              slots="cutoff" if tight else "skin", list_headroom=head,
                              aceplugin=os.environ.get("ACE_PLUGIN"))))
