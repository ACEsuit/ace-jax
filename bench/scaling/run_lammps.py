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
          "mace": "newton on neigh half",                      # gap: CPU only, no KOKKOS
          "mace-xl": "newton on neigh half", "mace-xl-f32": "newton on neigh half",
          # GRACE (the yury-lysogorskiy/lammps `grace` branch, bench/scaling/envs/sulis-grace.sh): native Kokkos 1L
          # (gracemaker export_kokkos .npz; full float64 by default) and the C++ GRACE-FS evaluator; both need newton on
          "grace": "newton on neigh half", "grace-fs": "newton on neigh half",
          "grace-2l": "newton on neigh half"}   # GRACE-2L: grace/2l/kk (GPU), grace/2l/cpu/kk (CPU)
# Symmetrix-XL (bench/scaling/envs/sulis-xl.sh): its fast path is pair_style symmetrix/mace[/float32]/kk with
# a JIT-compiled R1 artifact, so it runs Kokkos on the CPU too (the non-kk style is the old evaluator)
XL = {"mace-xl": ("symmetrix/mace/kk", "float64"), "mace-xl-f32": ("symmetrix/mace/float32/kk", "float32")}


def _pair(style, model_path, elements, device, extra=None):
    els = " ".join(elements)
    if style == "acejax":
        return f"pair_style jax/kk ${{pjrt}}\npair_coeff * * {model_path}\n"
    if style == "mlpace":
        return ("pair_style pace product\n" if device == "gpu" else "pair_style pace\n") + \
               f"pair_coeff * * {model_path} {els}\n"
    if style == "mace":                     # Symmetrix: element-specific .json
        return f"pair_style symmetrix/mace\npair_coeff * * {model_path} {els}\n"
    if style in XL:                         # extra: the R1 artifact fragment (xl_pair_args); GPU one rank
        dd = "no_domain_decomposition" if device == "gpu" else "mpi_message_passing"
        return (f"pair_style {XL[style][0]} {dd} streamed_edges direct profile speed {extra}\n"
                f"pair_coeff * * {model_path} {els}\n")
    if style == "grace":                    # GRACE-1L .npz: GPU kernel, or its CPU-tuned twin (Kokkos OpenMP)
        # the chunk-size sweep (envs/sulis-gpu-grace-chunk.sbatch): $GRACE_CHUNKSIZE (default 4096 atoms per kernel
        # batch) and $GRACE_PRECISION ("" float64, "/mixed", "/fp32"); unset = the published rows' defaults
        prec, chunk = os.environ.get("GRACE_PRECISION", ""), os.environ.get("GRACE_CHUNKSIZE")
        return (f"pair_style {('grace/1l/kk' if device == 'gpu' else 'grace/1l/cpu/kk') + prec}"
                f"{f' chunksize {int(chunk)}' if chunk else ''}\n"
                f"pair_coeff * * {model_path} {els}\n")
    if style == "grace-2l":                 # GRACE-2L .npz (export_kokkos): the same pair, two layers; the same
        prec, chunk = os.environ.get("GRACE_PRECISION", ""), os.environ.get("GRACE_CHUNKSIZE")   # overrides as 1L's
        return (f"pair_style {('grace/2l/kk' if device == 'gpu' else 'grace/2l/cpu/kk') + prec}"
                f"{f' chunksize {int(chunk)}' if chunk else ''}\n"
                f"pair_coeff * * {model_path} {els}\n")
    if style == "grace-fs":                 # GRACE-FS C++ YAML: Kokkos on the GPU, the plain MPI evaluator on the CPU
        return f"pair_style {'grace/fs/kk' if device == 'gpu' else 'grace/fs'}\npair_coeff * * {model_path} {els}\n"
    if style == "nep":                      # NEP_CPU's USER-NEP (float64, MPI): pair_style takes no arguments
        return f"pair_style nep\npair_coeff * * {model_path} {els}\n"
    if style == TRIM:                       # PR 309 plugin, then the juliac --trim library
        return f"plugin load ${{aceplugin}}\npair_style ace\npair_coeff * * {model_path} {els}\n"
    if style == "gap":                       # QUIP takes atomic numbers, and the xml label picks the Potential
        if not extra:                        # "xml_label=None" would load nothing QUIP can name
            raise ValueError("pair quip needs the GAP xml_label (row['quip_label'])")
        from ase.data import atomic_numbers
        zs = " ".join(str(atomic_numbers[e]) for e in elements)
        return f'pair_style quip\npair_coeff * * {model_path} "Potential xml_label={extra}" {zs}\n'
    raise ValueError(style)


TRIM = "acepotentials-trim"


def xl_pair_args(model_json, device, style):
    """(pair_style fragment naming the R1 JIT artifact, None) for a Symmetrix-XL model on this device at the
    style's precision, else (None, why): sulis-xl.sh's model-xl.jit-{cpu,cuda}.json next to model-xl.json."""
    m = pathlib.Path(model_json)
    side = m.with_name(m.stem + (".jit-cuda.json" if device == "gpu" else ".jit-cpu.json"))
    if not m.is_file() or not side.exists():
        return None, f"no Symmetrix-XL model/R1 artifact: {side if m.is_file() else m}"
    e = json.loads(side.read_text()).get(XL[style][1], {})
    if not e.get("args"):
        return None, f"R1 JIT refused: {e.get('error', 'no entry')}"[:1500]
    t = e["args"].split()        # R1 only: M0/R0 operator modules are capacity-profile accelerators (`profile speed`)
    return " ".join(f"{k} {v}" for k, v in zip(t[::2], t[1::2]) if k in XL_R1_KEYS), None


XL_R1_KEYS = ("jit_host_artifact", "jit_device_artifact", "jit_device_blocks_per_compute_unit")


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
                 warmup=50, pair_extra=None, procs=None):
    txt = ("units metal\natom_style atomic\nboundary p p p\natom_modify map yes\n"
           + (f"processors {' '.join(map(str, procs))}\n" if procs else "") +
           f"read_data {data_path}\n" + _pair(style, model_path, elements, device, pair_extra) +
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
    builds = re.findall(r"Neighbor list builds = (\d+)", text)   # one per run: the last is the timed one
    return {"loop_s": float(loop_s), "step_s": float(loop_s) / int(steps), "procs": int(procs),
            "n_atoms_end": int(natoms), "nan": bool(re.search(r"\bnan\b", text, re.I)),
            **({"neighbor_builds": int(builds[-1])} if builds else {})}


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


def capacity(at, rcut, skin=1.0, tight=False, list_headroom=0.5, procs=None):
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
    k(rcut + skin); 0.5 from one observed overflow (neighbour_capacity).

    procs: one MPI rank's capacities on that `processors` grid (a CPU run)."""
    from ace_jax.export.lammps import neighbour_capacity
    c = neighbour_capacity(at, rcut, skin=skin, slots="cutoff" if tight else "skin", margin=8,
                           list_headroom=list_headroom, **({"procs": procs} if procs else {}))
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
    procs = row.get("procs")                         # a CPU run: one rank's subdomain, lowered for XLA:CPU
    cap = capacity(at, float(meta["rcut"]), tight=slots == "cutoff", list_headroom=list_headroom, procs=procs)
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    b = export_lammps(model, meta, pathlib.Path(workdir) / "bundle.json", max_atoms=cap["max_atoms"],
                      max_edges=cap["max_edges"], k_dense=cap["k_dense"], dtype=dtype, layout=layout,
                      type_elements=[atomic_numbers[e] for e in row["elements"]],  # data-file order
                      max_owned=cap["max_owned"], max_neighbors=cap["max_neighbors"],
                      **({"spline_tol": row["spline_tol"]} if "spline_tol" in row else {}),
                      **({"platform": "cpu", "device_memory": row["device_memory"]} if procs else {}),
                      # opt-in recursive product basis (ace_jax.eval.aa_dag; "greedy" | "prefix"), an ace-jax with it;
                      # a row's own (accuracy_cost's lammps-cueq line: "cueq") takes precedence
                      **({"aa_dag": row.get("aa_dag") or os.environ["ACEJAX_EXPORT_AA_DAG"]}
                         if row.get("aa_dag") or os.environ.get("ACEJAX_EXPORT_AA_DAG") else {}),
                      # opt-in C++ FFI kernel (ace_jax.eval.ffi; CPU matrix bundles, radial table on): LAMMPS needs
                      # LAMMPS_JAX_FFI_HANDLERS naming its library, set with this in the env json's os_env
                      **({"ffi": True} if os.environ.get("ACEJAX_EXPORT_FFI") else {}))
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
    keep = {k: row[k] for k in ("name", "system", "path", "elements", "spline_tol", "procs", "device_memory",
                                "aa_dag") if k in row}
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--export", json.dumps(keep),
           str(len(at)), dtype, str(workdir), layout, slots, str(list_headroom)]
    bench = str(pathlib.Path(__file__).resolve().parents[1])     # so the child imports scaling
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [bench] + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x]),
           # the child is in the timed process tree, so its 75% pool was the GPU peak (30.04 GiB for every bundle)
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"bundle export failed ({p.returncode}): {(p.stderr or p.stdout)[-2000:]}")
    out = json.loads(lines[-1])
    if info is not None:
        info["splined"] = out.get("splined")
    return out["bundle"], out["layout"], out["compile_s"]




def bundle_env(bundle, base=None):
    """The environment LAMMPS runs a bundle in: `base` (os.environ) plus, for a cueq bundle (its `ace_jax.cueq`
    record), the replay sidecar, NVRTC cache, FFI handler and libraries that ace_jax.eval.cueq.run_env names
    (LD_LIBRARY_PATH prepended); None for any other bundle (LAMMPS inherits the environment)."""
    try:
        rec = json.loads(pathlib.Path(bundle).read_text()).get("ace_jax", {}).get("cueq")
    except (OSError, ValueError):
        return None
    if not rec:
        return None
    from ace_jax.eval.cueq import run_env
    base = dict(os.environ if base is None else base)
    e = run_env(rec, bundle)
    libs = [x for x in (e.pop("LD_LIBRARY_PATH"), base.get("LD_LIBRARY_PATH")) if x]
    return {**base, **e, "LD_LIBRARY_PATH": ":".join(libs)}


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
    entry = _trim_record(lib)
    if entry.get("unsupported"):
        return "unsupported", f"trim export failed for {lib}: {entry['unsupported']}"
    return "error", f"no trim library {lib}: run `python bench/scaling/models.py acepotentials-trim`"


def _trim_record(lib):
    """The build record of a trim library: its `<lib>.build.json` sidecar (accuracy_cost/build_trim.py),
    else its entry in models/manifest.json (models.py acepotentials-trim), else {}."""
    side = pathlib.Path(lib).with_suffix(".build.json")
    try:
        if side.exists():
            return json.loads(side.read_text())
        from scaling import models
        return models.load_manifest().get(lib, {})
    except (OSError, ValueError):
        return {}


def _trim_build(row):
    """The build record of the row's trim library (provenance for the row)."""
    e = _trim_record(row["trim_lib"])
    return {k: e[k] for k in ("build_id", "sha256", "build_cpu", "cpu_target", "versions", "identity")
            if k in e} or None


def run_case(row, n_atoms, dtype, device, lmp, ranks, workdir, pjrt=None, prev=None,
             slots="skin", list_headroom=0.5, aceplugin=None, threads=1):
    """One LAMMPS case.  CPU: `ranks` MPI ranks of `threads` cores each on the
    proc_grid(ranks) `processors` grid, every code alike; ace-jax runs lammps-jax's
    host backend with a bundle lowered for XLA:CPU and sized for one rank's
    subdomain, its dense budget a rank's share of the node's RAM."""
    work = pathlib.Path(workdir); work.mkdir(parents=True, exist_ok=True)
    procs = proc_grid(ranks) if device == "cpu" and ranks > 1 else None
    if device == "cpu" and row["code"].startswith("acejax"):
        procs = procs or (1, 1, 1)
        row = {**row, "procs": list(procs),
               "device_memory": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // ranks}
    kw = {"threads": threads, "procs": procs}
    at = supercell(row["system"], n_atoms, rattle=row.get("rattle", 0.0))
    data = work / "x.data"
    write(data, at, format="lammps-data", specorder=row["elements"], masses=True)
    style = ("acejax" if row["code"].startswith("acejax")      # mlpace, mace, mace-xl
             else "mlpace" if row["code"] == "mlpace-mixed" else row["code"])   # the same pair style, another LAMMPS
    if style in XL:
        args, why = xl_pair_args(row.get("symmetrix_xl") or "", device, style)
        if why:
            return {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
                    "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
                    "status": "unsupported", "error": why}
        row = {**row, "pair_extra": args}
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
                           aceplugin=aceplugin, **kw)
    if style != "acejax":
        model = row.get("symmetrix") if style == "mace" else row.get("symmetrix_xl") if style in XL else row["path"]
        return _run_lammps(row, style, model, data, work, n_atoms, dtype, device, lmp, ranks,
                           pjrt, steps, warmup, {}, **kw)
    if row.get("bundle"):                       # a prebuilt bundle (accuracy_cost's cueq line): run it as is
        aj = json.loads(pathlib.Path(row["bundle"]).read_text())["ace_jax"]
        extra = {"layout": aj["layout"], "spline_tol": row.get("spline_tol", "auto"), "splined": splined_meta(aj),
                 "bundle": row["bundle"], "block_rows": aj.get("block_rows")}
        return _run_lammps(row, style, row["bundle"], data, work, n_atoms, dtype, device, lmp, ranks, pjrt,
                           steps, warmup, extra, **kw)
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
                          steps, warmup, extra, **kw)
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


def proc_grid(ranks):
    """LAMMPS `processors` grid for `ranks` on a cubic cell: the most nearly equal
    three factors, larger first.  Every CPU LAMMPS code runs the same grid, and an
    ace-jax bundle is sized for one rank's subdomain of it
    (neighbour_capacity(procs=))."""
    best = None
    for a in range(1, ranks + 1):
        for b in range(1, ranks // a + 1):
            if ranks % (a * b) == 0:
                g = tuple(sorted((a, b, ranks // (a * b)), reverse=True))
                if best is None or g[0] / g[2] < best[0] / best[2]:
                    best = g
    return best


def cpu_mpirun(ranks, threads, style):
    """mpirun for a CPU run.  One thread per rank: the host's mpirun_args (sweep.HOSTS;
    lestrade binds one rank per P-core), else one core per rank for every style.
    Unbound ranks are wrong twice over: XLA sizes its thread pool from the affinity
    mask (every rank starts a pool for the whole node), and under OMP_PROC_BIND the
    OpenMP runtime in a Kokkos-OpenMP LAMMPS pins each rank's thread to place 0 of
    the full mask, the same hardware thread for all of them (ML-PACE at 128 ranks on
    Sulis: 1268 s for 50 steps, against 0.07 s bound).  threads > 1: each rank bound
    to that many cores of its own, which Kokkos OpenMP and XLA's pool then use."""
    if threads > 1:
        bind = ["--bind-to", "core", "--map-by", f"slot:PE={threads}"]
    elif mpirun_args():
        bind = mpirun_args()
    else:
        bind = ["--bind-to", "core", "--map-by", "core"]
    return ["mpirun", "-np", str(ranks), *bind]


def mpirun_args():
    """The host's extra mpirun arguments (sweep.HOSTS `mpirun_args`, handed over
    as BENCH_MPIRUN_ARGS, a JSON list: lestrade binds one rank per P-core)."""
    return json.loads(os.environ.get("BENCH_MPIRUN_ARGS") or "[]")


def kokkos_args(style, device, threads=1):
    """The KOKKOS command-line switches.  GPU: every style but gap (pair quip
    has no /kk variant, so it is CPU only).  CPU: host Kokkos (OpenMP) for
    lammps-jax, whose pair style is jax/kk only, and for a KOKKOS style run with
    threads > 1 (ML-PACE's pace/kk); plain MPI otherwise."""
    if device != "gpu":
        if style in ("acejax", "grace", "grace-2l") or style in XL or (threads > 1 and style in KOKKOS):   # grace: /kk only
            return ["-k", "on", "t", str(threads), "-sf", "kk", "-pk", "kokkos", *KOKKOS[style].split()]
        return []
    if style not in KOKKOS:
        raise ValueError(f"pair style {style!r} is CPU only: no KOKKOS settings")
    return ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", *KOKKOS[style].split()]


def _run_lammps(row, style, model, data, work, n_atoms, dtype, device, lmp, ranks, pjrt, steps,
                warmup, extra, aceplugin=None, threads=1, procs=None):
    kk = kokkos_args(style, device, threads)        # before any file is written
    (work / "in.bench").write_text(lammps_input(style, model, row["elements"], data, device, steps,
                                                warmup=warmup, pair_extra=row.get("pair_extra") or row.get("quip_label"),
                                                procs=procs))
    cmd = [lmp, "-in", "in.bench", "-log", "log.lammps", "-nocite", *kk]
    env = None
    if device == "cpu" and (ranks > 1 or threads > 1 or style == "acejax"):
        cmd = cpu_mpirun(ranks, threads, style) + cmd
        env = {**os.environ, "OMP_NUM_THREADS": str(threads), "OMP_PROC_BIND": "spread", "OMP_PLACES": "threads"}
    cmd += lammps_vars(style, pjrt, aceplugin)
    out = {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
           "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
           "ranks": ranks, "status": "ok", **extra}
    if style in XL:
        out["xl_pair_style"] = XL[style][0] + " " + row["pair_extra"]   # names the artifact (content-addressed)
    if device == "cpu":
        out.update(threads=threads, proc_grid=list(procs) if procs else None, kokkos=bool(kk))   # parse_log owns "procs"
    if style == "acejax" and os.environ.get("ACEJAX_EXPORT_AA_DAG"):
        out["aa_dag"] = os.environ["ACEJAX_EXPORT_AA_DAG"]
    cueq = bundle_env(model, env) if style == "acejax" else None
    if cueq:                                     # its replay sidecar, cache and libraries (not the env-var hooks)
        env, out["aa_dag"] = cueq, "cueq"
    if style == "acejax" and os.environ.get("ACEJAX_EXPORT_FFI"):
        out["ffi"] = True
    if cmd[0] == "mpirun" and mpirun_args():
        out["mpirun_args"] = mpirun_args()
    (work / "log.lammps").unlink(missing_ok=True)
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=3600, env=env)
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
