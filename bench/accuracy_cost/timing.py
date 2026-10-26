"""Time every fitted model on lestrade with bench/scaling's runners: each code
in its best mode per device, one size plus an N/8 plateau check, 3 repeats.
Waits for an idle box before every case and gate, up to 30 min, then
stops (--force-busy skips the wait; rows record it and their own load sample); gates each
LAMMPS model on LAMMPS-vs-own-evaluator parity first.

    python -m accuracy_cost.timing {gpu,cpu} [--only fam[/size]] [--n MAIN CHECK]
        [--codes C ...] [--modes M ...] [--repeats K] [--results PATH] [--work DIR] [--force-busy] [--redo]

Run from bench/ (or with it on PYTHONPATH).  Each case and each gate runs in a
fresh child process (`--child`, the env JSON's `python`) so no JAX/torch state
or GPU pool outlives it; the parent imports neither.  Rows go to
results/lestrade-<device>.jsonl, one per (code, family, size, dtype, mode);
gate rows are mode "parity"."""
import json
import os
import pathlib
import socket
import statistics
import subprocess
import sys

from . import common
from .common import RESULTS
from .build_trim import trim_lib

sys.path.insert(0, str(common.HERE.parent))
N = {"gpu": (32768, 4096), "cpu": (4096, 512)}
RATTLE, REPEATS = 0.05, 3
# CPU LAMMPS layout, every code alike: MPI ranks x cores per rank (Kokkos OpenMP threads; ace-jax's XLA
# pool).  Rows record both; give each layout its own --results file (KEY has neither)
CPU_RANKS = int(os.environ.get("ACEJAX_AC_CPU_RANKS", 32))
CPU_THREADS = int(os.environ.get("ACEJAX_AC_CPU_THREADS", 1))
RSS_CAP_GB = float(os.environ.get("ACEJAX_AC_RSS_CAP_GB", 48))   # lestrade: 62 GB; the case's process tree pays, not the box
                                                  # (Sulis GPU nodes: 512 GB; ML-PACE's f3200 gate passes 52 GB)
KEY = ("code", "family", "size", "dtype", "mode", "device")
# family -> [(code, {device: mode}, dtypes)]: the spec's code table.  ace_lr is
# timed as deployed (ACECalculator / export_lammps default: lean, spline_tol="auto").
# gap18_published is a reference line (metrics.jsonl), not a swept family.
# ACEJAX_AC_ACE_DTYPES (e.g. "float64,float32"): the ace family's ace-jax precisions; float32 bundles are
# exported as true float32 programs (export_lammps casts the traced model; docs/dev/fp32-results.md)
ACE_DTYPES = tuple(os.environ.get("ACEJAX_AC_ACE_DTYPES", "float64").split(","))
# JAX-MD (scaling/run_jaxmd.py): the whole MD step in one jit on the device, against the same
# models' LAMMPS rows; opt-in (--modes jaxmd jaxmd-tight lammps-tight), GPU only, in the ace family's precisions.
# A further spec of the same code: rows are keyed on mode
# JAX-MD's cell list; "-tight": model slots for rc pairs ("cutoff" slots)
JAXMD_MODES = ("jaxmd", "jaxmd-tight")
LAMMPS_TIGHT = "lammps-tight"   # lammps-jax with run_lammps' "cutoff" slots (opt-in, as the JAX-MD modes)
# (The ace-jax cueq line, lammps-cueq, is gone with aa_dag="cueq": its rows stay in the results as data.  The CUDA
# kernel that replaced it is the "lammps" mode with ACEJAX_EXPORT_FFI set, as the CPU kernel's line.)
OPT_IN_MODES = (*JAXMD_MODES, LAMMPS_TIGHT)
JAXMD = [("acejax-ace", {"gpu": m}, ACE_DTYPES) for m in (*JAXMD_MODES, LAMMPS_TIGHT)]
PLAN = {"ace": [("acejax-ace", {"cpu": "standalone", "gpu": "lammps"}, ACE_DTYPES), *JAXMD,
                # ACEpotentials.jl's PR 309 LAMMPS export of the same fit (accuracy_cost/build_trim.py: the
                # juliac --trim library, pair_style ace; CPU only); only where the env json names `lmp_ace`
                ("acepotentials-trim", {"cpu": "lammps"}, ("float64",))],
        "ace_lr": [("acejax-ace", {"cpu": "standalone", "gpu": "lammps"}, ("float64",)),
                   *[("acejax-ace", {"gpu": m}, ("float64",)) for m in (*JAXMD_MODES, LAMMPS_TIGHT)]],
        # PACE-shaped ACE (pacemaker's per-order caps, SBessel radials; feat/sbessel-radials + basis caps):
        # timed as deployed, like ace / ace_lr (lean; an SBessel radial is never splined, learned or not)
        "ace_wl": [("acejax-ace", {"cpu": "standalone", "gpu": "lammps"}, ("float64",))],   # wl_ladder/PLAN.md
        "ace_pshape": [("acejax-ace", {"cpu": "standalone", "gpu": "lammps"}, ("float64",))],
        "ace_pshape_lr": [("acejax-ace", {"cpu": "standalone", "gpu": "lammps"}, ("float64",))],
        "ace_gp": [("acejax-gp", {"cpu": "standalone", "gpu": "standalone"}, ("float64",))],
        "pace": [("acejax-pace", {"cpu": "standalone", "gpu": "lammps"}, ("float64",)),
                 ("mlpace", {"cpu": "lammps", "gpu": "lammps"}, ("float64",))],
        # linear ACE converted to an upstream C-tilde .yace (ace_to_yace.py: radials refitted to SBessel):
        # the ACE model run by ML-PACE; ace-jax on the same file on the GPU only (CPU: too slow for devel)
        "ace_yace": [("acejax-pace", {"gpu": "lammps"}, ("float64",)),
                     ("mlpace", {"cpu": "lammps", "gpu": "lammps"}, ("float64",))],
        # the same models exactly: `Tabulated` radials (write_yace; ace_to_yace.py's default), which need ML-PACE with the
        # tabulated-radial patch (env json `lmp`: envs/sulis-pacetab.sh); gated against the source .npz
        "ace_tab": [("mlpace", {"cpu": "lammps", "gpu": "lammps"}, ("float64",)),
                    # the same in a KOKKOS_PREC=mixed LAMMPS (env json `lmp_mixed`; main drops it otherwise)
                    ("mlpace-mixed", {"gpu": "lammps"}, ("mixed",))],
        "mace": [("mace", {"cpu": "lammps", "gpu": "lammps"}, ("float64",)),
                 ("mace", {"cpu": "standalone", "gpu": "standalone"}, ("float32",)),
                 # Symmetrix-XL (pair_style symmetrix/mace[/float32]/kk + JIT R1 artifact; envs/sulis-xl.sh):
                 # only where the env json names its LAMMPS (`lmp_xl`, main drops it otherwise)
                 ("mace-xl", {"cpu": "lammps", "gpu": "lammps"}, ("float64",)),
                 ("mace-xl-f32", {"cpu": "lammps", "gpu": "lammps"}, ("float32",))],
        # NEP: pair_style nep (NEP_CPU, float64, MPI) on the CPU; on the GPU GPUMD's own MD code, whose NEP is
        # mixed precision (dtype "mixed": float descriptors/network, double accumulation; scaling/run_gpumd.py)
        "nep": [("nep", {"cpu": "lammps"}, ("float64",)),
                ("nep-gpumd", {"gpu": "gpumd"}, ("mixed",))],
        # GRACE (fit_grace.py; LAMMPS from the yury-lysogorskiy/lammps `grace` branch, scaling/envs/sulis-grace.sh):
        # GRACE-1L as pair_style grace/1l/kk (GPU) / grace/1l/cpu/kk (CPU), float64; GRACE-FS as grace/fs/kk / grace/fs.
        # Seed 0 of each size is timed (the seeds share the architecture); grace_abl is accuracy only
        "grace": [("grace", {"cpu": "lammps", "gpu": "lammps"}, ("float64",))],
        "grace_fs": [("grace-fs", {"cpu": "lammps", "gpu": "lammps"}, ("float64",))],
        "grace_2l": [("grace-2l", {"cpu": "lammps", "gpu": "lammps"}, ("float64",))],   # grace/2l/kk, grace/2l/cpu/kk
        "gap": [("gap", {"cpu": "lammps"}, ("float64",))],         # pair quip: no KOKKOS, CPU only
        "gap_noglue": [("gap", {"cpu": "lammps"}, ("float64",))],  # the XML has no Glue term; same pair style
        "gap_E30": [("gap", {"cpu": "lammps"}, ("float64",))]}     # GAP-18's command, energy sigmas x10/3
STYLE = {"acejax-ace": "acejax", "acejax-pace": "acejax", "mlpace": "mlpace", "mlpace-mixed": "mlpace",
         "mace": "mace", "gap": "gap", "nep": "nep", "nep-gpumd": "nep-gpumd",
         "mace-xl": "mace-xl", "mace-xl-f32": "mace-xl-f32", "acepotentials-trim": "acepotentials-trim",
         "grace": "grace", "grace-fs": "grace-fs", "grace-2l": "grace-2l"}
GRACE_CODES = ("grace", "grace-fs", "grace-2l")
TRIM = "acepotentials-trim"
XL_CODES = ("mace-xl", "mace-xl-f32")
GATE_N, GATE_CELLS = 64, 3                        # three rattled 64-atom Si cells


def _load1():
    return float(pathlib.Path("/proc/loadavg").read_text().split()[0])


def _nvidia_pids():
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in q.split() if x.strip().isdigit()]


def _own_pids():
    """This driver and every process descended from it (its cases, LAMMPS, MPI ranks)."""
    import psutil
    me = psutil.Process()
    try:
        return {me.pid} | {q.pid for q in me.children(recursive=True)}
    except psutil.Error:
        return {me.pid}


def _gpu_procs():
    """GPU compute processes that are not this timing run's own."""
    pids = _nvidia_pids()
    own = _own_pids() if pids else set()
    return len([p for p in pids if p not in own])


def idle(load_max=None):
    """load_max: lestrade's 2 by default; $ACEJAX_AC_MAX_LOAD1 for other hosts (an exclusive
    128-core Sulis node idles at load1 ~6 from its system services)."""
    load_max = float(os.environ.get("ACEJAX_AC_MAX_LOAD1", 2.0)) if load_max is None else load_max
    load1, procs = _load1(), _gpu_procs()
    return {"ok": load1 <= load_max and not procs, "load1": load1, "gpu_procs": procs}


POLL_S, MAX_WAIT_S = 60, 1800


def wait_idle(sample=None, clock=None, sleep=None, poll=None, max_wait=None, force=False):
    """{"load1", "gpu_procs"[, "waited_s"]} from an idle sample.  Forced: one sample,
    no waiting (the caller records forced_busy).  Otherwise poll every `poll` s
    until the box is idle; past `max_wait` s, stop the run naming the load.
    (Note load1 is a 1-minute average: after a 32-rank case it takes ~3 min to
    fall below 2, which this wait absorbs as a cool-down.)"""
    import time
    sample, clock, sleep = sample or idle, clock or time.monotonic, sleep or time.sleep
    poll = POLL_S if poll is None else poll
    max_wait = MAX_WAIT_S if max_wait is None else max_wait
    s = sample()
    if force:
        return {"load1": s["load1"], "gpu_procs": s["gpu_procs"]}
    t0 = clock()
    while not s["ok"]:
        if clock() - t0 >= max_wait:
            raise SystemExit(f"{socket.gethostname()} busy for {max_wait:.0f} s: load1 {s['load1']}, gpu_procs {s['gpu_procs']} "
                             "(other GPU processes, or load above 2); stopped -- re-run to resume "
                             "(--force-busy for a smoke run only)")
        sleep(poll)
        s = sample()
    return {"load1": s["load1"], "gpu_procs": s["gpu_procs"], "waited_s": float(clock() - t0)}


def plateau_flag(main, check, tol=0.2):
    """True when per-atom throughput at the check size (N/8) differs from the
    main size by more than tol: N is not yet on the throughput plateau."""
    per = lambda r: r["n_atoms"] / r["step_s"]
    return bool(abs(per(main) - per(check)) / per(main) > tol)


def _sizes(fam, e):
    if fam in ("grace", "grace_fs", "grace_2l"):
        return [f"{e['size']}_s0"] if e.get("timing", True) else []   # c32l4_long: c32l4's architecture                     # fit_grace's run name: <size>_s<seed>
    return [f"f{n}" for n in e["take"]] if fam == "pace" else [e["size"]]


def _elements(fam, e):
    """The species of a sweep family's models: the benchmark is silicon throughout."""
    return ["Si"]


def _local(path):
    """A status.json model path as written on the desktop, re-rooted under this host's RUNS
    when the runs were copied elsewhere (Sulis: $ACEJAX_AC_ROOT)."""
    p = pathlib.Path(path)
    if p.exists() or "accuracy_cost" not in p.parts:
        return str(p)
    return str(common.RUNS.joinpath(*p.parts[p.parts.index("accuracy_cost") + 1:]))


def cases(device, only=None):
    """One case per (family, size, code, mode, dtype) with an ok status.json;
    `only` is "family" or "family/size"."""
    from scaling.sweep import lammps_supported
    for code, modes, _ in (s for specs in PLAN.values() for s in specs):   # before any status is read
        if modes.get(device) == "lammps" and not lammps_supported(code, device):
            raise ValueError(f"{code}: LAMMPS on {device} unsupported (gap is CPU only)")
    want_f, _, want_s = (only or "").partition("/")
    sweep, out = common.load_sweep(), []
    for fam, specs in PLAN.items():
        if want_f and fam != want_f:
            continue
        for e in sweep[fam]:
            for size in _sizes(fam, e):
                if want_s and size != want_s:
                    continue
                d = common.run_dir(fam, size)
                st = common.read_status(d)
                if not st or st.get("status") != "ok":
                    continue
                for code, modes, dtypes in specs:
                    if device not in modes:
                        continue
                    mode = modes[device]
                    if device == "cpu" and code.startswith("acejax") and os.environ.get("BENCH_LAMMPS_JAX_CPU"):
                        mode = "lammps"                       # lammps-jax's host backend (sweep.lammps_supported)
                    for dt in dtypes:
                        path = _local(st["model"])
                        if code == "mace":
                            path = str(d / ("model-f32.model" if dt == "float32" else
                                            "model-cpu.model" if device == "cpu" and (d / "model-cpu.model").exists()
                                            else "model.model"))      # as XL's: model.model holds CUDA tensors (L1)
                        if code in XL_CODES:                  # the gate's PyTorch reference: float64, and on a
                            path = str(d / ("model-cpu.model" if device == "cpu" and (d / "model-cpu.model").exists()
                                            else "model.model"))   # GPU-less node the CPU copy (fit_mace.cpu_copy)
                        row = {"code": code, "family": fam, "system": "Si", "size": size, "elements": _elements(fam, e),
                               "path": path, "name": f"{fam}/{size}/{code}", "rattle": RATTLE,
                               "symmetrix": str(d / "model.json"), "symmetrix_xl": str(d / "model-xl.json"),
                               "quip_label": st.get("quip_label"),
                               "ref": _local(st["source"]) if st.get("source") else None,   # gate reference
                               "n_params": st.get("n_params")}
                        if code == TRIM:
                            row["trim_lib"] = str(trim_lib(d))
                        if code in GRACE_CODES:                # LAMMPS reads the export; the gate's reference is the SavedModel
                            row["path"] = _local(st.get("fs_yaml" if code == "grace-fs" else "kokkos_npz") or "")
                            row["ref"] = path
                        if fam in ("ace", "ace_wl") and os.environ.get("ACEJAX_AC_ACE_SPLINE_TOL"):   # analytic radials splined
                            row["spline_tol"] = float(os.environ["ACEJAX_AC_ACE_SPLINE_TOL"])  # like ace_lr's (opt-in)
                        out.append({"row": row, "mode": mode, "dtype": dt})
    return out


# ---- child side: one gate or one timed case, in a fresh process -------------

def gate_cells():
    from scaling.structures import supercell
    return [supercell("Si", GATE_N, seed=k, rattle=RATTLE) for k in range(GATE_CELLS)]


def gate_model(row, device, env, work):
    """LAMMPS E/F of this model against its own evaluator on three rattled
    64-atom Si cells (parity.compare at parity.TOL[style]): acejax and mlpace
    against ace-jax (ACECalculator; PACEModel for a .yace -- pyace lives only in
    the pacemaker venv), mace against MACE in PyTorch (float64), gap against
    quippy with the same xml_label.  ace-jax gates every bundle layout on the
    first cell (run_lammps falls back from matrix/dense to dense/sparse) and the
    auto layout on the others."""
    from scaling import parity
    from scaling.run_lammps import export_bundle
    from scaling.sweep import lammps_supported
    code, style, work = row["code"], STYLE[row["code"]], pathlib.Path(work)
    out = {"gate": style, "n_atoms": GATE_N, "cells": GATE_CELLS}
    if style == "nep-gpumd":
        return gate_gpumd(row, env, work, out)
    if not lammps_supported(code, device):
        return {**out, "status": "unsupported", "error": f"no LAMMPS {code} on {device}"}
    if style == "mace" and not pathlib.Path(row["symmetrix"]).exists():
        return {**out, "status": "unsupported", "error": f"no Symmetrix model {row['symmetrix']}"}
    if style in XL_CODES:
        from scaling.run_lammps import xl_pair_args
        xl_args, why = xl_pair_args(row["symmetrix_xl"], device, style)
        if why:
            return {**out, "status": "unsupported", "error": why}
    lmp, checks = env["lmp"], []
    for k, at in enumerate(gate_cells()):
        layouts = (parity.BUNDLE_LAYOUTS if k == 0 else ("auto",)) if style == "acejax" else (None,)
        if style == "acejax" and os.environ.get("ACEJAX_EXPORT_FFI"):
            layouts = ("matrix",)                              # the FFI kernel: the matrix layout only
        for lay in layouts:
            w = work / f"cell{k}" / (lay or style)
            chk = {"cell": k, **({"layout": lay} if lay else {})}
            try:
                if style == "acejax":
                    E0, F0 = parity._acejax_ef(row["path"], at, **parity._tol_kw(row))   # the model the bundle exports
                    cpu = {"procs": [1, 1, 1], "device_memory": 4 * 2**30} if device == "cpu" else {}
                    if row.get("bundle"):
                        bundle = row["bundle"]
                        used = json.loads(pathlib.Path(bundle).read_text())["ace_jax"]["layout"]
                    else:
                        bundle, used, _ = export_bundle({**row, **cpu}, at, "float64", w, layout=lay)
                    chk["layout_used"] = used
                    lmp_aj, pjrt = lammps_for(code, device, env)
                    E1, F1 = parity._lammps_ef("acejax", bundle, row["elements"], at, device, lmp_aj, w, pjrt)
                elif style == "mlpace":                                # a converted model: its source
                    E0, F0 = parity._acejax_ef(row.get("ref") or row["path"], at)
                    E1, F1 = parity._lammps_ef("mlpace", row["path"], row["elements"], at, device,
                                               lammps_for(code, device, env)[0], w)
                elif style == TRIM:                                  # vs the fit itself, unsplined
                    E0, F0 = parity._acejax_ef(row["path"], at)
                    E1, F1 = parity._lammps_ef(TRIM, row["trim_lib"], row["elements"], at, device,
                                               env["lmp_ace"], w, aceplugin=env["ace_plugin"])
                elif style == "mace":
                    E0, F0 = parity._mace_ef(row["path"], at, device)
                    E1, F1 = parity._lammps_ef("mace", row["symmetrix"], row["elements"], at, device, lmp, w)
                elif style in XL_CODES:                                    # the same reference as mace
                    E0, F0 = parity._mace_ef(row["path"], at, device)
                    E1, F1 = parity._lammps_ef(style, row["symmetrix_xl"], row["elements"], at, device,
                                               env["lmp_xl"], w, pair_extra=xl_args)
                elif style in GRACE_CODES:                             # TPCalculator in the grace venv (env `grace_python`)
                    E0, F0 = parity._grace_ef(row["ref"], at, env["grace_python"], w)
                    E1, F1 = parity._lammps_ef(style, row["path"], row["elements"], at, device, lmp, w)
                elif style == "nep":                                   # NEP_CPU both sides, unshifted energies
                    E0, F0 = parity._nep_ef(row["path"], at)
                    E1, F1 = parity._lammps_ef("nep", row["path"], row["elements"], at, device, lmp, w)
                else:                                              # gap: the label picks the Potential
                    E0, F0 = parity._quip_ef(row["path"], at, row["quip_label"])
                    E1, F1 = parity._lammps_ef("gap", row["path"], row["elements"], at, device, lmp, w,
                                               pair_extra=row["quip_label"])
                tol = parity.TOL["acejax-splined" if style == "acejax" and "spline_tol" in row
                                 else code if code in parity.TOL else style]
                chk.update(parity.compare(E0, F0, E1, F1, len(at), tol), tol=list(tol))
            except Exception as ex:                                # a missing style is a failed gate
                chk.update({"status": "error", "error": parity.error_summary(ex)})
            checks.append(chk)
    st = {c["status"] for c in checks}
    status = "parity_ok" if st == {"parity_ok"} else "parity_fail" if "parity_fail" in st else "error"
    dE = [c["dE_per_atom"] for c in checks if "dE_per_atom" in c]
    dF = [c["max_dF"] for c in checks if "max_dF" in c]
    return {**out, "status": status, "checks": checks, "max_dE_per_atom": max(dE, default=None),
            "max_dF": max(dF, default=None)}


def gate_gpumd(row, env, work, out):
    """GPUMD (mixed precision) against calorine's CPUNEP (float64) on the gate cells, at parity.TOL["nep-gpumd"]."""
    from scaling import parity
    from scaling.run_gpumd import gpumd_ef
    checks = []
    for k, at in enumerate(gate_cells()):
        chk = {"cell": k}
        try:
            E0, F0 = parity._nep_ef(row["path"], at)
            E1, F1 = gpumd_ef(row["path"], at, env["gpumd"], work / f"cell{k}")
            chk.update(parity.compare(E0, F0, E1, F1, len(at), parity.TOL["nep-gpumd"]))
        except Exception as ex:
            chk.update({"status": "error", "error": parity.error_summary(ex)})
        checks.append(chk)
    st = {c["status"] for c in checks}
    status = "parity_ok" if st == {"parity_ok"} else "parity_fail" if "parity_fail" in st else "error"
    return {**out, "status": status, "checks": checks,
            "max_dE_per_atom": max((c["dE_per_atom"] for c in checks if "dE_per_atom" in c), default=None),
            "max_dF": max((c["max_dF"] for c in checks if "max_dF" in c), default=None)}


def time_case(c, device, n, env, work, prev=None):
    """One timed run: run_standalone.run_case, run_gpumd.run_case or run_lammps.run_case."""
    from scaling import run_gpumd, run_lammps, run_standalone
    row = c["row"]
    if c["mode"] in JAXMD_MODES:
        from scaling import run_jaxmd
        return run_jaxmd.run_case(row, n, c["dtype"], device, prev=prev, mode=c["mode"])
    if c["mode"] == "standalone":
        return run_standalone.run_case(row, n, c["dtype"], device)
    slots = "cutoff" if c["mode"] == LAMMPS_TIGHT else "skin"
    if c["mode"] == "gpumd":
        return run_gpumd.run_case(row, n, device, env["gpumd"], work, prev)
    lmp, pjrt = lammps_for(row["code"], device, env)
    ranks, threads = (CPU_RANKS, CPU_THREADS) if device == "cpu" else (1, 1)
    return run_lammps.run_case(row, n, c["dtype"], device, lmp, ranks, work, pjrt=pjrt, prev=prev,
                               threads=threads, slots=slots, aceplugin=env.get("ace_plugin"))


def lammps_for(code, device, env):
    """(LAMMPS wrapper, PJRT plugin) for a code: ace-jax on the CPU runs the host-Kokkos
    lammps-jax build with XLA's CPU plugin (env `lmp_jax_cpu`, `pjrt_cpu`)."""
    if code in XL_CODES:
        return env["lmp_xl"], None
    if code == TRIM:
        return env["lmp_ace"], None
    if code == "mlpace-mixed":
        return env["lmp_mixed"], None
    if not code.startswith("acejax"):
        return env["lmp"], env.get("pjrt")
    if device == "cpu":
        return env["lmp_jax_cpu"], env["pjrt_cpu"]
    return env.get("lmp_jax", env["lmp"]), env.get("pjrt")


def _child_main(kind, payload):
    p = json.loads(payload)
    if kind == "gate" and p.get("mode") in JAXMD_MODES:
        from scaling import run_jaxmd
        r = run_jaxmd.gate(p["row"], p["mode"])
    elif kind == "gate":
        r = gate_model(p["row"], p["device"], p["env"], p["work"])
    else:
        r = time_case(p["case"], p["device"], p["n"], p["env"], p["work"], p.get("prev"))
    print(json.dumps(r, default=str))


# ---- parent side --------------------------------------------------------------

def subprocess_runner(env, device):
    """(runner(c, n, work, prev), gate(c)) that each run one child process under
    bench/scaling's child_env (PYTHONPATH, PJRT_PLUGIN; standalone gets every
    core, MPI LAMMPS one thread per rank) and its RSS cap.  The lmp wrappers set
    LAMMPS_PLUGIN_PATH and the modules themselves."""
    from scaling.sweep import child_env, run_capped
    py = env.get("python", sys.executable)
    cpus = len(os.sched_getaffinity(0))

    def child(kind, payload, mode):
        cmd = [py, "-m", "accuracy_cost.timing", "--child", kind, json.dumps(payload)]
        gpu = {} if device == "gpu" and kind == "case" else None      # GPU memory of the timed run
        rc, out, err, peak, capped = run_capped(cmd, child_env(env, mode=mode, cpus=cpus), timeout=7200,
                                                cap_bytes=RSS_CAP_GB * 2**30, gpu=gpu)
        lines = [ln for ln in out.splitlines() if ln.startswith("{")]
        if lines and not capped:
            r = json.loads(lines[-1])
            r["peak_rss_bytes"] = peak                                    # host memory of the case's process tree
            if gpu is not None:
                r.update({f"gpu_{k}": v for k, v in gpu.items()})
            return r
        return {"status": "oom" if (capped or rc == -9) else "error", "returncode": rc,
                "error": (err or out)[-1500:], **({"capped_rss": peak} if capped else {})}

    def runner(c, n, work, prev):
        r = child("case", {"case": c, "device": device, "n": n, "env": env, "work": str(work), "prev": prev},
                  c["mode"])
        r.setdefault("n_atoms", n)
        return r

    def gate(c):
        work = pathlib.Path(env["_work"]) / "parity" / c["row"]["name"]
        return child("gate", {"row": c["row"], "device": device, "env": env, "work": str(work),
                              "mode": c["mode"]}, None)
    return runner, gate


def _median_row(rows):
    """The first ok row with step_s the median over the ok repeats, else the first row."""
    for r in rows:                                # standalone: one calculator call is one MD step
        if r.get("status") == "ok" and "step_s" not in r:
            r["step_s"] = r["call_s"]
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return rows[0]
    return {**ok[0], "step_s": statistics.median(r["step_s"] for r in ok),
            "step_s_reps": [r["step_s"] for r in ok], "repeats_ok": len(ok)}


def _done(results):
    p = pathlib.Path(results)
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []
    return {tuple(r.get(f) for f in KEY): r for r in rows}


def run_cases(device, cs, runner, gate, results, n, repeats, work, meta, redo=False, ready=None):
    """Gate (LAMMPS rows, once per model and code), then time each case: N/8
    first (its step time and bundle layout carry into the main run), then N,
    `repeats` times each, median step_s.  `ready()` runs before each gate and
    each case (wait_idle: it may wait, or stop the run) and its sample is
    recorded on that row.  Resumable: ok and unsupported rows, and ok and
    unsupported gates, are kept unless `redo`."""
    work, ready = pathlib.Path(work), ready or dict
    done = {} if redo else _done(results)
    host = socket.gethostname()
    keep = ("ok", "unsupported")
    for c in cs:
        r0 = c["row"]
        ident = {"code": r0["code"], "family": r0["family"], "size": r0["size"], "dtype": c["dtype"],
                 "device": device}
        key = tuple({**ident, "mode": c["mode"]}.get(f) for f in KEY)
        if done.get(key, {}).get("status") in keep:
            continue
        base = {**ident, "mode": c["mode"], "n_params": r0.get("n_params"), "host": host, **meta}
        d = work / device / r0["name"].replace("/", "_") / c["mode"] / c["dtype"]
        if c["mode"] in ("lammps", LAMMPS_TIGHT, "gpumd", *JAXMD_MODES):   # gated: LAMMPS, GPUMD against CPUNEP, JAX-MD
            # one gate per JAX-MD line
            gmode = f"parity-{c['mode']}" if c["mode"] in JAXMD_MODES else "parity"
            gkey = tuple({**ident, "dtype": "float64", "mode": gmode}.get(f) for f in KEY)
            g = done.get(gkey) if done.get(gkey, {}).get("status") in ("parity_ok", "unsupported") else None
            if g is None:
                s = ready()
                g = {**ident, "dtype": "float64", "mode": gmode, "host": host, **meta, **s, **gate(c)}
                common.upsert_jsonl(results, g, KEY); done[gkey] = g
            if g["status"] != "parity_ok":
                st = "unsupported" if g["status"] == "unsupported" else "parity_failed"
                row = {**base, "status": st, "parity": g["status"],
                       "parity_error": g.get("error") or {"max_dE_per_atom": g.get("max_dE_per_atom"),
                                                          "max_dF": g.get("max_dF")}}
                common.upsert_jsonl(results, row, KEY)
                print(row["code"], row["family"], row["size"], row["dtype"], row["mode"], st, flush=True)
                continue
        s = ready()
        n_main, n_check = n
        checks = [runner(c, n_check, d / str(n_check) / str(k), None) for k in range(repeats)]
        check = _median_row(checks) if checks else {"status": "skipped"}
        prev = None
        if check.get("status") == "ok":           # same bundle layout at N as at N/8 (run_lammps.bundle_layout)
            prev = {"step_s": check["step_s"], "n_atoms": n_check,
                    **{f: check[f] for f in ("layout", "matrix_overflow") if f in check}}
        mains = [runner(c, n_main, d / str(n_main) / str(k), prev) for k in range(repeats)]
        main = _median_row(mains)
        row = {**main, **base, **s, "parity": "parity_ok" if c["mode"] in ("lammps", LAMMPS_TIGHT, "gpumd", *JAXMD_MODES) else "not_applicable",
               "n_atoms": n_main, "check_n_atoms": n_check, "check_status": check.get("status"),
               "check_step_s": check.get("step_s"), "check_layout": check.get("layout"),
               # GPU memory (NVML) and host RSS: the largest over the repeats (scaling.sweep.run_capped)
               **{f"{pre}{k}": max(v) for pre, rs in (("", mains), ("check_", checks))
                  for k in ("gpu_peak_process_bytes", "gpu_peak_device_delta_bytes", "peak_rss_bytes")
                  if (v := [r[k] for r in rs if r.get(k) is not None])}}
        if main.get("status") == "ok":
            row["atom_steps_per_s"] = n_main / main["step_s"]
            row["not_plateau"] = plateau_flag(main, check) if check.get("status") == "ok" else "unknown"
        common.upsert_jsonl(results, row, KEY)
        print(row["code"], row["family"], row["size"], row["dtype"], row["mode"], row.get("status"),
              row.get("atom_steps_per_s"), flush=True)


def select(cs, codes=None, modes=None):
    """The cases of these codes and modes; without `modes`, every mode but JAX-MD's (opt-in)."""
    return [c for c in cs if (not codes or c["row"]["code"] in codes)
            and (c["mode"] in modes if modes else c["mode"] not in OPT_IN_MODES)]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", choices=sorted(N))
    ap.add_argument("--only", help="family or family/size")
    ap.add_argument("--n", nargs=2, type=int, metavar=("MAIN", "CHECK"), help="override N (multiples of 8)")
    ap.add_argument("--codes", nargs="+", help="only these codes (e.g. mace-xl mace-xl-f32)")
    ap.add_argument("--modes", nargs="+", help="only these modes; the opt-in modes (JAX-MD's and lammps-tight) "
                    "run only when named here")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--results", help="default results/lestrade-<device>.jsonl")
    ap.add_argument("--work", help="default $ACEJAX_AC_ROOT/runs/accuracy_cost/timing/<results stem>")
    ap.add_argument("--env", help="default envs/lestrade-<device>.json")
    ap.add_argument("--force-busy", action="store_true", help="time a busy box anyway (recorded per row; smoke only)")
    ap.add_argument("--redo", action="store_true", help="re-time ok rows and re-gate")
    a = ap.parse_args([argv] if isinstance(argv, str) else argv)
    ready = lambda: wait_idle(force=a.force_busy)
    ready()                                       # wait (or stop) before anything starts
    env = json.loads(pathlib.Path(a.env or common.HERE / "envs" / f"lestrade-{a.device}.json").read_text())
    env.setdefault("pythonpath", str(common.HERE.parent))
    # per results file: two jobs at once (one per layout or family file) each rewrite in.bench and bundle.json
    # in a shared work dir, and one read the other's processors grid (2026-10-12)
    work = pathlib.Path(a.work or common.RUNS / "timing" / pathlib.Path(a.results or f"lestrade-{a.device}").stem)
    env["_work"] = str(work)
    from scaling.sweep import device_name
    meta = {"forced_busy": a.force_busy,
            "device_name": device_name(a.device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    runner, gate = subprocess_runner(env, a.device)
    cs = [c for c in cases(a.device, only=a.only) if (c["row"]["code"] not in XL_CODES or env.get("lmp_xl"))
          and (c["row"]["code"] != TRIM or env.get("lmp_ace"))
          and (c["row"]["code"] != "mlpace-mixed" or env.get("lmp_mixed"))
          and (c["mode"] != "gpumd" or env.get("gpumd"))]    # optional codes: only where the env json names them
    cs = select(cs, a.codes, a.modes)
    run_cases(a.device, cs, runner, gate,
              pathlib.Path(a.results or RESULTS / f"lestrade-{a.device}.jsonl"),
              n=tuple(a.n) if a.n else N[a.device], repeats=a.repeats, work=work, meta=meta, redo=a.redo,
              ready=ready)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        _child_main(sys.argv[2], sys.argv[3])
    else:
        main()
