"""Time the multi-component (Cantor, CrMnFeCoNi) random-weight models of bench/scaling in LAMMPS, with the silicon
benchmark's driver: the same gate (LAMMPS against the code's own evaluator, before any timing), the same N/8
plateau check and repeats, the same LAMMPS builds and node layouts (timing.run_cases, timing.subprocess_runner).

    python -m accuracy_cost.multicomponent {gpu,cpu} --env ENV.json --results PATH [--codes C ...] [--n MAIN CHECK]

Models (bench/scaling/models.py; $BENCH_MODELS_DIR): the medium linear ACE .npz (ace-jax), the medium PACE .yace
(ace-jax and ML-PACE on the same file) and MACE-MP-0b2 medium (Symmetrix v0.3.0: `mace-symx3`, its own extraction
`mace_medium_Cantor-symx3.json`).  Random weights: speed only.  The ace-jax kernel rows are the same command with
ACEJAX_EXPORT_FFI=1 and LAMMPS_JAX_FFI_HANDLERS in the environment, in their own results file (the row key has no
kernel field, as timing.py's)."""
import json
import pathlib
import sys

from . import common, timing

SYSTEM, SIZE, FAMILY = "Cantor", "medium", "cantor"
N = {"gpu": (8192, 1024), "cpu": (8192, 1024)}    # MACE-MP-0b2 medium in float64 runs out of an A100's 40 GB above 8,192
GATE_N = 256                                      # a 4 x 4 x 4 fcc cell (14.4 A): wider than twice any cutoff here
CODES = {"acejax-ace": "acejax-ace", "acejax-pace": "acejax-pace", "mlpace": "mlpace", "mace": "mace-symx3"}


def cases():
    """One LAMMPS float64 case per code, on the bench/scaling model of (SYSTEM, SIZE)."""
    from scaling import models
    out = []
    for m in models.planned_models():
        if (m["system"], m["size"]) != (SYSTEM, SIZE) or m["code"] not in CODES:
            continue
        code = CODES[m["code"]]
        row = {"code": code, "family": FAMILY, "system": SYSTEM, "size": SIZE, "elements": m["elements"],
               "path": m["path"], "name": f"{FAMILY}/{SIZE}/{code}", "rattle": timing.RATTLE, "gate_n": GATE_N,
               "ref": None, "n_params": None}
        if code == "mace-symx3":
            row["symmetrix"] = str(models.DIR / f"mace_{SIZE}_{SYSTEM}-symx3.json")
        out.append({"row": row, "mode": "lammps", "dtype": "float64"})
    return out


def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", choices=sorted(N))
    ap.add_argument("--env", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--codes", nargs="+")
    ap.add_argument("--n", nargs=2, type=int, metavar=("MAIN", "CHECK"))
    ap.add_argument("--repeats", type=int, default=timing.REPEATS)
    ap.add_argument("--work")
    ap.add_argument("--force-busy", action="store_true")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)
    ready = lambda: timing.wait_idle(force=a.force_busy)
    ready()
    env = json.loads(pathlib.Path(a.env).read_text())
    env.setdefault("pythonpath", str(common.HERE.parent))
    work = pathlib.Path(a.work or common.RUNS / "timing" / pathlib.Path(a.results).stem)
    env["_work"] = str(work)
    from scaling.sweep import device_name
    meta = {"forced_busy": a.force_busy, "device_name": device_name(a.device), "system": SYSTEM,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "kernel": bool(os.environ.get("ACEJAX_EXPORT_FFI") or env.get("os_env", {}).get("ACEJAX_EXPORT_FFI"))}
    runner, gate = timing.subprocess_runner(env, a.device)
    cs = [c for c in timing.select(cases(), a.codes)
          if c["row"]["code"] not in timing.OPTIONAL_LMP or env.get(timing.OPTIONAL_LMP[c["row"]["code"]])]
    timing.run_cases(a.device, cs, runner, gate, pathlib.Path(a.results), n=tuple(a.n) if a.n else N[a.device],
                     repeats=a.repeats, work=work, meta=meta, redo=a.redo, ready=ready)


if __name__ == "__main__":
    main(sys.argv[1:])
