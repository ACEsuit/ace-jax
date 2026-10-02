"""Running the ACEpotentials.jl drivers in bench/scaling/julia/ from Python.

The Julia, its depot and the pinned project come from the host env json (keys
`julia`, `julia_depot`, `julia_project`) -- `sweep.child_env` hands them to a case
as ACEPOT_JULIA, ACEPOT_JULIA_DEPOT and ACEPOT_JULIA_PROJECT -- never from the
user's default depot: the env is pinned to ACEpotentials.jl PR 309 and must not
touch ~/.julia.  Each driver prints one JSON line last on stdout.
"""
import json
import os
import pathlib
import shlex
import subprocess

JULIA_DIR = pathlib.Path(__file__).resolve().parent / "julia"


def julia_config(env=None):
    """{"julia": argv list, "depot", "project"} from a host env json dict, or (None)
    from the ACEPOT_* environment variables.  The depot is required."""
    if env is not None:
        julia, depot, project = env.get("julia"), env.get("julia_depot"), env.get("julia_project")
    else:
        julia, depot, project = (os.environ.get(f"ACEPOT_JULIA{k}") for k in ("", "_DEPOT", "_PROJECT"))
    if not depot:
        raise RuntimeError("no Julia depot for the ACEpotentials.jl lines: set the host env json's "
                           "`julia_depot` (ACEPOT_JULIA_DEPOT); the default ~/.julia is never used")
    return {"julia": shlex.split(julia or "julia"), "depot": depot, "project": project or str(JULIA_DIR)}


def julia_cmd(cfg, script, *args):
    return [*cfg["julia"], "--startup-file=no", f"--project={cfg['project']}",
            str(JULIA_DIR / script), *map(str, args)]


def julia_env(cfg, threads=None):
    out = {**os.environ, "JULIA_DEPOT_PATH": cfg["depot"]}
    if threads:
        out["JULIA_NUM_THREADS"] = str(threads)
    return out


def run_julia(cfg, script, args, threads=None, timeout=7200):
    """Run a driver; its last `{` line of stdout, parsed.  RuntimeError with the
    stderr tail when it fails (an identity or gate assertion, a refused export)."""
    cmd = julia_cmd(cfg, script, *args)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env=julia_env(cfg, threads))
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        err = p.stderr or p.stdout
        i = err.find("ERROR:")                    # Julia's message, ahead of its stack trace
        raise RuntimeError(f"{script} failed ({p.returncode}): " + (err[i:i + 1500] if i >= 0 else err[-1500:]))
    return json.loads(lines[-1])


def roundtrip(at, path):
    """Write `at` as extxyz and read it back: the writer rounds positions to 1e-8 A,
    so Python, Julia and LAMMPS must all evaluate the re-read structure."""
    from ase.io import read, write
    write(str(path), at, format="extxyz")
    return read(str(path), format="extxyz")


def julia_ef(cfg, spec, xyz, which, work):
    """(E, F (n, 3)) of the model `spec` on the extxyz at `xyz`: which = "splined"
    (the npz's ace1_model) or "etace" (its exact twin, what the trim library compiles)."""
    import numpy as np
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    sp = work / "spec.json"
    sp.write_text(json.dumps(spec))
    out = run_julia(cfg, "eval_ef.jl", [str(sp), str(xyz), which], threads=1, timeout=3600)
    return out["E"], np.asarray(out["F"], dtype=np.float64)
