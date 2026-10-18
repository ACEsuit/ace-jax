"""NVE energy conservation of lammps-jax bundles in float64 and float32 (docs/dev/fp32-results.md,
"When float32 is safe").  A float32 force carries rounding noise that is not the gradient of any energy,
so velocity Verlet no longer conserves the (float64) total energy: the noise adds a random walk on top
of the integrator's O(dt^2) error.  Per (model, dtype, dt, size): diamond Si, rattled, velocities at
`--temp`, `--ps` of NVE; the thermo series (%.17g, every `--every` steps) and the fitted drift.

    BUNDLE_PYTHON=<venv python> PYTHONPATH=bench:src python bench/fp32/nve_drift.py MODEL [MODEL ...] \
        --dtypes float64 float32 --dts 1.0 --sizes 4096 --ps 40 --out nve.jsonl

The float64 bundle sets the integrator floor; float32 minus that is the precision effect.  lammps-jax
hands a float32 bundle float32 absolute positions, so a larger box (--sizes) adds rij rounding.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle_prec import EXPORT, LMP_JAX, PJRT, cell, child  # noqa: E402

INPUT = """units metal
atom_style atomic
boundary p p p
atom_modify map yes
read_data data.lmp
pair_style jax/kk {pjrt}
pair_coeff * * {bundle}
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
timestep {dt}
velocity all create {temp} 4928 loop geom mom yes rot yes
fix nve all nve
thermo_style custom step time temp pe ke etotal
thermo_modify format float %.17g norm no
thermo {every}
run {steps}
"""


def run(bundle, at, dt_fs, temp, steps, every):
    with tempfile.TemporaryDirectory(dir=os.environ.get("BUNDLE_TMP")) as d:
        d = Path(d)
        write(d / "data.lmp", at, format="lammps-data", specorder=["Si"], masses=True)
        (d / "in.nve").write_text(INPUT.format(pjrt=PJRT, bundle=bundle, dt=dt_fs / 1000, temp=temp,
                                               steps=steps, every=every))
        p = subprocess.run(["bash", "-c", f"cd {d} && {LMP_JAX} -in in.nve -log log.lammps -nocite "
                            "-k on g 1 -sf kk -pk kokkos newton on neigh half"],
                           capture_output=True, text=True, timeout=6 * 3600)
        log = (d / "log.lammps").read_text() if (d / "log.lammps").exists() else ""
        if p.returncode != 0 or "Loop time" not in log:
            errs = [ln for ln in (p.stdout + p.stderr + log).splitlines() if "ERROR" in ln]
            return {"error": "\n".join(errs[-5:]) or (p.stdout + p.stderr)[-800:]}
        block = re.split(r"^\s*Step\s+Time.*$", log, flags=re.M)[1].split("Loop time")[0]
        rows = np.array([[float(x) for x in ln.split()] for ln in block.strip().splitlines()
                         if re.match(r"^\s*\d", ln)])
        loop = float(re.findall(r"Loop time of (\S+)", log)[-1])
        return {"thermo": rows, "loop_s": loop}


def drift(th, n, skip_ps=1.0):
    """Etot per atom (meV) against time (ps): the linear drift after `skip_ps` (equipartition), the RMS
    about that line and the max excursion from the start."""
    t, T, etot = th[:, 1], th[:, 2], th[:, 5]
    e = 1e3 * (etot - etot[0]) / n
    m = t >= skip_ps
    slope, icpt = np.polyfit(t[m], e[m], 1)
    return {"drift_meV_atom_ps": float(slope), "rms_meV_atom": float(np.std(e[m] - (slope * t[m] + icpt))),
            "max_dev_meV_atom": float(np.abs(e).max()), "T_mean_K": float(T[m].mean()),
            "t_ps": float(t[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--dtypes", nargs="+", default=["float64", "float32"])
    ap.add_argument("--dts", nargs="+", type=float, default=[1.0], help="timesteps, fs")
    ap.add_argument("--sizes", nargs="+", type=int, default=[4096])
    ap.add_argument("--temp", type=float, default=1200.0, help="initial K (about half after equipartition)")
    ap.add_argument("--ps", type=float, default=40.0)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for model in a.models:
        for n in a.sizes:
            at = cell(n)
            with tempfile.TemporaryDirectory(dir=os.environ.get("BUNDLE_TMP")) as d:
                d = Path(d)
                write(d / "at.extxyz", at)
                for dtype in a.dtypes:
                    bundle = d / f"bundle_{dtype}.json"
                    info = child(EXPORT, model, d / "at.extxyz", bundle, dtype)
                    for dt in a.dts:
                        steps = round(a.ps * 1000 / dt)
                        r = run(bundle, at, dt, a.temp, steps, a.every)
                        row = {"model": model, "n": len(at), "box_A": float(at.cell[0, 0]), "dtype": dtype,
                               "dt_fs": dt, "temp0_K": a.temp, "layout": info["layout"]}
                        if "error" in r:
                            row["error"] = r["error"]
                        else:
                            row |= drift(r["thermo"], len(at)) | {"atom_steps_s": len(at) * steps / r["loop_s"],
                                                                  "thermo": r["thermo"][:, [1, 2, 5]].tolist()}
                        print(json.dumps({k: v for k, v in row.items() if k != "thermo"}), flush=True)
                        with open(a.out, "a") as f:
                            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
