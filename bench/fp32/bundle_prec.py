"""Plan step 2: lammps-jax bundles (float64 / float32) and KOKKOS ML-PACE (double / mixed / single)
in LAMMPS on one GPU, against the ace-jax float64 calculator, over box sizes.

lammps-jax hands a float32 bundle float32 *absolute* positions, so rij carries ~ulp(|x|) of
rounding that grows with the box; the float32 calculator forms displacements in float64.  The
bundle-vs-calculator float32 difference at growing N isolates that.

    uv run --no-sync python bench/fp32/bundle_prec.py MODEL [MODEL ...] --sizes 4096 32768 > out.json

MODEL: an ace-jax .npz or a .yace (only .yace models also run in ML-PACE).  Per (model, size):
energy and forces at step 0 (%.17g dumps) and the time per NVE step after a warm-up.
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
from ase.build import bulk
from ase.io import write

HERE = Path(__file__).resolve().parent
MLPACE = Path(os.environ.get("MLPACE_BUILDS", "/storage/eng/essswb/projects/ace-jax/runs/fp32_step1/mlpace"))
LMP_JAX = os.environ.get("LMP_JAX", "/home/eng/essswb/bench-scaling/lmp-jax.sh")
PJRT = os.environ.get("PJRT_PLUGIN", "/home/eng/essswb/bench-scaling/venv/lib/python3.12/site-packages/"
                      "jax_plugins/xla_cuda12/xla_cuda_plugin.so")
ENV = "source /etc/profile.d/modules.sh; module purge; module load gompi/2023a CUDA/12.4.0; "
STEPS = int(os.environ.get("BUNDLE_STEPS", 100))

INPUT = """units metal
atom_style atomic
boundary p p p
atom_modify map yes
read_data data.lmp
mass 1 28.0855
{pair}
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
timestep 0.0005
thermo_style custom step pe
thermo_modify format float %.17g
dump f all custom 1 forces.dump id fx fy fz
dump_modify f format float %.17g sort id
run 0
undump f
velocity all create 300 1 loop geom
fix nve all nve
run 50
run {steps}
"""


def cell(n):
    reps = round((n / 8) ** (1 / 3))
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((reps,) * 3)
    at.rattle(0.05, seed=1)
    return at


def lammps(cmd_prefix, pair, at, kk="-k on g 1 -sf kk -pk kokkos newton on neigh half", extra=""):
    with tempfile.TemporaryDirectory(dir=os.environ.get("BUNDLE_TMP")) as d:
        d = Path(d)
        write(d / "data.lmp", at, format="lammps-data", specorder=["Si"])
        (d / "in.bench").write_text(INPUT.format(pair=pair, steps=STEPS))
        p = subprocess.run(["bash", "-c", f"cd {d} && {cmd_prefix} -in in.bench -log log.lammps -nocite {kk} {extra}"],
                           capture_output=True, text=True, timeout=7200)
        log = (d / "log.lammps").read_text() if (d / "log.lammps").exists() else ""
        if p.returncode != 0 or "Loop time" not in log:
            errs = [ln for ln in (p.stdout + p.stderr + log).splitlines() if "ERROR" in ln or "rror" in ln]
            return {"error": "\n".join(errs[-5:]) or (p.stdout + p.stderr)[-800:]}
        pe = float(re.findall(r"^\s+0\s+(\S+)\s*$", log, re.M)[0])
        loops = [float(x) for x in re.findall(r"Loop time of (\S+)", log)]
        F = np.loadtxt(d / "forces.dump", skiprows=9)[:, 1:]
        return {"E": pe, "F": F, "t_step": loops[-1] / STEPS}


def child(code, *args):
    """Run `code` in a fresh process of this venv (JAX state, and its GPU pool, end with it)."""
    env = os.environ | {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    p = subprocess.run(["uv", "run", "--no-sync", "python", "-c", code, *map(str, args)],
                       capture_output=True, text=True, env=env, timeout=7200)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


EXPORT = """
import json, sys, jax
jax.config.update("jax_enable_x64", True)          # as `aj export`: the cast makes float32 a float32 program
from ase.io import read
from ace_jax.eval import load
from ace_jax.export.lammps import export_lammps, neighbour_capacity
model, meta, _ = load(sys.argv[1]); at = read(sys.argv[2]); dtype = sys.argv[4]
cap = neighbour_capacity(at, float(meta["rcut"]), skin=1.0)
b = export_lammps(model, meta, sys.argv[3], max_atoms=cap["max_atoms"], max_edges=cap["max_edges"],
                  k_dense=cap["k_dense"], max_owned=cap["max_owned"], max_neighbors=cap["max_neighbors"],
                  dtype=dtype, type_elements=[14])
print(json.dumps({"layout": b["ace_jax"]["layout"]}))
"""

CALC = """
import json, sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from ase.io import read
from ace_jax.calc import ACECalculator
at = read(sys.argv[1]); dt = None if sys.argv[3] == "float64" else np.dtype(sys.argv[3])
at.calc = ACECalculator(sys.argv[2], dtype=dt)
E = at.get_potential_energy(); np.save(sys.argv[4], at.get_forces()); print(json.dumps({"E": E}))
"""


def compare(r, ref, n):
    if "error" in r:
        return {"error": r["error"]}
    dF = r["F"] - ref["F"]
    return {"dE_meV_atom": 1e3 * abs(r["E"] - ref["E"]) / n, "dF_meV_A_rms": 1e3 * float(np.sqrt(np.mean(dF ** 2))),
            "dF_meV_A_max": 1e3 * float(np.abs(dF).max()), "dF_rel": float(np.linalg.norm(dF) / np.linalg.norm(ref["F"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--sizes", nargs="+", type=int, default=[4096, 32768])
    a = ap.parse_args()
    out = []
    for model in a.models:
        for n in a.sizes:
            at = cell(n)
            rows = {}
            with tempfile.TemporaryDirectory(dir=os.environ.get("BUNDLE_TMP")) as d:
                d = Path(d)
                write(d / "at.extxyz", at)
                try:
                    for dt in ("float64", "float32"):
                        r = child(CALC, d / "at.extxyz", model, dt, d / f"F_{dt}.npy")
                        rows[f"calc_{dt}"] = {"E": r["E"], "F": np.load(d / f"F_{dt}.npy")}
                except RuntimeError as e:          # e.g. the float64 reference out of GPU memory
                    print(json.dumps({"model": model, "n": len(at), "error": str(e)[-300:]}), file=sys.stderr)
                    continue
                for dt in ("float64", "float32"):
                    try:
                        info = child(EXPORT, model, d / "at.extxyz", d / f"bundle_{dt}.json", dt)
                        r = lammps(LMP_JAX, f"pair_style jax/kk {PJRT}\npair_coeff * * {d / f'bundle_{dt}.json'}", at)
                        rows[f"bundle_{dt}"] = r | {"layout": info["layout"]}
                    except RuntimeError as e:
                        rows[f"bundle_{dt}"] = {"error": str(e)[-800:]}
            if model.endswith(".yace"):
                for prec in ("double", "mixed", "single"):
                    rows[f"mlpace_{prec}"] = lammps(ENV + str(MLPACE / f"build-{prec}" / "lmp"),
                                                    f"pair_style pace product\npair_coeff * * {model} Si", at)
            ref = rows["calc_float64"]
            row = {"model": model, "n": len(at), "box_A": float(at.cell[0, 0]),
                   "t_step_ms": {k: 1e3 * v["t_step"] for k, v in rows.items() if "t_step" in v},
                   "layout": {k: v.get("layout") for k, v in rows.items() if "layout" in v},
                   "vs_calc_float64": {k: compare(v, ref, len(at)) for k, v in rows.items() if k != "calc_float64"},
                   "bundle32_vs_calc32": (compare(rows["bundle_float32"], rows["calc_float32"], len(at))
                                          if "error" not in rows["bundle_float32"] else None)}
            print(json.dumps(row), file=sys.stderr, flush=True)
            out.append(row)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
