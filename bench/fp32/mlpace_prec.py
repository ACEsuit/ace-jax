"""Plan step 1.6: KOKKOS ML-PACE (pair_style pace product, one GPU) at KOKKOS_PREC double / mixed /
single, against ace-jax's PACEModel calculator (fp64 / fp32) on the same .yace and cells.

Per (model, cell): energy and forces at step 0 (dumped at %.17g) and the time per MD step of a timed
NVE segment after a warm-up.  Errors are taken against ML-PACE double (and ace-jax fp64 against it,
as the parity check that both codes evaluate the same model).

    uv run --no-sync python bench/fp32/mlpace_prec.py MODEL.yace [MODEL.yace ...] > out.json
"""
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

BUILDS = Path(os.environ.get("MLPACE_BUILDS", "/storage/eng/essswb/projects/ace-jax/runs/fp32_step1/mlpace"))
ENV = "source /etc/profile.d/modules.sh; module purge; module load gompi/2023a CUDA/12.4.0; "
STEPS = int(os.environ.get("MLPACE_STEPS", 200))


def cells():
    out = {}
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((8, 8, 8))
    at.rattle(0.05, seed=1)
    out["diamond4096_r0.05"] = at
    hot = bulk("Si", "diamond", a=5.43, cubic=True).repeat((8, 8, 8))
    hot.rattle(0.25, seed=2)                         # strongly distorted: short bonds (0.81 A), large forces;
    out["diamond4096_r0.25"] = hot                   # accuracy only: NVE at 1 fs loses atoms from here
    return out


INPUT = """units metal
atom_style atomic
boundary p p p
read_data data.lmp
mass 1 28.0855
pair_style pace product
pair_coeff * * {model} Si
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
thermo_style custom step pe
thermo_modify format float %.17g
dump f all custom 1 forces.dump id fx fy fz
dump_modify f format float %.17g sort id
run 0
undump f
{md}"""
MD = """velocity all create 300 1 loop geom
fix nve all nve
run 50
run {steps}
"""


def run_lammps(prec, model, at, timed=True):
    lmp = BUILDS / f"build-{prec}" / "lmp"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write(d / "data.lmp", at, format="lammps-data", specorder=["Si"])
        md = MD.format(steps=STEPS) if timed else ""
        (d / "in.bench").write_text(INPUT.format(model=model, md=md))
        cmd = (ENV + f"cd {d} && {lmp} -in in.bench -log log.lammps -nocite -k on g 1 -sf kk "
               "-pk kokkos newton on neigh half")
        p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        log = (d / "log.lammps").read_text() if (d / "log.lammps").exists() else p.stdout
        if p.returncode != 0:
            return dict(error="\n".join(ln for ln in (p.stdout + log).splitlines() if "ERROR" in ln)[-1500:])
        pe = float(re.findall(r"^\s+0\s+(\S+)\s*$", log, re.M)[0])
        loops = [float(x) for x in re.findall(r"Loop time of (\S+)", log)]
        F = np.loadtxt(d / "forces.dump", skiprows=9)[:, 1:]
        return dict(E=pe, F=F) | (dict(t_step=loops[-1] / STEPS) if timed else {})


def run_acejax(dtype, model, at):
    code = f"""
import json, sys, time, numpy as np, jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
from ase.io import read
from ace_jax.calc import ACECalculator
at = read(sys.argv[1]); at.calc = ACECalculator("{model}", dtype=np.dtype("{dtype}"))
E = at.get_potential_energy(); F = at.get_forces()
rng = np.random.default_rng(0); ts = []
for i in range(25):
    at.positions += 1e-3 * rng.standard_normal(at.positions.shape)
    t = time.perf_counter(); at.get_potential_energy(); at.get_forces(); ts.append(time.perf_counter() - t)
np.save(sys.argv[2], F); print(json.dumps(dict(E=E, t_step=float(np.median(ts[5:])))))
"""
    with tempfile.TemporaryDirectory() as d:
        write(f"{d}/at.extxyz", at)
        env = os.environ | {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        p = subprocess.run(["uv", "run", "--no-sync", "python", "-c", code, f"{d}/at.extxyz", f"{d}/F.npy"],
                           capture_output=True, text=True, env=env)
        if p.returncode != 0:
            return dict(error=p.stderr[-1500:])
        r = json.loads(p.stdout.strip().splitlines()[-1])
        return r | dict(F=np.load(f"{d}/F.npy"))


def compare(r, ref, n):
    if "error" in r or "error" in ref:
        return None
    dF = r["F"] - ref["F"]
    return dict(dE_meV_atom=1e3 * abs(r["E"] - ref["E"]) / n, dF_meV_A_rms=1e3 * float(np.sqrt(np.mean(dF ** 2))),
                dF_meV_A_max=1e3 * float(np.abs(dF).max()), F_rel=float(np.linalg.norm(dF) / np.linalg.norm(ref["F"])))


out = []
for model in sys.argv[1:]:
    for name, at in cells().items():
        rows = {f"mlpace_{p}": run_lammps(p, model, at, timed="r0.25" not in name)
                for p in ("double", "mixed", "single")}
        rows |= {f"acejax_{d}": run_acejax(d, model, at) for d in ("float64", "float32")}
        ref = rows["mlpace_double"]
        out.append(dict(model=model, cell=name, n=len(at),
                        t_step_ms={k: (1e3 * v["t_step"] if "t_step" in v else None) for k, v in rows.items()},
                        vs_mlpace_double={k: compare(v, ref, len(at)) for k, v in rows.items() if k != "mlpace_double"},
                        errors={k: v["error"] for k, v in rows.items() if "error" in v}))
        print(json.dumps(out[-1]), file=sys.stderr, flush=True)
print(json.dumps(out, indent=1))
