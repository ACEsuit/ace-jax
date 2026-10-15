"""Plan step 1.4/1.5: ACECalculator energies, forces and stresses on the Si test set, and the
model-step time on a large cell, at dtype float32 or float64 (the calculator's existing `dtype=`).
x64 is on in both, so float32 here is the calculator's own fp32 path (positions in fp64 on the
host).  --matmul default leaves JAX's GPU default (TF32 on Ampere) in force.

    uv run --no-sync python bench/fp32/infer_dump.py --dtype float32 --model M.npz --test T.xyz --out o.npz
"""
import argparse
import json
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--dtype", choices=["float32", "float64"], required=True)
p.add_argument("--matmul", default="highest")
p.add_argument("--model", required=True)
p.add_argument("--test", required=True)
p.add_argument("--big", type=int, default=8, help="diamond cubic repeat for the timing cell (8 -> 4096 atoms)")
p.add_argument("--out", required=True)
a = p.parse_args()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
if a.matmul != "default":
    jax.config.update("jax_default_matmul_precision", a.matmul)

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from ace_jax.calc import ACECalculator  # noqa: E402
from ace_jax.fit.xyz import read_extxyz  # noqa: E402

dt = np.dtype(a.dtype)
calc = ACECalculator(a.model, dtype=dt)
frames = [Atoms(f.numbers, f.positions, cell=f.cell, pbc=f.pbc) for f in read_extxyz(a.test)]
E, F, S, nat = [], [], [], []
for at in frames:
    at.calc = calc
    E.append(at.get_potential_energy()); F.append(at.get_forces()); nat.append(len(at))
    S.append(at.get_stress() if at.cell.volume > 0 and all(at.pbc) else np.zeros(6))

at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((a.big,) * 3)
at.rattle(0.05, seed=1)
calc_big = ACECalculator(a.model, dtype=dt)
at.calc = calc_big
rng = np.random.default_rng(0)
ts, Es = [], []
for i in range(25):                                   # small moves: the skin list is reused
    at.positions += 1e-3 * rng.standard_normal(at.positions.shape)
    t = time.perf_counter()
    Es.append(at.get_potential_energy()); at.get_forces()
    ts.append(time.perf_counter() - t)
info = dict(dtype=a.dtype, matmul=a.matmul, device=str(jax.devices()[0]), n_test=len(frames),
            big_atoms=len(at), t_step_median=float(np.median(ts[5:])),
            model_s=calc_big.last_timing.get("model_s"), layout=calc_big.last_layout,
            rebuilds=calc_big.last_timing.get("rebuilds"))
np.savez(a.out, E=np.array(E), F=np.concatenate(F), S=np.array(S), nat=np.array(nat), Ebig=np.array(Es),
         Fbig=at.get_forces(), info=json.dumps(info))
print(json.dumps(info), file=sys.stderr)
