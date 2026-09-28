"""Langevin MD driven by an ACE model (ace-jax ACECalculator), for the
learned-radial transfer check.  Starts from structures of an xyz file (by
index), writes every `--every`-th frame to an extxyz, and stops a trajectory
early if it becomes unphysical (max|F| above --fmax-stop or a pair closer than
--rmin-stop).  Frame info carries T, step, max|F|, min pair distance and
whether the run was stopped.

    uv run python bench/learn_radial/md/md_run.py --model m.npz --starts val.xyz \
        --idx 0,1,2,3 --T 1200 --steps 1000 --every 20 --out traj.xyz
"""
import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list

import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from padded_calc import PaddedACECalculator as ACECalculator  # fixed-shape jit; identical results

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True); p.add_argument("--starts", required=True)
p.add_argument("--idx", default="0"); p.add_argument("--T", type=float, default=1200.0)
p.add_argument("--steps", type=int, default=1000); p.add_argument("--dt", type=float, default=1.0, help="fs")
p.add_argument("--every", type=int, default=20); p.add_argument("--friction", type=float, default=0.01, help="1/fs")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--fmax-stop", type=float, default=50.0); p.add_argument("--rmin-stop", type=float, default=1.0)
p.add_argument("--out", required=True)
a = p.parse_args()

calc = ACECalculator(a.model)
starts = read(a.starts, ":")
frames = []
for k, i in enumerate(int(x) for x in a.idx.split(",")):
    at = starts[i].copy()
    at.calc = calc
    rng = np.random.default_rng(a.seed + k)
    MaxwellBoltzmannDistribution(at, temperature_K=a.T, rng=rng)
    dyn = Langevin(at, a.dt * units.fs, temperature_K=a.T, friction=a.friction / units.fs, rng=rng)
    t0, stopped = time.time(), False
    for step in range(0, a.steps + 1, a.every):
        if step:
            dyn.run(a.every)
        f = at.get_forces()
        d = neighbor_list("d", at, 3.0)
        rmin = float(d.min()) if d.size else np.inf
        fmax = float(np.abs(f).max())
        fr = at.copy()
        fr.calc = None
        fr.info.update(start=i, T=a.T, step=step, ace_energy=float(at.get_potential_energy()),
                       fmax=fmax, rmin=rmin, T_inst=float(at.get_temperature()))
        fr.arrays["ace_force"] = f.copy()
        if fmax > a.fmax_stop or rmin < a.rmin_stop:
            stopped = True
        fr.info["stopped"] = stopped
        frames.append(fr)
        if stopped:
            break
    print(f"start {i} T={a.T:.0f}K: {step} steps in {time.time() - t0:.0f}s "
          f"({'STOPPED: unphysical' if stopped else 'ok'}; last max|F| {fmax:.2f} eV/A, min r {rmin:.2f} A)",
          flush=True)
write(a.out, frames)
