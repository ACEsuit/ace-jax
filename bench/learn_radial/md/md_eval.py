"""Errors of ACE models against the MACE reference on MACE-labelled MD frames
(md_label.py output), plus how much of the frames' pair-distance mass falls in
the training data's gaps.  Every model is evaluated on every trajectory, so a
model's error on its own trajectory and on the others' can be compared.

    uv run python bench/learn_radial/md/md_eval.py --models init.npz,learned.npz \
        --trajs md_init.xyz,md_learned.xyz --train train.xyz --out md_eval.json
"""
import argparse
import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

import sys as _sys
_sys.path.insert(0, __import__('os').path.dirname(__file__))
from padded_calc import PaddedACECalculator as ACECalculator  # fixed-shape jit; identical results

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--models", required=True); p.add_argument("--trajs", required=True)
p.add_argument("--train", required=True, help="training xyz, for the pair-distance density")
p.add_argument("--ntrain", type=int, default=200); p.add_argument("--seed", type=int, default=0)
p.add_argument("--rcut", type=float, default=5.5)
p.add_argument("--out", required=True)
a = p.parse_args()


def pair_r(frames, rcut):
    return np.concatenate([neighbor_list("d", f, rcut) for f in frames])


train = read(a.train, ":")
perm = np.random.default_rng(a.seed).permutation(len(train))
bins = np.linspace(0.5, a.rcut, 101)
h_train, _ = np.histogram(pair_r([train[i] for i in perm[:a.ntrain]], a.rcut), bins=bins, density=True)
gap = h_train < 0.05 * h_train.max()                       # bins the training data barely populates

res = {"gap_definition": "training pair-density < 5% of its max", "trajs": {}}
calcs = {pathlib.Path(m).stem: ACECalculator(m) for m in a.models.split(",")}
for tpath in a.trajs.split(","):
    frames = read(tpath, ":")
    tname = pathlib.Path(tpath).stem
    h, _ = np.histogram(pair_r(frames, a.rcut), bins=bins)
    entry = {"n_frames": len(frames), "n_stopped_frames": int(sum(bool(f.info.get("stopped")) for f in frames)),
             "gap_pair_fraction": float(h[gap].sum() / max(h.sum(), 1)), "models": {}}
    for mname, calc in calcs.items():
        eE, eF = [], []
        for f in frames:
            b = f.copy()
            b.calc = calc
            eE.append((b.get_potential_energy() - f.info["mace_energy"]) / len(f))
            eF.append((b.get_forces() - f.arrays["mace_force"]).ravel())
        eE, eF = np.array(eE), np.concatenate(eF)
        entry["models"][mname] = {"E_rmse_meV_atom": float(1e3 * np.sqrt(np.mean(eE ** 2))),
                                  "F_rmse_meV_A": float(1e3 * np.sqrt(np.mean(eF ** 2)))}
        print(f"traj {tname:28s} model {mname:24s} E {entry['models'][mname]['E_rmse_meV_atom']:8.3f} meV/atom"
              f"  F {entry['models'][mname]['F_rmse_meV_A']:8.2f} meV/A   (gap pair fraction "
              f"{entry['gap_pair_fraction']:.3f}, {entry['n_frames']} frames)", flush=True)
    res["trajs"][tname] = entry
res["train_gap_pair_fraction"] = float((h_train[gap] * np.diff(bins)[gap]).sum())
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
