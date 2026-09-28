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
p.add_argument("--rcut", type=float, default=None, help="histogram range; default: the first model's cutoff")
p.add_argument("--out", required=True)
a = p.parse_args()


def pair_r(frames, rcut):
    return np.concatenate([neighbor_list("d", f, rcut) for f in frames])


from ace_jax.eval import load as _load
if a.rcut is None:
    a.rcut = float(_load(a.models.split(",")[0])[1]["rcut"])
train = read(a.train, ":")
perm = np.random.default_rng(a.seed).permutation(len(train))
bins = np.linspace(0.0, a.rcut, 111)
h_train, _ = np.histogram(pair_r([train[i] for i in perm[:a.ntrain]], a.rcut), bins=bins, density=True)
low = h_train < 0.05 * h_train.max()                       # bins the training data barely populates
centres = 0.5 * (bins[1:] + bins[:-1])
r_first = centres[np.argmax(h_train > 0.05 * h_train.max())]
core = low & (centres < r_first)                            # compressed pairs, below the first shell
gap = low & (centres >= r_first)                            # between (and beyond) shells

res = {"gap_definition": "bins with training pair-density < 5% of its max; 'core' = below the first "
                         "populated shell, 'gap' = at or beyond it", "rcut": a.rcut, "trajs": {}}
calcs = {pathlib.Path(m).stem: ACECalculator(m) for m in a.models.split(",")}
for tpath in a.trajs.split(","):
    frames = read(tpath, ":")
    tname = pathlib.Path(tpath).stem
    h, _ = np.histogram(pair_r(frames, a.rcut), bins=bins)
    entry = {"n_frames": len(frames), "n_stopped_frames": int(sum(bool(f.info.get("stopped")) for f in frames)),
             "gap_pair_fraction": float(h[gap].sum() / max(h.sum(), 1)),
             "core_pair_fraction": float(h[core].sum() / max(h.sum(), 1)), "models": {}}
    for mname, calc in calcs.items():
        eE, eF, ok = [], [], []
        for f in frames:
            b = f.copy()
            b.calc = calc
            eE.append((b.get_potential_energy() - f.info["mace_energy"]) / len(f))
            eF.append((b.get_forces() - f.arrays["mace_force"]).ravel())
            ok.append(not bool(f.info.get("stopped")))
        eE, ok = np.array(eE), np.array(ok)
        rE = lambda e: float(1e3 * np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        entry["models"][mname] = {"E_rmse_meV_atom": rE(eE), "F_rmse_meV_A": rE(np.concatenate(eF)),
                                  "E_rmse_meV_atom_unstopped": rE(eE[ok]),
                                  "F_rmse_meV_A_unstopped": rE(np.concatenate([e for e, k in zip(eF, ok) if k]))
                                  if ok.any() else float("nan")}
        print(f"traj {tname:28s} model {mname:24s} E {entry['models'][mname]['E_rmse_meV_atom']:8.3f} meV/atom"
              f"  F {entry['models'][mname]['F_rmse_meV_A']:8.2f} meV/A   (gap pair fraction "
              f"{entry['gap_pair_fraction']:.3f}, core {entry['core_pair_fraction']:.3f}, {entry['n_frames']} frames,"
              f" {entry['n_stopped_frames']} stopped)", flush=True)
    res["trajs"][tname] = entry
res["train_gap_pair_fraction"] = float((h_train[gap] * np.diff(bins)[gap]).sum())
res["train_core_pair_fraction"] = float((h_train[core] * np.diff(bins)[core]).sum())
pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
