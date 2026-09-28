"""Pacemaker datasets for the same seed-0 train/val split as bench/learn_radial/run.py.
Reference energies: per-element least-squares fit on the train split (applied to both),
so the linear PACE model does not have to absorb ~eV/atom offsets."""
import sys, json, numpy as np, pandas as pd
from ase.io import read
data, ntrain, nval, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ats = read(data, ":")
perm = np.random.default_rng(0).permutation(len(ats))
tr, va = [ats[i] for i in perm[:ntrain]], [ats[i] for i in perm[ntrain:ntrain + nval]]
els = sorted({s for a in ats for s in a.get_chemical_symbols()})
X = np.array([[a.get_chemical_symbols().count(e) for e in els] for a in tr], float)
y = np.array([a.info["mace_energy"] for a in tr])
e_ref = dict(zip(els, np.linalg.lstsq(X, y, rcond=None)[0].tolist()))
def df(frames):
    rows = []
    for a in frames:
        b = a.copy(); b.calc = None
        eref = sum(e_ref[s] for s in a.get_chemical_symbols())
        rows.append(dict(ase_atoms=b, energy=a.info["mace_energy"], forces=a.arrays["mace_force"],
                         energy_corrected=a.info["mace_energy"] - eref))
    return pd.DataFrame(rows)
df(tr).to_pickle(f"{out}_train.pckl.gzip", compression="gzip")
df(va).to_pickle(f"{out}_val.pckl.gzip", compression="gzip")
json.dump({"elements": els, "e_ref": e_ref, "val_idx": perm[ntrain:ntrain + nval].tolist()}, open(f"{out}_meta.json", "w"))
print(out, "train", len(tr), "val", len(va), "e_ref", e_ref)
