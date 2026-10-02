"""Validation RMSE of a pacemaker potential on the same split as bench/learn_radial/rmse.py
(E per atom, F per component; total energy = corrected + per-element reference)."""
import sys, json, numpy as np
from ase.io import read
from pyace import PyACECalculator
pot, data, meta_json, ntrain = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
meta = json.load(open(meta_json))
ats = read(data, ":")
calc = PyACECalculator(pot)
out = {}
perm = np.random.default_rng(0).permutation(len(ats))
for split, idx in (("val", meta["val_idx"]), ("train", perm[:ntrain].tolist())):
    eE, eF = [], []
    for i in idx:
        a = ats[i].copy(); a.calc = calc
        eref = sum(meta["e_ref"][s] for s in a.get_chemical_symbols())
        eE.append((a.get_potential_energy() + eref - ats[i].info["mace_energy"]) / len(a))
        eF.append((a.get_forces() - ats[i].arrays["mace_force"]).ravel())
    eE, eF = np.array(eE), np.concatenate(eF)
    out[split] = {"E_rmse_meV_atom": float(1e3 * np.sqrt(np.mean(eE**2))), "F_rmse_meV_A": float(1e3 * np.sqrt(np.mean(eF**2)))}
print(pot, json.dumps(out))
