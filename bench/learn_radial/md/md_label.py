"""Label MD frames with the MACE reference (mace-torch).  Run in an environment
with mace-torch; the default model/head (MACE-MH-1, head matpes_r2scan)
reproduces the stored mace_energy/mace_force of the SiGe and Cantor training
sets (checked: 0.00 / 0.04 meV/atom).

    python bench/learn_radial/md/md_label.py traj.xyz traj_labelled.xyz
"""
import argparse

from ase.io import read, write
from mace.calculators import MACECalculator

p = argparse.ArgumentParser()
p.add_argument("inp"); p.add_argument("out")
p.add_argument("--model", default="/home/eng/essswb/.cache/mace/mace-mh-1.model")
p.add_argument("--head", default="matpes_r2scan"); p.add_argument("--device", default="cuda")
a = p.parse_args()

calc = MACECalculator(model_paths=a.model, device=a.device, default_dtype="float64", head=a.head)
frames = read(a.inp, ":")
for fr in frames:
    b = fr.copy()
    b.calc = calc
    fr.info["mace_energy"] = float(b.get_potential_energy())
    fr.arrays["mace_force"] = b.get_forces().copy()
write(a.out, frames)
print(f"labelled {len(frames)} frames -> {a.out}")
