"""The aluminium supercells of the ACEhamiltonians paper (arXiv:2111.13736) in the DeepH layout,
for codes that read it (MACE-H): on-site blocks and every bond within a cutoff, H in eV, orbital
phases as ace-jax (`aitb.FHI_SIGNS` applied), the MD files' factor of 27 (k-grid) divided out.
    uv run --extra ham python bench/ham/al_to_deeph.py <out dir> [cutoff=9.0] [files...]
Default files: the four 500 K MD supercells (729 atoms) of training_data."""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aitb
import offsite_accuracy as OA

from ace_jax.operators import bond as B
from ace_jax.ham import deeph

N = 14


def convert(name, out, rcut):
    d = aitb.load(os.path.join(OA.DATA, name), operators=("H", "S"), raw=True)
    g = B.bond_graph(types.SimpleNamespace(rcut_bond=rcut), d["positions"], d["cell"], d["numbers"])
    s = aitb.atom_signs(aitb.AL.ls(13))
    scale = np.rint(d["S_full"][0, 0])                  # 27 for the MD files, 1 for reference_data
    assert abs(d["S_full"][0, 0] / scale - 1) < 1e-3
    ss = s[:, None] * s[None, :] / scale
    keys = [(0, 0, 0, i, i) for i in range(len(d["positions"]))]
    keys += [(int(R[0]), int(R[1]), int(R[2]), int(i), int(j)) for R, i, j in zip(g.R, g.i, g.j)]
    blocks = {c: {k: d[c + "_full"][k[3] * N:(k[3] + 1) * N, k[4] * N:(k[4] + 1) * N] * ss for k in keys} for c in ("H", "S")}
    deeph.write(out, deeph.OperatorData(d["cell"], d["positions"], d["numbers"], aitb.AL, blocks))
    print(f"{name}: {len(d['positions'])} atoms, {len(g.i)} bonds within {rcut} angstrom -> {out}")


if __name__ == "__main__":
    out = sys.argv[1]
    rcut = float(sys.argv[2]) if len(sys.argv) > 2 else 9.0
    for name in sys.argv[3:] or OA.TRAIN + OA.TEST:
        convert(name, os.path.join(out, name.replace("/", "_").replace(".h5", "")), rcut)
