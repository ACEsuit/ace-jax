"""extxyz -> ase.Atoms through ace_jax's reader (never ase.io.read: it moves label keys into a calculator and
rejects non-9-vector `virial` info).  info and per-atom arrays are attached under the names they were written with."""
import numpy as np


def read_atoms(path):
    from ase import Atoms

    from ace_jax.fit.xyz import read_extxyz
    out = []
    for f in read_extxyz(path):
        at = Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
        at.info.update(f.info)
        for k, v in f.arrays.items():
            if k not in at.arrays:
                at.new_array(k, np.asarray(v))
        out.append(at)
    return out
