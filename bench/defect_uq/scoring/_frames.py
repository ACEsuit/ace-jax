"""Shared bench helpers.  extxyz -> ase.Atoms through ace_jax's reader (never ase.io.read: it moves label keys into a calculator and
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


def sig_round(x, digits=5):
    """x rounded to `digits` significant figures, for rank statistics: symmetry-equivalent atoms (equal up to
    roundoff) stay tied, so a roundoff-level difference cannot reorder them and pull Spearman's rho below 1."""
    x = np.asarray(x, float)
    e = np.floor(np.log10(np.where(x > 0, x, 1.0)))
    return np.round(x / 10 ** e, digits - 1) * 10 ** e
