"""Big-cell OOD set v3 (2026-09-30): the v2 cracks and dissociated dislocations (big2.py), made
THICK along the line so the chemistry is random along it.

The v2 cells were one period thick along the line: 2.58 A (screw, one Burgers vector), 4.47 A
(edge), 5.16 A (crack).  With the Cantor model's rcut = 6.25 A, every atom then sees its own
periodic images -- the screw ones as nearest neighbours -- so each atom sat in a single-species
column, which no random-alloy training config has.  Every atom seeing at most one copy of any
neighbour needs a line period > 2 rcut = 12.5 A:
  screw 5 x 2.58 = 12.9 A,   edge 3 x 4.47 = 13.4 A,   crack 6 x 2.58 = 15.5 A (3 x the v2 cell).
Radii shrink to keep ~4k atoms (MACE-MH-1 float64 fails on a 7.2k-atom cell).  Species are
assigned AFTER the repeat, so they are random along the line too.
"""
import numpy as np

import big2

# per family: (line repeats of the v2 cell, cylinder radius A); ~4k atoms each
THICK = {"crack": (3, 34.0), "edge": (3, 29.0), "screw": (5, 31.0)}
PARTIAL = {"edge": 12, "screw": 8}       # partial separation (glide distances): fits the smaller R


def _thicken(at, n):
    """Repeat along the line (z) n times and carry the per-atom arrays; species re-drawn later."""
    fixed, rcore = at.arrays["fixed"].copy(), at.arrays["r_core"].copy()
    out = at.repeat((1, 1, n))                     # ase: n consecutive copies of the atom list
    out.arrays["fixed"] = np.tile(fixed, n)
    out.arrays["r_core"] = np.tile(rcore, n)
    out.info = dict(at.info, line_repeats=int(n), line_period=float(out.cell[2, 2]))
    return out


def crack(rng, a0, C, gamma, k_rel):
    n, R = THICK["crack"]
    return big2.species(rng, _thicken(big2.crack(rng, a0, C, gamma, k_rel, R=R), n))


def dislocation(rng, a0, C, kind):
    n, R = THICK[kind]
    return big2.species(rng, _thicken(big2.dislocation(rng, a0, C, kind, R=R,
                                                       partial_distance=PARTIAL[kind]), n))


def self_image_check(at, rcut=6.25):
    """Max number of periodic copies of one atom within rcut of any atom (1 = none repeated)."""
    from ase.neighborlist import neighbor_list
    i, j = neighbor_list("ij", at, rcut)
    pair = i.astype(np.int64) * len(at) + j
    return int(np.bincount(pair).max(initial=1)), int(np.sum(i == j))
