"""Deterministic benchmark cells: SiGe (random 50/50 diamond) and Cantor
(random equiatomic CrMnFeCoNi fcc), at any power-of-two atom count."""
import itertools

import numpy as np
from ase.build import bulk

SYSTEMS = {
    "SiGe": {"lattice": "diamond", "a": 5.54, "elements": ("Si", "Ge"), "per_cell": 8},
    "Cantor": {"lattice": "fcc", "a": 3.59, "elements": ("Cr", "Mn", "Fe", "Co", "Ni"),
               "per_cell": 4},
}


def n_ladder(system, n_max, n_min=256):
    out, n = [], n_min
    while n <= n_max:
        out.append(n)
        n *= 2
    return out


def _reps(m):
    """Three near-equal integer factors of m (cells along x, y, z)."""
    best = None
    for a in range(1, int(round(m ** (1 / 3))) + 2):
        if m % a:
            continue
        for b in range(a, int(np.sqrt(m // a)) + 2):
            if (m // a) % b:
                continue
            c = m // a // b
            spread = max(a, b, c) - min(a, b, c)
            if best is None or spread < best[0]:
                best = (spread, (a, b, c))
    return best[1]


def supercell(system, n_atoms, seed=0):
    sp = SYSTEMS[system]
    if n_atoms % sp["per_cell"]:
        raise ValueError(f"{system}: n_atoms must be a multiple of {sp['per_cell']}")
    cell = bulk(sp["elements"][0], sp["lattice"], a=sp["a"], cubic=True)
    at = cell.repeat(_reps(n_atoms // sp["per_cell"]))
    k = len(sp["elements"])
    symbols = [sp["elements"][i % k] for i in range(n_atoms)]
    np.random.default_rng(seed).shuffle(symbols)
    at.set_chemical_symbols(symbols)
    return at
