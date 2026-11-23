"""One structure with the blocks of its operators: what the block fit reads and the readers of
any file format return."""
from collections import OrderedDict
from typing import NamedTuple

import numpy as np

from .orbitals import OrbitalBasis


class OperatorData(NamedTuple):
    """One structure with its operators.

    cell: (3, 3), lattice vectors in ROWS (ASE's layout); positions (N, 3);
    numbers (N,); basis: `OrbitalBasis`; blocks: {name: {(Rx, Ry, Rz, i, j): (n_i, n_j) array}},
    0-based atoms, ace-jax orbital order; names are operator channels ("H", "S", ...)."""

    cell: np.ndarray
    positions: np.ndarray
    numbers: np.ndarray
    basis: OrbitalBasis
    blocks: dict

    def bond(self, key):
        """Vector from atom i to atom j in cell R, for a block key."""
        Rx, Ry, Rz, i, j = key
        return self.positions[j] + np.array([Rx, Ry, Rz]) @ self.cell - self.positions[i]


class CentredData(NamedTuple):
    """One structure with atom-centred coefficients (`operators.centred`): what `fit_centred` reads.

    cell, positions, numbers as `OperatorData`; basis: the `OrbitalBasis` of the auxiliary
    functions; coefficients: {channel: (n_total,)}, the atoms in order, within an atom its shells
    in the basis' order, within a shell m = -l..l (`centred.offsets`): the layout of one atom's
    rows in an `OperatorData` block.  pbc: as ASE's."""

    cell: np.ndarray
    positions: np.ndarray
    numbers: np.ndarray
    basis: OrbitalBasis
    coefficients: dict
    pbc: object = True


class LazyData:
    """`n` structures read when asked for: `read(i)` gives the `OperatorData` of structure i.

    What the fits take in place of a list when the blocks of all the structures do not fit in
    memory (the triangular factors of a fit do not grow with the structures; a list of them
    does).  The last ones read are kept up to `budget` bytes of blocks, the newest always: a set
    within the budget is read once, a larger one once for each pass a fit makes over it, in
    order (`fit_onsite` one, `fit_offsite` two).  `reads` counts the calls of `read`."""

    def __init__(self, n, read, budget=0.0):
        self.n, self.read, self.budget, self.reads = int(n), read, float(budget), 0
        self._kept, self._bytes = OrderedDict(), 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        i = range(self.n)[i]
        if i in self._kept:
            self._kept.move_to_end(i)
            return self._kept[i][0]
        d = self.read(i)
        self.reads += 1
        size = sum(np.asarray(v).nbytes for b in d.blocks.values() for v in b.values())
        self._kept[i] = (d, size)
        self._bytes += size
        while len(self._kept) > 1 and self._bytes > self.budget:
            self._bytes -= self._kept.popitem(last=False)[1][1]
        return d

    def __iter__(self):
        return (self[i] for i in range(self.n))
