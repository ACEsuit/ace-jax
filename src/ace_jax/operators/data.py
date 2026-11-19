"""One structure with the blocks of its operators: what the block fit reads and the readers of
any file format return."""
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
