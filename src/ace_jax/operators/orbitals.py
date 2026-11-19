"""The atomic-orbital basis of an operator, as data: per species, the angular
momentum of each shell.  Carried in data sets and model files; no shell tables in code."""
from typing import NamedTuple

import numpy as np


class OrbitalBasis(NamedTuple):
    """shells: ((z, (l, l, ...)), ...) sorted by atomic number z; a shell of
    angular momentum l holds 2l+1 orbitals in the order m = -l..l of
    `eval/harmonics.py`."""

    shells: tuple

    @classmethod
    def of(cls, shells):
        return cls(tuple((int(z), tuple(int(l) for l in ls)) for z, ls in sorted(dict(shells).items())))

    def ls(self, z):
        for zz, ls in self.shells:
            if zz == int(z):
                return ls
        raise KeyError(f"no orbital basis for atomic number {int(z)}")

    @property
    def species(self):
        return tuple(z for z, _ in self.shells)

    def n_orb(self, z):
        return sum(2 * l + 1 for l in self.ls(z))

    def offsets(self, z):
        """Start of each shell of species z in its atom block, and the total, (n_shell + 1,)."""
        return np.cumsum([0] + [2 * l + 1 for l in self.ls(z)])

    def shell_slice(self, z, s):
        o = self.offsets(z)
        return slice(int(o[s]), int(o[s + 1]))

    @property
    def lmax(self):
        return max(l for _, ls in self.shells for l in ls)
