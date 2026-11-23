"""The atomic-orbital basis of an operator, as data: per species, the angular
momentum of each shell.  Carried in data sets and model files; no shell tables in code."""
from typing import NamedTuple

import numpy as np


class Radials(dict):
    """`OrbitalBasis.radials`: a dictionary {z: (grid, values)}.  A type of its own so that the
    tables are one opaque leaf wherever a basis is a static field of a model, not arrays of it."""


class OrbitalBasis(NamedTuple):
    """shells: ((z, (l, l, ...)), ...) sorted by atomic number z; a shell of
    angular momentum l holds 2l+1 orbitals in the order m = -l..l of
    `eval/harmonics.py`.

    radials (optional): {z: (grid (n_r,), values (n_shells, n_r))}, the radial function of every
    shell of species z tabulated on a radial grid: function (s, m) of an atom at the origin is
    values[s](|r|) Y_{l m}(r / |r|), Y the real spherical harmonic of `eval/harmonics.py` (unit
    L2 norm on the sphere).  values is the radial function itself, u(r) / r, not u(r); the grid is
    increasing, in the length unit of the positions, and values are in the normalisation the
    data's coefficients or matrix elements refer to (an orbital has int values^2 r^2 dr = 1; an
    auxiliary function is whatever the code that made the data uses, and its charge is
    sqrt(4 pi) int values r^2 dr for l = 0).  Nothing here evaluates them: they are carried
    with the basis for integrals between the functions.

    Two bases are equal, and hash equal, when their shells are: models and data compare bases by
    their shells, with or without tables."""

    shells: tuple
    radials: dict = None

    @classmethod
    def of(cls, shells, radials=None):
        sh = tuple((int(z), tuple(int(l) for l in ls)) for z, ls in sorted(dict(shells).items()))
        if radials is None:
            return cls(sh)
        rad = {int(z): (np.asarray(g, np.float64), np.atleast_2d(np.asarray(v, np.float64))) for z, (g, v) in sorted(dict(radials).items())}
        for z, ls in sh:
            if z not in rad or rad[z][1].shape != (len(ls), len(rad[z][0])):
                raise ValueError(f"radials of Z = {z}: need a grid (n_r,) and values ({len(ls)}, n_r), one row per shell")
        return cls(sh, Radials((z, rad[z]) for z, _ in sh))

    def __eq__(self, other):
        return isinstance(other, OrbitalBasis) and self.shells == other.shells

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.shells)

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
