"""ace-jax: fit and evaluate ACE interatomic potentials in Python/JAX.

Build a basis (`ace_jax.basis`), fit it (`ace_jax.fit.pipeline`, or `aj fit`) and
evaluate it (`ACECalculator`, `export_lammps`). Documentation:
https://acesuit.github.io/ace-jax/
"""
import importlib.metadata as _md

try:
    __version__ = _md.version("ace-jax")
except _md.PackageNotFoundError:      # a source tree that was never installed
    __version__ = "0+unknown"
from .eval import (ACEModel, load, highest_precision, site_descriptors, species_indices,
                   sparse_graph, dense_graph)

__all__ = ["__version__", "ACEModel", "load", "highest_precision", "site_descriptors", "species_indices",
           "sparse_graph", "dense_graph", "ACECalculator", "GPCalculator"]


def __getattr__(name):
    # ase-dependent calculators kept out of the core import path
    if name == "ACECalculator":
        from .calc.point import ACECalculator
        return ACECalculator
    if name == "GPCalculator":
        from .calc.gp import GPCalculator
        return GPCalculator
    raise AttributeError(name)
