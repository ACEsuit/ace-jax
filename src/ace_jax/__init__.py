"""ace-jax: fit and evaluate ACE interatomic potentials in Python/JAX.

Model definitions and splined radials are exported from ACEpotentials.jl to npz;
ace-jax consumes them (no Julia needed to fit or evaluate).  See README.
"""
from .eval import (ACEModel, load, highest_precision, site_descriptors, species_indices,
                   sparse_graph, dense_graph)

__all__ = ["ACEModel", "load", "highest_precision", "site_descriptors", "species_indices",
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
