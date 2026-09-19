from .model import (ACEModel, calibrate_edge_a, fold_readout, highest_precision,
                    pool_dense, pool_sparse, with_edge_a_kind)
from .api import site_descriptors, species_indices
from .io import load
from .nlist import DenseGraph, SparseGraph, dense_graph, dense_to_sparse, sparse_graph

__all__ = ["calibrate_edge_a", "with_edge_a_kind", "fold_readout", "ACEModel", "load",
           "highest_precision", "pool_sparse", "pool_dense", "SparseGraph", "DenseGraph",
           "sparse_graph", "dense_graph", "dense_to_sparse", "site_descriptors", "species_indices"]

__all__.append("ACECalculator")


def __getattr__(name):
    # ase is only needed for the calculator; keep it out of the core import path
    if name == "ACECalculator":
        from ..calc.point import ACECalculator
        return ACECalculator
    raise AttributeError(name)
