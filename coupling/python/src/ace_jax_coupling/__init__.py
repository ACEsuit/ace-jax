"""EquivariantTensors.jl SO(3) coupling tables for ace-jax, from a
juliac-compiled library (no Julia needed at runtime)."""
from ._api import MAX_L, MAX_ORDER, RawCoupling, RawEquivariantCoupling, couple_equivariant_raw, couple_raw
from ._loader import ABI_VERSION, CouplingLibError, build_info

__version__ = "0.2.0"
__all__ = ["ABI_VERSION", "MAX_L", "MAX_ORDER", "CouplingLibError", "RawCoupling", "RawEquivariantCoupling",
           "build_info", "couple_equivariant_raw", "couple_raw", "__version__"]
