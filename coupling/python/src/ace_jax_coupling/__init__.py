"""EquivariantTensors.jl SO(3) coupling tables for ace-jax, from a
juliac-compiled library (no Julia needed at runtime)."""
from ._api import MAX_ORDER, RawCoupling, couple_raw
from ._loader import ABI_VERSION, CouplingLibError, build_info

__version__ = "0.2.0"
__all__ = ["ABI_VERSION", "MAX_ORDER", "CouplingLibError", "RawCoupling", "build_info", "couple_raw", "__version__"]
