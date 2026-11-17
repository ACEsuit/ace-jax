"""Stubs for modules that moved (docs/dev/specs/2026-11-17-package-structure-design.md).

The old path is a module whose `__getattr__` warns and serves the new module's public names:
`from ace_jax.fit.stats import linear_statistics` keeps working until the stub is removed (0.8).
Names with a leading underscore are not served: they were never API."""
import importlib
import warnings

REMOVED_IN = "0.8"


def moved(old, new, *also):
    """`also`: modules that took some of the old module's names when it was split."""
    def __getattr__(name):
        if name.startswith("_"):
            raise AttributeError(f"module {old!r} has no attribute {name!r} (it moved to {new})")
        for target in (new, *also):
            mod = importlib.import_module(target)
            if hasattr(mod, name):
                warnings.warn(f"{old} moved to {target}; the old name will be removed in ace-jax {REMOVED_IN}",
                              DeprecationWarning, stacklevel=2)
                return getattr(mod, name)
        raise AttributeError(f"module {old!r} has no attribute {name!r} (it moved to {new})")
    return __getattr__
