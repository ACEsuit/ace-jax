"""Composable per-quantity weight factors for fitting.

A `WeightFactor` maps (config metadata, quantity name) -> a scalar multiplier.
`compose` chains several factors into a single function returning their
PRODUCT; a factor that has no opinion about a given (meta, quantity) pair
returns 1.0 (neutral), so factors can be freely combined and reordered.

Quantity names are the short forms used throughout the fitting stack:
"E" (energy), "F" (forces), "V" (virial).

These built-ins are shipped FROZEN (`learnable=False`, `params()` returns
None): a future VarOpt consumer may want to learn some of these factors
(e.g. per-config-type weights), so the protocol carries the hooks now, but
no learning is wired up here -- see Task 13+.
"""
from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class WeightFactor(Protocol):
    """A single weight contribution. `learnable`/`params()` are forward
    hooks for a future optimiser; frozen built-ins just return None."""

    learnable: bool

    def weight(self, meta: dict, quantity: str) -> float:
        ...

    def params(self):
        return None


class Structural:
    """1 / n_atoms**exp[quantity], the classic per-structure size weighting.
    `meta["n_atoms"]` absent -> neutral 1.0 (no structural information)."""

    learnable = False

    def __init__(self, exp: Optional[dict] = None):
        self.exp = {"E": 0.5, "V": 0.5, "F": 0.0} if exp is None else dict(exp)

    def weight(self, meta: dict, quantity: str) -> float:
        e = self.exp.get(quantity, 0.0)
        if e == 0.0:
            return 1.0
        n = meta.get("n_atoms")
        if n is None:
            return 1.0
        return float(n) ** (-e)

    def params(self):
        return None


class Quantity:
    """A flat per-quantity multiplier, e.g. {"E": 1.0, "F": 0.1, "V": 0.01}.
    Quantity absent from `w` -> neutral 1.0."""

    learnable = False

    def __init__(self, w: dict):
        self.w = dict(w)

    def weight(self, meta: dict, quantity: str) -> float:
        return float(self.w.get(quantity, 1.0))

    def params(self):
        return None


class ConfigType:
    """Per-config-type per-quantity multiplier, looked up via
    `meta[key]` (default key "config_type"). The lookup is CASE-INSENSITIVE,
    matching `data.py`'s `_get`/`type_idx` resolution, so a config_type that
    differs only in case from a table key still resolves to that entry rather
    than silently falling back to `default`. Unknown/absent config_type falls
    back to `default`; an unknown quantity within the resolved entry is
    neutral 1.0."""

    learnable = False

    def __init__(self, table: dict, key: str = "config_type", default: Optional[dict] = None):
        self.table = {str(k).lower(): v for k, v in table.items()}
        self.key = key
        self.default = {} if default is None else dict(default)

    def weight(self, meta: dict, quantity: str) -> float:
        ctype = meta.get(self.key)
        entry = self.table.get(str(ctype).lower(), self.default) if ctype is not None else self.default
        return float(entry.get(quantity, 1.0))

    def params(self):
        return None


class PerConfig:
    """Reads a scalar override straight from the config's own metadata,
    e.g. a user-supplied per-structure weight. `meta[key]` absent -> 1.0."""

    learnable = False

    def __init__(self, key: str = "weight"):
        self.key = key

    def weight(self, meta: dict, quantity: str) -> float:
        return float(meta.get(self.key, 1.0))

    def params(self):
        return None


class Custom:
    """Wraps an arbitrary `fn(meta, quantity) -> float` as a WeightFactor."""

    learnable = False

    def __init__(self, fn: Callable[[dict, str], float]):
        self.fn = fn

    def weight(self, meta: dict, quantity: str) -> float:
        return float(self.fn(meta, quantity))

    def params(self):
        return None


def compose(factors) -> Callable[[dict, str], float]:
    """Chain `factors` into a single fn(meta, quantity) -> float returning
    their PRODUCT. An empty factor list is neutral (always 1.0)."""
    factors = list(factors)

    def f(meta: dict, quantity: str) -> float:
        out = 1.0
        for factor in factors:
            out *= factor.weight(meta, quantity)
        return out

    return f
