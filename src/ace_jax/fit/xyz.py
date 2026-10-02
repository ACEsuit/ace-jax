"""extxyz -> plain frames, read with libAtoms `extxyz` (the C parser), not ase.io.

Why not ase.io: since 3.23 ASE moves info/array keys that are calculator
property names (`energy`, `forces`, `stress`, ...) into a SinglePointCalculator,
so a lookup in `atoms.info` / `atoms.arrays` finds nothing -- a fit using those
label names silently had no energies or forces.  Here every key comes back
under the name it was written with, and ace-jax owns the conventions:

* 9-element info values: `extxyz` returns them as 3x3 in Fortran order.  Keys
  in ASE's SPECIAL_3_3_KEYS (`virial`, `stress`) stay 3x3 -- that is what ASE
  writes them as -- and every other key is returned flat, in file order, as
  ase.io did (callers reshape).  Files read identically to before.
* `_JSON <json>` info values are ASE's encoding of 2-D (non-special) arrays,
  e.g. a (3, 3) `dft_virial`; neither extxyz parser decodes them, so they are
  decoded here.  A `_JSON` payload that is not valid JSON raises.
* No `Lattice`: extxyz reports pbc T with a zero cell; ASE (and a neighbour
  list) wants a non-periodic cluster, so a zero cell means pbc F.

`read_extxyz` never falls back to ase.io: a file the C parser rejects is retried
with extxyz's pure-Python parser, and if that fails too the error names the file.
"""
import json
from typing import Any, NamedTuple

import numpy as np

SPECIAL_3x3 = {"virial", "stress"}          # ase.io.extxyz.SPECIAL_3_3_KEYS minus Lattice


class Frame(NamedTuple):
    numbers: np.ndarray                     # (n,) atomic numbers
    positions: np.ndarray                   # (n, 3)
    cell: np.ndarray                        # (3, 3), rows are lattice vectors (ASE convention)
    pbc: np.ndarray                         # (3,) bool
    info: dict[str, Any]                    # per-config values, keys as written
    arrays: dict[str, np.ndarray]           # per-atom columns, keys as written (minus species/pos/Z)


def _info_value(key, v, where):
    if isinstance(v, str) and v.startswith("_JSON "):
        try:
            return np.asarray(json.loads(v[len("_JSON "):]))
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}: info {key!r} is an unreadable _JSON value {v[:80]!r}") from e
    if isinstance(v, np.ndarray) and v.shape == (3, 3) and key not in SPECIAL_3x3:
        return v.ravel(order="F")           # back to file order (extxyz reshapes 9-vectors, Fortran)
    return v


def _numbers(arrays, where):
    if "Z" in arrays:
        return np.asarray(arrays["Z"], int)
    from ase.data import atomic_numbers
    try:
        return np.array([atomic_numbers[str(s)] for s in arrays["species"]], int)
    except KeyError as e:
        raise ValueError(f"{where}: unknown species {e.args[0]!r} (and no Z column)") from None


def _frame(f, where):
    arrays = dict(f.arrays)
    if "pos" not in arrays:
        raise ValueError(f"{where}: no pos column in Properties")
    numbers = _numbers(arrays, where)
    positions = np.asarray(arrays.pop("pos"), float)
    arrays.pop("species", None); arrays.pop("Z", None)
    cell = np.asarray(f.cell, float).T      # extxyz: Lattice read column-major; ASE rows = vectors
    pbc = np.asarray(f.pbc, bool) if cell.any() else np.zeros(3, bool)
    info = {k: _info_value(k, v, where) for k, v in f.info.items()}
    return Frame(numbers, positions, cell, pbc, info, arrays)


def read_raw(path):
    """The extxyz library's own frames (info/arrays exactly as written), for writing back."""
    import extxyz
    path = str(path)
    try:
        return list(extxyz.iread_dicts(path, use_cextxyz=True))
    except Exception:                       # C parser is strict; the Python grammar is the reference
        try:
            return list(extxyz.iread_dicts(path, use_cextxyz=False))
        except Exception as e:
            raise ValueError(f"{path}: not a readable extxyz file ({type(e).__name__}: {e})") from e


def write_raw(path, frames):
    """Write read_raw-style frames (libAtoms extxyz; floats at its %16.8f)."""
    import extxyz
    extxyz.write_dicts(str(path), frames)


def read_extxyz(path):
    """All frames of an extxyz file, as `Frame`s (see the module docstring)."""
    return [_frame(f, f"{path} frame {i}") for i, f in enumerate(read_raw(path))]
