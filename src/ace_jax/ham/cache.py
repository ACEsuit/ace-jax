"""The frozen operator cache: one structure's `OperatorData` as one npz.

It is what the Quoll ingestion script (`ham/julia/ingest.jl`) writes and what
fitting reads, so neither h5py nor Julia is needed after ingestion.  Blocks are
in ace-jax's orbital order.  Layout: `cell` (3, 3, vectors in rows),
`positions`, `numbers`, `meta_json`, and per operator channel `<name>__keys`
(n, 5) int64 (Rx, Ry, Rz, i, j; 0-based atoms), `<name>__off` (n + 1,) and
`<name>__data`, the row-major blocks end to end."""
import json

import numpy as np

from .deeph import OperatorData
from .orbitals import OrbitalBasis

SCHEMA = 1


def save(path, data, meta=None):
    """Write `data`; `meta` (a JSON-able dict, e.g. the Quoll commit and source
    format) is kept under "source"."""
    out = {"cell": np.asarray(data.cell, np.float64), "positions": np.asarray(data.positions, np.float64),
           "numbers": np.asarray(data.numbers, np.int64)}
    for name, blocks in data.blocks.items():
        if "__" in name:
            raise ValueError(f"operator channel name {name!r} must not contain '__'")
        keys = sorted(blocks)
        out[f"{name}__keys"] = np.asarray(keys, np.int64).reshape(-1, 5)
        out[f"{name}__off"] = np.cumsum([0] + [blocks[k].size for k in keys]).astype(np.int64)
        out[f"{name}__data"] = (np.concatenate([np.asarray(blocks[k], np.float64).ravel() for k in keys])
                                if keys else np.zeros(0))
    out["meta_json"] = np.frombuffer(json.dumps({
        "schema": SCHEMA, "kind": "ace-jax operator cache", "channels": list(data.blocks),
        "basis": [[z, list(ls)] for z, ls in data.basis.shells], "orbital_order": "m=-l..l (ace-jax)",
        "source": meta or {}}).encode(), np.uint8)
    with open(path, "wb") as f:
        np.savez_compressed(f, **out)


def load(path):
    """(`OperatorData`, meta) from a cache file."""
    z = np.load(path)
    meta = json.loads(bytes(z["meta_json"]).decode())
    if meta.get("schema") != SCHEMA or meta.get("kind") != "ace-jax operator cache":
        raise ValueError(f"{path}: not an ace-jax operator cache of schema {SCHEMA} (found {meta.get('schema')!r})")
    numbers = np.asarray(z["numbers"], np.int64)
    basis = OrbitalBasis.of({int(zz): ls for zz, ls in meta["basis"]})
    blocks = {}
    for name in meta["channels"]:
        keys, off, flat = z[f"{name}__keys"], z[f"{name}__off"], z[f"{name}__data"]
        out = {}
        for k, (Rx, Ry, Rz, i, j) in enumerate(keys):
            shape = (basis.n_orb(numbers[i]), basis.n_orb(numbers[j]))
            if off[k + 1] - off[k] != shape[0] * shape[1]:
                raise ValueError(f"{path}: block {name} {tuple(int(v) for v in keys[k])} does not match the orbital bases")
            out[(int(Rx), int(Ry), int(Rz), int(i), int(j))] = flat[off[k]:off[k + 1]].reshape(shape)
        blocks[name] = out
    return OperatorData(np.asarray(z["cell"]), np.asarray(z["positions"]), numbers, basis, blocks), meta
