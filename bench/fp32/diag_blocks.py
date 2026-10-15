"""Step-2 open issue: LAMMPS step time of a lammps-jax bundle against its dense/matrix block_rows
(one model, one size), to find where the large-N superlinearity of ACE o4d16 comes from.

    uv run --no-sync python bench/fp32/diag_blocks.py MODEL N DTYPE BLOCK [BLOCK ...]
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bundle_prec import LMP_JAX, PJRT, cell, child, lammps  # noqa: E402

EXPORT = """
import json, sys, jax
jax.config.update("jax_enable_x64", True)
from ase.io import read
from ace_jax.eval import load
from ace_jax.export.lammps import export_lammps, neighbour_capacity
model, meta, _ = load(sys.argv[1]); at = read(sys.argv[2])
cap = neighbour_capacity(at, float(meta["rcut"]), skin=1.0)
b = export_lammps(model, meta, sys.argv[3], max_atoms=cap["max_atoms"], max_edges=cap["max_edges"],
                  k_dense=cap["k_dense"], max_owned=cap["max_owned"], max_neighbors=cap["max_neighbors"],
                  dtype=sys.argv[4], type_elements=json.loads(sys.argv[7]), layout=sys.argv[6],
                  block_rows=int(sys.argv[5]) or None)
print(json.dumps({k: b["ace_jax"][k] for k in ("layout", "block_rows")}))
"""


def main():
    model, n, dtype, blocks = sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(b) for b in sys.argv[4:]]
    system = os.environ.get("DIAG_SYSTEM", "Si")
    at = cell(n, system)
    els = sorted(set(at.get_chemical_symbols()), key=lambda e: __import__("ase.data").data.atomic_numbers[e])
    zs = [int(__import__("ase.data").data.atomic_numbers[e]) for e in els]
    out = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("BUNDLE_TMP")) as d:
        d = Path(d)
        from ase.io import write
        write(d / "at.extxyz", at)
        for br in blocks:
            try:
                info = child(EXPORT, model, d / "at.extxyz", d / "b.json", dtype, br, os.environ.get("DIAG_LAYOUT", "matrix"),
                             json.dumps(zs))
            except RuntimeError as e:
                out.append({"block_rows": br, "error": str(e)[-300:]})
                continue
            r = lammps(LMP_JAX, f"pair_style jax/kk {PJRT}\npair_coeff * * {d / 'b.json'}", at, elements=els)
            row = {"n": len(at), "dtype": dtype, **info, "requested": br,
                   **({"t_step_ms": 1e3 * r["t_step"]} if "t_step" in r else {"error": r["error"][-300:]})}
            print(json.dumps(row), file=sys.stderr, flush=True)
            out.append(row)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
