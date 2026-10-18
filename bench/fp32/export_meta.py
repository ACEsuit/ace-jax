"""Export one bundle as bundle_prec.py does and print its sizing metadata (open issues of step 2)."""
import json
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from bundle_prec import cell  # noqa: E402

from ace_jax.eval import load  # noqa: E402
from ace_jax.export.lammps import export_lammps, neighbour_capacity  # noqa: E402

model, meta, _ = load(sys.argv[1])
at = cell(int(sys.argv[2]))
cap = neighbour_capacity(at, float(meta["rcut"]), skin=1.0)
b = export_lammps(model, meta, sys.argv[4], max_atoms=cap["max_atoms"], max_edges=cap["max_edges"],
                  k_dense=cap["k_dense"], max_owned=cap["max_owned"], max_neighbors=cap["max_neighbors"],
                  dtype=sys.argv[3], type_elements=[14])
aj = b["ace_jax"]
print(json.dumps({"n": len(at), "dtype": sys.argv[3], "cap": cap,
                  **{k: aj.get(k) for k in ("layout", "block_rows", "device_memory", "dense_budget", "k_dense", "owned_rows")}}))
