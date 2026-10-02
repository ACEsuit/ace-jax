"""Extra v3 crack realisations labelled with MACE-MH-1 on a LOCAL GPU (same recipe and seeding scheme as
modal/modal_big3.py, realisations 2..R-1; cracks only).  Resumes from the output file.

    MACE_MODEL=... python big3_local.py <out.xyz> [--first R0] [--last R1] [--test]

--test builds and labels one crack cell (no write) and prints the peak GPU memory.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import big2  # noqa: E402
import big3  # noqa: E402

out_path = sys.argv[1]
argv = sys.argv[2:]
first = int(argv[argv.index("--first") + 1]) if "--first" in argv else 2
last = int(argv[argv.index("--last") + 1]) if "--last" in argv else 9
props = json.load(open(os.environ.get("BIG2_PROPS", "/storage/eng/essswb/acegp-data/cantor/big3/big2_props.json")))
a0, C, gamma = props["a0"], tuple(props["C"]), props["gamma111"]

import torch  # noqa: E402
from ase.io import read, write  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

calc = MACECalculator(model_paths=os.environ["MACE_MODEL"], device="cuda", default_dtype="float64",
                      head="matpes_r2scan")
log = lambda s: print(s, flush=True)
if "--test" in argv:
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(12345)
    at = big3.crack(rng, a0, C, gamma, 1.2)
    t = time.time()
    res = big2.relax_label(at, calc, rng, log=log)
    log(f"test: {len(at)} atoms, {len(res)} labelled configs, {time.time() - t:.0f} s, "
        f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    sys.exit(0)
out = read(out_path, ":") if os.path.exists(out_path) else []
todo = [(r, k) for r in range(first, last + 1) for k in (1.1, 1.2, 1.3)]
done = len(out) // 3
log(f"{done} cells already done; {len(todo) - done} to go")
for r, k in todo[done:]:
    rng = np.random.default_rng(27 + 100 * r + int(10 * k) + len("crack"))     # modal_big3 seeding scheme
    at = big3.crack(rng, a0, C, gamma, k)
    for b in big2.relax_label(at, calc, rng, log=log):
        b.info["realisation"] = int(r)
        out.append(b)
    write(out_path, out, format="extxyz")
log(f"wrote {len(out)} configs")
