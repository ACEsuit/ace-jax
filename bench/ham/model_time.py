"""Time of predicting every H (and S) block of a structure with a model file (`operators.io`):
    python bench/ham/model_time.py <model.npz> <structure: operator cache or DeepH directory> [threads=1] [repeats=5]
The smallest of `repeats` calls after one that compiles; neighbour lists included."""
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from ace_jax.ham.cli import structures
from ace_jax.ham.operator import HamiltonianPredictor

kw = {k: int(v) for k, v in (x.split("=") for x in sys.argv[3:])}
threads = kw.get("threads", 1)
p = HamiltonianPredictor.from_file(sys.argv[1], threads=threads)
_, at, _ = structures(sys.argv[2], 0)[0]
args = (at.positions, np.asarray(at.cell), at.numbers)


def run():
    return p.onsite(*args), p.offsite(*args)


run()
ts = []
for _ in range(kw.get("repeats", 5)):
    t = time.perf_counter(); out = run(); ts.append(time.perf_counter() - t)
n_blocks = sum(len(b) for o in out for b in list(o.values())[:1])
n_coef = sum(np.asarray(c).size for m in (p.onsite.model, p.offsite.model) for c in m.coef.values())
print(f"ace-jax {jax.default_backend()} threads {threads}: {len(at)} atoms, {n_blocks} blocks, {n_coef} coefficients; "
      f"{1e3 * min(ts):.1f} ms per structure, {1e6 * min(ts) / len(at):.0f} us per atom", flush=True)
