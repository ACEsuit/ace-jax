"""Time of predicting every H (and S) block of a structure with a model file (`operators.io`):
    python bench/ham/model_time.py <model.npz> <structure: operator cache or DeepH directory> [threads=1] [repeats=5]
        [save=<blocks.npz>] [ref=<blocks.npz>]
The smallest of `repeats` calls after one that compiles; neighbour lists included; on the device
JAX runs on (JAX_PLATFORMS).  HAM_X64=0 in the environment: float32.  save= writes the predicted
H blocks, ref= prints the difference from blocks another run saved (float32 against float64)."""
import os
import sys
import time

import jax

X64 = os.environ.get("HAM_X64", "1") != "0"
jax.config.update("jax_enable_x64", X64)
import numpy as np

from ace_jax.ham.cli import structures
from ace_jax.ham.operator import HamiltonianPredictor

kw = {k: (v if k in ("save", "ref") else int(v)) for k, v in (x.split("=") for x in sys.argv[3:])}
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
print(f"ace-jax {jax.default_backend()} {'float64' if X64 else 'float32'} threads {threads}: {len(at)} atoms, {n_blocks} blocks, {n_coef} coefficients; "
      f"{1e3 * min(ts):.1f} ms per structure, {1e6 * min(ts) / len(at):.0f} us per atom", flush=True)
ch = "H" if "H" in p.channels else p.channels[0]
H = {**out[1][ch], **out[0][ch]}
if "save" in kw:
    np.savez(kw["save"], **{",".join(map(str, k)): np.asarray(v) for k, v in H.items()})
if "ref" in kw:
    z = np.load(kw["ref"])
    d = np.concatenate([(np.asarray(v, float) - z[",".join(map(str, k))]).ravel() for k, v in H.items()])
    print(f"  against {kw['ref']}: mean |difference| {1e3 * np.abs(d).mean():.5f}, largest {1e3 * np.abs(d).max():.4f} (1e-3 of the data's unit)")
