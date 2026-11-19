"""Time against system size of the steps from positions to the bands at one k-point, for a
DeepH-layout structure (a supercell from `tile_deeph.py`):
    python bench/ham/size_time.py <structure dir> <out.jsonl> [model=<model.npz>] [threads=1] [repeats=3]
model=: ace-jax's prediction of every H block (neighbour lists included; `model_time.py`).
Always, from the structure's own H and S blocks (the same for any code's prediction): the Bloch
sums H(k) and S(k) at a general k-point (`ham.operator.BlockSum`, on the device JAX runs on) and
the generalised Hermitian eigenvalues, by LAPACK through SciPy on a CPU and by
`ham.operator.eigvals` on a GPU.  HAM_X64=0: float32 (prediction and Bloch sums; the eigensolver
too on a GPU).  One JSON record per run; the smallest of `repeats` after a first call.  On a GPU the record
also holds the peak device memory after each step (`*_peak_bytes`, XLA's allocator statistics;
they need XLA_PYTHON_CLIENT_PREALLOCATE=false to mean what the program used)."""
import json
import os
import sys
import time

import jax

X64 = os.environ.get("HAM_X64", "1") != "0"
jax.config.update("jax_enable_x64", X64)
import numpy as np

from ace_jax.ham import deeph
from ace_jax.ham.operator import BlockSum, HamiltonianPredictor, eigvals

src, out = sys.argv[1:3]
kw = dict(x.split("=") for x in sys.argv[3:])
threads, repeats = int(kw.get("threads", 1)), int(kw.get("repeats", 3))


def best(f):
    f()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return min(ts), r


d = deeph.read(src)
gpu = jax.default_backend() != "cpu"
peak = lambda: int((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)) if gpu else 0
rec = dict(atoms=len(d.numbers), orbitals=int(sum(d.basis.n_orb(int(z)) for z in d.numbers)), blocks=len(d.blocks["H"]), backend=jax.default_backend(),
           precision="float64" if X64 else "float32", threads=threads)
if "model" in kw:
    p = HamiltonianPredictor.from_file(kw["model"], threads=threads)
    args = (d.positions, d.cell, d.numbers)
    rec["predict_s"], _ = best(lambda: jax.block_until_ready((p.onsite(*args), p.offsite(*args))))
    rec["model"] = os.path.basename(kw["model"])
    rec["predict_peak_bytes"] = peak()
bs = BlockSum(sorted(d.blocks["H"]), d.numbers, d.basis)
k = np.array([[0.13, 0.27, -0.4]])
dt = np.float64 if X64 else np.float32
H = {key: v.astype(dt) for key, v in d.blocks["H"].items()}
S = {key: v.astype(dt) for key, v in d.blocks["S"].items()}
t_h, Hk = best(lambda: jax.block_until_ready(bs.matrices(H, k)))
t_s, Sk = best(lambda: jax.block_until_ready(bs.matrices(S, k)))
rec["bloch_s"] = t_h + t_s
rec["bloch_peak_bytes"] = peak()
if gpu:
    rec["eig_s"], e = best(lambda: np.asarray(eigvals(Hk, Sk)))
else:
    from scipy.linalg import eigh
    h, s = np.asarray(Hk[0], np.complex128), np.asarray(Sk[0], np.complex128)
    if len(h) > 3000:                                     # minutes on one core: once
        t = time.perf_counter(); e = eigh(h, s, eigvals_only=True); rec["eig_s"] = time.perf_counter() - t
    else:
        rec["eig_s"], e = best(lambda: eigh(h, s, eigvals_only=True))
rec["e_min"] = float(np.min(e))
rec["eig_peak_bytes"] = peak()
print(json.dumps(rec), flush=True)
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
