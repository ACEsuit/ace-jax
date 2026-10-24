"""Batched evaluation benchmark (docs/dev/batched-evaluation-results.md): µs per atom of
`BatchCalculator.calculate` against the number of cells per call, for rattled Si cells of 2, 8, 64 and 216
atoms, against `ACECalculator` one cell per call.  One JSON line per point on stdout (and --out).

    python bench/batch/run.py --model <o4d12 model.npz> --dtype float32 float64 [--builder auto jax msn host] [--ffi]

The timed call is the whole `calculate`: packing on the host, the device neighbour build, the model, and
the copy back.  Each point is the median of --reps calls after one warm-up (the compile)."""
import argparse
import json
import platform
import statistics
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from ase.build import bulk

from ace_jax.batch import BatchCalculator, bucket_size
from ace_jax.calc.point import ACECalculator

CELLS = {2: lambda: bulk("Si", "diamond", 5.43), 8: lambda: bulk("Si", "diamond", 5.43, cubic=True),
         64: lambda: bulk("Si", "diamond", 5.43, cubic=True).repeat(2),
         216: lambda: bulk("Si", "diamond", 5.43, cubic=True).repeat(3)}


def cells(n, count, seed=0):
    out = []
    for k in range(count):
        a = CELLS[n]()
        a.rattle(0.05, seed=seed + k)
        out.append(a)
    return out


def timed(f, reps):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 8, 64, 216])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    ap.add_argument("--dtype", nargs="+", default=["float64"])
    ap.add_argument("--builder", nargs="+", default=["auto"])
    ap.add_argument("--ffi", action="store_true", help="the C++ kernel (CPU, float64)")
    ap.add_argument("--max-atoms", type=int, default=2 ** 18, help="skip points with more atoms per call")
    ap.add_argument("--loop-cells", type=int, default=32, help="cells timed through ACECalculator")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    host = dict(host=platform.node(), backend=jax.default_backend(), device=str(jax.devices()[0].device_kind))
    out = open(a.out, "a") if a.out else None

    def emit(rec):
        line = json.dumps({**host, **rec})
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()

    for dtype in a.dtype:
        dt = None if dtype == "float64" else np.float32
        for n in a.sizes:
            calc = ACECalculator(a.model, dtype=dt, ffi=a.ffi)
            loop = cells(n, a.loop_cells, seed=10_000)

            def run_loop(loop=loop, calc=calc):
                for at in loop:
                    at.calc = calc
                    at.get_forces()
            t = timed(run_loop, max(1, a.reps // 2))
            emit(dict(kind="loop", dtype=dtype, n=n, cells=len(loop), ffi=a.ffi, s=t,
                      us_per_atom=1e6 * t / (n * len(loop))))
            for builder in a.builder:
                for B in a.batch:
                    if B * n > a.max_atoms:
                        continue
                    bc = BatchCalculator(a.model, dtype=dt, ffi=a.ffi, builder=builder,
                                         max_batch_atoms=B * bucket_size(n))
                    ats = cells(n, B)
                    t = timed(lambda bc=bc, ats=ats: bc.calculate(ats), a.reps)
                    tm = bc.last_timing
                    emit(dict(kind="batch", dtype=dtype, n=n, cells=B, builder=builder, ffi=a.ffi, s=t,
                              us_per_atom=1e6 * t / (n * B), batches=tm["batches"],
                              pack_s=tm["pack_s"], batched_s=tm["batched_s"]))


if __name__ == "__main__":
    main()
