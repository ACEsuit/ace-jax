"""radial_table= on the CPU: the calculator's compiled skin step, off vs on.

    PYTHONPATH=bench:src taskset -c 2 python bench/perf/radial_table.py <model> <system> <n> \
        [--radial-table 4000] [--reps 30] [--threads 1]
    ... --accuracy           # both variants in one process: |dE|/atom, max|dF| / max|F|

As `bench/perf/cpu_gap.py` (perf/cpu-product-basis) times it: an
`ACECalculator` (dense, skin 1.0, lean) on `supercell(system, n)`, a first call
to compile, MD-like displaced calls, then the compiled skin step alone with
its inputs on the device (`step_us` = median per atom-step, microseconds).
`--threads 1` adds `--xla_cpu_multi_thread_eigen=false`; the thread pool is
the taskset CPU set.  Interleave variants across processes for A/B rounds.
Prints one JSON object.
"""
import argparse
import json
import os
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--radial-table", type=int, default=0, help="intervals; 0 = off")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.threads == 1:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                                   + " --xla_cpu_multi_thread_eigen=false").strip()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from ase.calculators.calculator import all_changes

    from ace_jax.calc.point import ACECalculator
    from scaling.structures import supercell

    at = supercell(a.system, a.n)
    rng = np.random.default_rng(0)
    at.positions += rng.normal(0, 0.05, at.positions.shape)      # off-lattice: forces nonzero
    out = {"model": os.path.basename(a.model), "system": a.system, "n": a.n,
           "radial_table": a.radial_table or None, "threads": a.threads, "tag": a.tag,
           "cpus": sorted(os.sched_getaffinity(0)),
           "loadavg_before": open("/proc/loadavg").read().split()[:3]}

    def calc_for(rt):
        t0 = time.perf_counter()
        c = ACECalculator(a.model, layout="dense", skin=1.0, radial_table=rt or None)
        c.calculate(at, ["energy", "forces", "stress"], all_changes)
        return c, time.perf_counter() - t0

    if a.accuracy:
        c0, _ = calc_for(0)
        E0, F0, S0 = c0.results["energy"], c0.results["forces"].copy(), c0.results["stress"].copy()
        t0 = time.perf_counter()
        c1, first = calc_for(a.radial_table or 4000)
        out.update(build_and_first_s=first, info=c1.radial_table,
                   dE_per_atom=abs(c1.results["energy"] - E0) / a.n,
                   dE_rel=abs(c1.results["energy"] - E0) / abs(E0),
                   dF_max=float(np.abs(c1.results["forces"] - F0).max()),
                   F_max=float(np.abs(F0).max()),
                   dS_max=float(np.abs(c1.results["stress"] - S0).max()),
                   S_max=float(np.abs(S0).max()))
        out["dF_rel"] = out["dF_max"] / out["F_max"]
        print(json.dumps(out))
        return
    calc, out["first_s"] = calc_for(a.radial_table)
    st = calc._skin_state
    u = jax.device_put(st.displacements(at.positions, np.dtype("float64")))
    step = calc._step()
    for _ in range(3):
        jax.block_until_ready(step(u, st.arrays, K=st.K))
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); jax.block_until_ready(step(u, st.arrays, K=st.K))
        ts.append(time.perf_counter() - t0)
    out["step_us"] = statistics.median(ts) / a.n * 1e6
    out["step_us_min"] = min(ts) / a.n * 1e6
    out["loadavg_after"] = open("/proc/loadavg").read().split()[:3]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
