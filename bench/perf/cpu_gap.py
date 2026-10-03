"""ace-jax on the CPU: one MD-like calculator case, with the knobs the CPU-gap
study turns (docs/dev/cpu-gap-profile.md).

    PYTHONPATH=bench:src taskset -c 0 python bench/perf/cpu_gap.py <model> <system> <n> \
        [--dtype float64] [--reps 20] [--layout auto|dense|sparse] [--skin 1.0] \
        [--no-lean] [--edge-a-kind auto|gather|matmul] [--chunk N] [--threads T]

As `bench/scaling/run_standalone.py` runs the `acejax-*` lines: an
`ACECalculator` on `supercell(system, n)`, the first call undisplaced (compile),
then MD-like calls, each after displacing every atom by N(0, 1e-3 A).  Reports
medians of
  call_s  the whole calculator call (energy + forces + stress)
  model_s `last_timing["model_s"]`: the compiled step with its H2D / D2H
  step_s  the compiled skin step alone, inputs on the device (skin > 0)
and `us_per_atom` = call_s / n * 1e6.  XLA's CPU threading is set from
`--threads` before JAX starts: 1 adds `--xla_cpu_multi_thread_eigen=false`.
jaxlib 0.11.2 has no XLA flag for the intra-op pool size: `intra_op_parallelism_threads`
is a TF session option (`--intra_op_parallelism_threads` is a fatal unknown flag,
and without the dashes XLA silently ignores it), and the PjRt CPU client sizes its
pool from the process's CPU affinity -- so the thread count IS the `taskset` CPU
set (recorded, with the process's OS thread count after the run).  `--chunk` overrides `CHUNK_NODES` (the dense block size)
by rebinding the default of `energy_forces_virial_dense` in this process only.
Prints one JSON object.
"""
import argparse
import json
import os
import statistics
import sys
import time


def xla_threads(threads, extra=""):
    """XLA_FLAGS for `threads` XLA CPU threads (None: leave XLA's default)."""
    flags = os.environ.get("XLA_FLAGS", "")
    if threads == 1:
        flags += " --xla_cpu_multi_thread_eigen=false"
    return (flags + " " + extra).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--layout", default="auto")
    ap.add_argument("--skin", type=float, default=1.0)
    ap.add_argument("--no-lean", action="store_true")
    ap.add_argument("--edge-a-kind", default="auto")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--xla-extra", default="", help="extra XLA_FLAGS")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.environ["XLA_FLAGS"] = xla_threads(a.threads, a.xla_extra)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    jax.config.update("jax_enable_x64", a.dtype == "float64")
    import jax.numpy as jnp
    import numpy as np
    from ase.calculators.calculator import all_changes

    if a.chunk:
        from ace_jax.eval import edge_model
        f = edge_model.EdgeSiteModel.energy_forces_virial_dense
        d = list(f.__defaults__)
        d[0] = a.chunk                        # chunk is the first defaulted argument
        f.__defaults__ = tuple(d)
    from ace_jax.calc.point import ACECalculator
    from scaling.structures import supercell

    at = supercell(a.system, a.n)
    out = {"model": os.path.basename(a.model), "system": a.system, "n": a.n, "dtype": a.dtype,
           "reps": a.reps, "layout_req": a.layout, "skin": a.skin, "lean": not a.no_lean,
           "edge_a_kind_req": a.edge_a_kind, "chunk": a.chunk, "threads": a.threads,
           "cpus": sorted(os.sched_getaffinity(0)), "XLA_FLAGS": os.environ["XLA_FLAGS"],
           "jax": jax.__version__, "tag": a.tag, "loadavg_before": open("/proc/loadavg").read().split()[:3]}
    calc = ACECalculator(a.model, dtype=getattr(jnp, a.dtype), layout=a.layout, skin=a.skin,
                         lean=not a.no_lean, edge_a_kind=a.edge_a_kind)
    call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
    t0 = time.perf_counter(); call(); out["first_s"] = time.perf_counter() - t0
    out["energy"] = calc.results["energy"]
    rng = np.random.default_rng(0)
    ts, splits = [], []
    for i in range(a.reps + 2):                       # two warm-ups
        at.positions += rng.normal(0, 1e-3, at.positions.shape)
        t0 = time.perf_counter(); call(); dt = time.perf_counter() - t0
        if i >= 2:
            ts.append(dt)
            splits.append(dict(calc.last_timing))
    out["call_s"] = statistics.median(ts)
    out["call_s_min"] = min(ts)
    out["model_s"] = statistics.median(s["model_s"] for s in splits)
    out["nlist_s"] = statistics.median(s["nlist_s"] for s in splits)
    out["rebuilds"] = splits[-1]["rebuilds"]
    out["layout"], out["edge_a_kind"] = calc.last_layout, calc.last_edge_a_kind
    out["n_edges"] = int(getattr(calc, "last_n_edges", 0) or 0)
    st = getattr(calc, "_skin_state", None)
    if st is not None and calc.last_layout == "dense" and a.skin > 0:
        out["K"], out["K_skin"] = st.K, st.K_skin
        u = jax.device_put(st.displacements(at.positions, np.dtype(a.dtype)))
        step = lambda: jax.block_until_ready(calc._step()(u, st.arrays, K=st.K))
        step()
        ss = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); step(); ss.append(time.perf_counter() - t0)
        out["step_s"] = statistics.median(ss)
    out["us_per_atom"] = out["call_s"] / a.n * 1e6
    out["atom_steps_per_s"] = a.n / out["call_s"]
    out["os_threads"] = len(os.listdir("/proc/self/task"))
    import resource
    out["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out["loadavg_after"] = open("/proc/loadavg").read().split()[:3]
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
