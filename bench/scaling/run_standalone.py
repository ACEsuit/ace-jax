"""One standalone benchmark case -> one JSON row (stdout).

    python bench/scaling/run_standalone.py <model-name> <n_atoms> <dtype> <device>
call_s: median ASE calculator call (energy+forces+stress) incl. neighbour list,
for every code.  ace-jax also, from ACECalculator.last_timing: force_s (the
compiled model call alone), nlist_s (list + layout + host->device), the
neighbour-list backend and `rebuilds` (calls that built a neighbour list).

The timed calls are MD-like (`md_like: true`): each first displaces every atom
by N(0, 1e-3 A), a random walk seeded at 0, as consecutive MD steps would, so
ace-jax takes its skin-list reuse path (the default skin) and MACE sees moving
atoms too.  The first (compile) call, and `energy`, use the undisplaced
structure.
"""
import json
import platform
import statistics
import sys
import time

from scaling.structures import supercell


def _versions():
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for pkg in ("ace-jax", "jax", "jaxlib", "mace-torch", "torch", "ase", "matscipy",
                "matscipy-neighbours"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def _median_time(f, reps, setup=None):
    """Median seconds of f(); `setup()` (untimed) runs before each call."""
    ts = []
    for _ in range(reps):
        if setup:
            setup()
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _peak(device, code="acejax"):
    """Peak device memory from the framework the code ran on (torch for MACE,
    JAX for ace-jax), or peak RSS on the CPU."""
    if device == "gpu":
        if code == "mace":
            import torch
            return torch.cuda.max_memory_allocated()
        import jax
        return jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def _threads():
    import os
    out = {"cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()}
    for v in ("OMP_NUM_THREADS", "XLA_FLAGS"):
        if os.environ.get(v):
            out[v] = os.environ[v]
    return out


def run_case(row, n_atoms, dtype, device, reps=10):
    from ase.calculators.calculator import all_changes
    import numpy as np
    at = supercell(row["system"], n_atoms)
    rng = np.random.default_rng(0)

    def displace():                       # one MD-like step before each timed call (untimed)
        at.positions += rng.normal(0, 1e-3, at.positions.shape)

    out = {"code": row["code"], "mode": "standalone", "model": row["name"], "size": row["size"],
           "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
           "versions": _versions(), "status": "ok", "threads": _threads()}
    try:
        if row["code"].startswith("acejax"):
            import jax
            if device == "cpu":                   # jax[cuda12] would pick the GPU
                jax.config.update("jax_platforms", "cpu")
            jax.config.update("jax_enable_x64", dtype == "float64")
            out["platform"] = jax.default_backend()
            import jax.numpy as jnp
            from ace_jax.calc.point import ACECalculator
            from ace_jax.eval import sparse_graph
            calc = ACECalculator(row["path"], dtype=getattr(jnp, dtype))   # default skin
            out["skin"] = calc.skin
            call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
            t0 = time.perf_counter(); call(); out["compile_s"] = time.perf_counter() - t0
            out["energy"] = calc.results["energy"]                         # undisplaced
            n_edges = int(len(sparse_graph(at.positions, at.cell.array, at.pbc,
                                           calc.cutoff).senders))
            call()                                # K learnt: the steady-state path from here
            splits = []

            def timed():
                call()
                splits.append(calc.last_timing)

            out["call_s"] = _median_time(timed, reps, setup=displace)
            # the calculator's own split: nlist_s = list + layout + host->device,
            # force_s = the compiled model call alone
            out["nlist_s"] = statistics.median(t["nlist_s"] for t in splits)
            out["force_s"] = statistics.median(t["model_s"] for t in splits)
            out["nlist_backend"] = splits[-1]["nlist_backend"]
            out["layout"], out["edge_a_kind"] = calc.last_layout, calc.last_edge_a_kind
            out["n_edges"] = n_edges
            out["rebuilds"] = splits[-1]["rebuilds"]     # of all calls, the first included
            out["md_like"] = True
        elif row["code"] == "mace":
            import torch
            from mace.calculators import mace_mp
            calc = mace_mp(model=row["path"], default_dtype=dtype,
                           device="cuda" if device == "gpu" else "cpu",
                           enable_cueq=(device == "gpu"),
                           **({"head": row["head"]} if row.get("head") else {}))
            out["platform"] = "cuda" if device == "gpu" else "cpu"
            if device == "cpu":                   # MKL_NUM_THREADS=1 would pin torch to 1
                torch.set_num_threads(out["threads"]["cpus"])
            out["threads"]["torch"] = torch.get_num_threads()
            call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
            t0 = time.perf_counter(); call(); out["compile_s"] = time.perf_counter() - t0
            out["energy"] = calc.results["energy"]                         # undisplaced
            out["call_s"] = _median_time(call, reps, setup=displace)
            out["rebuilds"], out["md_like"] = None, True
            if device == "gpu":
                torch.cuda.synchronize()
        else:
            raise ValueError(f"no standalone runner for {row['code']}")
    except Exception as ex:                                        # OOM etc. are data
        msg = repr(ex)
        out["status"] = "oom" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()) else "error"
        out["error"] = msg[:300]
    out["peak_bytes"] = _peak(device, row["code"])
    return out


if __name__ == "__main__":
    from scaling.models import planned_models
    name, n, dtype, device = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    row = next(r for r in planned_models() if r["name"] == name)
    print(json.dumps(run_case(row, n, dtype, device)))
