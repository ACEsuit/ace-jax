"""One standalone benchmark case -> one JSON row (stdout).

    python bench/scaling/run_standalone.py <model-name> <n_atoms> <dtype> <device>
call_s: median ASE calculator call (energy+forces+stress) incl. neighbour list,
for every code.  ace-jax also, from ACECalculator.last_timing: force_s (the
compiled model call alone), nlist_s (list + layout + host->device) and the
neighbour-list backend.
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


def _median_time(f, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _peak(device):
    if device == "gpu":
        try:
            import jax
            return jax.devices()[0].memory_stats().get("peak_bytes_in_use")
        except Exception:
            import torch
            return torch.cuda.max_memory_allocated()
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
    at = supercell(row["system"], n_atoms)
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
            calc = ACECalculator(row["path"], dtype=getattr(jnp, dtype))
            call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
            t0 = time.perf_counter(); call(); out["compile_s"] = time.perf_counter() - t0
            call()                                # K learnt: the steady-state path from here
            splits = []

            def timed():
                call()
                splits.append(calc.last_timing)

            out["call_s"] = _median_time(timed, reps)
            # the calculator's own split: nlist_s = list + layout + host->device,
            # force_s = the compiled model call alone
            out["nlist_s"] = statistics.median(t["nlist_s"] for t in splits)
            out["force_s"] = statistics.median(t["model_s"] for t in splits)
            out["nlist_backend"] = splits[-1]["nlist_backend"]
            out["layout"], out["edge_a_kind"] = calc.last_layout, calc.last_edge_a_kind
            out["n_edges"] = int(len(sparse_graph(at.positions, at.cell.array, at.pbc,
                                                  calc.cutoff).senders))
        elif row["code"] == "mace":
            import torch
            from mace.calculators import mace_mp
            calc = mace_mp(model=row["path"], default_dtype=dtype,
                           device="cuda" if device == "gpu" else "cpu",
                           enable_cueq=(device == "gpu"),
                           **({"head": row["head"]} if row.get("head") else {}))
            out["platform"] = "cuda" if device == "gpu" else "cpu"
            out["threads"]["torch"] = torch.get_num_threads()
            call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
            t0 = time.perf_counter(); call(); out["compile_s"] = time.perf_counter() - t0
            out["call_s"] = _median_time(call, reps)
            if device == "gpu":
                torch.cuda.synchronize()
        else:
            raise ValueError(f"no standalone runner for {row['code']}")
    except Exception as ex:                                        # OOM etc. are data
        msg = repr(ex)
        out["status"] = "oom" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()) else "error"
        out["error"] = msg[:300]
    out["peak_bytes"] = _peak(device)
    return out


if __name__ == "__main__":
    from scaling.models import load_manifest, planned_models
    name, n, dtype, device = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    row = next(r for r in planned_models() if r["name"] == name)
    print(json.dumps(run_case(row, n, dtype, device)))
