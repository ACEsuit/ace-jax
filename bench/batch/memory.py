"""Batch size against device memory (docs/dev/batched-evaluation-results.md, `MAX_BATCH_ATOMS`): peak device
bytes and µs per atom of ONE batched call as the padded atoms per call double, for rattled Si cells of 2, 8,
64 and 216 atoms, until the device runs out of memory.  Each point runs in its own process
(`peak_bytes_in_use` is a process maximum; XLA_PYTHON_CLIENT_PREALLOCATE=false), so an OOM ends that
series and nothing else.  Small cells stop at --max-cells before the device fills; their bytes per
atom then give the limit by extrapolation.  One JSON line per point on stdout (and --out).

    python bench/batch/memory.py --model <o4d12 model.npz> --kind points md --dtype float64 [--builder msn]

kind "points": `BatchCalculator.calculate` (pack, list build, energy/forces/virial), max_batch_atoms set so
the whole list is one call.  kind "md": vmapped JAX-MD NVE over `BatchNeighbours` (compacted K, batch-sized
row blocks; lists fixed), --steps steps per jitted chunk, as bench/batch/md.py.  The process peak also
holds compile-time autotuning scratch, so `in_use` (resident after the run: lists, state, outputs) is
reported beside it."""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time

SIZES = (2, 8, 64, 216)


def child(case):
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from ase.build import bulk

    from ace_jax import batch
    from ace_jax.batch import BatchCalculator, bucket_size

    make = {2: lambda: bulk("Si", "diamond", 5.43), 8: lambda: bulk("Si", "diamond", 5.43, cubic=True),
            64: lambda: bulk("Si", "diamond", 5.43, cubic=True).repeat(2),
            216: lambda: bulk("Si", "diamond", 5.43, cubic=True).repeat(3)}[case["n"]]
    n, B, reps = case["n"], case["cells"], case["reps"]
    ats = []
    for k in range(B):
        at = make()
        at.rattle(0.05, seed=k)
        ats.append(at)
    dt = None if case["dtype"] == "float64" else np.float32
    bc = BatchCalculator(case["model"], dtype=dt, builder=case["builder"], max_batch_atoms=B * bucket_size(n))
    dev = jax.devices()[0]
    stats = lambda: dev.memory_stats() or {}
    if case["kind"] == "points":
        run = lambda: bc.calculate(ats)
    else:
        from jax_md import simulate, space
        _, shift = space.free()
        nb = bc.neighbour_lists(ats, skin=1.0)
        nl = nb.allocate()
        init, apply = simulate.nve(nb.energy_fn(nl), shift, dt=1.0 / 10.18)
        p0 = batch.initial_momenta(jax.random.PRNGKey(0), nb.live, 0.01, 28.0855)
        st = [jax.jit(jax.vmap(lambda R, p, nn: init(None, R, kT=0.0, mass=28.0855, momenta=p, neighbor=nn)))(
            nb.positions, p0, nl)]
        step = jax.jit(jax.vmap(lambda s, nn: jax.lax.fori_loop(0, case["steps"], lambda i, x: apply(x, neighbor=nn), s)))

        def run():
            st[0] = jax.block_until_ready(step(st[0], nl))
    run()                                          # the compile
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t0)
    t = statistics.median(ts)
    s = stats()
    per = n * B * (case["steps"] if case["kind"] == "md" else 1)
    out = dict(s=t, us_per_atom=1e6 * t / per, peak_bytes=s.get("peak_bytes_in_use"),
               in_use=s.get("bytes_in_use"), bytes_limit=s.get("bytes_limit"))
    if case["kind"] == "points":
        out["batches"] = bc.last_timing["batches"]
    print("RESULT " + json.dumps(out), flush=True)


def run_case(case, timeout):
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, __file__, "_child", json.dumps(case)], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(status="timeout", wall_s=timeout)
    wall = time.perf_counter() - t0
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return dict(status="ok", wall_s=wall, **json.loads(line[7:]))
    err = p.stderr[-4000:]
    oom = any(s in err for s in ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory", "OOM"))
    return dict(status="oom" if oom else "error", wall_s=wall, rc=p.returncode,
                error=None if oom else err[-1500:])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "_child":
        return child(json.loads(sys.argv[2]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", nargs="+", default=["points", "md"])
    ap.add_argument("--dtype", nargs="+", default=["float64"])
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--builder", default="msn")
    ap.add_argument("--min-atoms", type=int, default=2 ** 12, help="padded atoms per call, first point")
    ap.add_argument("--max-atoms", type=int, default=2 ** 22, help="padded atoms per call, last point")
    ap.add_argument("--max-cells", type=int, default=2 ** 16,
                    help="cells per call, last point (building millions of ASE Atoms dominates on the host)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out")
    a = ap.parse_args()
    from ace_jax.batch import bucket_size
    host = dict(host=platform.node())
    out = open(a.out, "a") if a.out else None
    for kind in a.kind:
        for dtype in a.dtype:
            for n in a.sizes:
                N = bucket_size(n)
                A = a.min_atoms
                while A <= a.max_atoms and A // N <= a.max_cells:
                    case = dict(model=a.model, kind=kind, dtype=dtype, n=n, cells=max(1, A // N),
                                builder=a.builder, steps=a.steps, reps=a.reps)
                    r = run_case(case, a.timeout)
                    rec = {**host, **{k: v for k, v in case.items() if k != "model"}, "padded_atoms": A, **r}
                    if r.get("peak_bytes"):
                        rec["peak_bytes_per_atom"] = r["peak_bytes"] / A
                    line = json.dumps(rec)
                    print(line, flush=True)
                    if out:
                        out.write(line + "\n")
                        out.flush()
                    if r["status"] != "ok":
                        break                  # the next size would fail too
                    A *= 2


if __name__ == "__main__":
    main()
