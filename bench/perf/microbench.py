"""Before/after micro-benchmark of the ace-jax calculator (perf-optimisation Task 10).

    PYTHONPATH=bench:src python bench/perf/microbench.py time <model> <system> <n> [--dtype float64] [--reps 20]
    PYTHONPATH=bench:src python bench/perf/microbench.py memory <model> <system> <n> [--dtype float64]

`time`: one ACECalculator on `supercell(system, n)`, timed MD-like: the first
(compile) call and `energy` use the undisplaced structure; every later call
first displaces each atom by N(0, 1e-3 A) (a random walk seeded at 0), so the
calculator reuses its skin list where it has one.  The skin is the
calculator's default when it takes a `skin` argument (this branch, 1.0 A) and
absent otherwise (the merge base, which rebuilds its list every call).
Reports, as medians over `reps` calls:
  model_s  -- `last_timing["model_s"]`, the compiled model call (with the skin
              list: the step, its displacement H2D and the packed result D2H)
  step_s   -- (skin list only) the compiled step alone, inputs on the device
  call_s   -- the whole calculator call, end to end
and atom-steps/s = n / time for both.

`memory`: XLA's compiled temp size for the dense E/F/V call
(`jax.jit(...).lower(...).compile().memory_analysis().temp_size_in_bytes`)
next to `estimate_a_bytes`, for refitting the latter (HEAD only).

Prints one JSON object.  Run on Modal by `modal_profile.py::microbench` (and
`::microbench_memory`), which runs every case in its own process.
"""
import argparse
import inspect
import json
import statistics
import time


def _setup(dtype):
    import jax
    jax.config.update("jax_enable_x64", dtype == "float64")
    return jax


def time_case(model, system, n, dtype, reps):
    import numpy as np
    from ase.calculators.calculator import all_changes

    jax = _setup(dtype)
    import jax.numpy as jnp

    from ace_jax.calc.point import ACECalculator
    from scaling.structures import supercell

    at = supercell(system, n)
    out = {"model": model, "system": system, "n": n, "dtype": dtype, "reps": reps,
           "device": jax.devices()[0].device_kind, "md_like": True, "status": "ok"}
    try:
        takes_skin = "skin" in inspect.signature(ACECalculator.__init__).parameters
        calc = ACECalculator(model, dtype=getattr(jnp, dtype))    # skin: the default, if any
        out["skin"] = calc.skin if takes_skin else None
        call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
        t0 = time.perf_counter(); call(); out["first_s"] = time.perf_counter() - t0
        out["energy"] = calc.results["energy"]                    # undisplaced
        rng = np.random.default_rng(0)
        ts, splits = [], []
        for i in range(reps + 1):                                 # the first is a warm-up
            at.positions += rng.normal(0, 1e-3, at.positions.shape)
            t0 = time.perf_counter(); call(); dt = time.perf_counter() - t0
            if i:
                ts.append(dt)
                splits.append(dict(calc.last_timing))
        out["call_s"] = statistics.median(ts)
        out["model_s"] = statistics.median(s["model_s"] for s in splits)
        out["nlist_s"] = statistics.median(s["nlist_s"] for s in splits)
        out["call_atom_steps_per_s"] = n / out["call_s"]
        out["model_atom_steps_per_s"] = n / out["model_s"]
        out["nlist_backend"] = splits[-1]["nlist_backend"]
        out["layout"] = calc.last_layout
        out["rebuilds"] = splits[-1].get("rebuilds")              # None before (no counter)
        out["step_s"] = _step_s(calc, at, dtype, reps)
    except Exception as ex:                                       # OOM etc. are data
        msg = repr(ex)
        out["status"] = "oom" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()) else "error"
        out["error"] = msg[:300]
    return out


def _step_s(calc, at, dtype, reps):
    """The skin list's compiled step alone, inputs already on the device (no
    H2D / D2H): what `model_s` costs without the transfers.  None without a
    skin list (the merge base)."""
    st = getattr(calc, "_skin_state", None)
    if st is None or getattr(calc, "_skin_step", None) is None:
        return None
    import jax
    import numpy as np
    u = jax.device_put(st.displacements(at.positions, np.dtype(dtype)))
    step = lambda: jax.block_until_ready(calc._skin_step(u, st.arrays, K=st.K))
    step()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); step(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def memory_case(model, system, n, dtype):
    import numpy as np

    jax = _setup(dtype)
    import jax.numpy as jnp

    from ace_jax.eval import load, sparse_graph
    from ace_jax.eval.edge_model import estimate_a_bytes
    from ace_jax.eval.nlist import dense_from_sparse
    from scaling.structures import supercell

    m, meta, _ = load(model, dtype=getattr(jnp, dtype))
    at = supercell(system, n)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    d = dense_from_sparse(g, meta["rcut"])
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    idx = jnp.asarray(d.idx)
    nn, K = idx.shape
    args = (jnp.asarray(d.rij, getattr(jnp, dtype)), jnp.broadcast_to(nz[:, None], idx.shape),
            nz[idx], idx, jnp.asarray(d.mask), nz)
    ma = jax.jit(lambda mm, *a: mm.energy_forces_virial_dense(*a)).lower(m, *args).compile() \
        .memory_analysis()
    itemsize = np.dtype(dtype).itemsize
    out = {"model": model, "system": system, "n": nn, "K": int(K), "dtype": dtype,
           "device": jax.devices()[0].device_kind, "n_edges": int(len(g.senders)),
           "C": int(m.a_channels), "n_a": int(m.aspec_r.shape[0]),
           "uses_edge_a": bool(m.uses_edge_a),
           "temp_bytes": None if ma is None else int(ma.temp_size_in_bytes),
           "estimate_bytes": int(estimate_a_bytes(m, "dense", nn, len(g.senders), K, itemsize))}
    if not m.uses_edge_a:
        out["n_b"], out["n_y"] = (int(v) for v in m.pool_first_widths())
        out["n_p"] = int(m.product_basis_width())
    else:
        out["n_cols"], out["n_y"] = (int(v) for v in m.edge_a_widths())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["time", "memory"])
    ap.add_argument("model")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    if a.what == "time":
        print(json.dumps(time_case(a.model, a.system, a.n, a.dtype, a.reps)))
    else:
        print(json.dumps(memory_case(a.model, a.system, a.n, a.dtype)))


if __name__ == "__main__":
    main()
