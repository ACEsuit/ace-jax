"""Profile one ace-jax PACE case: time split, per-stage timings, cost analysis,
and (optionally) a per-kernel GPU profile mapped back to HLO / JAX source.

Targets the pre-optimisation API (the merge base with feat/bench-scaling, 2eb629f):
it calls PACE methods removed by the pool-first rewrite (`_node_energies`,
`edge_a_factors`), so it no longer runs on this branch.  Kept as the evidence
behind docs/pace-performance-gap.md; bench/perf/microbench.py is the maintained
harness.

    PYTHONPATH=bench:src python bench/perf/profile_acejax.py <yace> <system> <n> \
        [--dtype float64] [--layout dense|sparse] [--kind gather|matmul] \
        [--trace DIR] [--variant NAME]

Prints one JSON object.  Used by bench/perf/modal_profile.py on the A100; runs
on the CPU too (for checking the script).

Stages (dense layout), each timed forward and forward+VJP inside one jit:
  factors  edge_a_factors: radial basis g, R = g . crad[zi,zj], Y_lm, per slot
  pool     pool_a_dense: one-hot channel expansion + (n,K,C*nc) x (n,K,ny) einsum + select
  node     _node_energies: AA products, AA @ ctilde, embedding, core
  core     _edge_core (per-edge core repulsion)
"""
import argparse
import json
import statistics
import sys
import time


def timeit(fn, *a, reps=20, warm=3):
    import jax
    for _ in range(warm):
        jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def cost(fn, *a):
    try:
        c = fn.lower(*a).compile().cost_analysis()
        if isinstance(c, list):
            c = c[0]
        return {"flops": float(c.get("flops", -1)), "bytes": float(c.get("bytes accessed", -1))}
    except Exception as ex:                                        # noqa: BLE001
        return {"error": repr(ex)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yace")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--layout", default="dense")
    ap.add_argument("--kind", default="gather")
    ap.add_argument("--trace", default="")
    ap.add_argument("--variant", default="baseline",
                    help="model variant from bench/perf/variants.py (baseline = unmodified)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-stages", action="store_true")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    import jax.numpy as jnp
    import numpy as np
    from ase.calculators.calculator import all_changes

    from ace_jax.calc.point import ACECalculator
    from ace_jax.eval.model import highest_precision
    from ace_jax.eval.nlist import dense_graph, sparse_graph
    from ace_jax.eval.pace_model import load_yace
    from scaling.structures import supercell

    dt = getattr(jnp, args.dtype)
    out = {"yace": args.yace, "system": args.system, "n": args.n, "dtype": args.dtype,
           "layout": args.layout, "variant": args.variant, "platform": jax.default_backend(),
           "device": str(jax.devices()[0].device_kind)}
    at = supercell(args.system, args.n)
    model, meta, _ = load_yace(args.yace, dtype=dt, edge_a_kind=args.kind)
    base_model = model
    use_rev = "rev" in args.variant.split("+")
    if args.variant != "baseline":
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        import variants
        model = variants.make(args.variant, model)
    rc = float(meta["rcut"])

    # ---------------- 1. the calculator end to end, as the benchmark runs it
    calc = ACECalculator(model, meta, dtype=dt, layout=args.layout, edge_a_kind=args.kind)
    call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
    t0 = time.perf_counter(); call(); out["first_call_s"] = time.perf_counter() - t0
    call()
    splits, tot = [], []
    for _ in range(args.reps):
        t0 = time.perf_counter(); call(); tot.append(time.perf_counter() - t0)
        splits.append(calc.last_timing)
    out["calc"] = {"call_s": statistics.median(tot),
                   "nlist_s": statistics.median(s["nlist_s"] for s in splits),
                   "model_s": statistics.median(s["model_s"] for s in splits),
                   "layout": calc.last_layout, "backend": splits[-1]["nlist_backend"]}
    E_ref = calc.results["energy"]; F_ref = calc.results["forces"]
    out["E"] = E_ref

    # ---------------- 2. the calculator's pre-model part, split
    pos, cell, pbc = at.get_positions(), at.cell.array, at.pbc
    node_z_np = calc._species_index(at.get_atomic_numbers())
    K = calc._k_hint
    split = {}
    if args.layout == "dense":
        gpu = jax.default_backend() == "gpu"
        dev = "cuda" if gpu else None
        f = lambda: dense_graph(pos, cell, pbc, rc, K, device=dev)
        split["neighbour_matrix_s"] = timeit(f, reps=args.reps)
        dg = f()
        node_z = jnp.asarray(node_z_np)

        def build():
            idx = jnp.asarray(dg.idx, dtype=jnp.int32)
            count = jnp.asarray(dg.count, dtype=jnp.int32)
            return (jnp.asarray(dg.rij, dtype=dt), jnp.broadcast_to(node_z[:, None], idx.shape),
                    node_z[idx], idx, jnp.arange(idx.shape[1])[None, :] < count[:, None], node_z)
        split["build_args_s"] = timeit(build, reps=args.reps)
        split["species_index_s"] = timeit(lambda: jnp.asarray(calc._species_index(
            at.get_atomic_numbers())), reps=args.reps)
        dargs = build()
        n, Kd = dargs[4].shape
        out["K"] = int(Kd); out["n_slots"] = int(n * Kd); out["n_edges"] = int(np.asarray(dg.count).sum())
    else:
        split["sparse_graph_s"] = timeit(lambda: sparse_graph(pos, cell, pbc, rc), reps=5)
        g = sparse_graph(pos, cell, pbc, rc)
        out["n_edges"] = int(len(g.senders))
    out["pre_model_split"] = split

    # device -> host of the results (what calculate() does after the call)
    if args.layout == "dense":
        efv = jax.jit(lambda m, *a: m.energy_forces_virial_dense(*a))
        with highest_precision():
            res = efv(model, *dargs)
            split["d2h_results_s"] = timeit(lambda: (float(res[0]), np.asarray(res[1]),
                                                     np.asarray(res[2])), reps=args.reps)

    if args.layout != "dense":
        print(json.dumps(out))
        return

    # ---------------- 3. compiled model: forces vs energy only; cost analysis
    rij, zi, zj, idx, mask, node_z = dargs
    energy = jax.jit(lambda m, rij, zi, zj, mask, nz: jnp.sum(
        m.site_energies_dense(rij, zi, zj, mask, nz)))
    with highest_precision():
        E, F, V = efv(model, *dargs)
        out["parity_vs_calc"] = {"dE": abs(float(E) - E_ref),
                                 "max_dF": float(np.max(np.abs(np.asarray(F) - F_ref)))}
        m = {"efv_s": timeit(efv, model, *dargs, reps=args.reps),
             "energy_only_s": timeit(energy, model, rij, zi, zj, mask, node_z, reps=args.reps)}
        m["efv_cost"] = cost(efv, model, *dargs)
        m["energy_cost"] = cost(energy, model, rij, zi, zj, mask, node_z)
        if args.variant != "baseline":
            # parity against the unmodified PACEModel on the same inputs
            Eb, Fb, Vb = efv(base_model, *dargs)
            out["parity_vs_baseline"] = {
                "dE_per_atom": abs(float(E) - float(Eb)) / args.n,
                "max_dF": float(np.max(np.abs(np.asarray(F) - np.asarray(Fb)))),
                "max_dV": float(np.max(np.abs(np.asarray(V) - np.asarray(Vb))))}
        if use_rev:
            t0 = time.perf_counter()
            rev = jnp.asarray(variants.reverse_slots(dg.idx, dg.rij, dg.count))
            m["reverse_slots_host_s"] = time.perf_counter() - t0
            efv_rev = jax.jit(lambda m_, *a: m_.energy_forces_virial_dense_rev(*a))
            rargs = (rij, zi, zj, idx, rev, mask, node_z)
            Er, Fr, Vr = efv_rev(model, *rargs)
            m["efv_rev_s"] = timeit(efv_rev, model, *rargs, reps=args.reps)
            m["efv_rev_cost"] = cost(efv_rev, model, *rargs)
            Eb, Fb, Vb = efv(base_model, *dargs)
            out["parity_rev_vs_baseline"] = {
                "dE_per_atom": abs(float(Er) - float(Eb)) / args.n,
                "max_dF": float(np.max(np.abs(np.asarray(Fr) - np.asarray(Fb)))),
                "max_dV": float(np.max(np.abs(np.asarray(Vr) - np.asarray(Vb))))}
            efv = efv_rev                      # the trace below profiles this path
            dargs = rargs
    out["model"] = m

    # ---------------- 4. stages (forward and forward+VJP)
    if not args.no_stages:
        out["stages"] = stages(model, rij, zi, zj, mask, node_z, args.reps)

    # ---------------- 5. trace
    if args.trace:
        with highest_precision():
            jax.block_until_ready(efv(model, *dargs))
            with jax.profiler.trace(args.trace, create_perfetto_trace=True):
                for _ in range(5):
                    jax.block_until_ready(efv(model, *dargs))
        out["trace"] = args.trace
    try:
        out["peak_bytes"] = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception:                                              # noqa: BLE001
        pass
    print(json.dumps(out))


def stages(model, rij, zi, zj, mask, node_z, reps):
    import jax
    import jax.numpy as jnp

    from ace_jax.eval.model import highest_precision

    n, K = mask.shape
    flat = lambda a: a.reshape(n * K, *a.shape[2:])

    def f_factors(m, r):
        cols, Y = m.edge_a_factors(flat(r), flat(zi), flat(zj), flat(mask))
        return cols.reshape(n, K, -1), Y.reshape(n, K, -1)

    def f_pool(m, cols, Y):
        return m.pool_a_dense(cols, Y, zj)

    seg = jnp.repeat(jnp.arange(n), K)

    def f_core(m, r):
        return m._edge_core(flat(r), flat(zi), flat(zj), flat(mask))

    def f_node(m, A, cr, d, dcin):
        return m._node_energies(A, cr, d, dcin, seg, n, node_z)

    out = {}
    with highest_precision():
        cols, Y = jax.jit(f_factors)(model, rij)
        A = jax.jit(f_pool)(model, cols, Y)
        cr, d, dcin = jax.jit(f_core)(model, rij)

        def fwd_bwd(f, *xs):
            def g(m, *xs):
                y, vjp = jax.vjp(lambda *xx: f(m, *xx), *xs)
                ct = jax.tree_util.tree_map(jnp.ones_like, y)
                return y, vjp(ct)
            return jax.jit(g)

        cases = {
            "factors": (f_factors, (rij,)),
            "pool": (f_pool, (cols, Y)),
            "core": (f_core, (rij,)),
            "node": (f_node, (A, cr, d, dcin)),
        }
        for name, (f, xs) in cases.items():
            fwd = jax.jit(f)
            fb = fwd_bwd(f, *xs) if name != "node" else fwd_bwd(
                lambda m, A: f_node(m, A, cr, d, dcin), A)
            fb_args = xs if name != "node" else (A,)
            out[name] = {"fwd_s": timeit(fwd, model, *xs, reps=reps),
                         "fwd_vjp_s": timeit(fb, model, *fb_args, reps=reps),
                         "fwd_cost": cost(fwd, model, *xs),
                         "fwd_vjp_cost": cost(fb, model, *fb_args)}
        out["shapes"] = {"cols": list(cols.shape), "Y": list(Y.shape), "A": list(A.shape)}
    return out


if __name__ == "__main__":
    main()
