"""Profile `ACEModel.energy_forces_virial_dense` (the linear ACE model) on one
benchmark structure: whole-call timing, per-stage timings, and (with --trace /
--dump) a per-kernel GPU profile attributed to the model's stages.

    PYTHONPATH=bench:src python bench/perf/profile_ace.py <npz> <system> <n> \
        [--dtype float64] [--reps 20] [--no-stages] [--trace DIR --dump DIR]

Prints one JSON object.  Used by bench/perf/modal_profile.py::ace on the A100;
runs on the CPU too (for checking the script).  The model is loaded as the
benchmark loads it (`ace_jax.eval.io.load`, readout folded into ctilde).

Stages of the folded dense energy path (site_energies_dense), each timed on its
own inside one jit, forward and forward+VJP:
  radial   ACEModel.radial: transform, envelope, Rnl (spline / analytic /
           factorised) and the pair radial, per slot
  angular  the Y_lm per slot
  pool     A assembly: pool_a_dense (batched outer product + select) and the
           pair channel's pool_dense
  aa       the product basis `_aa` (forward, and its adjoint in the VJP)
  readout  AA . ctilde[:, z] + Apair . Wpair[:, z] + E0
  force    the force scatter-add of energy_forces_virial_dense (post-grad; no VJP)

With --trace and --dump (XLA_FLAGS=--xla_dump_to is set here, before JAX starts)
the whole call is traced and every GPU kernel is attributed to a stage by the
call chains its fused HLO instructions carry (see `kernel_stages`).  A kernel
whose fusion spans several stages is split by its instruction count per stage
(`frac`), bounded below by kernels wholly in the stage (`pure_frac`) and above
by every kernel touching it (`touch_frac`); the time in such kernels is
`mixed_stage_us_per_call`.  `--reattribute` redoes this from a saved trace + dump.
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


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


# ------------------------------------------------------------------ attribution
# This JAX (0.11) writes no source_file/source_line into the optimised HLO: each
# instruction carries a stack_frame_id into the module's FileNames /
# FunctionNames / FileLocations / StackFrames tables.  A kernel is attributed by
# walking each of its instructions' call chains innermost first to the first
# function below that names a stage.
_FN_STAGE = {"ACEModel._aa": "aa", "ACEModel._readout_folded": "readout",
             "EdgeSiteModel.pool_a_dense": "pool", "pool_dense": "pool",
             "ACEModel.angular": "angular",
             "ACEModel.radial": "radial", "ACEModel._radial_one": "radial",
             "ACEModel._edge_factors_and_pair": "radial",      # the mask on Rnl
             "ACEModel.site_energies_dense": "other",          # reshapes
             "EdgeSiteModel.energy_forces_virial_dense.<locals>.total": "strain",
             # reached without passing `total`: after value_and_grad, i.e. the
             # mask on g_r and the force scatter-add
             "EdgeSiteModel.energy_forces_virial_dense": "force"}
_FILE_STAGE = {"radial.py": "radial", "harmonics.py": "angular"}


def _tables(txt):
    """{frame id: [(file basename, function name), ... innermost first]}."""
    sec, tab = None, {"FileNames": {}, "FunctionNames": {}, "FileLocations": {},
                      "StackFrames": {}}
    for line in txt.splitlines():
        if line in tab:
            sec = line
            continue
        if sec is None:
            continue
        if not line.strip():
            if tab[sec]:
                sec = None
            continue
        if line.startswith(("HloModule", "ENTRY", "%")):
            break
        k, _, v = line.partition(" ")
        tab[sec][int(k)] = v
    files = {k: os.path.basename(v.strip('"')) for k, v in tab["FileNames"].items()}
    fns = {k: v.strip('"') for k, v in tab["FunctionNames"].items()}
    locs = {}
    for k, v in tab["FileLocations"].items():
        f = int(re.search(r"file_name_id=(\d+)", v).group(1))
        fn = int(re.search(r"function_name_id=(\d+)", v).group(1))
        locs[k] = (files.get(f, "?"), fns.get(fn, "?"))
    parent, loc = {}, {}
    for k, v in tab["StackFrames"].items():
        loc[k] = int(re.search(r"file_location_id=(\d+)", v).group(1))
        # printed parent ids are offset by one; 0 (printed 1 on the root) = none
        parent[k] = int(re.search(r"parent_frame_id=(\d+)", v).group(1)) - 1
    chains = {}
    for k in loc:
        ch, f, seen = [], k, set()
        while f in loc and f not in seen:
            seen.add(f)
            ch.append(locs[loc[f]])
            f = parent[f]
        chains[k] = ch
    return chains


def _stage_of_chain(chain):
    for f, fn in chain:
        if fn in _FN_STAGE:
            return _FN_STAGE[fn]
        if f in _FILE_STAGE:
            return _FILE_STAGE[f]
    return "other"


def kernel_stages(path):
    """{HLO instruction name: Counter(stage -> instruction count)} for every
    instruction in the optimised module at `path`; a fusion counts the
    instructions of its body, which is what weights a kernel spanning stages."""
    txt = open(path).read()
    chains = _tables(txt)
    comps, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$", line)
        if m and not line.startswith(" "):
            cur = m.group(1)
            comps[cur] = []
            continue
        if cur is not None and line.strip() and line.strip() != "}":
            comps[cur].append(line)

    def own(line):
        f = re.search(r"stack_frame_id=(\d+)", line)
        return collections.Counter(
            [_stage_of_chain(chains.get(int(f.group(1)), []))] if f else [])

    memo = {}

    def comp_stages(c, depth=0):
        if c in memo or depth > 20:
            return memo.get(c, collections.Counter())
        memo[c] = collections.Counter()
        st = collections.Counter()
        for ln in comps.get(c, []):
            st += own(ln)
            for callee in re.findall(r"(?:calls|to_apply|body|condition)=%?([\w.\-]+)", ln):
                st += comp_stages(callee, depth + 1)
        memo[c] = st
        return st

    out = {}
    for c, lines in comps.items():
        for ln in lines:
            m = re.match(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=", ln)
            if not m:
                continue
            st = own(ln)
            for callee in re.findall(r"(?:calls|to_apply)=%?([\w.\-]+)", ln):
                st += comp_stages(callee)
            del st["other"]
            if not st:
                # no frame metadata (XLA drops it on some rewritten dots): take
                # the stages of the operands, which precede it in the schedule
                ops = re.search(r"\w+\((.*?)\)(?:,|$)", ln.split("=", 1)[1])
                for o in re.findall(r"%([\w.\-]+)", ops.group(1)) if ops else []:
                    st += collections.Counter({k: 1 for k in out.get(o, {}) if k != "other"})
            out[m.group(1)] = st or collections.Counter(["other"])
    return out


def _weights(st):
    """Stage weights of one kernel from its instruction counts.  `strain` (the
    r + r @ sym of the virial trick) is elementwise glue that XLA fuses into
    whatever consumes the edge vectors, so it keeps weight only on its own."""
    st = collections.Counter(st)
    if len(st) > 1:
        del st["strain"]
    tot = sum(st.values())
    return {k: v / tot for k, v in st.items()}


def attribute(trace_dir, dump_dir, n_calls, top=40):
    """Per-stage GPU time per call from the trace, and the top kernels.

    Per stage: `us` weights each kernel by the share of its fused instructions
    in the stage (the estimate); `pure_us` counts only kernels wholly in the
    stage (a lower bound) and `touch_us` every kernel with any instruction in
    it (an upper bound)."""
    sys.path.insert(0, HERE)
    import hlo_trace
    ev, _, gpu, tname = hlo_trace.load_trace(trace_dir)
    agg = collections.defaultdict(lambda: [0.0, 0, None])
    for e in ev:
        if e.get("ph") != "X" or e.get("pid") not in gpu:
            continue
        tn = tname.get((e["pid"], e["tid"]), "")
        if "stream" not in tn.lower():
            continue
        a = agg[e["name"]]
        a[0] += float(e.get("dur", 0.0))
        a[1] += 1
        a[2] = a[2] or (e.get("args") or {}).get("hlo_op")    # names library kernels
    info, path = hlo_trace.parse_hlo(dump_dir, list(agg))
    kst = kernel_stages(path) if path else {}
    gemms = re.findall(r"^\s*(?:ROOT\s+)?%([\w.\-]+) = .*custom_call_target=\"__cublas",
                       open(path).read(), re.M) if path else []
    total = sum(v[0] for v in agg.values())
    z = lambda: {"fwd_us": 0.0, "bwd_us": 0.0, "pure_us": 0.0, "touch_us": 0.0}
    by_stage = collections.defaultdict(z)
    mixed_us, rows = 0.0, []
    for k, (us, cnt, hop) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        base = k.split("(")[0].strip()
        cands = [base, re.sub(r"_(\d+)$", r".\1", base)]
        if hop:
            cands = [hop, re.sub(r"_(\d+)$", r".\1", hop)] + cands
        hlo = next((c for c in cands if c in info or c in kst), None)
        if hlo is None and "gemm" in k and len(gemms) == 1:
            # a library GEMM inside a CUDA command buffer (hlo_op names the
            # buffer, not the op): the module's only cuBLAS custom-call
            hlo = gemms[0]
        inf = info.get(hlo, {}) if hlo else {}
        w = _weights(kst.get(hlo, {"other": 1}) if hlo else {"unmatched": 1})
        # a transpose(...) op_name marks the adjoint
        side = "bwd_us" if inf.get("bwd") else "fwd_us"
        per = us / n_calls
        for s_, f in w.items():
            by_stage[s_][side] += per * f
            by_stage[s_]["touch_us"] += per
            if f == 1.0:
                by_stage[s_]["pure_us"] += per
        if len(w) > 1:
            mixed_us += per
        b = inf.get("bytes_in", 0) + inf.get("bytes_out", 0)
        rows.append({"kernel": k[:80], "hlo": hlo, "us_per_call": round(per, 2),
                     "frac": round(us / total, 4) if total else 0,
                     "launches_per_call": cnt / n_calls,
                     "stages": {s_: round(f, 3) for s_, f in w.items()}, "bwd": inf.get("bwd"),
                     "GBps": round(b / (per * 1e-6) / 1e9, 1) if per and b else None,
                     "hlo_op": inf.get("hlo_op"), "out_shape": inf.get("out_shape")[:120]
                     if inf.get("out_shape") else None})
    tot_call = total / n_calls
    stages = {}
    for s_, v in sorted(by_stage.items()):
        us = v["fwd_us"] + v["bwd_us"]
        stages[s_] = {"fwd_us": round(v["fwd_us"], 1), "bwd_us": round(v["bwd_us"], 1),
                      "us": round(us, 1), "frac": round(us / tot_call, 4) if tot_call else 0,
                      "pure_frac": round(v["pure_us"] / tot_call, 4) if tot_call else 0,
                      "touch_frac": round(v["touch_us"] / tot_call, 4) if tot_call else 0}
    return {"total_gpu_us_per_call": round(tot_call, 1), "n_kernels": len(agg),
            "launches_per_call": sum(v[1] for v in agg.values()) / n_calls,
            "mixed_stage_us_per_call": round(mixed_us, 1), "hlo_module": path,
            "stages": stages, "rows": rows[:top]}


# ------------------------------------------------------------------ widths
def widths(model, K):
    """The widths the pool-first rules compare (docs/perf-ace-profile.md)."""
    n_rnl, n_y = model.edge_a_widths()
    nz = int(model.E0.shape[0])
    kind = model.radial_kind
    if kind == "spline":
        n_b = int(model.rnl_coefs.shape[-2])                      # ncoef
    elif kind == "analytic":
        n_b = int(model.rnl_Wnlq.shape[-1])                       # n_q
    else:                        # spline_factorised: the table's columns P(x), n1
        n_b = int(model.rnl_coefs_single.shape[-1])
    return {"radial_kind": kind, "pair_radial_kind": model.pair_radial_kind, "NZ": nz,
            "n_rnl": int(n_rnl), "n_Y": int(n_y), "n_b": n_b, "n_A": int(model.aspec_r.shape[0]),
            "n_AA": int(sum(g.shape[0] for g in model.aa_specs)),
            "aa_orders": [int(g.shape[1]) for g in model.aa_specs],
            "n_pair": int(model.pair_coefs.shape[-1] if model.pair_radial_kind == "spline"
                          else model.pair_Wnlq.shape[-2]),
            "K": int(K),
            # what A assembly holds per node today, and what pool-first would:
            # the radial weights depend on (zi, zj), zi is the node's own species
            # but zj varies over the neighbours, so pool-first needs one pooled
            # block per neighbour species (C = NZ), as PACE's crad[zi, zj] does
            "pooled_now_per_node": int(n_rnl * n_y),
            "pooled_pool_first_per_node_C1": int(n_b * n_y),
            "pooled_pool_first_per_node_CNZ": int(nz * n_b * n_y)}


# ------------------------------------------------------------------ stages
def stages(model, rij, zi, zj, idx, mask, node_z, reps):
    import jax
    import jax.numpy as jnp

    from ace_jax.eval.model import highest_precision, pool_dense

    n, K = mask.shape
    flat = lambda a: a.reshape(n * K, *a.shape[2:])
    un = lambda a: a.reshape(n, K, -1)

    def f_radial(m, r):
        Rnl, Rpair = m.radial(flat(r), flat(zi), flat(zj))
        return jnp.where(flat(mask)[:, None], Rnl, 0.0), Rpair

    def f_angular(m, r):
        return m.angular(flat(r))

    def f_pool(m, Rnl, Y, Rpair):
        return m.pool_a_dense(un(Rnl), un(Y), None), pool_dense(un(Rpair), mask)

    def f_aa(m, A):
        return m._aa(A)

    def f_readout(m, AA, Apair):                 # ACEModel._readout_folded after _aa
        e = jnp.einsum("ia,ai->i", AA, m.ctilde[:, node_z])
        e = e + jnp.einsum("ip,pi->i", Apair, m.Wpair[:, node_z])
        return e + m.E0[node_z]

    def f_force(m, g_r):          # the rev=None assembly of energy_forces_virial_dense
        return (jnp.zeros((n, 3), g_r.dtype).at[jnp.arange(n)].add(g_r.sum(axis=1))
                .at[idx.reshape(-1)].add(-g_r.reshape(-1, 3)))

    def fwd_vjp(f):
        def g(m, *xs):
            y, vjp = jax.vjp(lambda *xx: f(m, *xx), *xs)
            return y, vjp(jax.tree_util.tree_map(jnp.ones_like, y))
        return jax.jit(g)

    out = {}
    with highest_precision():
        Rnl, Rpair = jax.jit(f_radial)(model, rij)
        Y = jax.jit(f_angular)(model, rij)
        A, Apair = jax.jit(f_pool)(model, Rnl, Y, Rpair)
        AA = jax.jit(f_aa)(model, A)
        cases = {"radial": (f_radial, (rij,)), "angular": (f_angular, (rij,)),
                 "pool": (f_pool, (Rnl, Y, Rpair)), "aa": (f_aa, (A,)),
                 "readout": (f_readout, (AA, Apair))}
        for name, (f, xs) in cases.items():
            fwd, fb = jax.jit(f), fwd_vjp(f)
            out[name] = {"fwd_s": timeit(fwd, model, *xs, reps=reps),
                         "fwd_vjp_s": timeit(fb, model, *xs, reps=reps),
                         "fwd_cost": cost(fwd, model, *xs), "fwd_vjp_cost": cost(fb, model, *xs)}
        fwd = jax.jit(f_force)                   # rij stands in for g_r (same shape)
        out["force"] = {"fwd_s": timeit(fwd, model, rij, reps=reps), "fwd_vjp_s": None,
                        "fwd_cost": cost(fwd, model, rij)}
        out["shapes"] = {"Rnl": list(Rnl.shape), "Y": list(Y.shape), "A": list(A.shape),
                         "AA": list(AA.shape), "Apair": list(Apair.shape)}
    tot = sum(v["fwd_vjp_s"] or v["fwd_s"] for k, v in out.items() if k != "shapes")
    out["sum_s"] = tot
    for k, v in out.items():
        if isinstance(v, dict) and "fwd_s" in v:
            v["frac_of_sum"] = (v["fwd_vjp_s"] or v["fwd_s"]) / tot
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-stages", action="store_true")
    ap.add_argument("--trace", default="", help="jax.profiler trace dir (the whole call)")
    ap.add_argument("--dump", default="", help="XLA HLO dump dir, for kernel attribution")
    ap.add_argument("--trace-calls", type=int, default=5)
    ap.add_argument("--reattribute", action="store_true",
                    help="only re-attribute an existing --trace / --dump pair (no JAX run)")
    args = ap.parse_args()
    if args.reattribute:
        print(json.dumps(attribute(args.trace, args.dump, args.trace_calls)))
        return
    if args.dump:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   f" --xla_dump_to={args.dump} --xla_dump_hlo_as_text").strip()

    import jax
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    import jax.numpy as jnp
    import numpy as np

    from ace_jax.eval.io import load
    from ace_jax.eval.model import highest_precision
    from ace_jax.eval.nlist import dense_graph
    from scaling.structures import supercell

    dt = getattr(jnp, args.dtype)
    model, meta, _ = load(args.npz, dtype=dt)
    rc = float(meta["rcut"])
    at = supercell(args.system, args.n)
    out = {"npz": os.path.basename(args.npz), "system": args.system, "n": args.n,
           "dtype": args.dtype, "platform": jax.default_backend(),
           "device": str(jax.devices()[0].device_kind), "jax": jax.__version__,
           "folded": bool(model.folded), "rcut": rc}

    # the dense graph on the host (the model call is what is profiled); K = the
    # largest neighbour count, as the calculator sizes it
    from ase.neighborlist import neighbor_list
    i = neighbor_list("i", at, rc)
    K = int(np.bincount(i, minlength=len(at)).max())
    g = dense_graph(at.get_positions(), at.cell.array, at.pbc, rc, K)
    z2i = {int(z): k for k, z in enumerate(meta["elements"])}
    nz_np = np.array([z2i[int(z)] for z in at.get_atomic_numbers()], np.int32)
    idx = jnp.asarray(g.idx, jnp.int32)
    count = jnp.asarray(g.count, jnp.int32)
    node_z = jnp.asarray(nz_np)
    rij = jnp.asarray(g.rij, dt)
    zi = jnp.broadcast_to(node_z[:, None], idx.shape)
    zj = node_z[idx]
    mask = jnp.arange(idx.shape[1])[None, :] < count[:, None]
    out["n_edges"] = int(np.asarray(g.count).sum())
    out["widths"] = widths(model, K)

    efv = jax.jit(lambda m, *a: m.energy_forces_virial_dense(*a))
    energy = jax.jit(lambda m, r, a, b, mk, nz: jnp.sum(m.site_energies_dense(r, a, b, mk, nz)))
    dargs = (rij, zi, zj, idx, mask, node_z)
    with highest_precision():
        t0 = time.perf_counter()
        E, F, V = jax.block_until_ready(efv(model, *dargs))
        out["first_call_s"] = time.perf_counter() - t0
        out["E_per_atom"] = float(E) / args.n
        out["max_abs_F"] = float(jnp.max(jnp.abs(F)))
        out["model"] = {"efv_s": timeit(efv, model, *dargs, reps=args.reps),
                        "energy_only_s": timeit(energy, model, rij, zi, zj, mask, node_z,
                                                reps=args.reps),
                        "efv_cost": cost(efv, model, *dargs)}

    if not args.no_stages:
        out["stages"] = stages(model, rij, zi, zj, idx, mask, node_z, args.reps)

    if args.trace:
        with highest_precision():
            jax.block_until_ready(efv(model, *dargs))
            with jax.profiler.trace(args.trace, create_perfetto_trace=True):
                for _ in range(args.trace_calls):
                    jax.block_until_ready(efv(model, *dargs))
        if args.dump:
            out["kernels"] = attribute(args.trace, args.dump, args.trace_calls)
    try:
        out["peak_bytes"] = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception:                                              # noqa: BLE001
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
