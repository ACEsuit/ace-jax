"""Per-stage CPU cost of the ace-jax force call (docs/dev/cpu-gap-profile.md).

    PYTHONPATH=bench:src taskset -c 0 python bench/perf/cpu_stages.py <model> <system> <n> \
        [--dtype float64] [--reps 10] [--threads 1] [--dump DIR] [--trace DIR]

The model is the one the calculator evaluates (`ACECalculator`'s `eval_model`:
`lean(model)` for a linear ACE `.npz`, the `PACEModel` as loaded for a `.yace`),
on the compact (n, K) in-cutoff graph the skin step hands it (K = the row
capacity `round_k(max neighbours)`).  Each stage runs on its own inside one
jit, inputs precomputed, forward and forward + VJP (`jax.vjp` with a random
cotangent, so the VJP time includes recomputing that stage's forward):

ACE (lean, l-blocked, species-compact):
  radial    r, transform, envelope, the R_nl spline (blk_rnl_coefs) per slot
  pair      the pair radial per slot
  angular   Y_lm per slot
  A         the l-blocked A assembly (one-hot z_j expansion + per-l einsum)
  AA        the product basis prod A[spec] (`_aa`, blk_aa_specs)
  readout   AA . ctilde[:, z] + pooled pair . Wpair + E0
PACE:
  radbase   bond parameters, r, the radial basis g_k (SBessel/...) per slot
  core      core repulsion / ZBL per slot
  angular   Y_lm per slot
  A         pool-first A (one-hot channel expansion, einsum, sel_y, W[z])
  AA        the product basis prod At[spec]
  rho       ctilde_real, rho = ct^T AA, gather by node species
  tail      embedding F(rho), the core switch, E0

and the whole: `efv` (energy_forces_virial_dense, return_edge_grad=True: the
energy, its gradient wrt every rij and the virial), `energy` (forward only) and
`step` (the calculator's compiled skin step: compaction + efv + force gather).
For every jitted function XLA's cost analysis (flops, bytes accessed) and the
number of top-level thunks of the compiled CPU program are recorded.

`--dump DIR` sets --xla_dump_to (with LLVM IR) for the `efv` program, and the
script then counts, per fusion, whether its optimised IR uses SIMD vectors
(<N x double> / <N x float>) -- the "does XLA:CPU vectorise it" check.
`--trace DIR` records a jax.profiler trace of `efv` for `cpu_trace.py`.
Prints one JSON object.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time


def timeit(fn, *a, reps=10, warm=2):
    import jax
    for _ in range(warm):
        jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def analyse(fn, *a):
    """XLA cost analysis and the compiled program's top-level instruction count."""
    try:
        comp = fn.lower(*a).compile()
        c = comp.cost_analysis()
        c = c[0] if isinstance(c, list) else c
        txt = comp.as_text()
        entry = txt[txt.find("ENTRY"):]
        body = entry[:entry.find("\n}")]
        ops = [ln for ln in body.splitlines()[1:] if "=" in ln]
        kinds = {}
        for ln in ops:
            m = re.search(r"=\s*[\w\[\]{},:\s\d]*?\b([a-z][\w\-]*)\(", ln)
            k = m.group(1) if m else "?"
            if k in ("parameter", "get-tuple-element", "tuple", "constant", "bitcast"):
                continue
            kinds[k] = kinds.get(k, 0) + 1
        return {"flops": float(c.get("flops", -1)), "bytes": float(c.get("bytes accessed", -1)),
                "transcendentals": float(c.get("transcendentals", -1)),
                "thunks": sum(kinds.values()), "thunk_kinds": kinds}
    except Exception as ex:                                        # noqa: BLE001
        return {"error": repr(ex)[:200]}


def stage_fns(model, kind):
    """{name: (f(*inputs) -> output, input builder from the cache)} in pipeline
    order; each builder takes the dict of earlier stages' outputs."""
    import jax
    import jax.numpy as jnp
    hi = jax.lax.Precision.HIGHEST
    if kind == "ace":
        from ace_jax.eval.model import pool_dense
        from ace_jax.eval.radial import agnesi_normalized, env_poly2sx
        m = model

        def radial(r3, zi, zj, mk):
            r = jnp.linalg.norm(r3, axis=-1)
            env = env_poly2sx(agnesi_normalized(r, m.rnl_transform[zi, zj]), m.rnl_envelope[zi, zj])
            coefs = m.blk_rnl_coefs if m.blk_compact else m.rnl_coefs
            R = m._radial_one(r, zi, zj, "spline", m.rnl_transform, coefs, m.rnl_grid,
                              None, None, env)
            return jnp.where(mk[:, None], R, 0.0)

        def pair(r3, zi, zj):
            return m.pair_radial(r3, zi, zj)

        def angular(r3):
            return m.angular(r3)

        def assemble(R, Y, zj, n, K):
            E = n * K
            nz = m.E0.shape[0]
            oh = jax.nn.one_hot(zj, nz, dtype=R.dtype) if m.blk_compact else None
            At = []
            for l, off, w in m.blk:
                Rl = R[:, off:off + w]
                if m.blk_compact:
                    Rl = (oh[:, :, None] * Rl[:, None, :]).reshape(E, nz * w)
                Yl = Y[:, l * l:(l + 1) ** 2]
                At.append(jnp.einsum("nkr,nky->ryn", Rl.reshape(n, K, -1), Yl.reshape(n, K, -1),
                                     precision=hi).reshape(-1, n))
            return jnp.concatenate(At, axis=0).T

        def aa(A):
            return m._aa(A, m.blk_aa_specs)

        def readout(AA, Rpair, mask, node_z):
            n, K = mask.shape
            P = pool_dense(Rpair.reshape(n, K, -1), mask)
            e = jnp.einsum("ia,ai->i", AA, m.ctilde[:, node_z])
            e = e + jnp.einsum("ip,pi->i", P, m.Wpair[:, node_z])
            return e + m.E0[node_z]

        return [("radial", radial, lambda c: (c["r3"], c["zi"], c["zj"], c["mk"])),
                ("pair", pair, lambda c: (c["r3"], c["zi"], c["zj"])),
                ("angular", angular, lambda c: (c["r3"],)),
                ("A", lambda R, Y, zj: assemble(R, Y, zj, c_n[0], c_n[1]),
                 lambda c: (c["radial"], c["angular"], c["zj"])),
                ("AA", aa, lambda c: (c["A"],)),
                ("readout", readout, lambda c: (c["AA"], c["pair"], c["mask"], c["node_z"]))]
    m = model

    def radbase(r3, zi, zj, mk):
        from ace_jax.eval.pace_radial import radbase as rb
        bp, valid, r, rij_s = m._geometry(r3, zi, zj, mk)
        lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
        g = rb(r, m.radbasename, m.inner_cutoff_type, lam, rc, dcut, cin, dcin, m.nradbase,
               m.sbessel_form)
        return jnp.where(valid[:, None], g, 0.0)

    def angular(r3, zi, zj, mk):
        from ace_jax.eval.harmonics import real_spherical_harmonics
        _, _, _, rij_s = m._geometry(r3, zi, zj, mk)
        return real_spherical_harmonics(rij_s, m.lmax)

    def core(r3, zi, zj, mk):
        cr, d, dcin = m._edge_core(r3, zi, zj, mk)
        return cr, d, dcin

    def A(g, Y, zj2, node_z):
        n, K = zj2.shape
        return m.pool_first_dense(g.reshape(n, K, -1), Y.reshape(n, K, -1), zj2, node_z)

    def AA(At):
        return jnp.concatenate([jnp.prod(At[s.T], axis=0) for s in m.aa_specs], axis=0)

    def rho(AAv, node_z):
        n = AAv.shape[1]
        ct = m.ctilde_real().reshape(m.n_aa, -1)
        rho_all = jnp.matmul(ct.T, AAv, precision=hi)
        return rho_all.reshape(m.nz, m.ndensity, n)[node_z, :, jnp.arange(n)]

    def tail(rh, crdd, node_z, seg):
        cr, d, dcin = crdd
        return m._energy_tail(rh, cr, d, dcin, seg, rh.shape[0], node_z)

    return [("radbase", radbase, lambda c: (c["r3"], c["zi"], c["zj"], c["mk"])),
            ("core", core, lambda c: (c["r3"], c["zi"], c["zj"], c["mk"])),
            ("angular", angular, lambda c: (c["r3"], c["zi"], c["zj"], c["mk"])),
            ("A", A, lambda c: (c["radbase"], c["angular"], c["zj2"], c["node_z"])),
            ("AA", AA, lambda c: (c["A"],)),
            ("rho", rho, lambda c: (c["AA"], c["node_z"])),
            ("tail", tail, lambda c: (c["rho"], c["core"], c["node_z"], c["seg"]))]


c_n = [0, 0]           # (n, K) for the ACE assembly stage's static reshape


def vectorised_fusions(dump):
    """Per optimised-IR module of the efv program: the functions (one per fused
    kernel / loop) whose body uses SIMD vector types."""
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(dump, "*efv*ir-with-opt.ll"))
                    + glob.glob(os.path.join(dump, "*ir-with-opt.ll"))):
        if f in out:
            continue
        txt = open(f).read()
        fns = re.split(r"\ndefine ", txt)[1:]
        tot = len(fns)
        vec = sum(1 for b in fns if re.search(r"<(\d+) x (double|float)>", b))
        widths = {}
        for w, t in re.findall(r"<(\d+) x (double|float)>", txt):
            widths[f"{w}x{t}"] = widths.get(f"{w}x{t}", 0) + 1
        out[os.path.basename(f)] = {"functions": tot, "vectorised": vec, "vector_types": widths}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("system")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--xla-extra", default="")
    a = ap.parse_args()
    flags = os.environ.get("XLA_FLAGS", "")
    if a.threads == 1:
        flags += " --xla_cpu_multi_thread_eigen=false"
    if a.dump:
        flags += f" --xla_dump_to={a.dump} --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*efv.*"
    os.environ["XLA_FLAGS"] = (flags + " " + a.xla_extra).strip()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", a.dtype == "float64")
    import jax.numpy as jnp
    import numpy as np

    from ace_jax.calc.point import ACECalculator
    from ace_jax.calc.skin import round_k
    from ace_jax.eval.nlist import dense_from_sparse, sparse_graph
    from scaling.structures import supercell

    dt = getattr(jnp, a.dtype)
    calc = ACECalculator(a.model, dtype=dt)
    m = calc.eval_model
    kind = "pace" if a.model.endswith(".yace") else "ace"
    at = supercell(a.system, a.n)
    rc = calc.cutoff
    g = sparse_graph(at.positions, at.cell.array, at.pbc, rc)
    kmax = int(np.bincount(np.asarray(g.senders), minlength=a.n).max())
    K = round_k(kmax)
    d = dense_from_sparse(g, rc, max_neighbours=K)
    lut = {int(z): i for i, z in enumerate(calc.meta["elements"])}
    node_z = jnp.asarray([lut[int(z)] for z in at.numbers], jnp.int32)
    idx = jnp.asarray(d.idx, jnp.int32)
    n = a.n
    mask = jnp.arange(K)[None, :] < jnp.asarray(d.count)[:, None]
    pad = np.asarray([1.0, 0.0, 0.0]) * float(m.pad_cutoff())
    rij = jnp.asarray(np.where(np.asarray(mask)[..., None], d.rij, pad), dt)
    zi = jnp.broadcast_to(node_z[:, None], idx.shape)
    zj2 = node_z[idx]
    c_n[0], c_n[1] = n, K
    cache = {"r3": rij.reshape(n * K, 3), "zi": zi.reshape(-1), "zj": zj2.reshape(-1),
             "mk": mask.reshape(-1), "mask": mask, "node_z": node_z, "zj2": zj2,
             "seg": jnp.repeat(jnp.arange(n), K)}
    out = {"model": os.path.basename(a.model), "kind": kind, "system": a.system, "n": n, "K": K,
           "n_edges": int(np.asarray(d.count).sum()), "fill": float(np.asarray(d.count).sum() / (n * K)),
           "dtype": a.dtype, "cpus": sorted(os.sched_getaffinity(0)), "XLA_FLAGS": os.environ["XLA_FLAGS"],
           "jax": jax.__version__, "loadavg_before": open("/proc/loadavg").read().split()[:3]}
    if kind == "ace":
        out["dims"] = {"blk": [list(b) for b in m.blk], "blk_compact": bool(m.blk_compact),
                       "n_A": int(sum((m.E0.shape[0] if m.blk_compact else 1) * w * (2 * l + 1)
                                      for l, w_off, w in [(b[0], b[1], b[2]) for b in m.blk])),
                       "n_AA_per_order": [int(s.shape[0]) for s in m.blk_aa_specs],
                       "orders": [int(s.shape[1]) for s in m.blk_aa_specs],
                       "lmax": int(m.lmax), "NZ": int(m.E0.shape[0]),
                       "rnl_cols": int((m.blk_rnl_coefs if m.blk_compact else m.rnl_coefs).shape[-1]),
                       "ncoef": int(m.rnl_coefs.shape[2])}
    else:
        out["dims"] = {"nradbase": int(m.nradbase), "lmax": int(m.lmax), "NZ": int(m.nz),
                       "n_a_local": int(m.n_a_local), "n_aa": int(m.n_aa),
                       "n_AA_per_order": [int(s.shape[0]) for s in m.aa_specs],
                       "orders": [int(s.shape[1]) for s in m.aa_specs],
                       "ndensity": int(m.ndensity), "radbase": m.radbasename,
                       "sbessel_form": m.sbessel_form}
    stages = {}
    for name, f, inp in stage_fns(m, kind):
        args = inp(cache)
        fj = jax.jit(f)
        y = jax.block_until_ready(fj(*args))
        cache[name] = y
        diff = [i for i, x in enumerate(args) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)]

        def fwdbwd(*xs, f=f, diff=diff, y=y):
            def g(*dx):
                full = list(xs)
                for i, v in zip(diff, dx):
                    full[i] = v
                return f(*full)
            out_, vjp = jax.vjp(g, *[xs[i] for i in diff])
            ct = jax.tree.map(lambda o: jnp.ones_like(o) if jnp.issubdtype(o.dtype, jnp.floating)
                              else None, out_)
            return vjp(ct)
        fb = jax.jit(fwdbwd)
        t_f = timeit(fj, *args, reps=a.reps)
        t_fb = timeit(fb, *args, reps=a.reps) if diff else None
        stages[name] = {"fwd_s": t_f, "fwd_vjp_s": t_fb, "fwd": analyse(fj, *args),
                        "fwd_vjp": analyse(fb, *args) if diff else None,
                        "out_bytes": int(sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(y)))}
    out["stages"] = stages
    # the whole: energy, efv, and the calculator's step
    targs = (rij, zi, zj2, idx, mask, node_z)

    def efv(mm, *xs):
        with jax.default_matmul_precision("highest"):
            return mm.energy_forces_virial_dense(*xs, return_edge_grad=True)

    def energy(mm, rij, zi, zj, idx, mask, node_z):
        with jax.default_matmul_precision("highest"):
            return jnp.sum(mm.site_energies_dense(rij, zi, zj, mask, node_z))
    import equinox as eqx
    efv_j = eqx.filter_jit(efv)
    en_j = eqx.filter_jit(energy)
    out["efv_s"] = timeit(efv_j, m, *targs, reps=a.reps)
    out["energy_s"] = timeit(en_j, m, *targs, reps=a.reps)
    params, static = eqx.partition(m, eqx.is_array)
    efv_named = jax.jit(lambda p, *xs: efv(eqx.combine(p, static), *xs))
    efv_named.__name__ = "efv"
    out["efv_cost"] = analyse(efv_named, params, *targs)
    out["energy_cost"] = analyse(jax.jit(lambda p, *xs: energy(eqx.combine(p, static), *xs)),
                                 params, *targs)
    # the calculator step (compaction + efv + gather), timed as microbench does
    from ase.calculators.calculator import all_changes
    calc.calculate(at, ["energy", "forces", "stress"], all_changes)
    st = calc._skin_state
    u = jax.device_put(st.displacements(at.positions, np.dtype(a.dtype)))
    stepf = calc._step()
    out["step_s"] = timeit(lambda: stepf(u, st.arrays, K=st.K), reps=a.reps)
    out["step_K"], out["step_K_skin"] = st.K, st.K_skin
    out["stage_sum_fwd_s"] = sum(v["fwd_s"] for v in stages.values())
    out["stage_sum_fwd_vjp_s"] = sum((v["fwd_vjp_s"] or v["fwd_s"]) for v in stages.values())
    if a.trace:
        jax.block_until_ready(efv_named(params, *targs))
        with jax.profiler.trace(a.trace):
            for _ in range(5):
                jax.block_until_ready(efv_named(params, *targs))
        out["trace_calls"] = 5
    if a.dump:
        out["vectorisation"] = vectorised_fusions(a.dump)
    out["loadavg_after"] = open("/proc/loadavg").read().split()[:3]
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
