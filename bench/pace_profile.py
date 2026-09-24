"""Force-pass profile of PACEModel: stage-by-stage forward / forward+VJP times
and the top GPU kernels of energy_forces_virial, on an n_rep^3 diamond-carbon cell.

    python bench/pace_profile.py path/to/c_ace.yace [--n-rep 8] [--reps 10]

Runs on any JAX backend (moriarty's A4500, or Modal via bench/pace_modal/run.py
--only profile).  Prints one JSON document.
"""


def run_profile(yace, reps=10, n_rep=8):
    """Where does the force pass spend its time?  Stage-by-stage forward and
    VJP timings of PACEModel.site_energies on the 4k-atom carbon cell, plus the
    top GPU kernels from a jax.profiler trace of energy_forces_virial."""
    import glob, gzip, json, sys, time
    from collections import defaultdict
    import jax, jax.numpy as jnp
    import numpy as np
    from ase.build import bulk
    from ace_jax.eval import load, sparse_graph
    from ace_jax.eval.harmonics import real_spherical_harmonics
    from ace_jax.eval.pace_radial import radbase

    at = bulk("C", "diamond", a=3.567, cubic=True).repeat(n_rep)
    at.rattle(0.02, seed=3)
    n = len(at)

    def bench(f, *args):
        jax.block_until_ready(f(*args))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    def fwd_and_vjp(fn, x):
        """(forward s, forward+VJP s) for fn: x -> array.  The cotangent is a
        random array passed in as an argument: a constant ones cotangent lets
        XLA constant-fold whole backward passes (seen on the first profile)."""
        f = jax.jit(fn)
        y = f(x)
        ct = jax.random.normal(jax.random.PRNGKey(0), y.shape, y.dtype)
        def fb(x, ct):
            _, pull = jax.vjp(fn, x)
            return pull(ct)[0]
        return bench(f, x), bench(jax.jit(fb), x, ct)

    out = {"n_atoms": n}
    for dtype in ("float32", "float64"):
        dt = getattr(jnp, dtype)
        m, meta, _ = load(yace, dtype=dt)
        g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
        s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
        nz = jnp.zeros(n, jnp.int32); zi, zj = nz[s], nz[r]
        rij = jnp.asarray(g.rij, dt)
        bp = m.radparams[zi, zj]
        rr = jnp.linalg.norm(rij, axis=-1)

        # stage functions, each differentiable in its main input
        def st_g(r):                                  # radial base g_k(r)
            return radbase(r, m.radbasename, m.inner_cutoff_type, bp[:, 0], bp[:, 1],
                           bp[:, 2], bp[:, 3], bp[:, 4], m.nradbase)
        gk = jax.jit(st_g)(rr)
        def st_R_gather(gk):                          # current: per-edge crad gather
            return jnp.einsum("ek,enlk->enl", gk, m.crad[zi, zj])
        W = m.crad[0, 0].reshape(-1, m.crad.shape[-1]).T          # (K, n*l)
        def st_R_matmul(gk):                          # candidate: one bond type, matmul
            return gk @ W
        def st_Y(x):
            return real_spherical_harmonics(x, m.lmax)
        cols = jnp.concatenate([gk, jax.jit(st_R_gather)(gk).reshape(gk.shape[0], -1)], -1)
        Y = jax.jit(st_Y)(rij)
        def st_eA(cols):                              # A-basis column gather product
            return cols[:, m.a_rad] * Y[:, m.a_y]
        eA = jax.jit(st_eA)(cols)
        def st_pool(eA):
            return jax.ops.segment_sum(eA, s * m.nz + zj, num_segments=n * m.nz)
        A = jax.jit(st_pool)(eA).reshape(n, -1)
        def st_node(A):                               # AA products, rho, embedding
            AA = jnp.concatenate([jnp.prod(A[:, q], axis=-1) for q in m.aa_specs], -1)
            rho = (AA @ m.ctilde_real().reshape(m.n_aa, -1)).reshape(n, m.nz, m.ndensity)[:, 0]
            return m._embedding(rho, nz)
        res = {"n_edges": int(len(g.senders)), "n_a_local": m.n_a_local, "n_aa": m.n_aa}
        for name, fn, x in (("radial_g", st_g, rr), ("R_crad_gather", st_R_gather, gk),
                            ("R_matmul_candidate", st_R_matmul, gk), ("Ylm", st_Y, rij),
                            ("eA_gather", st_eA, cols), ("pool_segment_sum", st_pool, eA),
                            ("node_AA_rho_F", st_node, A)):
            f, fb = fwd_and_vjp(fn, x)
            res[name] = {"fwd_s": f, "fwd_plus_vjp_s": fb}
        from ace_jax.eval import with_edge_a_kind
        for kind in ("gather", "matmul"):          # the same call, both A-basis forms
            mk = with_edge_a_kind(m, kind)
            fk = jax.jit(lambda x, mk=mk: mk.energy_forces_virial(x, zi, zj, s, r, n, nz))
            res[f"efv_{kind}_s"] = bench(fk, rij)
        en = jax.jit(lambda x: m.site_energies(x, zi, zj, s, n, nz))
        res["total_energy_s"] = bench(en, rij)

        # kernel-level: perfetto trace of energy_forces_virial, top GPU ops, per form
        for kind in ("gather", "matmul"):
            mk = with_edge_a_kind(m, kind)
            efv = jax.jit(lambda x, mk=mk: mk.energy_forces_virial(x, zi, zj, s, r, n, nz))
            tdir = f"/tmp/trace_{dtype}_{kind}"
            jax.block_until_ready(efv(rij))
            with jax.profiler.trace(tdir, create_perfetto_trace=True):
                for _ in range(3):
                    jax.block_until_ready(efv(rij))
            files = glob.glob(f"{tdir}/**/*perfetto_trace.json.gz", recursive=True)
            top, total = [], None
            if files:
                ev = json.load(gzip.open(files[0]))
                ev = ev["traceEvents"] if isinstance(ev, dict) else ev
                gpu_pids = {e["pid"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"
                            and "GPU" in str(e.get("args", {}).get("name", ""))}
                tot = defaultdict(float)
                for e in ev:
                    if e.get("ph") == "X" and e.get("pid") in gpu_pids:
                        tot[e.get("name", "?")[:90]] += e.get("dur", 0) / 1e6 / 3
                top = sorted(tot.items(), key=lambda kv: -kv[1])[:12]
                total = sum(tot.values())
            res[f"gpu_kernels_{kind}"] = {"total_s": total, "top_s": top}
        out[dtype] = res
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("yace")
    ap.add_argument("--n-rep", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    print(json.dumps(run_profile(a.yace, a.reps, a.n_rep), indent=1, default=float))
