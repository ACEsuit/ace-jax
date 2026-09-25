"""Prototype: build PACE's A basis per node by a batched outer product.

Sparse (current):  A[i, a] = sum_{e in i} R_e[aspec_r[a]] * Y_e[aspec_y[a]]
                   -- per-edge (E, n_A) columns, then segment_sum.  The column
                   gathers' adjoints are the scatters that dominate GPU forces.
Dense (candidate): A_full[i] = sum_k R_ik (x) Y_ik  over a padded (n, K)
                   neighbour layout (batched GEMM), then A = A_full[:, sel]:
                   no per-edge column gather; the only gather is per node.

Both feed the same per-node energy (AA, rho, embedding), so any difference is
A construction.  Core/ZBL terms are left out of both (identical, per-edge
scalars).  Memory of the dense form grows as n * NZ * n_radial_cols * (lmax+1)^2
plus the (n, K) padding, so the sparse form stays for large or many-element
systems; this measures where the line is.

    python bench/pace_dense_proto.py check  c_ace.yace
    python bench/pace_dense_proto.py run    c_ace.yace          # all variants
Each timed case runs in its own process (peak GPU memory never resets).
"""
import json
import subprocess
import sys
import time

CASES = [(v, dt, n) for n in (8, 12) for dt in ("float32", "float64") for v in ("sparse", "dense")]
# integrated: the real model methods (sparse in both A-forms, and dense)
MODEL_CASES = [(v, dt, n) for n in (8, 12) for dt in ("float32", "float64")
               for v in ("sparse-gather", "sparse-matmul", "dense")]


def setup(yace, n_rep, dtype):
    import jax.numpy as jnp
    import numpy as np
    from ase.build import bulk
    from ace_jax.eval import load, sparse_graph
    from ace_jax.eval.nlist import dense_graph
    at = bulk("C", "diamond", a=3.567, cubic=True).repeat(n_rep)
    at.rattle(0.02, seed=3)
    dt = getattr(jnp, dtype)
    m, meta, _ = load(yace, dtype=dt)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    K = int(np.bincount(g.senders, minlength=len(at)).max())
    d = dense_graph(at.positions, at.cell.array, at.pbc, meta["rcut"], K)
    return m, at, g, d, dt


def cols_and_Y(m, rij, zi, zj, mask):
    """Per-edge radial columns [g_k | R_nl] and real Y, zero on invalid edges --
    the two factors PACEModel.edge_a_rows multiplies (prototype copy)."""
    import jax.numpy as jnp
    from ace_jax.eval.harmonics import real_spherical_harmonics
    from ace_jax.eval.pace_radial import radbase
    r2 = jnp.sum(rij * rij, axis=-1)                 # PACEModel._edges' geometry
    bp = m.radparams[zi, zj]
    lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
    valid = r2 < rc * rc
    if mask is not None:
        valid = valid & mask
    r = jnp.sqrt(jnp.where(valid, r2, 1.0)) * jnp.where(valid, 1.0, 0.5 * rc)
    rij_s = jnp.where(valid[:, None], rij, jnp.stack([r, 0 * r, 0 * r], -1))
    gk = radbase(r, m.radbasename, m.inner_cutoff_type, lam, rc, dcut, cin, dcin, m.nradbase)
    R = jnp.einsum("ek,enlk->enl", gk, m.crad[zi, zj]).reshape(gk.shape[0], -1)
    cols = jnp.where(valid[:, None], jnp.concatenate([gk, R], -1), 0.0)
    return cols, real_spherical_harmonics(rij_s, m.lmax)


def node_energy(m, A, node_z):
    """PACEModel.site_energies from the pooled A onward (core terms omitted)."""
    import jax.numpy as jnp
    n = A.shape[0]
    AA = jnp.concatenate([jnp.prod(A[:, s], axis=-1) for s in m.aa_specs], axis=-1)
    rho = (AA @ m.ctilde_real().reshape(m.n_aa, -1)).reshape(n, m.nz, m.ndensity)
    return m._embedding(rho[jnp.arange(n), node_z], node_z) + m.E0[node_z]


def energy_sparse(m, rij, zi, zj, seg, n, node_z):
    import jax
    cols, Y = cols_and_Y(m, rij, zi, zj, None)
    eA = m.edge_a(cols, Y)                           # the model's current A-form
    A = jax.ops.segment_sum(eA, seg * m.nz + zj, num_segments=n * m.nz)
    return node_energy(m, A.reshape(n, -1), node_z)


def energy_dense(m, rij_d, zj_d, mask_d, node_z):
    """rij_d (n, K, 3), zj_d / mask_d (n, K)."""
    import jax
    import jax.numpy as jnp
    n, K = mask_d.shape
    zi_d = jnp.broadcast_to(node_z[:, None], (n, K))
    cols, Y = cols_and_Y(m, rij_d.reshape(-1, 3), zi_d.reshape(-1), zj_d.reshape(-1),
                         mask_d.reshape(-1))
    nc, ny = cols.shape[1], Y.shape[1]
    cols = cols.reshape(n, K, nc)
    if m.nz > 1:                                  # neighbour-species channel
        oh = jax.nn.one_hot(zj_d, m.nz, dtype=cols.dtype)
        cols = (oh[..., :, None] * cols[..., None, :]).reshape(n, K, m.nz * nc)
    A_full = jnp.einsum("nkr,nky->nry", cols, Y.reshape(n, K, ny),
                        precision=jax.lax.Precision.HIGHEST).reshape(n, -1)
    mu = jnp.arange(m.nz)[:, None]
    sel = ((mu * nc + m.aspec_r[None, :]) * ny + m.aspec_y[None, :]).reshape(-1)
    return node_energy(m, A_full[:, sel], node_z)


def check(yace):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    m, at, g, d, dt = setup(yace, 2, "float64")
    nz = jnp.zeros(len(at), jnp.int32)
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    rij = jnp.asarray(g.rij, dt)
    E_s, G_s = jax.value_and_grad(lambda x: jnp.sum(energy_sparse(m, x, nz[s], nz[r], s, len(at), nz)))(rij)
    mask = jnp.asarray(d.mask); idx = jnp.asarray(d.idx)
    rd = jnp.asarray(d.rij, dt)
    E_d, G_d = jax.value_and_grad(lambda x: jnp.sum(energy_dense(m, x, nz[idx], mask, nz)))(rd)
    # per-atom forces from edge gradients, both layouts
    F_s = np.zeros((len(at), 3)); np.add.at(F_s, np.asarray(g.senders), np.asarray(G_s))
    np.add.at(F_s, np.asarray(g.receivers), -np.asarray(G_s))
    Gd = np.asarray(G_d); mk = np.asarray(d.mask)
    F_d = np.zeros((len(at), 3))
    ii = np.repeat(np.arange(len(at)), mk.shape[1]).reshape(mk.shape)
    np.add.at(F_d, ii[mk], Gd[mk]); np.add.at(F_d, np.asarray(d.idx)[mk], -Gd[mk])
    out = {"dE": float(abs(E_s - E_d)), "E": float(E_s), "max_dF": float(np.abs(F_s - F_d).max()),
           "max_F": float(np.abs(F_s).max())}
    print(json.dumps(out))
    assert out["dE"] < 1e-9 * abs(out["E"]) and out["max_dF"] < 1e-9 * out["max_F"], out


def one(yace, variant, dtype, n_rep, reps=10):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    m, at, g, d, dt = setup(yace, n_rep, dtype)
    n = len(at)
    nz = jnp.zeros(n, jnp.int32)
    if variant == "sparse":
        s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
        x = jnp.asarray(g.rij, dt)
        e = lambda x: jnp.sum(energy_sparse(m, x, nz[s], nz[r], s, n, nz))
    else:
        idx, mask = jnp.asarray(d.idx), jnp.asarray(d.mask)
        x = jnp.asarray(d.rij, dt)
        e = lambda x: jnp.sum(energy_dense(m, x, nz[idx], mask, nz))
    fe, fg = jax.jit(e), jax.jit(jax.value_and_grad(e))

    def bench(f):
        jax.block_until_ready(f(x))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); jax.block_until_ready(f(x)); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    res = {"variant": variant, "dtype": dtype, "n_atoms": n, "n_edges": int(len(g.senders)),
           "K": int(d.idx.shape[1])}
    try:
        res["energy_s"] = bench(fe)
        res["energy_forces_s"] = bench(fg)
    except Exception as ex:                                  # e.g. out of memory
        res["error"] = repr(ex)[:200]
    res["peak_gpu_bytes"] = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    print(json.dumps(res))


def one_model(yace, variant, dtype, n_rep, reps=10):
    """energy_forces_virial[_dense] of the model itself, with the layout's
    estimate_a_bytes next to the measured peak."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from ace_jax.eval import with_edge_a_kind
    from ace_jax.eval.edge_model import estimate_a_bytes
    from ace_jax.eval.nlist import dense_from_sparse
    m, at, g, _, dt = setup(yace, n_rep, dtype)
    n = len(at)
    nz = jnp.zeros(n, jnp.int32)
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    K = int(np.bincount(g.senders, minlength=n).max())
    if variant == "dense":
        d = dense_from_sparse(g, float(np.max(np.asarray(m.radparams[..., 1]))))
        idx, mask = jnp.asarray(d.idx), jnp.asarray(d.mask)
        x = jnp.asarray(d.rij, dt)
        f = jax.jit(lambda x: m.energy_forces_virial_dense(x, jnp.broadcast_to(nz[:, None], idx.shape),
                                                           nz[idx], idx, mask, nz))
        est = estimate_a_bytes(m, "dense", n, len(g.senders), K, np.dtype(dtype).itemsize)
    else:
        mk = with_edge_a_kind(m, variant.split("-")[1])
        x = jnp.asarray(g.rij, dt)
        f = jax.jit(lambda x: mk.energy_forces_virial(x, nz[s], nz[r], s, r, n, nz))
        est = estimate_a_bytes(m, "sparse", n, len(g.senders), K, np.dtype(dtype).itemsize)
    res = {"variant": variant, "dtype": dtype, "n_atoms": n, "n_edges": int(len(g.senders)),
           "estimate_bytes": est}
    try:
        jax.block_until_ready(f(x))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); jax.block_until_ready(f(x)); ts.append(time.perf_counter() - t0)
        res["efv_s"] = float(np.median(ts))
    except Exception as ex:
        res["error"] = repr(ex)[:200]
    res["peak_gpu_bytes"] = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    print(json.dumps(res))


if __name__ == "__main__":
    mode, yace = sys.argv[1], sys.argv[2]
    if mode == "check":
        check(yace)
    elif mode == "one":
        one(yace, sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif mode == "one_model":
        one_model(yace, sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        cases, sub = (MODEL_CASES, "one_model") if mode == "model" else (CASES, "one")
        for v, dt, nr in cases:
            p = subprocess.run([sys.executable, __file__, sub, yace, v, dt, str(nr)],
                               capture_output=True, text=True,
                               env={**__import__("os").environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            print(line[-1] if line else json.dumps({"variant": v, "dtype": dt, "n_rep": nr,
                                                    "error": p.stderr[-300:]}), flush=True)
