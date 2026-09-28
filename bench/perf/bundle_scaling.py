"""The lammps-jax bundle's energy+forces function, timed outside LAMMPS.

Why LAMMPS throughput of ace-jax falls above ~16k atoms (docs/perf-lammps-large-n.md).

For one model and atom count this builds the inputs lammps-jax hands the
bundle -- positions of owned atoms then ghosts (max_atoms rows), species,
nlocal, nghost and a padded (max_edges,) edge list of owned senders within
rcut, sized by bench/scaling/run_lammps.py::capacity -- and times the fused
program lammps-jax runs every step: value_and_grad of the masked energy sum
with respect to all max_atoms positions (a copy of
lammps_jax.export.wrap_energy_fn, which is not on PyPI), jitted and compiled
ahead of time, under jax.default_matmul_precision("highest") as the export is.

Edge order: lammps-jax (origin/HEAD, PackNeighborFunctor) appends edges with an
atomic counter over a flat index that runs atom-fastest, so consecutive edges
belong to different senders ("interleaved": sorted by (neighbour rank, atom));
"rows" groups each sender's edges, rows in random order.

Variants, "base[@chunk][!]" (bundle_scaling.parse_variant): "@chunk" evaluates
site_energies_dense in lax.map blocks of <= chunk rows under jax.checkpoint
(the calculator's scheme), "!" drops the checkpoint.  Full programs:
  stock          export.lammps.make_energy_fn (dense, n_rows = max_owned,
                 k_dense = k(rcut + skin) + 8), chunked if "@"
  k_tight        the same with k_dense = k(rcut) + 8
  filter_tight   k_tight, first masking packed pairs beyond rcut
  sparse         make_energy_fn(layout="sparse")
  chunked        legacy name for stock@16384 (CHUNK_NODES)
Components (not full programs):
  pack           the dense packing alone (argsort, scatters), forward
  model_fb       value_and_grad of sum(site_energies_dense) w.r.t. the packed
                 (n_rows, k_dense, 3) edge vectors ("@chunk" as above)
--trace adds a per-kernel GPU table (hlo_trace.kernel_table).

    PYTHONPATH=bench:src python bench/perf/bundle_scaling.py pace_Cantor_medium 1024 \
        [--variants stock,chunked] [--order interleaved]
Prints one JSON object per variant.  On Modal: modal_bundle_scaling.py.
"""
import argparse
import json
import statistics
import time

import numpy as np

CHUNK = 16384


# ----------------------------------------------------------------- inputs
def lammps_inputs(at, rcut, cap, order="interleaved", seed=0, edge_cut=None):
    """numpy arrays shaped as lammps-jax passes them, plus bookkeeping.
    edge_cut: the pair distance packed (default rcut; rcut + skin mimics a
    lammps-jax that packs the whole neighbour list, skin pairs included)."""
    from ace_jax.eval import sparse_graph
    N = len(at)
    cell = at.cell.array
    pos = at.positions
    k_rcut = int(np.bincount(sparse_graph(pos, cell, at.pbc, rcut).senders, minlength=N).max())
    g = sparse_graph(pos, cell, at.pbc, edge_cut or rcut)
    S = np.rint(g.shifts @ np.linalg.inv(cell)).astype(np.int64)
    assert np.abs(S).max() <= 1, "cell thinner than rcut: ghost images beyond one shell"
    code = (S[:, 0] + 1) * 9 + (S[:, 1] + 1) * 3 + (S[:, 2] + 1)
    gh = code != 13
    key = g.receivers[gh].astype(np.int64) * 27 + code[gh]
    uk, inv = np.unique(key, return_inverse=True)
    ng = len(uk)
    recv = g.receivers.astype(np.int64).copy()
    recv[gh] = N + inv
    c = uk % 27
    Su = np.stack([c // 9 - 1, (c // 3) % 3 - 1, c % 3 - 1], 1)
    max_atoms, max_edges = cap["max_atoms"], cap["max_edges"]
    # LAMMPS's ghost count: the rcut + skin shell (capacity sized max_atoms at 1.1x it)
    nghost = max(ng, int(max_atoms / 1.1) - N)
    assert N + nghost <= max_atoms, (N, nghost, max_atoms)
    P = np.zeros((max_atoms, 3))
    P[:N] = pos
    P[N:N + ng] = pos[uk // 27] + Su @ cell
    P[N + ng:N + nghost] = pos[0] + 1e4                 # unreferenced skin-shell ghosts
    send = g.senders.astype(np.int64)
    E = len(send)
    assert E <= max_edges, (E, max_edges)
    assert np.allclose(P[recv] - P[send], g.rij)
    rank = np.arange(E) - np.searchsorted(send, send)   # senders come sorted
    if order == "interleaved":
        o = np.lexsort((send, rank))
    elif order == "rows":
        perm = np.random.default_rng(seed).permutation(N)
        o = np.lexsort((rank, perm[send]))
    else:
        o = np.arange(E)
    s = np.full(max_edges, max_atoms, np.int32)
    r = np.full(max_edges, max_atoms, np.int32)
    m = np.zeros(max_edges, bool)
    s[:E], r[:E], m[:E] = send[o], recv[o], True
    species = np.zeros(max_atoms, np.int32)
    return {"positions": P, "species": species, "nlocal": N, "nghost": nghost,
            "senders": s, "receivers": r, "edge_mask": m, "n_edges": E, "n_ghost_used": ng,
            "k_rcut": k_rcut,
            "k_packed": int(np.bincount(send, minlength=N).max())}


# ----------------------------------------------------------------- lammps-jax wrapper
class Graph:
    def __init__(self, senders, receivers, edge_mask):
        self.senders, self.receivers, self.edge_mask = senders, receivers, edge_mask


def fused_program(energy_fn, max_atoms, dtype):
    """lammps_jax.export.wrap_energy_fn's energy_and_forces program (one rank,
    owned_rows_only False), as a plain function of the seven ABI arrays."""
    import jax
    import jax.numpy as jnp

    def anchor(*values):
        a = jnp.zeros((), dtype)
        for v in values:
            a += jnp.finfo(dtype).tiny * jnp.sum(jax.lax.stop_gradient(jnp.asarray(v, dtype)))
        return a

    def export_energy(positions, species, nlocal, nghost, senders, receivers, edge_mask):
        graph = Graph(senders, receivers, edge_mask)
        e = jnp.asarray(energy_fn(positions, species, graph), dtype)
        i = jnp.arange(max_atoms)
        zero = jnp.zeros((), dtype)
        local = jnp.sum(jnp.where(i < nlocal, e, zero))
        total = jnp.sum(jnp.where(i < nlocal + nghost, e, zero))
        ghost_sender = jnp.any(edge_mask & (senders >= nlocal))
        return jnp.where(ghost_sender, total, local) + anchor(
            positions, species, nlocal, nghost, senders, receivers, edge_mask)

    def energy_and_forces(*args):
        v, f = jax.value_and_grad(lambda p: -export_energy(p, *args[1:]))(args[0])
        return -v, f + anchor(*args[1:]) * jnp.zeros_like(f)

    return energy_and_forces


# ----------------------------------------------------------------- energy-fn variants
def chunked_rows(model, rd, zi, zj, md, zr, chunk=CHUNK, ckpt=True):
    """site_energies_dense in lax.map blocks of <= chunk rows; per-row energies."""
    import jax
    import jax.numpy as jnp
    n, K = md.shape
    nb = max(1, -(-n // chunk))
    if nb == 1:
        return model.site_energies_dense(rd, zi, zj, md, zr)
    B = -(-n // nb)
    pad = nb * B - n
    park = jnp.asarray([1.0, 0.0, 0.0], rd.dtype) * model.pad_cutoff()

    def blocks(a, fill):
        if pad:
            filler = (jnp.broadcast_to(fill, (pad,) + a.shape[1:]).astype(a.dtype))
            a = jnp.concatenate([a, filler])
        return a.reshape((nb, B) + a.shape[1:])

    f = lambda a: model.site_energies_dense(*a)
    if ckpt:
        f = jax.checkpoint(f)
    xs = (blocks(rd, park), blocks(zi, 0), blocks(zj, 0), blocks(md, False), blocks(zr, 0))
    return jax.lax.map(f, xs).reshape(-1)[:n]


def dense_pack(positions, species, graph, *, pad_cutoff, n_species, k_dense, n_rows, type_map):
    """make_energy_fn's dense packing, verbatim: (rd, zi, zj, md, zr, overflow, n, nr)."""
    import jax
    import jax.numpy as jnp
    n = positions.shape[0]
    nr = min(n_rows, n)
    m = graph.edge_mask
    s = jnp.where(m, graph.senders, 0)
    r = jnp.where(m, graph.receivers, 0)
    node_z = jnp.clip(species, 0, n_species - 1)
    if type_map is not None:
        node_z = jnp.asarray(type_map, jnp.int32)[node_z]
    pad = jnp.asarray([1.0, 0.0, 0.0], positions.dtype) * pad_cutoff
    rij = jnp.where(m[:, None], positions[r] - positions[s], pad)
    in_range = m & (s < nr)
    key = jnp.where(in_range, s, nr)
    order = jnp.argsort(key, stable=True)
    ks = key[order]
    counts = jax.ops.segment_sum(jnp.ones_like(ks), ks, num_segments=nr + 1)
    starts = jnp.cumsum(counts) - counts
    slot_sorted = jnp.arange(ks.shape[0]) - starts[ks]
    slot = jnp.zeros_like(slot_sorted).at[order].set(slot_sorted)
    ok = in_range & (slot < k_dense)
    row = jnp.where(ok, s, nr)
    col = jnp.where(ok, slot, 0)
    rd = jnp.broadcast_to(pad, (nr, k_dense, 3)).at[row, col].set(rij, mode="drop")
    idx = jnp.zeros((nr, k_dense), jnp.int32).at[row, col].set(r, mode="drop")
    md = jnp.zeros((nr, k_dense), bool).at[row, col].set(True, mode="drop")
    zr = node_z[:nr]
    overflow = jnp.any(m & ((s >= nr) | (slot >= k_dense)))
    return rd, jnp.broadcast_to(zr[:, None], idx.shape), node_z[idx], md, zr, overflow, n, nr


def parse_variant(v):
    """"base[@chunk][!]": chunk rows per lax.map block (none: unchunked), "!" no
    checkpoint.  Legacy names: chunked = stock@CHUNK, chunked_nockpt = stock@CHUNK!,
    filter_tight_chunked = filter_tight@CHUNK, model_fb_chunked = model_fb@CHUNK."""
    legacy = {"chunked": f"stock@{CHUNK}", "chunked_nockpt": f"stock@{CHUNK}!",
              "filter_tight_chunked": f"filter_tight@{CHUNK}",
              "model_fb_chunked": f"model_fb@{CHUNK}"}
    v = legacy.get(v, v)
    ckpt = not v.endswith("!")
    v = v.rstrip("!")
    base, _, chunk = v.partition("@")
    return base, int(chunk) if chunk else None, ckpt


def make_filtered_energy_fn(model, n_species, k_dense, n_rows, rcut, type_map=None,
                            chunk=None, ckpt=True):
    """Drop packed pairs beyond rcut (skin pairs, zero weight) from the mask,
    then make_energy_fn (or the chunked form) with k_dense sized for rcut."""
    import jax.numpy as jnp
    from ace_jax.export.lammps import make_energy_fn
    inner = (make_energy_fn(model, n_species, "dense", k_dense, type_map, n_rows=n_rows)
             if chunk is None else
             make_chunked_energy_fn(model, n_species, k_dense, n_rows, type_map, chunk, ckpt))

    def energy_fn(positions, species, graph):
        m = graph.edge_mask
        s = jnp.where(m, graph.senders, 0)
        r = jnp.where(m, graph.receivers, 0)
        d = positions[r] - positions[s]
        keep = m & (jnp.sum(d * d, axis=1) < rcut * rcut)
        return inner(positions, species, Graph(graph.senders, graph.receivers, keep))

    return energy_fn


def make_chunked_energy_fn(model, n_species, k_dense, n_rows, type_map=None, chunk=CHUNK,
                           ckpt=True):
    """make_energy_fn (dense) with the model evaluated in row blocks."""
    import jax.numpy as jnp

    def energy_fn(positions, species, graph):
        rd, zi, zj, md, zr, overflow, n, nr = dense_pack(
            positions, species, graph, pad_cutoff=model.pad_cutoff(), n_species=n_species,
            k_dense=k_dense, n_rows=n_rows, type_map=type_map)
        e = chunked_rows(model, rd, zi, zj, md, zr, chunk, ckpt)
        if nr < n:
            e = jnp.concatenate([e, jnp.zeros((n - nr,), e.dtype)])
        return e * jnp.where(overflow, jnp.nan, 1.0)

    return energy_fn


# ----------------------------------------------------------------- timing
def _time(compiled, args, budget_s=6.0, max_reps=20):
    import jax
    jax.block_until_ready(compiled(*args))
    t0 = time.perf_counter()
    out = jax.block_until_ready(compiled(*args))
    first = time.perf_counter() - t0
    reps = int(max(3, min(max_reps, budget_s / max(first, 1e-4))))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(compiled(*args))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), reps, out


def _compile(fn, args):
    import jax
    t0 = time.perf_counter()
    with jax.default_matmul_precision("highest"):
        c = jax.jit(fn).lower(*args).compile()
    ma = c.memory_analysis()
    mem = None
    if ma is not None:
        mem = {k: int(getattr(ma, k)) for k in ("temp_size_in_bytes", "argument_size_in_bytes",
                                                "output_size_in_bytes")
               if hasattr(ma, k)}
    return c, time.perf_counter() - t0, mem


def _trace(compiled, args, trace_dir, dump_dir, calls=3):
    """Per-kernel GPU time (hlo_trace.kernel_table) over `calls` calls; the
    process must have been started with XLA_FLAGS=--xla_dump_to=<dump_dir>."""
    import os
    import sys
    import jax
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hlo_trace
    jax.block_until_ready(compiled(*args))
    with jax.profiler.trace(trace_dir):
        for _ in range(calls):
            jax.block_until_ready(compiled(*args))
    return hlo_trace.kernel_table(trace_dir, dump_dir, calls, top=25)


def run(model_name, n, variants, order="interleaved", dtype="float64", skin=1.0,
        budget_s=6.0, edge_cut="rcut", trace=None):
    import jax
    jax.config.update("jax_enable_x64", dtype == "float64")
    import jax.numpy as jnp
    from ase.data import atomic_numbers

    from ace_jax.eval import load
    from ace_jax.export.lammps import make_energy_fn
    from scaling.models import ELEMENTS
    from scaling.run_lammps import capacity
    from scaling.structures import supercell

    kind, system, size = model_name.split("_")
    path = f"{__file__.rsplit('/', 2)[0]}/scaling/models/{model_name}" + \
        (".yace" if kind == "pace" else ".npz")
    model, meta, _ = load(path)
    rcut = float(meta["rcut"])
    at = supercell(system, n)
    cap = capacity(at, rcut, skin)
    inp = lammps_inputs(at, rcut, cap, order=order,
                        edge_cut=rcut + skin if edge_cut == "skin" else rcut)
    fdt = jnp.float64 if dtype == "float64" else jnp.float32
    args = (jnp.asarray(inp["positions"], fdt), jnp.asarray(inp["species"]),
            jnp.asarray(inp["nlocal"], jnp.int32), jnp.asarray(inp["nghost"], jnp.int32),
            jnp.asarray(inp["senders"]), jnp.asarray(inp["receivers"]),
            jnp.asarray(inp["edge_mask"]))
    model_z = [int(z) for z in meta["elements"]]
    type_elements = [atomic_numbers[e] for e in ELEMENTS[system]]
    type_map = [model_z.index(z) for z in type_elements]
    type_map = None if type_map == list(range(len(model_z))) else type_map
    ns = len(type_elements)
    max_atoms, k_dense, n_rows = cap["max_atoms"], cap["k_dense"], cap["max_owned"]
    base = {"model": model_name, "n": n, "dtype": dtype, "order": order, "edge_cut": edge_cut,
            "device": jax.devices()[0].device_kind, "max_atoms": max_atoms,
            "max_edges": cap["max_edges"], "k_dense": k_dense, "k_skin": cap["k_max"],
            "k_rcut": inp["k_rcut"], "k_packed": inp["k_packed"], "max_owned": n_rows, "n_edges": inp["n_edges"],
            "nghost": inp["nghost"]}
    ref = None
    rows = []
    for v in variants:
        row = {**base, "variant": v, "status": "ok"}
        try:
            base_v, chunk, ckpt = parse_variant(v)
            if base_v in ("stock", "k_tight", "filter_tight", "sparse"):
                k = k_dense if base_v == "stock" else inp["k_rcut"] + 8
                row["k_dense"], row["chunk"], row["ckpt"] = k, chunk, ckpt
                if base_v == "sparse":
                    efn = make_energy_fn(model, ns, "sparse", None, type_map)
                elif base_v == "filter_tight":
                    efn = make_filtered_energy_fn(model, ns, k, n_rows, rcut, type_map, chunk, ckpt)
                elif chunk is None:
                    efn = make_energy_fn(model, ns, "dense", k, type_map, n_rows=n_rows)
                else:
                    efn = make_chunked_energy_fn(model, ns, k, n_rows, type_map, chunk, ckpt)
                fn = fused_program(efn, max_atoms, fdt)
                c, cs, mem = _compile(fn, args)
                med, mn, reps, out = _time(c, args, budget_s)
                E, F = float(out[0]), np.asarray(out[1])[:n]
                row["energy"] = E
                if not np.isfinite(E):
                    row["status"] = "nan"
                if ref is None and row["status"] == "ok":
                    ref = (v, E, F)
                elif ref is not None:
                    row["vs"] = ref[0]
                    row["dE"] = abs(E - ref[1])
                    row["max_dF"] = float(np.abs(F - ref[2]).max())
            else:
                pk = dict(pad_cutoff=model.pad_cutoff(), n_species=ns, k_dense=k_dense,
                          n_rows=n_rows, type_map=type_map)
                packer = lambda p, sp, s, r, m: dense_pack(p, sp, Graph(s, r, m), **pk)[:5]  # noqa: B023 (used this iteration)
                pargs = (args[0], args[1], args[4], args[5], args[6])
                if base_v == "pack":
                    fn, fargs = packer, pargs
                else:
                    with jax.default_matmul_precision("highest"):
                        packed = jax.block_until_ready(jax.jit(packer)(*pargs))
                    row["chunk"], row["ckpt"] = chunk, ckpt

                    def fn(rd, zi, zj, md, zr, chunk=chunk or 10 ** 9):
                        return jax.value_and_grad(lambda x: jnp.sum(chunked_rows(
                            model, x, zi, zj, md, zr, chunk, ckpt)))(rd)  # noqa: B023 (used this iteration)
                    fargs = packed
                c, cs, mem = _compile(fn, fargs)
                med, mn, reps, _ = _time(c, fargs, budget_s)
            row.update({"compile_s": cs, "median_s": med, "min_s": mn, "reps": reps,
                        "mem": mem, "atom_steps_per_s": n / med})
            if trace:
                import os
                import tempfile
                row["kernels"] = _trace(c, fargs if base_v in ("pack", "model_fb") else args,
                                        tempfile.mkdtemp(), os.environ.get("ACEJAX_DUMP"))
            del c
        except Exception as e:                                        # noqa: BLE001
            msg = str(e)
            row["status"] = "oom" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()) \
                else "error"
            row["error"] = msg[:600]
        rows.append(row)
        jax.clear_caches()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("n", type=int)
    ap.add_argument("--variants", default="stock,chunked")
    ap.add_argument("--order", default="interleaved")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--budget", type=float, default=6.0)
    ap.add_argument("--edge-cut", default="rcut", choices=["rcut", "skin"])
    ap.add_argument("--trace", action="store_true",
                    help="per-kernel table; set XLA_FLAGS=--xla_dump_to=$ACEJAX_DUMP")
    a = ap.parse_args()
    for row in run(a.model, a.n, a.variants.split(","), a.order, a.dtype, budget_s=a.budget,
                   edge_cut=a.edge_cut, trace=a.trace):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
