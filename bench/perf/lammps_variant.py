"""ace-jax inside LAMMPS (lammps-jax, pair_style jax/kk): the stock dense bundle
vs a bundle that evaluates only the owned rows with cutoff-sized slots.

Targets the pre-optimisation API (the merge base with feat/bench-scaling, 2eb629f):
its variants (variants.py) call PACE methods removed by the pool-first rewrite
(`_node_energies`, `edge_a_factors`), so it no longer runs on this branch.  Kept as the evidence
behind docs/pace-performance-gap.md; bench/perf/microbench.py is the maintained
harness.

The stock bundle (export.lammps.make_energy_fn, sized by run_lammps.capacity)
runs the dense model on all max_atoms rows (owned + ghost capacity) with
k_dense = k(rcut + skin) + 8 slots.  But lammps-jax packs only owned senders
and drops pairs beyond the cutoff, so rows >= nlocal are empty and only
k(rcut) slots per row are ever live.  `owned` keeps n_rows = ceil(1.1 n_owned)
rows and k(rcut) + 4 slots, and returns NaN (never a silent truncation) if a
sender row or a slot overflows.

    PYTHONPATH=bench:src python bench/perf/lammps_variant.py Cantor_medium 8192 \
        --mode stock|owned [--variant rec+pool+fm] --work /tmp/lv
Prints one JSON row (run_lammps format + pe of the first thermo row).
"""
import argparse
import json
import math
import os
import pathlib
import sys

import numpy as np


def make_energy_fn_owned(model, n_species, k_dense, n_rows, type_map=None):
    import jax
    import jax.numpy as jnp

    def energy_fn(positions, species, graph):
        n = positions.shape[0]
        m = graph.edge_mask
        s = jnp.where(m, graph.senders, 0)
        r = jnp.where(m, graph.receivers, 0)
        node_z = jnp.clip(species, 0, n_species - 1)
        if type_map is not None:
            node_z = jnp.asarray(type_map, jnp.int32)[node_z]
        pad = jnp.asarray([1.0, 0.0, 0.0], positions.dtype) * model.pad_cutoff()
        rij = jnp.where(m[:, None], positions[r] - positions[s], pad)
        key = jnp.where(m & (s < n_rows), s, n_rows)
        order = jnp.argsort(key, stable=True)
        ks = key[order]
        counts = jax.ops.segment_sum(jnp.ones_like(ks), ks, num_segments=n_rows + 1)
        starts = jnp.cumsum(counts) - counts
        slot_sorted = jnp.arange(ks.shape[0]) - starts[ks]
        slot = jnp.zeros_like(slot_sorted).at[order].set(slot_sorted)
        ok = m & (s < n_rows) & (slot < k_dense)
        row = jnp.where(ok, s, n_rows)
        col = jnp.where(ok, slot, 0)
        rd = jnp.broadcast_to(pad, (n_rows, k_dense, 3)).at[row, col].set(rij, mode="drop")
        idx = jnp.zeros((n_rows, k_dense), jnp.int32).at[row, col].set(r, mode="drop")
        md = jnp.zeros((n_rows, k_dense), bool).at[row, col].set(True, mode="drop")
        zr = node_z[:n_rows]
        e = model.site_energies_dense(rd, jnp.broadcast_to(zr[:, None], idx.shape),
                                      node_z[idx], md, zr)
        e = jnp.concatenate([e, jnp.zeros((n - n_rows,), e.dtype)])
        overflow = jnp.any(m & ((s >= n_rows) | (slot >= k_dense)))
        return jnp.where(overflow, jnp.nan, e)

    return energy_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("n", type=int)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="stock", choices=["stock", "owned"])
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--work", default="/tmp/lammps_variant")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    from ase.data import atomic_numbers
    from ase.io import write

    from ace_jax.eval.nlist import sparse_graph
    from ace_jax.eval.pace_model import load_yace
    from ace_jax.export.lammps import export_lammps
    from scaling.models import planned_models
    from scaling.run_lammps import _run_lammps, capacity, read_pe
    from scaling.structures import supercell

    system = args.model.split("_")[0]
    row = next(r for r in planned_models()
               if r["name"] == f"acejax-pace/{system}/{args.model.split('_')[1]}")
    row["path"] = f"/ace-jax/bench/scaling/models/pace_{args.model}.yace" \
        if os.path.exists("/ace-jax") else row["path"]
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    at = supercell(system, args.n)
    data = work / "x.data"
    write(data, at, format="lammps-data", specorder=row["elements"], masses=True)

    model, meta, _ = load_yace(row["path"])
    if args.variant != "baseline":
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        import variants
        model = variants.make(args.variant, model)
    rc = float(meta["rcut"])
    cap = capacity(at, rc)
    type_elements = [atomic_numbers[e] for e in row["elements"]]
    bundle = work / "bundle.json"
    if args.mode == "stock":
        b = export_lammps(model, meta, bundle, max_atoms=cap["max_atoms"],
                          max_edges=cap["max_edges"], k_dense=cap["k_dense"], dtype=args.dtype,
                          layout="dense", type_elements=type_elements)
        extra = {"layout": b["ace_jax"]["layout"], "k_dense": cap["k_dense"],
                 "rows": cap["max_atoms"]}
    else:
        from lammps_jax.export import export_model
        g = sparse_graph(at.positions, at.cell.array, at.pbc, rc)
        k_cut = int(np.bincount(g.senders, minlength=len(at)).max()) + 4
        n_rows = int(math.ceil(1.1 * len(at)))
        model_z = [int(z) for z in meta["elements"]]
        type_map = [model_z.index(z) for z in type_elements]
        fn = make_energy_fn_owned(model, len(type_elements), k_cut, n_rows,
                                  None if type_map == list(range(len(model_z))) else type_map)
        b = export_model(energy_fn=fn, path=bundle, max_atoms=cap["max_atoms"],
                         max_edges=cap["max_edges"], cutoff=rc, unit_style="metal",
                         precision=args.dtype, n_species=len(type_elements))
        bundle.write_text(json.dumps(b, indent=2, sort_keys=True) + "\n")
        extra = {"layout": "dense-owned", "k_dense": k_cut, "rows": n_rows}
    import jax_plugins.xla_cuda12 as p
    pjrt = os.path.join(os.path.dirname(p.__file__), "xla_cuda_plugin.so")
    out = _run_lammps(row, "acejax", str(bundle), data, work, args.n, args.dtype, "gpu",
                      "/opt/lmp-jax.sh", 1, pjrt, args.steps, args.warmup,
                      {**extra, "mode": args.mode, "variant": args.variant})
    try:
        out["pe"] = read_pe((work / "log.lammps").read_text())
    except Exception:                                              # noqa: BLE001
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
