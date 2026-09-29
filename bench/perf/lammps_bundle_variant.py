"""One LAMMPS run (pair_style jax/kk) of an ace-jax Cantor model with a
bundle built from a bench/perf/bundle_scaling.py energy-function variant.

    PYTHONPATH=bench:src python bench/perf/lammps_bundle_variant.py pace_Cantor_medium 131072 \
        --variant stock|stock@65536|k_tight@65536|stock+tightcap|... --work /tmp/lbv

Variants as bundle_scaling.parse_variant.  Unchunked stock is
export.lammps.export_lammps(layout="dense") exactly as the benchmark sizes it
(run_lammps.capacity); the others go through lammps_jax's export_model with
the same capacities.  "+tightcap" sizes k_dense and max_edges for rcut pairs
only (k(rcut) + 8): LAMMPS aborts ("edge capacity exceeded") if lammps-jax
packs skin pairs too.  Steps: as run_lammps.choose_steps
would pick them, unless --steps.  Prints one JSON row (run_lammps format + pe
of the first thermo row, for a parity check between variants).
"""
import argparse
import json
import os
import pathlib
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("n", type=int)
    ap.add_argument("--variant", default="stock")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--work", default="/tmp/lammps_bundle_variant")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--thermo", type=int, default=50, help="thermo interval (the deck's is 50)")
    a = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", a.dtype == "float64")
    from ase.data import atomic_numbers
    from ase.io import write

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import bundle_scaling as bs
    from ace_jax.eval import load
    from ace_jax.export.lammps import export_lammps
    from scaling.models import ELEMENTS
    from scaling.run_lammps import capacity, read_pe
    from scaling.structures import supercell

    kind, system, size = a.model.split("_")
    path = f"/ace-jax/bench/scaling/models/{a.model}" + (".yace" if kind == "pace" else ".npz")
    model, meta, _ = load(path)
    rcut = float(meta["rcut"])
    work = pathlib.Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    at = supercell(system, a.n)
    elements = list(ELEMENTS[system])
    write(work / "x.data", at, format="lammps-data", specorder=elements, masses=True)
    cap = capacity(at, rcut)
    type_elements = [atomic_numbers[e] for e in elements]
    model_z = [int(z) for z in meta["elements"]]
    type_map = [model_z.index(z) for z in type_elements]
    type_map = None if type_map == list(range(len(model_z))) else type_map
    bundle = work / "bundle.json"
    extra = {"variant": a.variant, "k_dense": cap["k_dense"]}
    ns = len(type_elements)
    base, chunk, ckpt = bs.parse_variant(a.variant.removesuffix("+tightcap"))
    max_edges = cap["max_edges"]
    k = cap["k_dense"]
    if base in ("k_tight", "filter_tight") or a.variant.endswith("+tightcap"):
        from ace_jax.eval import sparse_graph
        import numpy as np
        g = sparse_graph(at.positions, at.cell.array, at.pbc, rcut)
        k = int(np.bincount(g.senders, minlength=len(at)).max()) + 8
    if a.variant.endswith("+tightcap"):      # edge buffer for rcut pairs only
        max_edges = cap["max_owned"] * k
    extra.update(k_dense=k, max_edges=max_edges, chunk=chunk, ckpt=ckpt)
    if base == "stock" and chunk is None:
        export_lammps(model, meta, bundle, max_atoms=cap["max_atoms"], max_edges=max_edges,
                      k_dense=k, dtype=a.dtype, layout="dense",
                      type_elements=type_elements, max_owned=cap["max_owned"])
    else:
        from lammps_jax.export import export_model
        from ace_jax.export.lammps import make_energy_fn
        if base == "filter_tight":
            fn = bs.make_filtered_energy_fn(model, ns, k, cap["max_owned"], rcut, type_map,
                                            chunk, ckpt)
        elif chunk is None:
            fn = make_energy_fn(model, ns, "dense", k, type_map, n_rows=cap["max_owned"])
        else:
            fn = bs.make_chunked_energy_fn(model, ns, k, cap["max_owned"], type_map, chunk, ckpt)
        b = export_model(energy_fn=fn, path=bundle, max_atoms=cap["max_atoms"],
                         max_edges=max_edges, cutoff=rcut, unit_style="metal",
                         precision=a.dtype, n_species=ns)
        bundle.write_text(json.dumps(b, indent=2, sort_keys=True) + "\n")
    # free this process's GPU pool before LAMMPS starts: run LAMMPS in a child
    import jax_plugins.xla_cuda12 as p
    pjrt = os.path.join(os.path.dirname(p.__file__), "xla_cuda_plugin.so")
    row = {"name": f"acejax-{kind}/{system}/{size}", "code": f"acejax-{kind}", "size": size,
           "system": system, "elements": elements}
    import re
    import scaling.run_lammps as rl
    deck = rl.lammps_input
    # thermo every a.thermo steps, with the largest coordination within rcut
    # (compute coord/atom), to see whether an atom outgrows its k_dense slots
    rl.lammps_input = lambda *x, **kw: deck(*x, **kw).replace(
        "thermo 50\n", f"thermo {a.thermo}\n").replace(
        "thermo_style custom step pe atoms\n",
        f"compute cn all coord/atom cutoff {rcut}\ncompute cnmax all reduce max c_cn\n"
        "thermo_style custom step pe atoms c_cnmax\n")
    contract = json.loads(bundle.read_text())["contract"]
    extra.update(contract_cutoff=contract["cutoff"], contract_max_edges=contract["max_edges"],
                 rcut=rcut, steps=a.steps, warmup=a.warmup)
    out = rl._run_lammps(row, "acejax", str(bundle), work / "x.data", work, a.n, a.dtype, "gpu",
                      "/opt/lmp-jax.sh", 1, pjrt, a.steps, a.warmup, extra)
    log = (work / "log.lammps").read_text() if (work / "log.lammps").exists() else ""
    try:
        out["pe"] = read_pe(log)
    except Exception:                                              # noqa: BLE001
        pass
    # thermo rows (step, pe) and neighbour-list stats, to place an abort in the run
    # (step, pe, max coordination within rcut); pe may be nan
    out["thermo"] = [[int(m.group(1)), float(m.group(2)), float(m.group(3))] for m in
                     re.finditer(r"^\s*(\d+)\s+(\S+)\s+\d+\s+(\S+)\s*$", log, re.M)]
    out["neigh"] = re.findall(r"(?:Total # of neighbors|Neighbor list builds|Dangerous builds)"
                              r"\s*=\s*\d+", log)
    out["log_errors"] = [l for l in log.splitlines() if "ERROR" in l or "capacity" in l][:5]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
