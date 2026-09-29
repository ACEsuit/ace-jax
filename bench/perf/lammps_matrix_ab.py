"""Neighbour-matrix bundle A/B in real LAMMPS (pair_style jax/kk): stock packed
dense against layout="matrix", and the matrix with tight (rcut) model slots.

    # export (CPU is enough: the bundle programs are lowered for CUDA)
    PYTHONPATH=bench:src python bench/perf/lammps_matrix_ab.py export pace_Cantor_medium 16384 \
        matrix /tmp/b.json [--ref]
    # run one bundle on one deck (GPU host with the lammps-jax plugin)
    PYTHONPATH=bench:src python bench/perf/lammps_matrix_ab.py run pace_Cantor_medium 16384 \
        /tmp/b.json /tmp/work --deck bench|static --steps 60 --warmup 10 --repeats 2

Variants (all sized by scaling.run_lammps.capacity, float64):

* stock:  export_lammps(layout="dense"), k_dense = max_edges / max_owned =
  k(rcut + skin) + 8: the benchmark's bundle before this change.
* matrix: layout="matrix", model slots = list slots = k(rcut + skin) + 8.
* tight:  layout="matrix", list slots k(rcut + skin) + 8, model slots
  k(rcut) + 8 (capacity(tight=True); compacted per step).

Decks: "bench" is scaling.run_lammps.lammps_input (fix nve from a perfect
lattice; the random-weight structures compress, so tight slots are not safe
on it).  "static" is the same deck without the integrator: atoms stay put,
every step still evaluates energy and forces -- the stable-MD proxy on which
tight slots are safe (no list rebuilds after the first, as in stable MD
between rebuilds).  Each run is one LAMMPS process: warm-up, then `repeats`
timed segments; the row keeps every segment's step time.  Prints one JSON row.

--ref (export, n small): also the ace-jax calculator's energy and forces on
the same structure, for a `run 0` parity check of the bundle in LAMMPS.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
MODELS = pathlib.Path("/ace-jax/bench/scaling/models")


def model_path(name, root=MODELS):
    kind = name.split("_")[0]
    return str(root / (name + (".yace" if kind == "pace" else ".npz")))


def export(name, n, variant, out, ref=False, root=MODELS):
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from ase.data import atomic_numbers
    from ace_jax.eval import load
    from ace_jax.export.lammps import export_lammps
    from scaling.models import ELEMENTS
    from scaling.run_lammps import capacity
    from scaling.structures import supercell
    system = name.split("_")[1]
    path = model_path(name, root)
    model, meta, _ = load(path)
    at = supercell(system, n)
    rcut = float(meta["rcut"])
    cap, tight = capacity(at, rcut), capacity(at, rcut, tight=True)
    types = [atomic_numbers[e] for e in ELEMENTS[system]]
    kw = dict(max_atoms=cap["max_atoms"], dtype="float64", type_elements=types,
              max_owned=cap["max_owned"])
    t0 = time.perf_counter()
    if variant == "stock":
        b = export_lammps(model, meta, out, layout="dense", k_dense=cap["k_dense"],
                          max_edges=cap["max_edges"], **kw)
    elif variant == "matrix":
        b = export_lammps(model, meta, out, layout="matrix", k_dense=cap["k_dense"],
                          max_neighbors=cap["max_neighbors"], **kw)
    elif variant == "tight":
        b = export_lammps(model, meta, out, layout="matrix", k_dense=tight["k_dense"],
                          max_neighbors=tight["max_neighbors"], **kw)
    else:
        raise ValueError(variant)
    row = {"model": name, "n": n, "variant": variant, "export_s": time.perf_counter() - t0,
           "layout": b["ace_jax"]["layout"], "k_dense": b["ace_jax"]["k_dense"],
           "input_layout": b["contract"]["input_layout"],
           "max_neighbors": b["contract"].get("max_neighbors"),
           "max_edges": b["contract"].get("max_edges"), "max_owned": cap["max_owned"],
           "max_atoms": cap["max_atoms"], "k_list": cap["k_max"],
           "k_cut": tight["k_dense"] - 8, "bundle_bytes": pathlib.Path(out).stat().st_size}
    if ref:
        from ace_jax.calc.point import ACECalculator
        a = at.copy()
        a.calc = ACECalculator(path)
        row["ref_E"] = float(a.get_potential_energy())
        row["ref_F"] = np.asarray(a.get_forces()).tolist()
    return row


def deck(bundle, elements, data, steps, warmup, repeats, kind, dump=None):
    from scaling.run_lammps import lammps_input
    txt = lammps_input("acejax", bundle, elements, data, "gpu", steps, warmup=warmup, dump=dump)
    if kind == "static":
        assert "fix 1 all nve\n" in txt
        txt = txt.replace("fix 1 all nve\n", "")
    elif kind != "bench":
        raise ValueError(kind)
    return txt + f"run {steps}\n" * (repeats - 1)


def run(name, n, bundle, work, kind="bench", steps=60, warmup=10, repeats=2,
        lmp="/opt/lmp-jax.sh", pjrt=None, parity=None):
    from ase.io import write
    import numpy as np
    from scaling.models import ELEMENTS
    from scaling.run_lammps import KOKKOS, read_dump_forces, read_pe
    from scaling.structures import supercell
    system = name.split("_")[1]
    elements = list(ELEMENTS[system])
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    at = supercell(system, n)
    write(work / "x.data", at, format="lammps-data", specorder=elements, masses=True)
    dump = str(work / "f.dump") if parity else None
    (work / "in.ab").write_text(deck(bundle, elements, work / "x.data", steps, warmup,
                                     repeats, kind, dump) if not parity else
                                deck(bundle, elements, work / "x.data", 0, 0, 1, kind, dump))
    if pjrt is None:
        import jax_plugins.xla_cuda12 as p
        pjrt = str(pathlib.Path(p.__file__).parent / "xla_cuda_plugin.so")
    cmd = [lmp, "-in", "in.ab", "-log", "log.lammps", "-nocite", "-k", "on", "g", "1", "-sf",
           "kk", "-pk", "kokkos", *KOKKOS["acejax"].split(), "-var", "pjrt", pjrt]
    (work / "log.lammps").unlink(missing_ok=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=3000)
    log = (work / "log.lammps").read_text() if (work / "log.lammps").exists() else p.stdout
    row = {"model": name, "n": n, "deck": kind, "wall_s": time.perf_counter() - t0,
           "returncode": p.returncode}
    loops = re.findall(r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", log)
    row["log_errors"] = [ln for ln in (log + p.stderr).splitlines()
                         if "ERROR" in ln or "exceeded" in ln][:5]
    row["widest_row"] = re.findall(r"widest neighbor row uses (\d+) of (\d+)", log)
    row["neigh"] = re.findall(r"(?:Neighbor list builds|Dangerous builds)\s*=\s*\d+", log)
    try:
        row["pe0"] = read_pe(log)
    except Exception:                                                   # noqa: BLE001
        row["pe0"] = None
    row["nan"] = bool(re.search(r"\bnan\b", log, re.I))
    if parity:
        E0, F0 = parity
        if row["pe0"] is None or not pathlib.Path(dump).exists():
            row["status"] = "error"
            row["stderr"] = (p.stderr or p.stdout)[-2000:]
            return row
        F = read_dump_forces(dump)
        row.update(status="ok", dE_per_atom=abs(row["pe0"] - E0) / n,
                   max_dF=float(np.abs(F - np.asarray(F0)).max()))
        return row
    timed = [float(t) / int(s) for t, s in loops[1:]] if warmup else \
            [float(t) / int(s) for t, s in loops]
    row["step_s"] = timed
    ok = p.returncode == 0 and len(timed) == repeats and not row["nan"]
    row["status"] = "ok" if ok else "error"
    if ok:
        row["atom_steps_per_s"] = n / min(timed)
    else:
        row["stderr"] = (p.stderr or p.stdout)[-2000:]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "run"])
    ap.add_argument("model")
    ap.add_argument("n", type=int)
    ap.add_argument("args", nargs="+")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--deck", default="bench")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "export":
        variant, out = a.args
        print(json.dumps(export(a.model, a.n, variant, out, a.ref)))
    else:
        bundle, work = a.args
        print(json.dumps(run(a.model, a.n, bundle, work, a.deck, a.steps, a.warmup, a.repeats)))


if __name__ == "__main__":
    main()
