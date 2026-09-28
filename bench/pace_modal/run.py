"""Task 9 of docs/pace-yace-plan.md: ace-jax PACEModel vs LAMMPS ML-PACE on a GPU.

Runs on Modal (one cheap GPU), so no local LAMMPS build or SSH is needed:

    modal run bench/pace_modal/run.py               # parity + timings + AA counts
    modal run bench/pace_modal/run.py --only parity
    modal run bench/pace_modal/run.py --only profile   # force-pass breakdown

The image builds LAMMPS (stable) with ML-PACE + KOKKOS/CUDA + the Python module
once and caches it.  ace-jax is mounted from this checkout (not installed), so
code changes need no image rebuild.  Results print as one JSON document; they
are summarised in docs/pace-yace-results.md.

Precision note: pace/kk runs in double, and so does the f64 ace-jax column.
Cheap cards (L4/T4) have ~1/64-rate FP64, so absolute timings here understate
what an A100/H100 would do; the f32 ace-jax column is the representative one.
"""
import json
import pathlib

import modal

# local paths only matter on the client; in the container this file is /root/run.py
ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")
LAMMPS_EX = pathlib.Path.home() / "gits" / "lammps" / "examples"
GPU = "L4"                       # Ada, sm_89
KOKKOS_ARCH = "Kokkos_ARCH_ADA89"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "cmake", "build-essential", "wget", "libopenmpi-dev", "openmpi-bin")
    .pip_install("numpy<2.3", "ase", "pyyaml", "matscipy", "equinox", "lineax", "scipy",
                 "jax[cuda12]")
    .run_commands(
        "git clone --depth 1 -b stable https://github.com/lammps/lammps.git /opt/lammps",
        "cmake -S /opt/lammps/cmake -B /opt/lammps/build -D CMAKE_BUILD_TYPE=Release"
        " -D BUILD_SHARED_LIBS=ON -D PKG_PYTHON=ON -D BUILD_MPI=OFF"
        " -D PKG_ML-PACE=ON -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON"
        f" -D {KOKKOS_ARCH}=ON -D Kokkos_ENABLE_SERIAL=ON"
        " -D CMAKE_CXX_COMPILER=/opt/lammps/lib/kokkos/bin/nvcc_wrapper"
        # the builder has no GPU driver: link the CUDA driver API against the stub
        # libcuda.so (SONAME libcuda.so.1); the real one is mounted at run time
        ' -D CMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"'
        ' -D CMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"',
        "cmake --build /opt/lammps/build -j 16",
        "cmake --build /opt/lammps/build --target install-python",
        "grep PACELIB_URL /opt/lammps/build/CMakeCache.txt > /opt/lammps/PACELIB.txt",
    )
    .env({"LD_LIBRARY_PATH": "/opt/lammps/build", "JAX_ENABLE_X64": "1"})
    .add_local_dir(ROOT / "src", "/ace-jax/src")
    .add_local_file(ROOT / "bench" / "pace_profile.py", "/ace-jax/bench/pace_profile.py")
    .add_local_file(ROOT / "bench" / "pace_dense_proto.py", "/ace-jax/bench/pace_dense_proto.py")
    .add_local_dir(ROOT / "fixtures" / "pace", "/data/fixtures")
    .add_local_file(LAMMPS_EX / "PACKAGES/dispersion/potential_files/c_ace.yace", "/data/c_ace.yace")
    .add_local_file(LAMMPS_EX / "PACKAGES/apip/Cu-1.yace", "/data/Cu-1.yace")
)
app = modal.App("ace-jax-pace-task9", image=image)


# ----------------------------------------------------------------- helpers
def _structures():
    import numpy as np
    from ase.build import bulk
    out = {}
    c = bulk("C", "diamond", a=3.567, cubic=True).repeat(3)          # 216
    c.rattle(0.03, seed=1)
    out["c_ace"] = ("/data/c_ace.yace", ["C"], c)
    cu = bulk("Cu", "fcc", a=3.615, cubic=True).repeat(3)            # 108
    cu.rattle(0.03, seed=2)
    out["Cu-1"] = ("/data/Cu-1.yace", ["Cu"], cu)
    r = np.load("/data/fixtures/sige_zbl_ref.npz")
    from ase import Atoms
    for s in ("bulk", "close"):
        at = Atoms(numbers=r[f"Z_{s}"], positions=r[f"pos_{s}"], cell=r[f"cell_{s}"], pbc=True)
        out[f"sige_zbl/{s}"] = ("/data/fixtures/sige_zbl.yace", ["Si", "Ge"], at)
    return out


def _lammps_efs(yace, elements, atoms, kokkos=False, evaluator=""):
    """E, F (by atom id), ASE-convention stress from LAMMPS pair_style pace[/kk]."""
    import os, tempfile
    import numpy as np
    from ase import units
    from ase.io import write
    from lammps import lammps
    d = tempfile.mkdtemp()
    data = os.path.join(d, "x.data")
    write(data, atoms, format="lammps-data", specorder=elements, masses=True)
    args = ["-log", "none", "-screen", "none", "-nocite"]
    if kokkos:
        args += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"]
        evaluator = "product"            # pace/kk on the GPU requires it
    lmp = lammps(cmdargs=args)
    lmp.commands_list([
        "units metal", "atom_style atomic", "boundary p p p",
        "atom_modify map array sort 0 0.0", f"read_data {data}",
        f"pair_style pace {evaluator}".strip(), f"pair_coeff * * {yace} {' '.join(elements)}",
        "thermo_style custom pe pxx pyy pzz pyz pxz pxy", "run 0"])
    E = lmp.get_thermo("pe")
    ids = np.asarray(lmp.numpy.extract_atom("id")).ravel()
    F = np.asarray(lmp.numpy.extract_atom("f"))[:len(atoms)].copy()
    F = F[np.argsort(ids[:len(atoms)])]
    p = np.array([lmp.get_thermo(k) for k in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")])
    lmp.close()
    return E, F, -p * units.bar


def _acejax_efs(yace, atoms, dtype="float64"):
    import sys
    sys.path.insert(0, "/ace-jax/src")
    import jax.numpy as jnp
    from ace_jax.eval import ACECalculator
    at = atoms.copy()
    at.calc = ACECalculator(yace, dtype=getattr(jnp, dtype))
    return at.get_potential_energy(), at.get_forces(), at.get_stress()


# ----------------------------------------------------------------- parity
@app.function(gpu=GPU, timeout=3600)
def parity():
    import sys
    sys.path.insert(0, "/ace-jax/src")
    import numpy as np
    from ace_jax.eval import load, write_yace
    res = {"pacelib": open("/opt/lammps/PACELIB.txt").read().strip()}
    for name, (yace, els, at) in _structures().items():
        Ej, Fj, Sj = _acejax_efs(yace, at)
        row = {"n_atoms": len(at), "E_acejax": Ej}
        for tag, kw in (("pace_recursive", {}), ("pace_product", {"evaluator": "product"}),
                        ("pace_kk", {"kokkos": True})):
            try:
                E, F, S = _lammps_efs(yace, els, at, **kw)
                row[tag] = {"dE_per_atom": abs(E - Ej) / len(at),
                            "max_dF": float(np.abs(F - Fj).max()),
                            "max_dS": float(np.abs(S - Sj).max())}
            except Exception as e:                       # report, do not hide
                row[tag] = {"error": repr(e)[:300]}
        res[name] = row
    # LAMMPS reads a file written by write_yace (c_ace, unchanged model)
    m, meta, spec = load("/data/c_ace.yace")
    write_yace(m, spec, "/tmp/c_ace_rt.yace")
    _, els, at = _structures()["c_ace"]
    E0 = _lammps_efs("/data/c_ace.yace", els, at)[0]
    E1 = _lammps_efs("/tmp/c_ace_rt.yace", els, at)[0]
    res["lammps_reads_write_yace"] = {"dE_total": abs(E1 - E0)}
    return res


# ----------------------------------------------------------------- timings
@app.function(gpu=GPU, timeout=3600)
def timings(reps: int = 20, n_rep: int = 8):
    """Force-call time for a ~4k-atom carbon cell: pace/kk vs jitted ace-jax."""
    import sys, time
    sys.path.insert(0, "/ace-jax/src")
    import jax, jax.numpy as jnp
    import numpy as np
    from ase.build import bulk
    from ace_jax.eval import load, sparse_graph
    at = bulk("C", "diamond", a=3.567, cubic=True).repeat(n_rep)           # 8 n^3 atoms
    at.rattle(0.02, seed=3)
    out = {"n_atoms": len(at), "gpu": GPU}

    # LAMMPS pace/kk: `run N` with a zero-velocity NVE; loop time / N per force call
    import os, tempfile
    from ase.io import write
    from lammps import lammps
    d = tempfile.mkdtemp(); data = os.path.join(d, "c.data")
    write(data, at, format="lammps-data", specorder=["C"], masses=True)
    for tag, args in (("pace_kk_gpu", ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos",
                                       "newton", "on", "neigh", "half"]),
                      ("pace_cpu_1core", [])):
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"] + args)
        steps = 50 if tag == "pace_kk_gpu" else 5
        lmp.commands_list(["units metal", "atom_style atomic", "boundary p p p",
                           f"read_data {data}",
                           "pair_style pace product" if tag == "pace_kk_gpu" else "pair_style pace",
                           "pair_coeff * * /data/c_ace.yace C", "timestep 0.0001",
                           "fix 1 all nve", "run 2"])
        t0 = time.perf_counter(); lmp.command(f"run {steps}"); dt = time.perf_counter() - t0
        out[tag] = {"s_per_step": dt / steps, "atom_steps_per_s": len(at) * steps / dt}
        lmp.close()

    for dtype in ("float32", "float64"):          # f32 first: peak_bytes never resets
        dt_ = getattr(jnp, dtype)
        model, meta, _ = load("/data/c_ace.yace", dtype=dt_)
        g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
        nz = jnp.zeros(len(at), jnp.int32)
        s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
        rij = jnp.asarray(g.rij, dt_)
        f = jax.jit(lambda x: model.energy_forces_virial(x, nz[s], nz[r], s, r, len(at), nz))
        try:
            jax.block_until_ready(f(rij))
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); jax.block_until_ready(f(rij)); ts.append(time.perf_counter() - t0)
            t = float(np.median(ts))
            e = jax.jit(lambda x: model.site_energies(x, nz[s], nz[r], s, len(at), nz))
            jax.block_until_ready(e(rij)); te = []
            for _ in range(reps):
                t0 = time.perf_counter(); jax.block_until_ready(e(rij)); te.append(time.perf_counter() - t0)
            out[f"acejax_{dtype}"] = {"s_per_call": t, "atom_steps_per_s": len(at) / t,
                                      "s_energy_only": float(np.median(te)),
                                      "n_edges": int(len(g.senders)),
                                      "peak_gpu_bytes": jax.devices()[0].memory_stats().get("peak_bytes_in_use")}
        except Exception as e:
            out[f"acejax_{dtype}"] = {"error": repr(e)[:300], "n_edges": int(len(g.senders))}
    return out


# ----------------------------------------------------------------- AA sizes
@app.function(timeout=600)
def aa_counts():
    """The spec's 'AA-union' assumption: real-basis products vs PACE ms-combs."""
    import sys
    sys.path.insert(0, "/ace-jax/src")
    from ace_jax.eval.pace_build import build_basis
    from ace_jax.eval.pace_io import parse_yace
    out = {}
    for name, p in (("c_ace", "/data/c_ace.yace"), ("Cu-1", "/data/Cu-1.yace"),
                    ("sige_zbl", "/data/fixtures/sige_zbl.yace")):
        spec, a = parse_yace(p)
        b = build_basis(a)
        out[name] = {"n_functions": len(a["funcs"]),
                     "n_ms_combs": int(sum(len(f["ms"]) for f in a["funcs"])),
                     "n_aa_real": b["n_aa"], "n_a_local": b["n_a_local"],
                     "n_T": int(len(b["T_vals"]))}
    return out


# ----------------------------------------------------------------- profile
@app.function(gpu=GPU, timeout=3600)
def profile(reps: int = 10, n_rep: int = 8):
    """Force-pass breakdown; the logic lives in bench/pace_profile.py."""
    import sys
    sys.path.insert(0, "/ace-jax/src"); sys.path.insert(0, "/ace-jax/bench")
    from pace_profile import run_profile
    return run_profile("/data/c_ace.yace", reps, n_rep)


# ----------------------------------------------------------------- dense prototype
@app.function(gpu="A100-80GB", timeout=3600)
def dense_proto(modes: str = "check,model"):
    """bench/pace_dense_proto.py on the given GPU; one JSON per line.  Modes:
    check (dense == sparse), run (prototype A construction), model (the real
    energy_forces_virial[_dense] with estimate_a_bytes vs measured peak)."""
    import os, subprocess, sys
    env = {**os.environ, "PYTHONPATH": "/ace-jax/src"}
    out = []
    for mode in modes.split(","):
        p = subprocess.run([sys.executable, "/ace-jax/bench/pace_dense_proto.py", mode,
                            "/data/c_ace.yace"], capture_output=True, text=True, env=env)
        out += [ln for ln in p.stdout.splitlines() if ln.startswith("{")] or [p.stderr[-500:]]
    return {"gpu": os.environ.get("MODAL_GPU", ""), "lines": out}


@app.local_entrypoint()
def main(only: str = ""):
    out = ROOT / "bench" / "pace_modal" / "last_results.json"
    res = json.loads(out.read_text()) if out.exists() else {}   # --only reruns merge in
    stages = (("aa_counts", aa_counts), ("parity", parity), ("timings", timings),
              ("profile", profile), ("dense_A100", dense_proto),
              ("dense_H100", dense_proto.with_options(gpu="H100")))
    for key, fn in stages:
        # profile / dense_* are opt-in (--only profile | dense); the default run
        # is the Task 9 set
        opt_in = key == "profile" or key.startswith("dense")
        if only != key.split("_")[0] and not (only == "" and not opt_in):
            continue
        try:
            res[key] = fn.remote()
        except Exception as e:                   # keep earlier stages' results
            res[key] = {"error": repr(e)[:500]}
        print(f"== {key}\n" + json.dumps(res[key], indent=2, default=float), flush=True)
        out.write_text(json.dumps(res, indent=2, default=float))
