"""Profiling on a Modal A100-80GB, reusing the benchmark image (bench/scaling/modal_app.py).

    uv run --with modal modal run bench/perf/modal_profile.py::probe
    uv run --with modal modal run bench/perf/modal_profile.py::acejax --model Cantor_medium --n 8192
    uv run --with modal modal run bench/perf/modal_profile.py::mlpace --model Cantor_medium --n 8192
    uv run --with modal modal run bench/perf/modal_profile.py::sweep_acejax
    uv run --with modal modal run bench/perf/modal_profile.py::sweep_mlpace
    uv run --with modal modal run bench/perf/modal_profile.py::ace      # ACEModel (npz), 6 models

Results are returned (and written under bench/perf/results/ locally by the
entrypoints); nothing is written to bench/scaling/results.
"""
import json
import pathlib
import sys

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scaling"))
sys.path.insert(0, str(HERE.parent))
from scaling.modal_app import base_image, with_sources  # noqa: E402

# kokkos-tools simple kernel timer, for ML-PACE per-kernel times
image = with_sources(base_image.run_commands(
    "git clone --depth 1 https://github.com/kokkos/kokkos-tools.git /opt/kokkos-tools",
    "cmake -S /opt/kokkos-tools -B /opt/kokkos-tools/build -D CMAKE_BUILD_TYPE=Release"
    " -D KokkosTools_ENABLE_MPI=OFF -D KokkosTools_ENABLE_PAPI=OFF",
    "cmake --build /opt/kokkos-tools/build -j 16",
    "find /opt/kokkos-tools/build -name 'libkp_kernel_timer*' -o -name 'kp_reader' | head",
))
app = modal.App("ace-jax-perf-profile", image=image)
OUT = HERE / "results"
ENV = {"PYTHONPATH": "/ace-jax/bench:/ace-jax/src"}


def _sh(cmd, env=None, cwd=None, timeout=3000):
    import os
    import subprocess
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
                       env={**os.environ, **ENV, **(env or {})}, cwd=cwd, timeout=timeout)
    return p


@app.function(gpu="A100-80GB", timeout=1200)
def probe():
    cmds = ["nvidia-smi", "ls /usr/local/cuda/bin", "which nsys ncu",
            "find / -name 'nsys' -type f 2>/dev/null | head -3",
            "find /opt/kokkos-tools/build -name '*.so' | head -30",
            "python -c 'import jax; print(jax.__version__, jax.devices())'",
            "nvidia-smi -q -d CLOCK | head -40"]
    return {c: (lambda p: (p.stdout + p.stderr)[-3000:])(_sh(c)) for c in cmds}


def _acejax(model, system, n, dtype="float64", layout="dense", kind="gather",
            variant="baseline", trace=True, stages=True, reps=20):
    import os
    import shutil
    import tempfile
    yace = f"/ace-jax/bench/scaling/models/pace_{model}.yace"
    base = ["python", "/ace-jax/bench/perf/profile_acejax.py", yace, system, str(n),
            "--dtype", dtype, "--layout", layout, "--kind", kind, "--variant", variant,
            "--reps", str(reps)]
    p = _sh(base + ([] if stages else ["--no-stages"]))
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:                                              # noqa: BLE001
        return {"error": (p.stdout[-2000:] + p.stderr[-4000:])}
    if trace:
        td, dd = tempfile.mkdtemp(), tempfile.mkdtemp()
        p2 = _sh(base + ["--no-stages", "--trace", td, "--reps", "3"],
                 env={"XLA_FLAGS": f"--xla_dump_to={dd} --xla_dump_hlo_as_text"})
        sys.path.insert(0, "/ace-jax/bench/perf")
        import hlo_trace
        try:
            out["kernels"] = hlo_trace.kernel_table(td, dd, 5, top=45)
        except Exception as ex:                                    # noqa: BLE001
            files = [os.path.join(r, f) for r, _, fs in os.walk(td) for f in fs]
            out["kernels"] = {"error": repr(ex), "stderr": p2.stderr[-3000:],
                              "stdout": p2.stdout[-1000:], "files": files[:20]}
        # keep the raw trace + optimised HLO on the volume for re-parsing
        keep = f"/vol/{model}_{n}_{dtype}_{layout}_{variant}"
        shutil.rmtree(keep, ignore_errors=True)
        shutil.copytree(td, keep + "/trace")
        os.makedirs(keep + "/hlo", exist_ok=True)
        for f in os.listdir(dd):
            if f.endswith("after_optimizations.txt") and os.path.getsize(os.path.join(dd, f)) > 20000:
                shutil.copy(os.path.join(dd, f), keep + "/hlo/")
        out["saved"] = keep
        shutil.rmtree(td, ignore_errors=True)
        shutil.rmtree(dd, ignore_errors=True)
    return out


vol = modal.Volume.from_name("ace-jax-perf-profile", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def acejax_remote(cases: list):
    out = [_acejax(**c) for c in cases]
    vol.commit()
    return out


def _mlpace(model, system, n, steps=100, warmup=20, ktimer=True, style="product",
            extra_pk=""):
    import glob
    import os
    import tempfile
    sys.path.insert(0, "/ace-jax/bench")
    from ase.io import write
    from scaling.run_lammps import lammps_input, parse_log
    from scaling.structures import SYSTEMS, supercell
    els = list(SYSTEMS[system]["elements"])
    work = tempfile.mkdtemp()
    at = supercell(system, n)
    write(f"{work}/x.data", at, format="lammps-data", specorder=els, masses=True)
    txt = lammps_input("mlpace", f"/ace-jax/bench/scaling/models/pace_{model}.yace", els,
                       f"{work}/x.data", "gpu", steps, warmup=warmup)
    if style != "product":
        txt = txt.replace("pair_style pace product", f"pair_style pace {style}")
    pathlib.Path(work, "in.bench").write_text(txt)
    env = {}
    if ktimer:
        lib = glob.glob("/opt/kokkos-tools/build/**/libkp_kernel_timer.so", recursive=True)
        env["KOKKOS_TOOLS_LIBS"] = lib[0]
    cmd = (f"/opt/lmp.sh -in in.bench -log log.lammps -nocite -k on g 1 -sf kk "
           f"-pk kokkos newton on neigh half {extra_pk}")
    p = _sh(cmd, env=env, cwd=work)
    log = pathlib.Path(work, "log.lammps").read_text() if pathlib.Path(work, "log.lammps").exists() else ""
    out = {"model": model, "n": n, "steps": steps, "warmup": warmup}
    try:
        out.update(parse_log(log))
        out["atom_steps_per_s"] = n / out["step_s"]
    except Exception:                                              # noqa: BLE001
        out["error"] = (p.stdout[-2000:] + p.stderr[-2000:] + log[-2000:])
        return out
    # LAMMPS timing breakdown of the last run
    i = log.rfind("MPI task timing breakdown")
    out["breakdown"] = log[i:i + 1200] if i >= 0 else ""
    if ktimer:
        dats = glob.glob(f"{work}/*.dat")
        reader = glob.glob("/opt/kokkos-tools/build/**/kp_reader*", recursive=True)
        # this kokkos-tools version prints the table to stdout at finalize
        i = p.stdout.find("Total wall time")
        out["kernel_timer_stdout"] = p.stdout[i:i + 8000] if i >= 0 else p.stdout[-8000:]
        if dats and reader:
            r = _sh(f"{reader[0]} {dats[0]}", env={"KOKKOS_TOOLS_LIBS": env["KOKKOS_TOOLS_LIBS"]})
            out["kernel_timer"] = (r.stdout + r.stderr)[:12000]
    return out


@app.function(gpu="A100-80GB", timeout=3600)
def mlpace_remote(cases: list):
    return [_mlpace(**c) for c in cases]


def _save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print("->", OUT / name)


def _system(model):
    return model.split("_")[0]


@app.local_entrypoint()
def acejax(model: str = "Cantor_medium", n: int = 8192, dtype: str = "float64",
           layout: str = "dense", kind: str = "gather", variant: str = "baseline",
           trace: bool = True, stages: bool = True, tag: str = ""):
    r = acejax_remote.remote([dict(model=model, system=_system(model), n=n, dtype=dtype,
                                   layout=layout, kind=kind, variant=variant, trace=trace,
                                   stages=stages)])
    _save(f"acejax_{model}_{n}_{dtype}_{layout}_{variant}{tag}.json", r[0])


@app.local_entrypoint()
def mlpace(model: str = "Cantor_medium", n: int = 8192, steps: int = 100, tag: str = ""):
    r = mlpace_remote.remote([dict(model=model, system=_system(model), n=n, steps=steps)])
    _save(f"mlpace_{model}_{n}{tag}.json", r[0])
    print(r[0].get("kernel_timer", r[0].get("error", ""))[:4000])


@app.local_entrypoint()
def sweep_acejax(models: str = "SiGe_small,SiGe_medium,SiGe_large,Cantor_small,Cantor_medium,Cantor_large",
                 ns: str = "1024,4096,8192,16384,32768", dtype: str = "float64",
                 variant: str = "baseline", tag: str = ""):
    groups = [[dict(model=m, system=_system(m), n=int(n), dtype=dtype, variant=variant,
                    trace=False, stages=False, reps=10) for n in ns.split(",")]
              for m in models.split(",")]
    res = list(acejax_remote.map(groups))
    _save(f"sweep_acejax_{dtype}_{variant}{tag}.json", [r for g in res for r in g])


@app.local_entrypoint()
def sweep_mlpace(models: str = "SiGe_small,SiGe_medium,SiGe_large,Cantor_small,Cantor_medium,Cantor_large",
                 ns: str = "1024,4096,8192,16384,32768", tag: str = ""):
    cases = [dict(model=m, system=_system(m), n=int(n), steps=100, ktimer=True)
             for m in models.split(",") for n in ns.split(",")]
    res = mlpace_remote.remote(cases)
    _save(f"sweep_mlpace{tag}.json", res)


@app.local_entrypoint()
def variants(model: str = "Cantor_medium", n: int = 8192, dtype: str = "float64",
             names: str = "baseline,rec,pool,rev,rec+pool+rev", trace: str = "rec+pool+rev",
             tag: str = ""):
    """Every variant on ONE container (same GPU), so the timings compare."""
    cases = [dict(model=model, system=_system(model), n=n, dtype=dtype, variant=v,
                  trace=(v in trace.split(",")), stages=False) for v in names.split(",")]
    res = acejax_remote.remote(cases)
    _save(f"variants_{model}_{n}_{dtype}{tag}.json", res)


@app.function(gpu="A100-80GB", timeout=1800)
def script_remote(argvs: list):
    """Run bench/perf scripts (each argv a list) and return their last stdout line."""
    out = []
    for argv in argvs:
        p = _sh(["python"] + argv, cwd="/ace-jax")
        try:
            out.append(json.loads(p.stdout.strip().splitlines()[-1]))
        except Exception:                                          # noqa: BLE001
            out.append({"argv": argv, "error": p.stdout[-1500:] + p.stderr[-3000:]})
    return out


@app.local_entrypoint()
def micro(models: str = "Cantor_medium,SiGe_medium,Cantor_large", ns: str = "8192,32768",
          script: str = "bench/perf/micro_scatter.py", tag: str = ""):
    argvs = [[script, m, n] for m in models.split(",") for n in ns.split(",")]
    res = script_remote.remote(argvs)
    _save(f"micro_{pathlib.Path(script).stem}{tag}.json", res)
    for r in res:
        print(json.dumps(r))


@app.local_entrypoint()
def e2e(model: str = "Cantor_medium", ns: str = "8192", variants: str = "baseline,rec+pool+fm",
        dtype: str = "float64", tag: str = ""):
    argvs = []
    for n in ns.split(","):
        for v in variants.split(","):
            argvs.append(["bench/perf/e2e.py", f"/ace-jax/bench/scaling/models/pace_{model}.yace",
                          _system(model), n, "--dtype", dtype, "--variant", v]
                         + ([] if v == "baseline" else ["--skip-ase"]))
    res = script_remote.remote(argvs)
    _save(f"e2e_{model}_{dtype}{tag}.json", res)
    for r in res:
        print(json.dumps(r))


@app.function(gpu="A100-80GB", timeout=3 * 3600, volumes={"/vol": vol})
def bigsweep_remote(models: list, ns: list, variant: str, extra: list, tag: str):
    """ML-PACE (LAMMPS pace/kk) and ace-jax (ASE calculator as benchmarked, the
    skin calculator, and the skin calculator with `variant`) on ONE A100, case
    by case; rows appended to /vol/bigsweep<tag>.jsonl as they finish."""
    import subprocess
    dev = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip()
    path = pathlib.Path(f"/vol/bigsweep{tag}.jsonl")
    rows = []

    def emit(r):
        r["gpu"] = dev
        rows.append(r)
        with path.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
        vol.commit()

    cases = [(m, n, "float64") for m in models for n in ns] + [tuple(e) for e in extra]
    for m, n, dtype in cases:
        n = int(n)
        if dtype == "float64":
            steps = 100 if n <= 16384 else 30
            r = _mlpace(model=m, system=_system(m), n=n, steps=steps, warmup=10, ktimer=False)
            emit({"code": "mlpace", "model": m, "n": n, "dtype": dtype,
                  **{k: r.get(k) for k in ("step_s", "atom_steps_per_s", "error")}})
        yace = f"/ace-jax/bench/scaling/models/pace_{m}.yace"
        for v in ("baseline", variant):
            argv = ["bench/perf/e2e.py", yace, _system(m), str(n), "--dtype", dtype,
                    "--variant", v, "--reps", "10"] + ([] if v == "baseline" else ["--skip-ase"])
            p = _sh(["python"] + argv, cwd="/ace-jax")
            try:
                r = json.loads(p.stdout.strip().splitlines()[-1])
            except Exception:                                      # noqa: BLE001
                r = {"error": (p.stdout[-800:] + p.stderr[-1500:])}
            emit({"code": "acejax", "model": m, "n": n, "dtype": dtype, "variant": v, **r})
    return rows


@app.local_entrypoint()
def bigsweep(models: str = "SiGe_small,SiGe_medium,SiGe_large,Cantor_small,Cantor_medium,Cantor_large",
             ns: str = "1024,4096,16384,65536", variant: str = "rec+pool+fm",
             extra: str = "Cantor_medium:8192:float64,SiGe_medium:8192:float64,"
                          "Cantor_medium:8192:float32,Cantor_medium:65536:float32",
             tag: str = ""):
    ex = [e.split(":") for e in extra.split(",") if e]
    rows = bigsweep_remote.remote(models.split(","), ns.split(","), variant, ex, tag)
    _save(f"bigsweep{tag}.json", rows)


@app.local_entrypoint()
def lammps_variants(model: str = "Cantor_medium", ns: str = "8192,32768",
                    modes: str = "stock,owned,owned:rec+pool+fm", dtype: str = "float64",
                    tag: str = ""):
    argvs = []
    for n in ns.split(","):
        for mv in modes.split(","):
            mode, _, var = mv.partition(":")
            argvs.append(["bench/perf/lammps_variant.py", model, n, "--dtype", dtype, "--mode", mode,
                          "--variant", var or "baseline", "--work", f"/tmp/lv_{n}_{mode}_{var}"])
    res = script_remote.remote(argvs)
    _save(f"lammps_variants_{model}_{dtype}{tag}.json", res)
    for r in res:
        print(json.dumps({k: r.get(k) for k in ("n_atoms", "mode", "variant", "layout", "rows",
                                                 "k_dense", "status", "step_s", "atom_steps_per_s",
                                                 "pe", "error")}))


@app.local_entrypoint()
def mlpace_chunk(model: str = "Cantor_medium", ns: str = "8192,65536",
                 chunks: str = "4096,16384,65536", tag: str = ""):
    """ML-PACE's own headroom: pace/kk processes `chunksize` atoms per kernel
    sequence (default 4096); the kernel timer adds fences, so it is off here."""
    cases = [dict(model=model, system=_system(model), n=int(n), steps=60, warmup=10,
                  ktimer=False, style=f"product chunksize {c}")
             for n in ns.split(",") for c in chunks.split(",")]
    res = mlpace_remote.remote(cases)
    for c, r in zip(cases, res):
        r["style"] = c["style"]
        print(r["n"], c["style"], r.get("atom_steps_per_s"), r.get("error", "")[:200])
    _save(f"mlpace_chunk_{model}{tag}.json", res)


def _ace(model, n=8192, dtype="float64", reps=20):
    """ACEModel (bench/perf/profile_ace.py): timings + stages, then a traced run
    with the HLO dump, whose kernels are attributed to stages.  The raw trace
    and optimised HLO are kept on the volume for re-parsing."""
    import os
    import shutil
    import tempfile
    npz = f"/ace-jax/bench/scaling/models/ace_{model}.npz"
    base = ["python", "/ace-jax/bench/perf/profile_ace.py", npz, _system(model), str(n),
            "--dtype", dtype]
    p = _sh(base + ["--reps", str(reps)])
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:                                              # noqa: BLE001
        return {"model": model, "error": (p.stdout[-2000:] + p.stderr[-4000:])}
    td, dd = tempfile.mkdtemp(), tempfile.mkdtemp()
    p2 = _sh(base + ["--reps", "3", "--no-stages", "--trace", td, "--dump", dd])
    try:
        out["kernels"] = json.loads(p2.stdout.strip().splitlines()[-1])["kernels"]
    except Exception:                                              # noqa: BLE001
        out["kernels"] = {"error": (p2.stdout[-1500:] + p2.stderr[-3000:])}
    keep = f"/vol/ace_{model}_{n}_{dtype}"
    shutil.rmtree(keep, ignore_errors=True)
    shutil.copytree(td, keep + "/trace")
    os.makedirs(keep + "/hlo", exist_ok=True)
    for f in os.listdir(dd):
        if f.endswith("after_optimizations.txt") and os.path.getsize(os.path.join(dd, f)) > 20000:
            shutil.copy(os.path.join(dd, f), keep + "/hlo/")
    out["saved"] = keep
    shutil.rmtree(td, ignore_errors=True)
    shutil.rmtree(dd, ignore_errors=True)
    return out


@app.function(gpu="A100-80GB", timeout=3600, volumes={"/vol": vol})
def ace_remote(models: list, n: int, dtype: str):
    """Every model on ONE container (one GPU); each result is written to the
    volume as it lands, so a failure part-way keeps the finished ones."""
    import subprocess
    gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip()
    out = []
    for m in models:
        r = _ace(m, n, dtype)
        r["gpu"] = gpu
        pathlib.Path(f"/vol/ace_{m}_{n}_{dtype}.json").write_text(json.dumps(r))
        vol.commit()
        out.append(r)
    return out


@app.local_entrypoint()
def ace(models: str = "SiGe_small,SiGe_medium,SiGe_large,Cantor_small,Cantor_medium,Cantor_large",
        n: int = 8192, dtype: str = "float64", tag: str = ""):
    """ACEModel energy_forces_virial_dense per-stage GPU profile (Task 8)."""
    res = ace_remote.remote(models.split(","), n, dtype)
    for m, r in zip(models.split(","), res):
        _save(f"ace_{m}_{n}_{dtype}{tag}.json", r)
        k = r.get("kernels", {})
        print(m, r.get("model", r.get("error", "")[:300]), k.get("total_gpu_us_per_call"),
              {s: v["frac"] for s, v in k.get("stages", {}).items()})


@app.local_entrypoint()
def probe_main():
    for k, v in probe.remote().items():
        print("==", k)
        print(v)
