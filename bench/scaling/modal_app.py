"""Benchmark sweep on a Modal A100-80GB (spec: docs/dev/benchmark-scaling-spec.md).

    modal run bench/scaling/modal_app.py --parity-only
    modal run bench/scaling/modal_app.py [--only acejax-pace]

The image mirrors bench/scaling/envs/moriarty.sh: LAMMPS develop with KOKKOS
(CUDA sm_80 + OpenMP), ML-PACE and Symmetrix; the lammps-jax plugin; a venv with
JAX CUDA, lammps-jax, mace-torch, cuequivariance and symmetrix.  ace-jax source,
the benchmark models are mounted from this checkout.

Rows are written to the `ace-jax-bench-scaling-results` Volume as they land
(committed every minute), so a preempted container resumes instead of starting
over.  A run is seeded with bench/scaling/results/modal-a100.jsonl plus
modal-snapshots/<code>.jsonl, and its rows are merged back into
modal-a100.jsonl (by case key) when it returns.
"""
import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")
RESULTS = ROOT / "bench" / "scaling" / "results" / "modal-a100.jsonl"
STUB = '-L/usr/local/cuda/lib64/stubs -lcuda'     # the image builder has no GPU driver

base_image = (
    # CUDA 12.9: nvcc 12.4 rejects the lammps-jax plugin (parenthesised aggregate
    # init in emplace_back); 12.9 compiles it unmodified, as on moriarty
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "wget", "libopenmpi-dev", "openmpi-bin", "gfortran",
                 "libopenblas-dev", "liblapack-dev")          # Symmetrix needs BLAS/LAPACK
    .pip_install("cmake>=3.27", "jax[cuda12]", "matscipy", "ase", "matplotlib", "pyyaml",
                 "equinox", "lineax", "scipy", "mace-torch", "cuequivariance-torch",
                 "cuequivariance-ops-torch-cu12")
    # the python base image exports CC for a compiler that is not in the container;
    # CMake (symmetrix wheel, LAMMPS) would fail on it
    .env({"CC": "gcc", "CXX": "g++"})
    .run_commands(
        "git clone --depth 1 -b patch_10Sep2025 https://github.com/lammps/lammps.git /opt/lammps",  # pinned (Symmetrix)
        "git clone --recursive https://github.com/wcwitt/symmetrix.git /opt/symmetrix",
        "git clone https://github.com/abhijeetgangan/lammps-jax.git /opt/lammps-jax",
        # second tree for the lammps-jax plugin (needs develop; Symmetrix needs 10Sep2025)
        "git clone --depth 1 -b develop https://github.com/lammps/lammps.git /opt/lammps-dev",
        "cd /opt/symmetrix/pair_symmetrix && ./install.sh /opt/lammps",
        "pip install -e /opt/lammps-jax",
        # the python package is only for symmetrix_extract_mace: build it CPU-only
        # (the builder has nvcc but no GPU, so Kokkos arch auto-detection fails)
        # and without ARCH_NATIVE (builder and runtime CPUs differ)
        "SKBUILD_CMAKE_ARGS='-DKokkos_ENABLE_CUDA=OFF;-DSYMMETRIX_SPHERICART_CUDA=OFF;-DKokkos_ARCH_NATIVE=OFF'"
        " pip install /opt/symmetrix/symmetrix",
        "cmake -S /opt/lammps/cmake -B /opt/lammps/build-kk -D CMAKE_BUILD_TYPE=Release"
        " -D CMAKE_CXX_STANDARD=20 -D CMAKE_CXX_STANDARD_REQUIRED=ON"
        " -D CMAKE_CXX_COMPILER=/opt/lammps/lib/kokkos/bin/nvcc_wrapper"
        " -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON"
        " -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON"
        " -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_AMPERE80=ON"
        " -D PKG_ML-PACE=ON -D PKG_PYTHON=ON -D Python_EXECUTABLE=/usr/local/bin/python"
        " -D SYMMETRIX_KOKKOS=ON -D SYMMETRIX_SPHERICART_CUDA=ON"
        f' -D CMAKE_SHARED_LINKER_FLAGS="{STUB}" -D CMAKE_EXE_LINKER_FLAGS="{STUB}"',
        "cmake --build /opt/lammps/build-kk -j 32",
        "cmake --build /opt/lammps/build-kk --target install-python",
        "cmake -S /opt/lammps-dev/cmake -B /opt/lammps-dev/build-kk -D CMAKE_BUILD_TYPE=Release"
        " -D CMAKE_CXX_COMPILER=/opt/lammps-dev/lib/kokkos/bin/nvcc_wrapper"
        " -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON"
        " -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON"
        " -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_AMPERE80=ON -D PKG_ML-PACE=ON"
        " -D PKG_PLUGIN=ON"                          # LAMMPS_PLUGIN_PATH auto-loading
        f' -D CMAKE_SHARED_LINKER_FLAGS="{STUB}" -D CMAKE_EXE_LINKER_FLAGS="{STUB}"',
        "cmake --build /opt/lammps-dev/build-kk -j 32",
        "INC=$(python -c \"import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))\");"
        " cmake -S /opt/lammps-jax/cpp -B /opt/lammps-jax/build-plugin-gpu-pjrt"
        " -D CMAKE_CXX_COMPILER=/opt/lammps-dev/lib/kokkos/bin/nvcc_wrapper -D CMAKE_BUILD_TYPE=Release"
        " -D CMAKE_CXX_FLAGS='-fno-lto -fopenmp' -D CMAKE_SHARED_LINKER_FLAGS='-fno-lto -fopenmp " + STUB + "'"
        " -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -D LAMMPS_HEADER_DIR=/opt/lammps-dev/src"
        " -D JAXLIB_INCLUDE_DIR=$INC -D KOKKOS_CONFIG_INCLUDE_DIR=/opt/lammps-dev/build-kk/lib/kokkos"
        # no driver in the builder: link the stub (the real libcuda is there at run time)
        " -D CUDA_DRIVER_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so",
        "cmake --build /opt/lammps-jax/build-plugin-gpu-pjrt -j 32",
        "printf '#!/usr/bin/env bash\\nexport LD_LIBRARY_PATH=/opt/lammps/build-kk:${LD_LIBRARY_PATH:-}\\n"
        "exec /opt/lammps/build-kk/lmp \"$@\"\\n' > /opt/lmp.sh && chmod +x /opt/lmp.sh",
        "printf '#!/usr/bin/env bash\\nexport LAMMPS_PLUGIN_PATH=/opt/lammps-jax/build-plugin-gpu-pjrt\\n"
        "export LD_LIBRARY_PATH=/opt/lammps-dev/build-kk:${LD_LIBRARY_PATH:-}\\n"
        "exec /opt/lammps-dev/build-kk/lmp \"$@\"\\n' > /opt/lmp-jax.sh && chmod +x /opt/lmp-jax.sh",
    )
    # torch's cu130 wheel JIT-compiles kernels with NVRTC, which dlopens
    # libnvrtc-builtins.so.13.0 by name: put the pip CUDA 13 libs on the loader
    # path (sonames .so.13, so the CUDA 12 stack of JAX and LAMMPS is unaffected)
    .env({"PYTHONPATH": "/ace-jax/src:/ace-jax/bench",
          "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib"
                             ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64"})
    # its own layer, after the LAMMPS builds, so adding it kept their cache:
    # matscipy-neighbours with CUDA (sm_80) -- the calculator's dense graph on the GPU
    .run_commands("pip install -C cmake.define.ENABLE_CUDA=ON -C cmake.define.CMAKE_CUDA_ARCHITECTURES=80"
                  " matscipy-neighbours==1.0.0")
)


def with_sources(img):
    """`img` with this checkout's ace-jax source, bench and julia dirs mounted
    (the last layers, so editing them never rebuilds the toolchain).

    BENCH_SRC_ROOT (default: this checkout) roots the `src` mount elsewhere, e.g.
    a worktree of an older commit for a before/after comparison; bench and julia
    stay this checkout's, so both sides run the same harness."""
    import os
    src_root = pathlib.Path(os.environ.get("BENCH_SRC_ROOT") or ROOT)
    return (img
    .add_local_dir(src_root / "src", "/ace-jax/src")
    .add_local_dir(ROOT / "bench", "/ace-jax/bench",
                   ignore=["**/__pycache__", "scaling/results/*"])
    .add_local_dir(ROOT / "julia", "/ace-jax/julia"))


image = with_sources(base_image)
app = modal.App("ace-jax-bench-scaling", image=image)
# rows land here as they are written: a preempted container is restarted with
# the same input, and resumes from the volume instead of from scratch
vol = modal.Volume.from_name("ace-jax-bench-scaling-results", create_if_missing=True)
SNAPSHOTS = ROOT / "bench" / "scaling" / "results" / "modal-snapshots"


@app.function(gpu="A100-80GB", timeout=24 * 3600, volumes={"/results": vol})
def sweep(seed: str = "", only: str = "", parity_only: bool = False):
    import os
    import subprocess
    import sys
    import time
    import jax_plugins.xla_cuda12 as p
    bench = pathlib.Path("/ace-jax/bench/scaling")
    pjrt = os.path.join(os.path.dirname(p.__file__), "xla_cuda_plugin.so")
    (bench / "envs" / "modal-a100.json").write_text(json.dumps(
        {"lmp": "/opt/lmp.sh", "lmp_jax": "/opt/lmp-jax.sh", "pjrt": pjrt, "pythonpath": "/ace-jax/bench:/ace-jax/src",
         "python": sys.executable}))
    models = bench / "models"
    if not any(models.glob("mace_*.model")):                 # MACE-MP-0b2 / MH-1 + Symmetrix
        subprocess.run([sys.executable, str(bench / "models.py"), "mace"], check=True)
    vol.reload()
    res = pathlib.Path(f"/results/modal-a100-{only or 'all'}{'-parity' if parity_only else ''}.jsonl")
    if not res.exists():                                    # first start; a restart keeps the volume copy
        res.write_text(seed)
        vol.commit()
    cmd = [sys.executable, str(bench / "sweep.py"), "modal-a100", "--results", str(res)]
    cmd += (["--only", only] if only else []) + (["--parity-only"] if parity_only else [])
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:                              # at most a minute lost to preemption
        time.sleep(60)
        vol.commit()
    vol.commit()
    return res.read_text()


@app.function(gpu="A100-80GB", timeout=3600)
def debug_case(model: str, n: int, dtype: str = "float64"):
    """One LAMMPS case, returning the full stderr/stdout/log (the sweep keeps a summary)."""
    import os
    import subprocess
    import sys
    bench = pathlib.Path("/ace-jax/bench/scaling")
    if not any((bench / "models").glob("mace_*.model")):
        subprocess.run([sys.executable, str(bench / "models.py"), "mace"], check=True,
                       capture_output=True)
    lmp = "/opt/lmp-jax.sh" if model.startswith("acejax") else "/opt/lmp.sh"
    work = "/tmp/debug_case"
    p = subprocess.run([sys.executable, str(bench / "run_lammps.py"), model, str(n), dtype, "gpu",
                        lmp, "1", work], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "/ace-jax/bench:/ace-jax/src"})
    log = pathlib.Path(work, "log.lammps")
    lmp_out = subprocess.run(["bash", "-c", f"cd {work} && {lmp} -in in.bench -nocite -log none "
                              "-k on g 1 -sf kk -pk kokkos newton on neigh half 2>&1 | tail -60"],
                             capture_output=True, text=True).stdout
    return {"stdout": p.stdout[-3000:], "stderr": p.stderr[-6000:],
            "log": log.read_text()[-3000:] if log.exists() else "", "rerun": lmp_out}


@app.local_entrypoint()
def main(only: str = "", parity_only: bool = False):
    import sys
    sys.path.insert(0, str(ROOT / "bench"))
    from scaling.sweep import merge_rows, seed_for
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    prior = RESULTS.read_text() if RESULTS.exists() else ""
    seed = seed_for(prior, SNAPSHOTS, only) if only else prior
    rows = sweep.remote(seed, only, parity_only)
    # re-read now: parallel runs (one per --only) each finish at their own time
    merged, added = merge_rows(RESULTS.read_text() if RESULTS.exists() else "", rows)
    RESULTS.write_text(merged)
    print(f"{added} new rows ->", RESULTS)
    for r in (json.loads(l) for l in merged.splitlines() if l.strip()):
        if r.get("mode") == "parity":
            print(f"  parity {r['gate']:7s} {r['system']:7s} {r['model']:28s} {r['status']}")
