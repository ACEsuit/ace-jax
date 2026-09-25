"""Benchmark sweep on a Modal A100-80GB (spec: docs/benchmark-scaling-spec.md).

    modal run bench/scaling/modal_app.py --parity-only
    modal run bench/scaling/modal_app.py [--only acejax-pace]

The image mirrors bench/scaling/envs/moriarty.sh: LAMMPS develop with KOKKOS
(CUDA sm_80 + OpenMP), ML-PACE and Symmetrix; the lammps-jax plugin; a venv with
JAX CUDA, lammps-jax, mace-torch, cuequivariance and symmetrix.  ace-jax source,
the benchmark models and any existing results are mounted from this checkout,
so the sweep resumes where the last run stopped.  Rows come back to
bench/scaling/results/modal-a100.jsonl.
"""
import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2] if modal.is_local() else pathlib.Path("/")
RESULTS = ROOT / "bench" / "scaling" / "results" / "modal-a100.jsonl"
STUB = '-L/usr/local/cuda/lib64/stubs -lcuda'     # the image builder has no GPU driver

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "wget", "libopenmpi-dev", "openmpi-bin", "gfortran")
    .pip_install("cmake>=3.27", "jax[cuda12]", "matscipy", "ase", "matplotlib", "pyyaml",
                 "equinox", "lineax", "scipy", "mace-torch", "cuequivariance-torch",
                 "cuequivariance-ops-torch-cu12")
    .run_commands(
        "git clone --depth 1 -b patch_10Sep2025 https://github.com/lammps/lammps.git /opt/lammps",  # pinned (Symmetrix)
        "git clone --recursive https://github.com/wcwitt/symmetrix.git /opt/symmetrix",
        "git clone https://github.com/abhijeetgangan/lammps-jax.git /opt/lammps-jax",
        # second tree for the lammps-jax plugin (needs develop; Symmetrix needs 10Sep2025)
        "git clone --depth 1 -b develop https://github.com/lammps/lammps.git /opt/lammps-dev",
        "cd /opt/symmetrix/pair_symmetrix && ./install.sh /opt/lammps",
        "pip install -e /opt/lammps-jax /opt/symmetrix/symmetrix",
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
        f' -D CMAKE_SHARED_LINKER_FLAGS="{STUB}" -D CMAKE_EXE_LINKER_FLAGS="{STUB}"',
        "cmake --build /opt/lammps-dev/build-kk -j 32",
        "INC=$(python -c \"import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))\");"
        " cmake -S /opt/lammps-jax/cpp -B /opt/lammps-jax/build-plugin-gpu-pjrt"
        " -D CMAKE_CXX_COMPILER=/opt/lammps-dev/lib/kokkos/bin/nvcc_wrapper -D CMAKE_BUILD_TYPE=Release"
        " -D CMAKE_CXX_FLAGS='-fno-lto -fopenmp' -D CMAKE_SHARED_LINKER_FLAGS='-fno-lto -fopenmp " + STUB + "'"
        " -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -D LAMMPS_HEADER_DIR=/opt/lammps-dev/src"
        " -D JAXLIB_INCLUDE_DIR=$INC -D KOKKOS_CONFIG_INCLUDE_DIR=/opt/lammps-dev/build-kk/lib/kokkos",
        "cmake --build /opt/lammps-jax/build-plugin-gpu-pjrt -j 32",
        "printf '#!/usr/bin/env bash\\nexport LD_LIBRARY_PATH=/opt/lammps/build-kk:${LD_LIBRARY_PATH:-}\\n"
        "exec /opt/lammps/build-kk/lmp \"$@\"\\n' > /opt/lmp.sh && chmod +x /opt/lmp.sh",
        "printf '#!/usr/bin/env bash\\nexport LAMMPS_PLUGIN_PATH=/opt/lammps-jax/build-plugin-gpu-pjrt\\n"
        "export LD_LIBRARY_PATH=/opt/lammps-dev/build-kk:${LD_LIBRARY_PATH:-}\\n"
        "exec /opt/lammps-dev/build-kk/lmp \"$@\"\\n' > /opt/lmp-jax.sh && chmod +x /opt/lmp-jax.sh",
    )
    .env({"PYTHONPATH": "/ace-jax/src:/ace-jax/bench"})
    .add_local_dir(ROOT / "src", "/ace-jax/src")
    .add_local_dir(ROOT / "bench", "/ace-jax/bench",
                   ignore=["**/__pycache__", "pace_modal/*.json*", "scaling/results/*"])
    .add_local_dir(ROOT / "julia", "/ace-jax/julia")
)
app = modal.App("ace-jax-bench-scaling", image=image)


@app.function(gpu="A100-80GB", timeout=24 * 3600)
def sweep(results_so_far: str = "", only: str = "", parity_only: bool = False):
    import os
    import subprocess
    import sys
    import jax_plugins.xla_cuda12 as p
    bench = pathlib.Path("/ace-jax/bench/scaling")
    pjrt = os.path.join(os.path.dirname(p.__file__), "xla_cuda_plugin.so")
    (bench / "envs" / "modal-a100.json").write_text(json.dumps(
        {"lmp": "/opt/lmp.sh", "lmp_jax": "/opt/lmp-jax.sh", "pjrt": pjrt, "pythonpath": "/ace-jax/bench:/ace-jax/src",
         "python": sys.executable}))
    models = bench / "models"
    if not any(models.glob("mace_*.model")):                 # MACE-MP-0 / MH-1 + Symmetrix
        subprocess.run([sys.executable, str(bench / "models.py"), "mace"], check=True)
    res = pathlib.Path("/tmp/modal-a100.jsonl")
    res.write_text(results_so_far)                          # resume: skip finished cases
    cmd = [sys.executable, str(bench / "sweep.py"), "modal-a100", "--results", str(res)]
    cmd += (["--only", only] if only else []) + (["--parity-only"] if parity_only else [])
    subprocess.run(cmd, check=False)
    return res.read_text()[len(results_so_far):]            # only the new rows


@app.local_entrypoint()
def main(only: str = "", parity_only: bool = False):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    prior = RESULTS.read_text() if RESULTS.exists() else ""
    new = sweep.remote(prior, only, parity_only)
    with RESULTS.open("a") as f:
        f.write(new)
    rows = [json.loads(l) for l in new.splitlines() if l.strip()]
    print(f"{len(rows)} new rows ->", RESULTS)
    for r in rows:
        if r.get("mode") == "parity":
            print(f"  parity {r['gate']:7s} {r['system']:7s} {r['model']:28s} {r['status']}")
