"""modal run tests/modal_gpu.py: the tests of the CUDA kernel on a Modal T4 (CI's `gpu` workflow; GitHub's
runners have no GPU). The image holds the lock's environment with the `cuda13` extra (JAX's CUDA 13 wheels
and NVIDIA's nvcc wheel, so no CUDA toolkit) and is rebuilt only when pyproject.toml or uv.lock change; the
checkout is mounted at run time. matscipy-neighbours' wheels are CPU-only, so the image rebuilds the locked
version with its CUDA backend (`MSN`; that build needs a CUDA toolkit, `TOOLKIT`: nvcc's wheel has no static
runtime for CMake to link), and the run fails if the GPU neighbour list is missing. The kernels' libraries (`ffi.library`'s cache is keyed by content) and JAX's
compile cache are kept in a Modal Volume. ACEJAX_REQUIRE_GPU: a GPU case that would skip fails. With empty
caches 23 min, about $0.35. Needs a Modal token (`modal token new`, or MODAL_TOKEN_ID and
MODAL_TOKEN_SECRET)."""
import pathlib
import subprocess
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ["tests/test_ffi.py", "tests/test_ffi_plan.py", "tests/test_batch.py", "tests/test_batch_md.py",
         "tests/test_export_lammps.py", "tests/test_calc_jit.py"]
SYNC = ("cd /deps && UV_PROJECT_ENVIRONMENT=/venv uv sync --frozen --no-install-project --python 3.12 "
        "--extra cuda13 --extra lammps --extra jaxmd --extra native")
# NVIDIA's apt repository for this Debian; the toolkit's minor version is that of the lock's CUDA 13 wheels
TOOLKIT = ("apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && . /etc/os-release && "
           "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian${VERSION_ID}/x86_64/cuda-keyring_1.1-1_all.deb && "
           "dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && "
           "apt-get install -y --no-install-recommends cuda-minimal-build-13-4")
MSN = ("V=$(/venv/bin/python -c \"from importlib.metadata import version; print(version('matscipy-neighbours'))\") && "
       "PATH=/usr/local/cuda-13.4/bin:$PATH uv pip install --python /venv/bin/python --no-deps "
       "--reinstall-package matscipy-neighbours --no-binary matscipy-neighbours matscipy-neighbours==$V "
       "-C cmake.define.ENABLE_CUDA=ON -C cmake.define.CMAKE_CUDA_ARCHITECTURES=75")      # 75: the T4
HAS_GPU_LIST = "import matscipy_neighbours._matscipy_neighbours as m; assert m._has_gpu, 'CPU-only matscipy-neighbours'"

image = (modal.Image.debian_slim(python_version="3.12").apt_install("build-essential").pip_install("uv")
         .add_local_file(ROOT / "pyproject.toml", "/deps/pyproject.toml", copy=True)
         .add_local_file(ROOT / "uv.lock", "/deps/uv.lock", copy=True)
         .run_commands(SYNC, TOOLKIT, MSN)
         .add_local_dir(ROOT, "/repo", ignore=lambda p: p.parts[0] not in ("src", "tests", "fixtures", "pyproject.toml")
                        or "__pycache__" in p.parts or p.suffix == ".so"))
app = modal.App("ace-jax-gpu-tests", image=image)
cache = modal.Volume.from_name("ace-jax-gpu-ci", create_if_missing=True)


@app.function(gpu="T4", cpu=4, memory=16384, timeout=2400, volumes={"/cache": cache})
def gpu_tests(args: list[str]) -> int:
    env = {"PATH": "/venv/bin:/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": "/repo/src", "HOME": "/root",
           "JAX_ENABLE_X64": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "ACEJAX_REQUIRE_GPU": "1",
           "ACEJAX_FFI_CACHE": "/cache/ffi", "JAX_COMPILATION_CACHE_DIR": "/cache/jax"}
    run = lambda c: subprocess.run(c, cwd="/repo", env=env).returncode
    run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if run(["python", "-c", "import jax; assert jax.default_backend() == 'gpu', jax.default_backend()"]):
        return 1                                            # a CPU JAX would pass with every GPU case skipped
    if run(["python", "-c", HAS_GPU_LIST]):
        return 1                                            # the calculator would fall back to the host list
    return run(["python", "-m", "pytest", "-n", "0", "-m", "not slow", "-p", "no:cacheprovider", "-q", *(args or TESTS)])


@app.local_entrypoint()
def main(*args: str):
    sys.exit(gpu_tests.remote(list(args)))
