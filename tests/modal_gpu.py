"""modal run tests/modal_gpu.py: the tests of the CUDA kernel on a Modal T4 (CI's `gpu` workflow; GitHub's
runners have no GPU). The image holds the lock's environment with the `cuda13` extra (JAX's CUDA 13 wheels
and NVIDIA's nvcc wheel, so no CUDA toolkit) and is rebuilt only when pyproject.toml or uv.lock change; the
checkout is mounted at run time. The kernels' libraries (`ffi.library`'s cache is keyed by content) and JAX's
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

image = (modal.Image.debian_slim(python_version="3.12").apt_install("build-essential").pip_install("uv")
         .add_local_file(ROOT / "pyproject.toml", "/deps/pyproject.toml", copy=True)
         .add_local_file(ROOT / "uv.lock", "/deps/uv.lock", copy=True)
         .run_commands(SYNC)
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
    return run(["python", "-m", "pytest", "-n", "0", "-m", "not slow", "-p", "no:cacheprovider", "-q", *(args or TESTS)])


@app.local_entrypoint()
def main(*args: str):
    sys.exit(gpu_tests.remote(list(args)))
