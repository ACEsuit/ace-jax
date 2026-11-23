"""Lean linear ACE energies and forces as one XLA FFI call: C++ on the CPU, or `ace_ffi_gpu.cu` on a CUDA or
ROCm GPU.

Two entry points of the C++ kernel (`ace_ffi.cc`, the same per-site evaluator):
- `AceDirect` (`lammps_fns`): lammps-jax's direct force path for a lean float64 ACEModel with a radial table:
  positions and LAMMPS's neighbour matrix in, atom energies and forces out (newton on). Single-threaded:
  LAMMPS's MPI ranks give the parallelism. LAMMPS loads it with
  LAMMPS_JAX_FFI_HANDLERS="ace_jax_direct=<lib>:AceDirect" (`handlers`).
- `AceEdges` (`KernelModel`, `calc.ACECalculator(ffi=...)`): the dense edge input of
  `EdgeSiteModel.energy_forces_virial_dense` (rij, species, slot mask) to site energies and dE_i/drij, so
  JAX assembles forces and the virial as for any model.  Rows run in parallel on XLA's intra-op thread pool.
  Positions only: no gradients in the model's parameters and no second derivatives (use the JAX model).
On a GPU, `ace_ffi_gpu.cu` has the same two entry points, float64 or float32: `AceDirectGpu`
(`lammps_fns(..., gpu=True)`, loaded with LAMMPS_JAX_FFI_HANDLERS="ace_jax_direct_gpu=<lib>:AceDirectGpu")
and `AceEdgesGpu` (`KernelModel.of(model, gpu=True)`), whose dE_i/drij are written once per slot, so its
forces are deterministic; AceDirectGpu's are summed with atomics.

Distribution: the CPU kernel ships compiled in ace-jax's platform wheels (Linux x86_64/aarch64, macOS
arm64; `prebuilt`, `build_prebuilt`, the wheel hook `hatch_build.py`), and as source in every wheel.  A plain
`library()` takes the prebuilt one; elsewhere (the pure wheel, a checkout, another source than the one that
was compiled), or with flags (`ACEJAX_FFI_FLAGS`, `flags=`), it compiles the source on first use with the
user's C++17 compiler against the installed jaxlib's FFI headers, into a content-keyed cache; neither is
`FFIUnavailable`.  The prebuilt x86_64 kernel is x86-64-v3 (AVX2 + FMA; within 5% of -march=native on a
Xeon Silver 4216 and 2% on an EPYC 7742; x86-64-v4 was 10-41% SLOWER than native on the Xeon, the x86-64
baseline 1.3-3.3x slower: docs/dev/kernel-wheels.md), with the baseline for CPUs without AVX2.  It is
built against the OLDEST supported jaxlib (`PREBUILT_JAXLIB`): XLA accepts an FFI handler of an older API
minor version and refuses a newer one (built against 0.11.2, API 0.4, jaxlib 0.10.1 refuses it).
The GPU kernel likewise, with nvcc (a CUDA toolkit's or module's, or the `cuda13` extra's: NVIDIA's CUDA 13 nvcc wheel), for the GPU
JAX sees.
"""
import ctypes
import functools
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .aa_dag import build_plan
from . import coupled
from .edge_model import EdgeSiteModel

TARGET = "ace_jax_direct"
SYMBOL = "AceDirect"
EDGE_TARGET, EDGE_SYMBOL = "ace_jax_edges", "AceEdges"
SOURCE = pathlib.Path(__file__).with_name("ace_ffi.cc")
TARGET_GPU = "ace_jax_direct_gpu"
SYMBOL_GPU = "AceDirectGpu"
EDGE_TARGET_GPU, EDGE_SYMBOL_GPU = "ace_jax_edges_gpu", "AceEdgesGpu"
GPU_SOURCE = pathlib.Path(__file__).with_name("ace_ffi_gpu.cu")
DEFAULT_FLAGS = ("-O3", "-march=native")
PREBUILT = pathlib.Path(__file__).with_name("_prebuilt")    # in a platform wheel: the kernel compiled at release
PREBUILT_JAXLIB = "0.10.1"                 # the jaxlib the wheels' kernel is built against: pyproject's jax floor
X86_64_V3 = frozenset("avx avx2 bmi1 bmi2 f16c fma abm movbe".split())      # /proc/cpuinfo names (abm: lzcnt)
# per machine: (variant, flags, the CPU flags it needs), the first this CPU can run is taken
VARIANTS = {"x86_64": (("x86-64-v3", ("-O3", "-march=x86-64-v3"), X86_64_V3), ("x86-64", ("-O3",), frozenset())),
            "aarch64": (("generic", ("-O3",), frozenset()),), "arm64": (("generic", ("-O3",), frozenset()),)}
REGISTERED = set()


class FFIUnavailable(RuntimeError):
    """The kernel cannot be compiled here: no C++17 compiler, or the compile failed."""


def build(path, cxx="c++", flags=("-O3",)):
    """Compile the kernel to the shared library `path`."""
    path = pathlib.Path(path)
    # -Wno-psabi: the kernel's 32-byte vector types never cross its (internal) functions' ABI; without AVX, GCC warns
    subprocess.run([cxx, *flags, "-std=c++17", "-shared", "-fPIC", "-Wno-psabi", "-isystem", jax.ffi.include_dir(),
                    str(SOURCE), "-o", str(path)], check=True)
    return path


def source_key():
    """The CPU kernel source's hash: a prebuilt library is used only with the source it was compiled from."""
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]


def _version(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _cpu_flags():
    """This CPU's feature flags (Linux /proc/cpuinfo; empty elsewhere, where only one variant ships)."""
    try:
        for ln in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("flags"):
                return frozenset(ln.split(":", 1)[1].split())
    except OSError:
        pass
    return frozenset()


def build_prebuilt(out, cxx=None, floor=True):
    """Compile this machine's `VARIANTS` into the directory `out`, with the manifest `prebuilt` reads: what a
    platform wheel carries (hatch_build.py; kernel-wheels.yml runs this in a manylinux container, or on
    macOS for 11.0).  The C++ runtime is linked statically on Linux, so the library needs only glibc.
    floor: refuse a jaxlib other than `PREBUILT_JAXLIB` (a newer one's kernel does not load under an older)."""
    import jaxlib
    if floor and jaxlib.__version__ != PREBUILT_JAXLIB:
        raise RuntimeError(f"the wheels' kernel is built against jaxlib {PREBUILT_JAXLIB}, not {jaxlib.__version__}")
    cxx = cxx or compiler()
    if cxx is None:
        raise FFIUnavailable("no C++17 compiler to build the prebuilt kernel")
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    link = (("-mmacosx-version-min=11.0",) if platform.system() == "Darwin"
            else ("-static-libstdc++", "-static-libgcc"))
    variants = []
    for name, flags, needs in VARIANTS[platform.machine()]:
        build(out / f"libace_jax_ffi-{name}.so", cxx, (*flags, *link))
        variants.append({"name": name, "file": f"libace_jax_ffi-{name}.so", "flags": [*flags, *link],
                         "needs": sorted(needs)})
    ver = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    info = {"source": source_key(), "jaxlib": jaxlib.__version__, "machine": platform.machine(),
            "system": platform.system(), "compiler": ver[0] if ver else cxx, "variants": variants}
    (out / "manifest.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def prebuilt(env=None):
    """(path, variant) of the kernel compiled into this installation (a platform wheel), or None: no `PREBUILT`
    directory (the pure wheel, a checkout), `ACEJAX_FFI_PREBUILT=0`, another source than the one it was compiled
    from (an edited checkout), another machine type, a jaxlib older than the one it was built against, or no
    variant this CPU can run.  `ACEJAX_FFI_VARIANT` names one variant (e.g. x86-64, to compare)."""
    import jaxlib
    env = os.environ if env is None else env
    manifest = PREBUILT / "manifest.json"
    if env.get("ACEJAX_FFI_PREBUILT") == "0" or not manifest.is_file():
        return None
    info = json.loads(manifest.read_text())
    if (info["source"] != source_key() or info["machine"] != platform.machine()
            or info["system"] != platform.system() or _version(jaxlib.__version__) < _version(info["jaxlib"])):
        return None
    want, have = env.get("ACEJAX_FFI_VARIANT"), None
    for v in info["variants"]:
        if want and v["name"] != want:
            continue
        if v["needs"] and have is None:
            have = _cpu_flags()
        if set(v["needs"]) <= (have or set()) and (PREBUILT / v["file"]).is_file():
            return PREBUILT / v["file"], v["name"]
    return None


def build_gpu(path, compiler="nvcc", flags=("-O3", "-arch=native")):
    """Compile the GPU kernel to the shared library `path` with nvcc, or with hipcc and ROCm flags such as
    -x hip --offload-arch=gfx90a -munsafe-fp-atomics."""
    path = pathlib.Path(path)
    pic = ["-Xcompiler", "-fPIC"] if pathlib.Path(compiler).name.startswith("nvcc") else ["-fPIC"]
    subprocess.run([compiler, *flags, "-std=c++17", "-shared", *pic, "-isystem", jax.ffi.include_dir(),
                    str(GPU_SOURCE), "-o", str(path)], check=True)
    return path


def compiler(env=None):
    """The C++ compiler: $CXX, else the first of c++, g++, clang++ on PATH; None if there is none."""
    env = os.environ if env is None else env
    for c in ([env["CXX"]] if env.get("CXX") else []) + ["c++", "g++", "clang++"]:
        if (w := shutil.which(c)):
            return w
    return None


def _wheel_nvcc():
    """nvcc from NVIDIA's CUDA 13 nvcc wheel (`nvidia-cuda-nvcc`, the `cuda13` extra), with its headers and
    cudart beside it; None without it.  (The CUDA 12 wheel ships only ptxas.)"""
    import importlib.util
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return None
    for d in (spec.submodule_search_locations or []) if spec else []:
        if (p := pathlib.Path(d) / "cu13" / "bin" / "nvcc").is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def cuda_compiler(env=None):
    """nvcc: $NVCC, else the CUDA 13 nvcc wheel's (`_wheel_nvcc`, matching JAX's CUDA 13 wheels), else on PATH,
    else under $CUDA_HOME; None if there is none."""
    env = os.environ if env is None else env
    home = [os.path.join(env["CUDA_HOME"], "bin", "nvcc")] if env.get("CUDA_HOME") else []
    for c in ([env["NVCC"]] if env.get("NVCC") else []) + [_wheel_nvcc()] + ["nvcc"] + home:
        if c and (w := shutil.which(c)):
            return w
    return None


def gpu_arch():
    """The compute capability of the first GPU JAX sees, as nvcc's sm_XY digits."""
    try:
        cc = getattr(jax.devices("gpu")[0], "compute_capability", None)
    except RuntimeError:
        cc = None
    if not cc:
        raise FFIUnavailable("no CUDA GPU to build the ace-jax GPU kernel for: pass flags such as "
                             "-arch=sm_80, or set ACEJAX_FFI_GPU_FLAGS")
    return cc.replace(".", "")


def cache_dir(env=None):
    """`ACEJAX_FFI_CACHE`, else `$XDG_CACHE_HOME/ace-jax/ffi` (default `~/.cache/...`), as the coupling cache."""
    env = os.environ if env is None else env
    if env.get("ACEJAX_FFI_CACHE"):
        return pathlib.Path(env["ACEJAX_FFI_CACHE"]).expanduser()
    return pathlib.Path(env.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")) / "ace-jax" / "ffi"


def _cpu():
    """This host's CPU model: part of the key when the flags tune for it (-march=native), since a cache in a
    shared home is read from login and compute nodes of other CPU types."""
    try:
        for ln in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def library_path(flags=None, cxx=None, cache=None, gpu=False):
    """(path, compiler, flags) of the kernel library for these settings, without building it: the cache key is
    the source, the jaxlib version (its FFI headers), the compiler and its version, the flags, the machine and,
    under -march=native, the CPU model.  flags: `ACEJAX_FFI_FLAGS` (space-separated), else DEFAULT_FLAGS.
    gpu: the GPU kernel, with `cuda_compiler` for `gpu_arch` and flags `ACEJAX_FFI_GPU_FLAGS`, else
    ("-O3", "-arch=sm_XY"), keyed on the GPU model under -arch=native; cxx="hipcc" with ROCm flags builds it
    for an AMD GPU.  No compiler raises FFIUnavailable."""
    import jaxlib
    var = "ACEJAX_FFI_GPU_FLAGS" if gpu else "ACEJAX_FFI_FLAGS"
    flags = tuple(flags if flags is not None else os.environ.get(var, "").split()
                  or (("-O3", f"-arch=sm_{gpu_arch()}") if gpu else DEFAULT_FLAGS))
    cxx = cxx or (cuda_compiler() if gpu else compiler())
    if cxx is None and gpu:
        raise FFIUnavailable("the ace-jax GPU kernel is compiled on first use and needs nvcc: load a CUDA "
                             "module, set NVCC or CUDA_HOME, or install the `cuda13` extra (driver >= 580); "
                             "or run without ffi")
    if cxx is None:
        raise FFIUnavailable("the ace-jax CPU kernel (ffi=True) is compiled on first use and needs a C++17 compiler; "
                             "none found (set CXX, or install g++ or clang++), or export without ffi")
    ver = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    native = any("native" in f for f in flags)
    source, host = (GPU_SOURCE, gpu_arch) if gpu else (SOURCE, _cpu)
    key = hashlib.sha256("\0".join([source.read_text(), jaxlib.__version__, cxx, *ver, *flags, platform.machine(),
                                     host() if native else ""]).encode()).hexdigest()[:16]
    d = pathlib.Path(cache) if cache else cache_dir()
    return d / f"libace_jax_ffi{'_gpu' if gpu else ''}-{key}.so", cxx, flags


def library(flags=None, cxx=None, cache=None, gpu=False):
    """The compiled kernel's path, built on first use (`library_path`'s key); pass e.g. flags=("-O3",
    "-march=x86-64-v3") to build on a login node for compute nodes of another CPU type.  `aj native --build`
    builds it ahead of use.  A plain call (no flags, compiler or cache, and no `ACEJAX_FFI_FLAGS`) for the
    CPU kernel returns the `prebuilt` one when this installation has it: nothing is compiled."""
    if not gpu and flags is None and cxx is None and cache is None and not os.environ.get("ACEJAX_FFI_FLAGS"):
        if (pre := prebuilt()) is not None:
            return pre[0]
    out, cxx, flags = library_path(flags, cxx, cache, gpu)
    if out.exists():
        return out
    d = out.parent
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".so.tmp")      # same dir: os.replace is atomic, so concurrent
    os.close(fd)                                             # ranks or jobs never load a half-written library
    try:
        (build_gpu if gpu else build)(tmp, cxx, flags)
        os.replace(tmp, out)
    except subprocess.CalledProcessError as e:
        what = f"the ace-jax {'GPU' if gpu else 'CPU'} kernel with {cxx} {' '.join(flags)}"
        raise FFIUnavailable(f"compiling {what} failed (exit {e.returncode}); see the compiler output "
                             "above") from None
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)
    return out


def handlers(path, gpu=False):
    """The LAMMPS_JAX_FFI_HANDLERS value that lets LAMMPS load the kernel at `path`, the GPU one if `gpu`."""
    return f"{TARGET_GPU if gpu else TARGET}={pathlib.Path(path).resolve()}:{SYMBOL_GPU if gpu else SYMBOL}"


def register(path):
    """Register the compiled kernel's entry points for in-process CPU programs (once per process: a later
    library, e.g. other flags, does not replace them)."""
    lib = None
    for target, symbol in ((TARGET, SYMBOL), (EDGE_TARGET, EDGE_SYMBOL)):
        if target not in REGISTERED:
            lib = lib or ctypes.cdll.LoadLibrary(str(path))
            jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, symbol)), platform="cpu")
            REGISTERED.add(target)


def register_gpu(path, platform="CUDA"):
    """Register the compiled GPU kernel's entry points for in-process programs; platform "CUDA" or "ROCM"
    (once per process, as `register`)."""
    lib = None
    for target, symbol in ((TARGET_GPU, SYMBOL_GPU), (EDGE_TARGET_GPU, EDGE_SYMBOL_GPU)):
        if (target, platform) not in REGISTERED:
            lib = lib or ctypes.cdll.LoadLibrary(str(path))
            jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, symbol)), platform=platform)
            REGISTERED.add((target, platform))


def plan(model, gpu=False):
    """(runs, groups, factor, ct, n_leaf, levels): A in runs of consecutive harmonics; the product basis as a
    flat DAG whose last n_leaf nodes are its deepest level. Each level holds a parent's children together, the
    largest groups first. groups (n_groups, 2) gives each group's parent and end,
    levels each level's first group and the end."""
    ar, ay = np.asarray(model.aspec_r), np.asarray(model.aspec_y)
    n_a = len(ar)
    order = np.lexsort((ay, ar))
    new = np.empty(n_a, np.int64); new[order] = np.arange(n_a)
    levels, readout, _ = build_plan([np.asarray(s) for s in model.aa_specs], n_a, "greedy")
    pars, facs, invs, lev = [], [], [new], [0]              # invs: old to new index, per level
    for p, a in levels:
        p, f = invs[-1][p], new[a]
        _, g, cnt = np.unique(p, return_inverse=True, return_counts=True)
        lev.append(lev[-1] + len(cnt))
        perm = np.lexsort((p, -cnt[g]))
        invs.append(np.empty(len(p), np.int64)); invs[-1][perm] = np.arange(len(p))
        pars.append(p[perm]); facs.append(f[perm])
    offs = np.cumsum([0, n_a] + [len(p) for p in pars])
    parent = np.concatenate([np.zeros(0, np.int64)] + [offs[k] + p if k else p for k, p in enumerate(pars)])
    ends = np.flatnonzero(np.diff(parent, append=-1)) + 1
    factor = np.concatenate([np.zeros(0, np.int64)] + facs)
    ctilde = np.asarray(model.ctilde, np.float64)
    ct = np.zeros((offs[-1], ctilde.shape[1]))
    for k, (c, nd) in enumerate(readout):
        np.add.at(ct, offs[k] + invs[k][nd], ctilde[c])
    rs, ys = ar[order], ay[order]
    deg = np.sqrt(ys).astype(np.int64) if gpu else 0 * ys     # the GPU kernel files each run under one l
    a0 = np.nonzero(np.r_[True, (rs[1:] != rs[:-1]) | (ys[1:] != ys[:-1] + 1) | (deg[1:] != deg[:-1])])[0]
    runs = np.stack([a0, rs[a0], ys[a0], np.diff(np.r_[a0, n_a])], axis=1)
    if not gpu:
        runs = runs[np.argsort(runs[:, 3], kind="stable")]   # by length: the C++ kernel switches on it
    return (runs, np.stack([parent[ends - 1], ends], axis=1), factor, np.ascontiguousarray(ct.T),
            len(levels[-1][0]) if levels else 0, np.asarray(lev))


PRECISIONS = ("float64", "mixed")
FORMS = ("auto", "dag", "coupled")
FORM = os.environ.get("ACEJAX_FFI_FORM") or None   # the product basis' form for every kernel_inputs: None, by argument
COUPLED_SPLIT = 1.25                                  # a task above this x the mean work of a thread is split
COUPLED_THREADS = None                                # the GPU pass's threads per block: None by size
COUPLED_SHARED = 90 << 10                             # bytes of z, g and pieces of one row a GPU block may take
COUPLED_GAIN = 0.8                                    # auto: coupled when its operations are below this x the DAG's
RECOUPLE = True                                       # `coupled.build`: candidates of every pairing (o4d20: X 1,879 -> 1,274)
ILP_SECONDS = 0.0                                     # `coupled.build`'s integer program (greedy alone: within ~5 %)
_COUPLED = {}                                         # spec digest -> coupled.Plan or the NoPlan


def _a_order(model):
    """The kernels' order of A, by radial column then harmonic: (order, old index -> new)."""
    ar, ay = np.asarray(model.aspec_r), np.asarray(model.aspec_y)
    order = np.lexsort((ay, ar))
    new = np.empty(len(ar), np.int64); new[order] = np.arange(len(ar))
    return order, new


def coupled_of(model):
    """`coupled.build`'s plan of the model's product basis (kept per spec: it does not depend on the readout);
    raises `coupled.NoPlan` where there is none."""
    ar, ay = np.asarray(model.aspec_r, np.int64), np.asarray(model.aspec_y, np.int64)
    specs = [np.ascontiguousarray(s, np.int64) for s in model.aa_specs]
    hh = hashlib.sha256()
    for a in (ar, ay, *specs):
        hh.update(str(a.shape).encode()); hh.update(np.ascontiguousarray(a).tobytes())
    key = (hh.hexdigest(), ILP_SECONDS, RECOUPLE)
    if key not in _COUPLED:
        if len(_COUPLED) > 16:
            _COUPLED.clear()
        try:
            _COUPLED[key] = coupled.build(ar, ay, specs, ilp_seconds=ILP_SECONDS, recouple=RECOUPLE)
        except coupled.NoPlan as e:
            _COUPLED[key] = e
    if isinstance(_COUPLED[key], coupled.NoPlan):
        raise _COUPLED[key]
    return _COUPLED[key]


def _coupled_terms(model, check=True):
    """The coupled plan of the model in the kernels' order of A (every block a run of z), as arrays: n_a, n_x,
    nz; xt (x, a, b) and xc, the CG terms sorted by x; per row entry its row's first z index ru, 2L + 1 rK, its
    column's first z index rv and its weights rw (nz, entries), sorted by row; tu (a, x, partner) and ac, the
    dE/dA terms sorted by a; lin (nz, n_a)."""
    import types
    P = coupled_of(model)
    ct = np.asarray(model.ctilde, np.float64)
    w = P.weights(ct)
    if check:                                              # a float32 readout is in the span to float32 only
        coupled.check(P, [np.asarray(s) for s in model.aa_specs], ct, w,
                      tol=coupled.RESIDUAL if model.ctilde.dtype == jnp.float64 else 1e-5)
    _, new = _a_order(model)
    n_a, nb, n_lin, nz = P.n_a, len(P.bl), len(P.lin), ct.shape[1]
    base = np.empty(nb + len(P.xb), np.int64)
    for b, q in enumerate(P.mem):
        qn = new[np.asarray(q)]
        if not np.array_equal(qn, qn[0] + np.arange(len(qn))):
            raise coupled.NoPlan("an A block is not a run of the kernels' A")
        base[b] = qn[0]
    base[nb:] = P.x_base
    K = np.r_[2 * P.bl + 1, 2 * P.xb[:, 2] + 1]
    xt, xc = P.cg_terms()
    xt = np.stack([xt[:, 0], new[xt[:, 1]], new[xt[:, 2]]], 1) if len(xt) else xt
    o = np.lexsort((xt[:, 2], xt[:, 1], xt[:, 0]))
    xt, xc = xt[o], xc[o]
    U, V = P.quad[:, 0], P.quad[:, 1]                      # rows: U gets (V, w), V gets (U, w); U = V: 2 w
    widx = np.arange(len(P.quad))
    same = U == V
    ru = np.r_[U, V[~same]]; rv = np.r_[V, U[~same]]; rw = np.r_[widx, widx[~same]]
    rf = np.r_[np.where(same, 2.0, 1.0), np.ones(int((~same).sum()))]
    o = np.lexsort((base[rv], ru))
    ru, rv, rw, rf = ru[o], rv[o], rw[o], rf[o]
    t = np.r_[np.stack([xt[:, 1], xt[:, 0], xt[:, 2]], 1), np.stack([xt[:, 2], xt[:, 0], xt[:, 1]], 1)]
    tu, tinv = np.unique(t, axis=0, return_inverse=True)   # sorted by target A, X, partner; equal terms merged
    ac = np.bincount(tinv.ravel(), np.r_[xc, xc], minlength=len(tu))
    lin = np.zeros((nz, n_a)); lin[:, new[P.lin]] = w[:n_lin].T
    return types.SimpleNamespace(n_a=n_a, n_x=P.n_x, nz=nz, xt=xt, xc=xc, ru=base[ru], rK=K[ru], rv=base[rv],
                                 rw=(w[n_lin:][rw] * rf[:, None]).T, tu=tu, ac=ac, lin=lin,
                                 ops=2 * len(xt) + int(K[ru].sum()) + 2 * len(tu))


def coupled_plan(model, check=True):
    """(ints (n, 2), reals (NZ, N), operations): the coupled form of the model's product basis for the C++
    kernel (`coupled.Plan`, A in the kernels' order, so every block is a run of z).

    ints, flat: a header (n_a, n_x, CG terms, block rows, row entries, dE/dA entries, 0, 0), then the X values
    as CSR (xptr, and each term's two A), the block rows (z index of the first component, 2L + 1, end of its
    entries), each entry's first z index, and dE/dA as CSR per A (aptr, the X in z and the partner A of each term).
    reals, per centre species: the linear weights (n_a), the entries' weights, the CG coefficients of the X
    terms and of the dE/dA terms.  operations: multiply-adds a site, to compare with 3 a DAG node."""
    c = _coupled_terms(model, check)
    n_a, nz = c.n_a, c.nz
    xptr = np.r_[0, np.cumsum(np.bincount(c.xt[:, 0] - n_a, minlength=c.n_x))]
    us, first, cnt = np.unique(c.ru, return_index=True, return_counts=True)
    rows = np.stack([us, c.rK[first], np.cumsum(cnt)], 1)
    aptr = np.r_[0, np.cumsum(np.bincount(c.tu[:, 0], minlength=n_a))]
    ints = np.concatenate([[n_a, c.n_x, len(c.xt), len(rows), len(c.rv), len(c.tu), 0, 0], xptr, c.xt[:, 1],
                           c.xt[:, 2], rows.ravel(), c.rv, aptr, c.tu[:, 1], c.tu[:, 2]]).astype(np.int64)
    if ints.max(initial=0) >= 2 ** 31:
        raise coupled.NoPlan("the coupled plan overflows int32")
    ints = np.r_[ints, np.zeros(len(ints) % 2, np.int64)].reshape(-1, 2)
    reals = np.concatenate([c.lin, c.rw, np.tile(c.xc, (nz, 1)), np.tile(c.ac, (nz, 1))], 1)
    return ints, np.ascontiguousarray(reals), c.ops


GPU_NC = 4                                            # components of a block row a thread takes together, at most
GPU_HEAD = 48                                         # ints before the streams' arrays
END = np.int64(1) << 31                               # an item that ends its task


def coupled_threads(ops, real=8):
    """Threads per block of the GPU kernel's coupled pass, by its operations a site and the bytes of its
    floats (A100, o4d20, the pass in ms at 128 and 256 threads: float64 5.5, 4.7; float32 2.9, 3.4)."""
    return 32 if ops < 2000 else 64 if ops < 8000 else 128 if ops < 30000 or real == 4 else 256


def _stream(tasks, T, n_tot, piece0, split, nz=1):
    """Tasks (dst, u, item ints, their coefficient columns (k, items)) dealt to T threads of near-equal work.
    A task above `split` items goes in pieces, the later ones to piece slots from n_tot + piece0 on (`nc`
    slots each, nc = the task's components), which a fix-up adds to dst.  Returns items (steps, T) (a thread's
    stream, END on a task's last item, padded with item 0), hdr (tasks, T) (dst | u << 16 per task in order),
    cols (k, steps, T) (padded with 0), fixes [(dst, first slot, pieces, stride)], piece slots used."""
    import heapq
    parts, fixes, used = [], [], 0
    for dst, u, nc, it, co in tasks:
        n = len(it)
        if split and n > split:
            k = -(-n // split)
            cuts = np.linspace(0, n, k + 1).astype(int)
            parts.append((dst, u, it[:cuts[1]], co[:, :cuts[1]]))
            for j in range(1, k):
                parts.append((n_tot + piece0 + used + (j - 1) * nc, u, it[cuts[j]:cuts[j + 1]], co[:, cuts[j]:cuts[j + 1]]))
            fixes += [(dst + m, n_tot + piece0 + used + m, k - 1, nc) for m in range(nc)]
            used += (k - 1) * nc
        else:
            parts.append((dst, u, it, co))
    order = sorted(range(len(parts)), key=lambda i: -len(parts[i][2]))
    heap = [(0, t) for t in range(T)]
    mine = [[] for _ in range(T)]
    for i in order:
        load, t = heapq.heappop(heap)
        mine[t].append(i)
        heapq.heappush(heap, (load + len(parts[i][2]), t))
    steps = max((sum(len(parts[i][2]) for i in m) for m in mine), default=0)
    steps = -(-steps // 8) * 8                              # the kernel loads 4 or 8 steps at a time
    ntask = max((len(m) for m in mine), default=0)
    k = tasks[0][4].shape[0] if tasks else nz
    items, hdr, cols = np.zeros((steps, T), np.int64), np.zeros((ntask, T), np.int64), np.zeros((k, steps, T))
    for t, m in enumerate(mine):
        j = 0
        for q, i in enumerate(sorted(m, key=lambda i: parts[i][0])):
            dst, u, it, co = parts[i]
            n = len(it)
            items[j:j + n, t] = it
            items[j + n - 1, t] |= END
            cols[:, j:j + n, t] = co
            hdr[q, t] = dst | (u << 16)
            j += n
    return items, hdr, cols, fixes, used


def coupled_gpu_plan(model, T=None, check=True, real=8):
    """(ints (n,), reals (NZ, N), threads): the coupled form for the GPU kernel's pass, blocks of T threads
    (`coupled_threads`), every sum by the thread that owns its target (no atomic adds).

    Six streams, each a list of items per thread stored by step (thread t's step j at T j + t), one coefficient
    an item: the X values (item a | b << 15: X += c z_a z_b); the block rows, by the number nc = 1..GPU_NC of
    components taken together (item v: g_{dst + m} += w z_{v + m}, m < nc, and the energy z_{u + m} g); dE/dA
    (item x | a << 15: g_dst += c g_x z_a).  A task's target, and for the rows the z index its energy reads,
    come from the stream's headers.  Long tasks are split (`_stream`), and after the rows and after dE/dA
    the fix-ups (dst, first slot, pieces, stride) add the pieces in.

    ints: T, n_a, n_x, piece slots, then per stream (steps, tasks, and the places of its items, headers and
    coefficients (in reals)), per fix-up list (count, place), then the arrays.  reals, per centre species: the
    linear weights, then each stream's coefficients."""
    c = _coupled_terms(model, check)
    n_a, n_tot, nz = c.n_a, c.n_a + c.n_x, c.nz
    if n_tot + 4 * GPU_NC >= 1 << 15:
        raise coupled.NoPlan(f"{n_tot} values of z: the GPU kernel's items take 15 bits")
    T = T or int(os.environ.get("ACEJAX_CPL_T", 0)) or coupled_threads(c.ops, real)   # (the variable: development)
    one = lambda v: np.tile(np.asarray(v, np.float64), (nz, 1))
    t1, e = [], np.r_[0, np.cumsum(np.bincount(c.xt[:, 0] - n_a, minlength=c.n_x))]
    for x in range(c.n_x):
        sl = slice(e[x], e[x + 1])
        t1.append((n_a + x, 0, 1, c.xt[sl, 1] | (c.xt[sl, 2] << 15), one(c.xc[sl])))
    us, first, cnt = np.unique(c.ru, return_index=True, return_counts=True)
    t2 = [[] for _ in range(GPU_NC)]
    for u, f, n in zip(us.tolist(), first.tolist(), cnt.tolist()):
        K = int(c.rK[f])
        parts = -(-K // GPU_NC)
        m0 = 0
        for j in range(parts):
            nc = K // parts + (j < K % parts)
            t2[nc - 1].append((u + m0, u + m0, nc, c.rv[f:f + n] + m0, c.rw[:, f:f + n]))
            m0 += nc
    t3, e = [], np.r_[0, np.cumsum(np.bincount(c.tu[:, 0], minlength=n_a))]
    for a in range(n_a):
        if e[a + 1] > e[a]:
            sl = slice(e[a], e[a + 1])
            t3.append((a, 0, 1, c.tu[sl, 1] | (c.tu[sl, 2] << 15), one(c.ac[sl])))
    work2 = sum(len(t[3]) for cl in t2 for t in cl)
    cap = lambda work: max(8, int(np.ceil(COUPLED_SPLIT * work / T)))
    streams, fixes, pieces = [_stream(t1, T, n_tot, 0, 0, nz)], [], 0
    f2, used = [], 0
    for cl in t2:
        st = _stream(cl, T, n_tot, used, cap(work2 / GPU_NC), nz)
        streams.append(st); f2 += st[3]; used += st[4]
    fixes.append(f2); pieces = used
    st = _stream(t3, T, n_tot, 0, cap(len(c.tu)), nz)
    streams.append(st); fixes.append(st[3]); pieces = max(pieces, st[4])
    if n_tot + pieces >= 1 << 16:
        raise coupled.NoPlan("the coupled plan's targets take 16 bits")
    head, ints, reals = [T, n_a, c.n_x, pieces, len(streams)], [], [c.lin]
    off, roff = GPU_HEAD, n_a
    for items, hdr, cols, _, _ in streams:
        head += [items.shape[0], hdr.shape[0], off, off + items.size, roff]
        ints += [items.ravel(), hdr.ravel()]
        reals.append(cols.reshape(nz, -1))
        off += items.size + hdr.size; roff += items.size
    for fx in fixes:
        fa = np.asarray(fx, np.int64).reshape(-1, 4)
        head += [len(fa), off]
        ints.append((np.stack([fa[:, 0] | (fa[:, 2] << 16), fa[:, 1] | (fa[:, 3] << 16)], 1)).ravel())
        off += 2 * len(fa)
    assert len(head) <= GPU_HEAD
    ints = np.concatenate([np.r_[head, np.zeros(GPU_HEAD - len(head), np.int64)]] + ints)
    return ints.astype(np.uint32).view(np.int32), np.ascontiguousarray(np.concatenate(reals, 1)), T


ANALYTIC_LMAX = 8                                     # the degrees the kernel's analytic force pass is made for
# The GPU kernel's radial: None or "auto" is analytic where the model has `analytic_plan` (a polynomial tensor
# radial and an analytic pair radial, evaluated by the kernel itself in its force pass, a lane per neighbour, and
# for few A in float64 in its A pass too), else the radial table; "table" always the table; "analytic" raises
# where there is no plan.  Where only the force pass is analytic the energy is the table's and the forces the
# model's: they differ from the table energy's gradient by the table's error (~1e-8 eV/A).
RADIAL = os.environ.get("ACEJAX_FFI_RADIAL") or None


def analytic_plan(model, runs):
    """(ints, reals, nq, npq) of the GPU kernel's analytic radial for a lean model whose tensor radial is the
    polynomial one (`radial_kind` "analytic", `rnl_basis` "poly": R_c = env(x) sum_q W_cq P_q(x), x the Agnesi
    transform), or None.  `runs`: `plan(model, gpu=True)`'s, each of one degree l.  ints: per species pair and
    degree the first entry of each q (NZ^2 (L + 1) (nq + 1)), then each entry's first A (of its run's 2l + 1);
    then the same per centre species (the A pass sums a warp's lanes together, so its entries cannot depend on
    the neighbour: their weights, last in reals, are per neighbour species); then per centre species each A
    value's first term (NZ (n_a + 1)) and the terms, entry << 14 | q << 8 | lm (what the A pass reads);
    reals: per pair the transform's 7 and the envelope's 5 parameters; per pair the folded pair radial's
    transform (7), its envelope's cutoff, r0 and power and its kind (0 "ace1_poly1sr", 1 "poly1sr"), then 0;
    the recursion's A, B, C (nq each); the pair radial's (npq each) and its coefficients per pair; the entries'
    W.  nq stops at the last polynomial any column reads.  The pair radial is analytic too: at o3d8 it and the
    tensor radial give forces of 600 eV/A that cancel to 2, and so do their errors in one table, which the
    table's pair radial beside an exact tensor radial would lose (1e-7 eV/A).  None for another radial or pair
    radial kind, above degree `ANALYTIC_LMAX`, or where a run is not a whole degree."""
    if (model.radial_kind != "analytic" or getattr(model, "rnl_basis", "poly") != "poly"
            or model.pair_radial_kind != "analytic" or model.pair_Wnlq.shape[2] != 1
            or model.pair_envelope_kind not in ("ace1_poly1sr", "poly1sr")):
        return None
    W = np.asarray(model.rnl_Wnlq, np.float64)             # (NZ, NZ, n_rnl, n_q)
    nz, _, n_rnl, _ = W.shape
    if n_rnl != model.rtab_coefs.shape[-1] - model.Wpair.shape[0]:
        return None
    used = np.flatnonzero(np.abs(W).sum(axis=(0, 1, 2)))
    nq = int(used.max()) + 1 if len(used) else 1
    deg = np.sqrt(runs[:, 2]).astype(np.int64)             # each run's degree
    L = int(model.lmax)
    if L > ANALYTIC_LMAX or nq > 63 or not (np.array_equal(runs[:, 2], deg * deg)
                                            and np.array_equal(runs[:, 3], 2 * deg + 1)):   # nq: 6 bits of a term
        return None
    ptr, ents, vals, off = [], [], [], 0
    for pz in range(nz * nz):
        Wp = W.reshape(nz * nz, n_rnl, -1)[pz][:, :nq]
        for l in range(L + 1):
            u = np.flatnonzero(deg == l)
            n, q = np.nonzero(Wp[runs[u, 1]])              # (run of this degree, q)
            o = np.lexsort((n, q))
            ptr.append(off + np.r_[0, np.cumsum(np.bincount(q, minlength=nq))])
            ents.append(runs[u, 0][n[o]]); vals.append(Wp[runs[u, 1]][n[o], q[o]])
            off += len(o)
    dptr, dents, dvals, off = [], [], [], 0                 # pass 1: per centre species, weights per neighbour's
    W4 = W[..., :nq]
    n_a = int((runs[:, 0] + runs[:, 3]).max())
    terms = [[[] for _ in range(n_a)] for _ in range(nz)]   # per centre species and A value: entry, q, lm
    for zc in range(nz):
        for l in range(L + 1):
            u = np.flatnonzero(deg == l)
            n, q = np.nonzero(np.abs(W4[zc][:, runs[u, 1]]).sum(0))
            o = np.lexsort((n, q))
            dptr.append(off + np.r_[0, np.cumsum(np.bincount(q, minlength=nq))])
            dents.append(runs[u, 0][n[o]]); dvals.append(W4[zc][:, runs[u, 1][n[o]], q[o]].T.ravel())
            for e, (first, qq) in enumerate(zip(runs[u, 0][n[o]], q[o]), off):   # the same entries by A value
                for mm in range(2 * l + 1):
                    terms[zc][first + mm].append(e << 14 | int(qq) << 8 | (l * l + mm))
            off += len(o)
    aptr = np.cumsum([0] + [len(t) for tz in terms for t in tz])
    aptr = np.concatenate([aptr[zc * n_a:(zc + 1) * n_a + 1] for zc in range(nz)])   # (NZ, n_a + 1)
    aitem = np.array([c for tz in terms for t in tz for c in t], np.int64)
    par = np.concatenate([np.asarray(model.rnl_transform, np.float64).reshape(nz * nz, 7),
                          np.asarray(model.rnl_envelope, np.float64).reshape(nz * nz, 5)], 1)
    abc = np.concatenate([np.asarray(a, np.float64)[:nq] for a in (model.polys_A, model.polys_B, model.polys_C)])
    pc = np.asarray(model.pair_Wnlq, np.float64).reshape(nz * nz, -1)      # folded: (pair, q)
    npq = pc.shape[1]
    pe = np.asarray(model.pair_envelope, np.float64).reshape(nz * nz, -1)
    ace1 = model.pair_envelope_kind == "ace1_poly1sr"       # (rcut, r0, p), or (rcut, p)
    penv = np.stack([pe[:, 0], pe[:, 1] if ace1 else pe[:, 0], pe[:, 2] if ace1 else pe[:, 1],
                     np.full(nz * nz, 0.0 if ace1 else 1.0), np.zeros(nz * nz)], 1)
    ppar = np.concatenate([np.asarray(model.pair_transform, np.float64).reshape(nz * nz, 7), penv], 1)
    pabc = np.concatenate([np.asarray(a, np.float64)[:npq]
                           for a in (model.pair_polys_A, model.pair_polys_B, model.pair_polys_C)])
    return (np.concatenate([np.concatenate(ptr), np.concatenate(ents), np.concatenate(dptr),
                            np.concatenate(dents), aptr, aitem]).astype(np.int32),
            np.concatenate([par.ravel(), ppar.ravel(), abc, pabc, pc.ravel(), np.concatenate(vals),
                            np.concatenate(dvals)]), nq, npq)


def choose_form(model, form, n_nodes):
    """"dag" or "coupled" for `form` (`FORMS`): auto is coupled where the model has a coupled plan with fewer
    operations than `COUPLED_GAIN` x the DAG's (3 a node).  Returns (form, the coupled arrays or None)."""
    form = FORM or form or "auto"
    if form not in FORMS:
        raise ValueError(f"ffi_form is one of {FORMS}, got {form!r}")
    if form == "dag":
        return "dag", None
    try:
        cp = coupled_plan(model)
    except coupled.NoPlan:
        if form == "coupled":
            raise
        return "dag", None
    if form == "auto" and cp[2] > COUPLED_GAIN * 3 * n_nodes:
        return "dag", None
    return "coupled", cp


def check_precision(precision):
    """`precision` as the kernels take it: None or "float64" (the default), or "mixed"."""
    if precision not in (None,) + PRECISIONS:
        raise ValueError(f"ffi_precision is one of {PRECISIONS}, got {precision!r}")
    return precision == "mixed"


def kernel_inputs(model, gpu=False, precision=None, form=None):
    """(attrs, consts) of the kernel for a lean ACEModel with a radial table, float64, or float32 on the GPU;
    the GPU kernel also takes the DAG's levels, n_st, the nodes before the last two levels, which it keeps,
    `pull_plan`'s lists and the leaves' coefficients in their two orders.

    precision="mixed" (a float64 model): the readout arrays in float32, which is how both kernels are told to
    run the product-basis DAG and its adjoint in float32 (twice the sites a CPU register; the GPU pass's
    float32 form) with everything else float64. Forces differ from float64 by ~1e-5 eV/A; energies carry
    noise from the float32 node values (~0.1 meV in 200 atoms at o4d20), so optimisers that line-search on
    energies can fail near a minimum (docs/dev/accuracy-cost-results.md).

    form (`FORMS`, None = "auto"): the product basis as the DAG or in coupled form (`coupled_plan`, whose arrays
    then take the DAG's operands: the plan for the groups, its reals for the readout; attribute form = 1).  On
    the GPU `coupled_gpu_plan`: its streams for `pull`, their coefficients for `lc`, attributes n_st and tile
    its X values and piece slots; auto also keeps the DAG where z does not fit a block's shared memory.
    The GPU's radial is analytic where the model has `analytic_plan` (`RADIAL` "table" keeps the table).
    `describe(attrs)` names what was taken."""
    from .model import ACEModel
    if not (isinstance(model, ACEModel) and model.energy_only and model.rtab_grid):
        raise ValueError("the FFI kernel needs a lean ACEModel with a radial table: "
                         "lean(model, radial_table=True)")
    if model.E0.dtype != jnp.float64 and not (gpu and model.E0.dtype == jnp.float32):
        raise ValueError(f"the FFI kernel takes float64, or float32 on the GPU, got {model.E0.dtype}")
    mixed = check_precision(precision)
    if mixed and model.E0.dtype != jnp.float64:
        raise ValueError(f"ffi_precision='mixed' is for a float64 model, got {model.E0.dtype}")
    dag = jnp.float32 if mixed else model.E0.dtype          # the readout's type is the DAG's
    x0, h, n = model.rtab_grid
    runs, groups, factor, ct, n_leaf, lev = plan(model, gpu)
    attrs = dict(x0=np.float64(x0), h=np.float64(h), n=np.int64(n), lmax=np.int64(model.lmax),
                 n_rnl=np.int64(model.rtab_coefs.shape[-1] - model.Wpair.shape[0]),
                 solid=np.int64(model.ysolid), n_leaf=np.int64(n_leaf))
    i32 = lambda a: jnp.asarray(a, jnp.int32)
    if gpu:                                                # the radial: the table, or (asked for) analytic
        if RADIAL not in (None, "auto", "table", "analytic"):
            raise ValueError(f"ffi.RADIAL is 'auto', 'table' or 'analytic', got {RADIAL!r}")
        an = analytic_plan(model, runs) if RADIAL != "table" else None      # where the model has one
        if an is None and RADIAL == "analytic":
            raise ValueError("ffi.RADIAL = 'analytic': this model has no analytic plan (`analytic_plan`)")
        radial = (i32(an[0]), jnp.asarray(an[1], model.E0.dtype)) if an else (i32(np.zeros(1)), jnp.zeros(1, model.E0.dtype))
        attrs["nq"], attrs["npq"] = np.int64(an[2] if an else 0), np.int64(an[3] if an else 0)
    if not gpu:
        attrs["form"] = np.int64(0)
        which, cp = choose_form(model, form, len(factor))
        if which == "coupled":
            attrs.update(form=np.int64(1), n_leaf=np.int64(0))
            return attrs, (model.rtab_coefs, model.table_cutoffs(), i32(runs), i32(cp[0]), i32(np.zeros(1)),
                           jnp.asarray(cp[1], dag), model.E0)
    consts = (model.rtab_coefs, model.table_cutoffs(), i32(runs), i32(groups), i32(factor),
              jnp.asarray(ct, dag), model.E0)
    if not gpu:
        return attrs, consts
    gform = FORM or form or "auto"
    if gform in ("coupled", "auto"):
        try:
            if choose_form(model, gform, len(factor))[0] == "coupled":
                ints, reals, T = coupled_gpu_plan(model, COUPLED_THREADS, real=jnp.dtype(dag).itemsize)   # the DAG's operands empty
                if (2 * int(ints[1]) + int(ints[2]) + int(ints[3])) * jnp.dtype(dag).itemsize > COUPLED_SHARED:
                    raise coupled.NoPlan("the coupled form's z does not fit a GPU block's shared memory")
                return ({**attrs, "n_leaf": np.int64(0), "n_st": np.int64(ints[2]), "tile": np.int64(ints[3]),
                         "lv": np.int64(0), "threads": np.int64(T), "form": np.int64(1)},   # n_st, tile: sizes
                        (model.rtab_coefs, model.table_cutoffs(), i32(runs), i32(np.zeros((1, 2))),
                         i32(np.zeros(0)), jnp.asarray(reals[:, :ints[1]], dag), model.E0, i32(np.zeros(1)),
                         i32(ints), jnp.asarray(reals, dag)) + radial)
        except coupled.NoPlan:
            if gform == "coupled":
                raise
    starts, n_a = np.r_[0, groups[:, 1]], ct.shape[1] - len(factor)
    n_st = starts[lev[-3]] if len(lev) > 2 else 0
    parent = np.repeat(groups[:, 0], np.diff(starts))
    T = pull_threads(len(factor))
    pull, pos, tile, flat = pull_plan(n_a, n_st, n_leaf, starts, lev, parent, factor,
                                      flat=PULL_LV == 0 or PULL_LV is None and len(lev) <= 4, T=T)
    lv = 0 if flat else PULL_LV or 1
    lc = np.zeros((len(ct), pos.max() + 1 if n_leaf else 0)); lc[:, pos] = np.tile(ct[:, ct.shape[1] - n_leaf:], 2)
    return ({**attrs, "n_st": np.int64(n_st), "tile": np.int64(tile), "lv": np.int64(lv),
             "threads": np.int64(T), "form": np.int64(0)},
            consts + (i32(lev), i32(pull), jnp.asarray(lc, dag)) + radial)


def describe(attrs):
    """{"form": "coupled" | "dag", "radial": "analytic" | "table"} of `kernel_inputs`' attributes (a dict or
    (name, value) pairs): what the kernel runs (the CPU kernel's radial is always the table)."""
    a = dict(attrs)
    return {"form": "coupled" if int(a.get("form", 0)) else "dag", "radial": "analytic" if int(a.get("nq", 0)) else "table"}


PULL_TILE = 2048                                      # leaf parents per tile
PULL_THREADS = None                                   # threads per block of the DAG pass: None by size
PULL_LV = None                                        # the kept values: None by depth and size (tests: 1 or 2)


_CANDIDATES = 48                                      # a lane's choices in the bank orders


class _Banks:
    """The shared-memory banks one load takes in a warp. The kernel loads a batch's values of an index, 16
    bytes side by side (4 float32 or 2 float64 rows), at once; a warp's load is served a word per bank (32
    banks of 4 bytes) at a time, lanes that read one index together. So the distinct indices should spread
    over the residues mod 8 (mod 16 for a batch of 8 bytes). `cost`: the conflicts each candidate adds to the
    lanes placed so far."""

    def __init__(self, key):
        self.key, self.seen = key, np.zeros(int(key.max()) + 1 if len(key) else 1, bool)
        self.k8, self.k16, self.o8, self.o16 = key % 8, key % 16, np.zeros(8), np.zeros(16)

    def clear(self):
        self.seen[:] = False; self.o8[:] = 0; self.o16[:] = 0

    def cost(self, cand):
        return np.where(self.seen[self.key[cand]], 0, self.o8[self.k8[cand]] + 0.25 * self.o16[self.k16[cand]])

    def take(self, i):
        k = self.key[i]
        if not self.seen[k]:
            self.seen[k] = True; self.o8[self.k8[i]] += 1; self.o16[self.k16[i]] += 1


def _spread(cut, last, keys, T):
    """src: an order of a piece list's items (range r's item j runs in lane r at step j) that permutes each
    piece (it ends where `last` is set) so that the indices `keys` (one array per load) of a warp's lanes
    fall in different banks, greedily, lane by lane."""
    n = len(last); src = np.arange(n)
    if not n:
        return src
    end = np.flatnonzero(last) + 1; start = np.r_[0, end[:-1]]
    rid = np.repeat(np.arange(len(end)), end - start)
    free = np.ones(n, bool); banks = [_Banks(k) for k in keys]
    for w in range(0, T, 32):
        for s in range(int(np.diff(cut[w:w + 33]).max())):
            for o in banks:
                o.clear()
            for r in range(w, w + 32):
                pos = cut[r] + s
                if pos >= cut[r + 1]:
                    continue
                a = start[rid[pos]]
                cand = a + np.flatnonzero(free[a:end[rid[pos]]])[:_CANDIDATES]
                c = cand[np.argmin(sum(o.cost(cand) for o in banks))] if len(cand) > 1 else cand[0]
                free[c] = False; src[pos] = c
                for o in banks:
                    o.take(c)
    return src


def _deal(g, f, T):
    """(count, place, last): the leaves of a tile, sorted by parent g (the parent's place in the tile), dealt
    to T lanes a parent at a time: each lane's count, each leaf's place (lane r's step j at T j + r) and
    whether it is its parent's last there. A lane takes the parent and leaf whose factor f, and at a
    parent's last leaf the parent, fall in the banks the warp's other lanes leave free (`_bank_cost`)."""
    n = len(g); place = np.zeros(n, np.int64); last = np.zeros(n, bool); count = np.zeros(T, np.int64)
    if not n:
        return count, place, last
    b = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, n]
    gi = np.repeat(np.arange(len(b) - 1), np.diff(b))
    left = np.diff(b).copy()
    free = np.ones(n, bool); pool = np.ones(n, bool)        # leaves to place; those of parents not yet begun
    cur = np.full(T, -1); done = 0
    of, og, size = _Banks(f), _Banks(g), np.diff(b)[gi]
    full = -(-n // T)                                       # a lane's share: it begins no parent that passes it
    while done < n:
        before = done
        for w in range(0, T, 32):
            of.clear(); og.clear()
            pi = np.flatnonzero(pool)[:32 * _CANDIDATES]
            for r in range(w, w + 32):
                if cur[r] >= 0:
                    a = b[cur[r]]; cand = a + np.flatnonzero(free[a:b[cur[r] + 1]])
                else:
                    cand = pi[pool[pi]]
                    cand = cand[size[cand] <= full - count[r]][:_CANDIDATES]
                    if not len(cand):
                        continue
                c = cand[np.argmin(of.cost(cand) + (left[gi[cand]] == 1) * og.cost(cand) - size[cand] / (4.0 * n))]
                q = gi[c]
                if cur[r] < 0:
                    pool[b[q]:b[q + 1]] = False
                free[c] = False; left[q] -= 1; done += 1
                place[c] = T * count[r] + r; count[r] += 1
                of.take(c)
                if left[q] == 0:
                    last[c] = True; cur[r] = -1; og.take(c)
                else:
                    cur[r] = q
        full += done == before                              # no parent fits a lane: a longer share
    return count, place, last


def pull_threads(n_nodes):
    """Threads per block of the GPU kernel's DAG pass for a DAG of n_nodes: the rows in flight per SM go as
    1 / threads (registers bound them) until shared memory does, so a small DAG takes few (A100, 32,768 Si,
    the pass in ms, float64, at 32, 64, 128, 256 threads: 76 nodes 0.09, 0.12, 0.20, 0.37; 1024: 0.55, 0.54,
    0.65, 0.86; 9280: 6.1, 3.9, 3.1, 3.3; 68,335: 49, 29, 20, 18)."""
    return PULL_THREADS or (32 if n_nodes < 1000 else 64 if n_nodes < 4000 else 128 if n_nodes < 24000 else 256)


def pull_plan(n_a, n_st, n_leaf, starts, lev, parent, factor, flat, T=256):
    """(pull, pos, tile, flat): the GPU kernel's adjoint pass for blocks of T threads (`pull_threads`), in
    which each thread sums into targets it alone owns (no atomic adds), from lists stored rank by range (range
    r's item j at T j + r: a warp's loads are neighbours).

    Tiles: the leaves' parents (the mids, or A without them) in runs of whole parent groups, PULL_TILE at
    most (or n_a), whose adjoints wms fit shared memory. Per tile, the leaves sorted by parent, as (factor,
    parent in the tile, ~ on a parent's last leaf), ranges ending with a parent.

    Piece lists, one per level then one per tile's mids: the nodes sorted by factor in equal ranges; a node
    is (x, y), its term's adjoint and its parent's value: a leaf's coefficient and Lv[x] Lv[y] (a mid's
    value, recomputed), a kept node's wv[x] Lv[y], a mid's wms[x] Lv[y]; y is ~y on the last node of a
    piece, a run of one factor in one range, whose sum goes to the next piece slot. `flat` (at most one
    kept level, n_lev <= 3): the kernel keeps no Lv, and a value is A[v & 65535] A[v >> 16] (`pair`: an A
    entry times the 1 at A[n_a], or a kept node's two factors), so a leaf is (pair, y) and a mid (x, pair).

    pull = header (tiles, leaf slots, lists, and the offsets of the arrays after it), tiles (u0, size, leaf
    item base, first and end group), lcut (each tile's ranges), the tiles' leaf items, then per list ib (its
    first item), ist (each range's first node), pst (its first piece) and fst (each factor's first piece),
    and the piece items. pos: each leaf's two coefficient slots, by parent then by factor; tile: the largest
    tile."""
    n_lev = len(lev) - 1
    if n_a >= 32768:
        raise ValueError(f"the GPU kernel takes up to 32,767 A entries, got {n_a}")
    mids, nL, n_nodes = n_lev > 1, n_a + n_st + 1, len(factor)
    if flat and n_lev > 3:
        raise ValueError("flat: one kept level at most")

    def pair(v):                                            # node v's value as two A entries
        k = np.maximum(v - n_a, 0)
        return np.where(v < n_a, v, parent[k]), np.where(v < n_a, n_a, factor[k])

    leaf = np.arange(n_nodes - n_leaf, n_nodes)
    u = parent[leaf] - (n_a + n_st if mids else 0)          # each leaf's parent, in the tiles' numbering
    if mids:                                                # tiles of whole groups of the mid level
        size = np.diff(starts)[lev[-3]:lev[-2]]
        tile = np.zeros(len(size), np.int64)
        fill = 0
        for g, c in enumerate(size):
            new = g and fill + c > max(PULL_TILE, n_a)
            fill, tile[g] = (c, tile[g - 1] + 1) if new else (fill + c, tile[g - 1] if g else 0)
        g0 = lev[-3] + np.flatnonzero(np.r_[True, tile[1:] != tile[:-1]])
        g1 = np.r_[g0[1:], lev[-2]]
        u0 = starts[g0] - n_st; n_u = starts[g1] - n_st - u0
    else:
        g0 = g1 = u0 = np.zeros(1, np.int64); n_u = np.array([n_a])
    if n_u.max() >= 65536:
        raise ValueError(f"the GPU kernel takes up to 65,535 children of a node, got {n_u.max()}")
    flat = bool(flat and n_a < 1024 and n_u.max() <= 2048 and nL < 1 << 21)   # an item's fields fit 32 bits
    word = lambda v, end: np.where(end, v - (1 << 31), v)   # an item: the top bit ends a parent or a piece
    by_u = np.argsort(u, kind="stable")
    tiles, lcut, litems = [], [], []
    n_l = 0
    pos = np.zeros((2, n_leaf), np.int64)

    for a, n, ga, gb in zip(u0, n_u, g0, g1):
        sel = by_u[np.searchsorted(u[by_u], a):np.searchsorted(u[by_u], a + n)]
        us = u[sel]
        count, place, end = _deal(us - a, factor[leaf[sel]], T)
        cut = np.r_[0, np.cumsum(count)]
        it = np.zeros(T * int(count.max()), np.int64)
        it[place] = word(factor[leaf[sel]] | (us - a) << 15, end)
        pos[0, sel] = n_l + place
        tiles.append([a, n, n_l, ga, gb]); lcut.append(cut); litems.append(it.ravel()); n_l += len(it)
    ib, ist, pst, fst, items = [], [], [], [], []
    n_items = n_pieces = 0

    def pieces(f, x, y, keys, packed=None):                 # a list; returns each node's place
        nonlocal n_items, n_pieces
        cut = (np.arange(T + 1) * len(f)) // T
        last = np.zeros(len(f), bool)
        if len(f):
            last[:-1] = f[1:] != f[:-1]; last[-1] = True; last[cut[1:-1][cut[1:-1] > 0] - 1] = True
        r = np.searchsorted(cut, np.arange(len(f)), side="right") - 1
        at = T * (np.arange(len(f)) - cut[r]) + r
        src = _spread(cut, last, keys, T)                     # each piece's nodes, in the order of the banks
        place = np.empty(len(f), np.int64); place[src] = at
        if flat:
            it = np.zeros(T * -(-len(f) // T), np.int64); it[at] = word(packed[src], last)
        else:
            it = np.zeros((T * -(-len(f) // T), 2), np.int64)
            it[at] = np.stack([x[src], np.where(last, ~y[src], y[src])], 1)
        ib.append([n_items]); ist.append(cut); pst.append(n_pieces + np.r_[0, np.cumsum(last)][cut])
        fst.append(n_pieces + np.searchsorted(f[last], np.arange(n_a + 1))); items.append(it.ravel())
        n_items += it.size; n_pieces += int(last.sum())
        return place

    for l in range(n_lev):
        k = np.arange(starts[lev[l]], starts[lev[l + 1]])
        if mids and l == n_lev - 2:
            k = k[:0]                                       # the mids are in the tiles' lists
        o = np.argsort(factor[k], kind="stable"); k = k[o]
        p = parent[k]
        if l == n_lev - 1:
            x, y = (parent[p - n_a], factor[p - n_a]) if mids else (p, np.full(len(p), nL - 1))
            if flat:
                x0, x1 = pair(x); y = np.where(y == nL - 1, n_a, y)
                pos[1, o] = n_l + pieces(factor[k], x, y, [x0, x1, y], x0 | x1 << 10 | y << 20)
            else:
                pos[1, o] = n_l + pieces(factor[k], x, y, [x, y])
        else:
            pieces(factor[k], n_a + k, p, [p], p | (n_a + k) << 10)
    for a, n in zip(u0, n_u) if mids else ():
        k = n_st + a + np.arange(n)
        o = np.argsort(factor[k], kind="stable")
        if flat:
            y0, y1 = pair(parent[k][o])
            pieces(factor[k][o], o, y0, [o, y0, y1], y0 | y1 << 10 | o << 20)
        else:
            pieces(factor[k][o], o, parent[k][o], [o, parent[k][o]])
    cat = lambda a: np.concatenate([np.ravel(v) for v in a]) if a else np.zeros(0, np.int64)
    parts = [cat(a) for a in (tiles, lcut, litems, ib, ist, pst, fst, items)]
    offs = 3 + len(parts) + np.cumsum([0] + [len(a) for a in parts])
    return (np.concatenate([[len(tiles), n_l, len(ib)], offs[:len(parts)]] + parts).astype(np.int64), pos.ravel(),
            int(n_u.max()), flat)


def lammps_fns(model, n_species, type_map, rcut, gpu=False, precision=None):
    """(energy_fn, force_fn) of lammps-jax's direct force path: one kernel call on the (K, rows) neighbour
    matrix, on the GPU kernel if `gpu`. type_map[t] is the model species of LAMMPS type t+1.  energy_fn has
    a custom VJP in positions (a second kernel call with the cotangent as site weights), so jax.grad /
    jax.vjp of any function of the atom energies is exact; nothing is differentiable in the model, and
    there is no second derivative. precision: `kernel_inputs`'."""
    attrs, consts = kernel_inputs(model, gpu, precision)
    tmap, target = jnp.asarray(type_map, jnp.int32), TARGET_GPU if gpu else TARGET

    def call(positions, species, graph, weights=None):
        z = tmap[jnp.clip(species, 0, n_species - 1)]
        w = jnp.ones(z.shape, positions.dtype) if weights is None else jnp.asarray(weights, positions.dtype)
        out = (jax.ShapeDtypeStruct(z.shape, positions.dtype),
               jax.ShapeDtypeStruct(positions.shape, positions.dtype))
        return jax.ffi.ffi_call(target, out)(positions, z, graph.neighbors, graph.num_neighbors, w, *consts,
                                             **attrs, rc2=np.float64(rcut * rcut))

    @jax.custom_vjp
    def energy_fn(positions, species, graph):
        return call(positions, species, graph)[0]

    def fwd(positions, species, graph):
        return energy_fn(positions, species, graph), (positions, species, graph)

    def bwd(res, ct):                                      # sum_i ct_i grad e_i = -kernel force at w = ct
        positions, species, graph = res
        return -call(positions, species, graph, ct)[1], None, None

    energy_fn.defvjp(fwd, bwd)
    force_fn = lambda *a: call(*a)[1]
    force_fn.kernel = describe(attrs)                      # what the kernel runs (export_lammps records it)
    return energy_fn, force_fn


@functools.lru_cache(maxsize=64)
def _edge_kernel(attrs):
    """The AceEdges call for these attributes (AceEdgesGpu when they name its target), with a batching rule:
    rows are independent, so under jax.vmap the batch is flattened into rows -- (B, rows, K, 3) -> (B rows,
    K, 3), ONE kernel call threaded over all of them (one launch on a GPU) -- and the outputs reshaped back.
    Exact (the same per-row arithmetic).  Unbatched row inputs are broadcast to the batch; batched model
    constants (a different model per batch element) fall back to one call per element (lax.map)."""
    target = dict(attrs).get("target", EDGE_TARGET)
    kw = {k: np.float64(v) if isinstance(v, float) else np.int64(v) for k, v in attrs if k != "target"}

    @jax.custom_batching.custom_vmap
    def call(rij, node_z, zj, mask, *consts):
        out = (jax.ShapeDtypeStruct(node_z.shape, rij.dtype), jax.ShapeDtypeStruct(rij.shape, rij.dtype))
        return jax.ffi.ffi_call(target, out)(rij, node_z, zj, mask, *consts, **kw)

    @call.def_vmap
    def rule(axis_size, in_batched, rij, node_z, zj, mask, *consts):
        if any(in_batched[4:]):
            rows = [a if b else jnp.broadcast_to(a, (axis_size,) + a.shape)
                    for a, b in zip((rij, node_z, zj, mask, *consts), in_batched)]
            return jax.lax.map(lambda xs: call(*xs), tuple(rows)), (True, True)
        rij, node_z, zj, mask = (a if b else jnp.broadcast_to(a, (axis_size,) + a.shape)
                                 for a, b in zip((rij, node_z, zj, mask), in_batched[:4]))
        lead = rij.shape[:2]
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        e, g = call(flat(rij), flat(node_z), flat(zj), flat(mask), *consts)
        return (e.reshape(lead), g.reshape(lead + g.shape[1:])), (True, True)

    return call


def _edge_call(attrs, rij, node_z, zj, mask, consts):
    """AceEdges: site energies (rows,) and dE_i / drij (rows, K, 3); vmappable (`_edge_kernel`)."""
    return _edge_kernel(attrs)(rij, node_z.astype(jnp.int32), zj.astype(jnp.int32), mask.astype(bool), *consts)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def edge_site_energies(attrs, rij, node_z, zj, mask, *consts):
    """Site energies of the dense edge input, by the kernel, differentiable in rij only: one call gives the
    energies and G = dE_i / drij, and the VJP of a cotangent c is c_i G_i (no second call).  attrs: the
    kernel's attributes as a hashable tuple of (name, value); consts: `kernel_inputs`' arrays (passed, not
    closed over, so new weights of a jitted caller do not retrace)."""
    return _edge_call(attrs, rij, node_z, zj, mask, consts)[0]


def _edge_fwd(attrs, rij, node_z, zj, mask, *consts):
    e, g = _edge_call(attrs, rij, node_z, zj, mask, consts)
    return e, g


def _edge_bwd(attrs, g, ct):
    gpu = dict(attrs).get("target") == EDGE_TARGET_GPU
    return (ct[:, None, None] * g,) + (None,) * 3 + (None,) * (N_CONSTS_GPU if gpu else N_CONSTS)


N_CONSTS, N_CONSTS_GPU = 7, 12                        # `kernel_inputs`' arrays (gpu=False, True)
edge_site_energies.defvjp(_edge_fwd, _edge_bwd)


class KernelModel(EdgeSiteModel):
    """A lean ACEModel with a radial table, evaluated by the kernel's `AceEdges` entry point (float64), or on
    a GPU by `AceEdgesGpu` (float64 or float32): an evaluation model for `energy_forces_virial_dense`
    (ACECalculator's dense layout and skin step, `eval.dense`), whose forces and virial JAX assembles from
    dE/drij as for the model itself.  Energy only, in positions only."""
    base: object                                      # the lean ACEModel (pad_cutoff, provenance)
    consts: tuple                                     # `kernel_inputs`' arrays
    attrs: tuple = eqx.field(static=True)             # the kernel's attributes, (name, value) pairs
    uses_edge_a = False                               # no edge_a form to calibrate (the sparse layout)
    energy_only = True

    @classmethod
    def of(cls, model, max_threads=0, gpu=False, precision=None):
        """`model` (a lean ACEModel with a radial table, float64, or float32 on a GPU) for the kernel, the
        GPU one if `gpu`; max_threads > 0 caps the CPU kernel's threads (0: XLA's intra-op pool);
        precision: `kernel_inputs`'."""
        attrs, consts = kernel_inputs(model, gpu, precision)
        attrs = tuple(sorted((k, float(v) if k in ("x0", "h") else int(v)) for k, v in attrs.items()))
        extra = (("target", EDGE_TARGET_GPU),) if gpu else (("max_threads", int(max_threads)),)
        return cls(model, tuple(consts), attrs + extra)

    def pad_cutoff(self):
        return self.base.pad_cutoff()

    @property
    def elements(self):
        return self.base.elements

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        return edge_site_energies(self.attrs, rij, node_z, zj, mask, *self.consts)

    def site_energies_dense_blocked(self, rij, zi, zj, mask, node_z, chunk):
        return self.site_energies_dense(rij, zi, zj, mask, node_z)   # one call: no per-block memory to bound
