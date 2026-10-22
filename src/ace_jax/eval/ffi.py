"""Lean linear ACE energies and forces on the CPU as one C++ XLA FFI call on lammps-jax's neighbour matrix.

`lammps_fns` gives lammps-jax's direct force path for a lean float64 ACEModel with a radial table:
positions in, atom energies and forces out. LAMMPS loads the kernel with
LAMMPS_JAX_FFI_HANDLERS="ace_jax_direct=<lib>:AceDirect" (`handlers`).

Distribution: the kernel ships as source in the (pure Python) wheel and `library` compiles it on first
use with the user's C++17 compiler against the installed jaxlib's FFI headers, into a content-keyed cache;
no compiler raises `FFIUnavailable`.  Anyone running it already builds LAMMPS and the lammps-jax plugin
from source, so a compiler is there; a prebuilt wheel would trade -march=native for a generic baseline.
"""
import ctypes
import hashlib
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from .aa_dag import build_plan

TARGET = "ace_jax_direct"
SYMBOL = "AceDirect"
SOURCE = pathlib.Path(__file__).with_name("ace_ffi.cc")
DEFAULT_FLAGS = ("-O3", "-march=native")
REGISTERED = set()


class FFIUnavailable(RuntimeError):
    """The kernel cannot be compiled here: no C++17 compiler, or the compile failed."""


def build(path, cxx="c++", flags=("-O3",)):
    """Compile the kernel to the shared library `path`."""
    path = pathlib.Path(path)
    subprocess.run([cxx, *flags, "-std=c++17", "-shared", "-fPIC", "-isystem", jax.ffi.include_dir(),
                    str(SOURCE), "-o", str(path)], check=True)
    return path


def compiler(env=None):
    """The C++ compiler: $CXX, else the first of c++, g++, clang++ on PATH; None if there is none."""
    env = os.environ if env is None else env
    for c in ([env["CXX"]] if env.get("CXX") else []) + ["c++", "g++", "clang++"]:
        if (w := shutil.which(c)):
            return w
    return None


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


def library(flags=None, cxx=None, cache=None):
    """The compiled kernel's path, built on first use: keyed on the source, the jaxlib version (its FFI headers),
    the compiler and its version, the flags, the machine and, under -march=native, the CPU model.  flags:
    `ACEJAX_FFI_FLAGS` (space-separated), else DEFAULT_FLAGS; pass e.g. ("-O3", "-march=x86-64-v3") to build on a
    login node for compute nodes of another CPU type."""
    import jaxlib
    flags = tuple(flags if flags is not None else os.environ.get("ACEJAX_FFI_FLAGS", "").split() or DEFAULT_FLAGS)
    cxx = cxx or compiler()
    if cxx is None:
        raise FFIUnavailable("the ace-jax CPU kernel (ffi=True) is compiled on first use and needs a C++17 compiler; "
                             "none found (set CXX, or install g++ or clang++), or export without ffi")
    ver = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    native = any("native" in f for f in flags)
    key = hashlib.sha256("\0".join([SOURCE.read_text(), jaxlib.__version__, cxx, *ver, *flags, platform.machine(),
                                     _cpu() if native else ""]).encode()).hexdigest()[:16]
    d = pathlib.Path(cache) if cache else cache_dir()
    out = d / f"libace_jax_ffi-{key}.so"
    if out.exists():
        return out
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".so.tmp")      # same dir: os.replace is atomic, so concurrent
    os.close(fd)                                             # ranks or jobs never load a half-written library
    try:
        build(tmp, cxx, flags)
        os.replace(tmp, out)
    except subprocess.CalledProcessError as e:
        raise FFIUnavailable(f"compiling the ace-jax CPU kernel with {cxx} {' '.join(flags)} failed "
                             f"(exit {e.returncode}); see the compiler output above") from None
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)
    return out


def handlers(path):
    """The LAMMPS_JAX_FFI_HANDLERS value that lets LAMMPS load the kernel at `path`."""
    return f"{TARGET}={pathlib.Path(path).resolve()}:{SYMBOL}"


def register(path):
    """Register the compiled kernel for in-process CPU programs."""
    if TARGET not in REGISTERED:
        jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(getattr(ctypes.cdll.LoadLibrary(str(path)), SYMBOL)),
                                    platform="cpu")
        REGISTERED.add(TARGET)


def plan(model):
    """(runs, groups, factor, ct, n_leaf): A in runs of consecutive harmonics; the product basis as a flat DAG
    whose last n_leaf nodes are its deepest level. Each level holds a parent's children together, the largest
    groups first; groups (n_groups, 2) gives each group's parent and end."""
    ar, ay = np.asarray(model.aspec_r), np.asarray(model.aspec_y)
    n_a = len(ar)
    order = np.lexsort((ay, ar))
    new = np.empty(n_a, np.int64); new[order] = np.arange(n_a)
    levels, readout, _ = build_plan([np.asarray(s) for s in model.aa_specs], n_a, "greedy")
    pars, facs, invs = [], [], [new]                        # invs: old to new index, per level
    for p, a in levels:
        p = invs[-1][p]
        _, g, cnt = np.unique(p, return_inverse=True, return_counts=True)
        perm = np.lexsort((p, -cnt[g]))
        invs.append(np.empty(len(p), np.int64)); invs[-1][perm] = np.arange(len(p))
        pars.append(p[perm]); facs.append(new[a[perm]])
    offs = np.cumsum([0, n_a] + [len(p) for p in pars])
    parent = np.concatenate([np.zeros(0, np.int64)] + [offs[k] + p if k else p for k, p in enumerate(pars)])
    ends = np.flatnonzero(np.diff(parent, append=-1)) + 1
    factor = np.concatenate([np.zeros(0, np.int64)] + facs)
    ctilde = np.asarray(model.ctilde, np.float64)
    ct = np.zeros((offs[-1], ctilde.shape[1]))
    for k, (c, nd) in enumerate(readout):
        np.add.at(ct, offs[k] + invs[k][nd], ctilde[c])
    rs, ys = ar[order], ay[order]
    a0 = np.nonzero(np.r_[True, (rs[1:] != rs[:-1]) | (ys[1:] != ys[:-1] + 1)])[0]
    runs = np.stack([a0, rs[a0], ys[a0], np.diff(np.r_[a0, n_a])], axis=1)
    return (runs, np.stack([parent[ends - 1], ends], axis=1), factor, np.ascontiguousarray(ct.T),
            len(levels[-1][0]) if levels else 0)


def kernel_inputs(model):
    """(attrs, consts) of the kernel for a lean float64 ACEModel with a radial table."""
    from .model import ACEModel
    if not (isinstance(model, ACEModel) and model.energy_only and model.rtab_grid):
        raise ValueError("the FFI kernel needs a lean ACEModel with a radial table: "
                         "lean(model, radial_table=True)")
    if model.E0.dtype != jnp.float64:
        raise ValueError(f"the FFI kernel is float64 only, got {model.E0.dtype}")
    x0, h, n = model.rtab_grid
    runs, groups, factor, ct, n_leaf = plan(model)
    attrs = dict(x0=np.float64(x0), h=np.float64(h), n=np.int64(n), lmax=np.int64(model.lmax),
                 n_rnl=np.int64(model.rtab_coefs.shape[-1] - model.Wpair.shape[0]),
                 solid=np.int64(model.ysolid), n_leaf=np.int64(n_leaf))
    i32 = lambda a: jnp.asarray(a, jnp.int32)
    return attrs, (model.rtab_coefs, model.table_cutoffs(), i32(runs), i32(groups), i32(factor),
                   jnp.asarray(ct), model.E0)


def lammps_fns(model, n_species, type_map, rcut):
    """(energy_fn, force_fn) of lammps-jax's direct force path: one kernel call on the (K, rows) neighbour
    matrix. type_map[t] is the model species of LAMMPS type t+1.  energy_fn has a custom VJP in positions
    (a second kernel call with the cotangent as site weights), so jax.grad / jax.vjp of any function of the
    atom energies is exact; nothing is differentiable in the model, and there is no second derivative."""
    attrs, consts = kernel_inputs(model)
    tmap = jnp.asarray(type_map, jnp.int32)

    def call(positions, species, graph, weights=None):
        z = tmap[jnp.clip(species, 0, n_species - 1)]
        w = jnp.ones(z.shape, positions.dtype) if weights is None else jnp.asarray(weights, positions.dtype)
        out = (jax.ShapeDtypeStruct(z.shape, positions.dtype),
               jax.ShapeDtypeStruct(positions.shape, positions.dtype))
        return jax.ffi.ffi_call(TARGET, out)(positions, z, graph.neighbors, graph.num_neighbors, w, *consts,
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
    return energy_fn, (lambda *a: call(*a)[1])
