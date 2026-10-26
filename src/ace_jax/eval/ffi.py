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
On a GPU, `AceDirectGpu` (`lammps_fns(..., gpu=True)`) is the direct force path, float64 or float32, loaded
with LAMMPS_JAX_FFI_HANDLERS="ace_jax_direct_gpu=<lib>:AceDirectGpu".

Distribution: the kernel ships as source in the (pure Python) wheel and `library` compiles it on first
use with the user's C++17 compiler against the installed jaxlib's FFI headers, into a content-keyed cache;
no compiler raises `FFIUnavailable`.  Anyone running it already builds LAMMPS and the lammps-jax plugin
from source, so a compiler is there; a prebuilt wheel would trade -march=native for a generic baseline.
The GPU kernel likewise, with nvcc, for the GPU JAX sees.
"""
import ctypes
import functools
import hashlib
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .aa_dag import build_plan
from .edge_model import EdgeSiteModel

TARGET = "ace_jax_direct"
SYMBOL = "AceDirect"
EDGE_TARGET, EDGE_SYMBOL = "ace_jax_edges", "AceEdges"
SOURCE = pathlib.Path(__file__).with_name("ace_ffi.cc")
TARGET_GPU = "ace_jax_direct_gpu"
SYMBOL_GPU = "AceDirectGpu"
GPU_SOURCE = pathlib.Path(__file__).with_name("ace_ffi_gpu.cu")
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


def cuda_compiler(env=None):
    """nvcc: $NVCC, else on PATH, else under $CUDA_HOME; None if there is none."""
    env = os.environ if env is None else env
    home = [os.path.join(env["CUDA_HOME"], "bin", "nvcc")] if env.get("CUDA_HOME") else []
    for c in ([env["NVCC"]] if env.get("NVCC") else []) + ["nvcc"] + home:
        if (w := shutil.which(c)):
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
        raise FFIUnavailable("the ace-jax GPU kernel is compiled on first use and needs nvcc: set NVCC or "
                             "CUDA_HOME, or export without ffi")
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
    builds it ahead of use."""
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
    """Register the compiled GPU kernel for in-process programs; platform "CUDA" or "ROCM"."""
    if (TARGET_GPU, platform) not in REGISTERED:
        handler = getattr(ctypes.cdll.LoadLibrary(str(path)), SYMBOL_GPU)
        jax.ffi.register_ffi_target(TARGET_GPU, jax.ffi.pycapsule(handler), platform=platform)
        REGISTERED.add((TARGET_GPU, platform))


def plan(model, gpu=False):
    """(runs, groups, factor, ct, n_leaf, levels): A in runs of consecutive harmonics; the product basis as a
    flat DAG whose last n_leaf nodes are its deepest level. Each level holds a parent's children together, the
    largest groups first; for the GPU kernel, groups of a size by their children's factors and children by
    factor, so a warp's lanes share dE/dA entries. groups (n_groups, 2) gives each group's parent and end,
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
        key = (p, -cnt[g])
        if gpu:                                             # each group's factors in a row, ascending
            o = np.lexsort((f, g))
            rows = np.full((len(cnt), cnt.max()), -1)
            rows[g[o], np.arange(len(o)) - (np.cumsum(cnt) - cnt)[g[o]]] = f[o]
            tie = np.empty(len(cnt), np.int64); tie[np.lexsort(rows.T[::-1])] = np.arange(len(cnt))
            key = (f, p, tie[g], -cnt[g])
        perm = np.lexsort(key)
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
    return (runs, np.stack([parent[ends - 1], ends], axis=1), factor, np.ascontiguousarray(ct.T),
            len(levels[-1][0]) if levels else 0, np.asarray(lev))


def kernel_inputs(model, gpu=False):
    """(attrs, consts) of the kernel for a lean ACEModel with a radial table, float64, or float32 on the GPU;
    the GPU kernel also takes the DAG's levels, each node's parent, n_st, the nodes before the last two
    levels, which it keeps, the mids without leaves, and the leaves copied rank by lane for chunks of 32
    groups."""
    from .model import ACEModel
    if not (isinstance(model, ACEModel) and model.energy_only and model.rtab_grid):
        raise ValueError("the FFI kernel needs a lean ACEModel with a radial table: "
                         "lean(model, radial_table=True)")
    if model.E0.dtype != jnp.float64 and not (gpu and model.E0.dtype == jnp.float32):
        raise ValueError(f"the FFI kernel takes float64, or float32 on the GPU, got {model.E0.dtype}")
    x0, h, n = model.rtab_grid
    runs, groups, factor, ct, n_leaf, lev = plan(model, gpu)
    attrs = dict(x0=np.float64(x0), h=np.float64(h), n=np.int64(n), lmax=np.int64(model.lmax),
                 n_rnl=np.int64(model.rtab_coefs.shape[-1] - model.Wpair.shape[0]),
                 solid=np.int64(model.ysolid), n_leaf=np.int64(n_leaf))
    i32 = lambda a: jnp.asarray(a, jnp.int32)
    consts = (model.rtab_coefs, model.table_cutoffs(), i32(runs), i32(groups), i32(factor),
              jnp.asarray(ct, model.E0.dtype), model.E0)
    if not gpu:
        return attrs, consts
    starts, n_a = np.r_[0, groups[:, 1]], ct.shape[1] - len(factor)
    n_st = starts[lev[-3]] if len(lev) > 2 else 0
    parent = np.repeat(groups[:, 0], np.diff(starts))
    kids = np.bincount(parent[parent >= n_a] - n_a, minlength=len(factor))
    lone = n_st + np.flatnonzero(kids[n_st:len(factor) - n_leaf] == 0)
    size = np.diff(starts)[lev[-2]:] if n_leaf else np.zeros(0, int)   # the leaf groups, largest first
    most = size[::32]; first = 32 * (np.cumsum(most) - most)
    g = np.repeat(np.arange(len(size)), size)
    slot = first[g // 32] + 32 * (np.arange(n_leaf) - np.repeat(np.cumsum(size) - size, size)) + g % 32
    lf = np.full(32 * most.sum(), -1); lf[slot] = factor[len(factor) - n_leaf:]
    lc = np.zeros((len(ct), len(lf))); lc[:, slot] = ct[:, ct.shape[1] - n_leaf:]
    return ({**attrs, "n_st": np.int64(n_st)},
            consts + (i32(lev), i32(parent), i32(lone), i32(np.stack([first, most], 1)), i32(lf),
                      jnp.asarray(lc, model.E0.dtype)))


def lammps_fns(model, n_species, type_map, rcut, gpu=False):
    """(energy_fn, force_fn) of lammps-jax's direct force path: one kernel call on the (K, rows) neighbour
    matrix, on the GPU kernel if `gpu`. type_map[t] is the model species of LAMMPS type t+1.  energy_fn has
    a custom VJP in positions (a second kernel call with the cotangent as site weights), so jax.grad /
    jax.vjp of any function of the atom energies is exact; nothing is differentiable in the model, and
    there is no second derivative."""
    attrs, consts = kernel_inputs(model, gpu)
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
    return energy_fn, (lambda *a: call(*a)[1])


@functools.lru_cache(maxsize=64)
def _edge_kernel(attrs):
    """The AceEdges call for these attributes, with a batching rule: rows are independent, so under
    jax.vmap the batch is flattened into rows -- (B, rows, K, 3) -> (B rows, K, 3), ONE kernel call threaded
    over all of them -- and the outputs reshaped back.  Exact (the same per-row arithmetic).  Unbatched row
    inputs are broadcast to the batch; batched model constants (a different model per batch element) fall
    back to one call per element (lax.map)."""
    kw = {k: np.float64(v) if isinstance(v, float) else np.int64(v) for k, v in attrs}

    @jax.custom_batching.custom_vmap
    def call(rij, node_z, zj, mask, *consts):
        out = (jax.ShapeDtypeStruct(node_z.shape, rij.dtype), jax.ShapeDtypeStruct(rij.shape, rij.dtype))
        return jax.ffi.ffi_call(EDGE_TARGET, out)(rij, node_z, zj, mask, *consts, **kw)

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
    return (ct[:, None, None] * g,) + (None,) * 3 + (None,) * N_CONSTS


N_CONSTS = 7                                          # `kernel_inputs`' arrays
edge_site_energies.defvjp(_edge_fwd, _edge_bwd)


class KernelModel(EdgeSiteModel):
    """A lean float64 ACEModel with a radial table, evaluated by the kernel's `AceEdges` entry point: an
    evaluation model for `energy_forces_virial_dense` (ACECalculator's dense layout and skin step), whose
    forces and virial JAX assembles from dE/drij as for the model itself.  Energy only, in positions only."""
    base: object                                      # the lean ACEModel (pad_cutoff, provenance)
    consts: tuple                                     # `kernel_inputs`' arrays
    attrs: tuple = eqx.field(static=True)             # the kernel's attributes, (name, value) pairs
    uses_edge_a = False                               # no edge_a form to calibrate (the sparse layout)
    energy_only = True

    @classmethod
    def of(cls, model, max_threads=0):
        """`model` (a lean ACEModel with a radial table, float64) for the kernel; max_threads > 0 caps
        the kernel's threads (0: XLA's intra-op pool)."""
        attrs, consts = kernel_inputs(model)
        attrs = tuple(sorted((k, float(v) if k in ("x0", "h") else int(v)) for k, v in attrs.items()))
        return cls(model, tuple(consts), attrs + (("max_threads", int(max_threads)),))

    def pad_cutoff(self):
        return self.base.pad_cutoff()

    @property
    def elements(self):
        return self.base.elements

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        return edge_site_energies(self.attrs, rij, node_z, zj, mask, *self.consts)

    def site_energies_dense_blocked(self, rij, zi, zj, mask, node_z, chunk):
        return self.site_energies_dense(rij, zi, zj, mask, node_z)   # one call: no per-block memory to bound
