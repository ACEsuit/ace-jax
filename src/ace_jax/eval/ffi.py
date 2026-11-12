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

Distribution: the kernel ships as source in the (pure Python) wheel and `library` compiles it on first
use with the user's C++17 compiler against the installed jaxlib's FFI headers, into a content-keyed cache;
no compiler raises `FFIUnavailable`.  Anyone running it already builds LAMMPS and the lammps-jax plugin
from source, so a compiler is there; a prebuilt wheel would trade -march=native for a generic baseline.
The GPU kernel likewise, with nvcc (a CUDA toolkit's or module's, or the `cuda13` extra's: NVIDIA's CUDA 13 nvcc wheel), for the GPU
JAX sees.
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
EDGE_TARGET_GPU, EDGE_SYMBOL_GPU = "ace_jax_edges_gpu", "AceEdgesGpu"
GPU_SOURCE = pathlib.Path(__file__).with_name("ace_ffi_gpu.cu")
DEFAULT_FLAGS = ("-O3", "-march=native")
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
    return (runs, np.stack([parent[ends - 1], ends], axis=1), factor, np.ascontiguousarray(ct.T),
            len(levels[-1][0]) if levels else 0, np.asarray(lev))


def kernel_inputs(model, gpu=False):
    """(attrs, consts) of the kernel for a lean ACEModel with a radial table, float64, or float32 on the GPU;
    the GPU kernel also takes the DAG's levels, n_st, the nodes before the last two levels, which it keeps,
    `pull_plan`'s lists and the leaves' coefficients in their two orders."""
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
    pull, pos, tile, flat = pull_plan(n_a, n_st, n_leaf, starts, lev, parent, factor,
                                      flat=PULL_LV == 0 or PULL_LV is None and len(lev) <= 4)
    lv = 0 if flat else PULL_LV or 1
    lc = np.zeros((len(ct), pos.max() + 1 if n_leaf else 0)); lc[:, pos] = np.tile(ct[:, ct.shape[1] - n_leaf:], 2)
    return ({**attrs, "n_st": np.int64(n_st), "tile": np.int64(tile), "lv": np.int64(lv)},
            consts + (i32(lev), i32(pull), jnp.asarray(lc, model.E0.dtype)))


PULL_THREADS, PULL_TILE = 256, 2048                   # ace_ffi_gpu.cu's kThreads; leaf parents per tile
PULL_LV = None                                        # the kept values: None by depth and size (tests: 1 or 2)


def _ranked(n, edge, T=None):
    """(cut, place): n sorted items in T ranges that end where `edge` (n,) is set, as equal as that allows,
    and each item's place rank by range: range r's item j at T j + r."""
    T = T or PULL_THREADS
    b = np.r_[0, np.flatnonzero(edge) + 1] if n else np.zeros(1, np.int64)
    cut = b[np.searchsorted(b, (np.arange(T + 1) * n) // T)]
    r = np.searchsorted(cut, np.arange(n), side="right") - 1
    return cut, T * (np.arange(n) - cut[r]) + r


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


def _spread(cut, last, keys, T=None):
    """src: an order of a piece list's items (range r's item j runs in lane r at step j) that permutes each
    piece (it ends where `last` is set) so that the indices `keys` (one array per load) of a warp's lanes
    fall in different banks, greedily, lane by lane."""
    T = T or PULL_THREADS
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


def _deal(g, f, T=None):
    """(count, place, last): the leaves of a tile, sorted by parent g (the parent's place in the tile), dealt
    to T lanes a parent at a time: each lane's count, each leaf's place (lane r's step j at T j + r) and
    whether it is its parent's last there. A lane takes the parent and leaf whose factor f, and at a
    parent's last leaf the parent, fall in the banks the warp's other lanes leave free (`_bank_cost`)."""
    T = T or PULL_THREADS
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


def pull_plan(n_a, n_st, n_leaf, starts, lev, parent, factor, flat):
    """(pull, pos, tile): the GPU kernel's adjoint pass, in which each thread sums into targets it alone owns (no
    atomic adds), from lists stored rank by range (`_ranked`: a warp's loads are neighbours).

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
    T, n_lev = PULL_THREADS, len(lev) - 1
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
        count, place, end = _deal(us - a, factor[leaf[sel]])
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
        src = _spread(cut, last, keys)                      # each piece's nodes, in the order of the banks
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


N_CONSTS, N_CONSTS_GPU = 7, 10                        # `kernel_inputs`' arrays (gpu=False, True)
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
    def of(cls, model, max_threads=0, gpu=False):
        """`model` (a lean ACEModel with a radial table, float64, or float32 on a GPU) for the kernel, the
        GPU one if `gpu`; max_threads > 0 caps the CPU kernel's threads (0: XLA's intra-op pool)."""
        attrs, consts = kernel_inputs(model, gpu)
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
