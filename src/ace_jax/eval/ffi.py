"""Lean linear ACE energies and forces on the CPU as one C++ XLA FFI call on lammps-jax's neighbour matrix.

`lammps_fns` gives lammps-jax's direct force path for a lean float64 ACEModel with a radial table:
positions in, atom energies and forces out. LAMMPS loads the kernel with
LAMMPS_JAX_FFI_HANDLERS="ace_jax_direct=<lib>:AceDirect".
"""
import ctypes
import pathlib
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from .aa_dag import build_plan

TARGET = "ace_jax_direct"
SOURCE = pathlib.Path(__file__).with_name("ace_ffi.cc")
REGISTERED = set()


def build(path, cxx="c++", flags=("-O3",)):
    """Compile the kernel to the shared library `path`."""
    path = pathlib.Path(path)
    subprocess.run([cxx, *flags, "-std=c++17", "-shared", "-fPIC", "-isystem", jax.ffi.include_dir(),
                    str(SOURCE), "-o", str(path)], check=True)
    return path


def register(path):
    """Register the compiled kernel for in-process CPU programs."""
    if TARGET not in REGISTERED:
        jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(ctypes.cdll.LoadLibrary(str(path)).AceDirect),
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
    matrix. type_map[t] is the model species of LAMMPS type t+1."""
    attrs, consts = kernel_inputs(model)
    tmap = jnp.asarray(type_map, jnp.int32)

    def call(positions, species, graph):
        z = tmap[jnp.clip(species, 0, n_species - 1)]
        out = (jax.ShapeDtypeStruct(z.shape, positions.dtype),
               jax.ShapeDtypeStruct(positions.shape, positions.dtype))
        return jax.ffi.ffi_call(TARGET, out)(positions, z, graph.neighbors, graph.num_neighbors, *consts,
                                             **attrs, rc2=np.float64(rcut * rcut))

    return (lambda *a: call(*a)[0]), (lambda *a: call(*a)[1])
