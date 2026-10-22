"""cuequivariance (NVIDIA cueq) as a GPU form of the lean ACE product basis (`aa_dag="cueq"`).

Design: docs/dev/specs/2026-10-22-cueq-product-basis-design.md.  The folded site energy
e_i = sum_v ctilde[v, z_i] prod_t A_i[s_v,t] is a cueq SegmentedPolynomial: one SegmentedTensorProduct per order
k ([A]*k -> out), one path per monomial, evaluated by cuequivariance_jax.segmented_polynomial(method="uniform_1d")
with atoms in the u mode (U atoms per group: the obvious batch-axis layout is 6x slower) and the polynomial split
into kernels of at most CHUNK monomials (NVRTC compile grows superlinearly with paths).  cueq is optional:
nothing here imports it at module level."""
CHUNK = 1000
U = 32
TARGET = "uniform_1d_cuda"
# the bundle's dense / matrix rows per block (`export.lammps.cueq_block_rows_for`): the first that fits the memory
# budget, then halvings to MIN_BLOCK_ROWS.  cueq wants large blocks: at the DAG's choice (1024 rows at o4d20) it is
# no faster than the DAG, at 8192 1.95x the best DAG (Sulis A100); 8192 ran out of an RTX 4000 Ada's 20 GB at o4d20
CUEQ_BLOCK_ROWS = (8192, 4096)
# the uniform_1d plan compile (recorded at export) and execute symbols of libcue_ops_jax, as lammps-jax's
# contrib/ffi-replay shim replays them
COMPILE, EXECUTE = "compile_uniform_1d_ctypes", "execute_uniform_1d_cuda_handler"
SHIM = "ffi_replay_shim.so"                   # built next to the recorder (contrib/ffi-replay/README.md)


class CueqUnavailable(ImportError):
    pass


def available():
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_jax  # noqa: F401
    except (ImportError, OSError):  # OSError: the ops wheel is installed but libnvrtc/libcue_ops is not on LD_LIBRARY_PATH
        return False
    return True


def require():
    if not available():
        raise CueqUnavailable("aa_dag='cueq' needs cuequivariance: pip install 'ace-jax[cueq]' (Linux, CUDA 12)")


class _Static:
    """Hash-by-identity wrapper so a polynomial can sit in a static eqx field."""
    def __init__(self, v):
        self.v = v

    def __hash__(self):
        return id(self)

    def __eq__(self, o):
        return self is o


def check_supported(model, folded=True):
    """Refuse a model aa_dag="cueq" cannot evaluate.  folded=False: also accept the unfolded model (`lean` folds
    it), counting its species from WB."""
    from .model import ACEModel
    if not isinstance(model, ACEModel):
        raise ValueError(f"aa_dag='cueq' needs a linear ACEModel, got {type(model).__name__}")
    if folded and not model.folded:
        raise ValueError("aa_dag='cueq' needs the folded (lean) model: apply lean() first")
    if getattr(model, "rnl_embedding", None) is not None:  # TODO(Task 8): remove with TrACE support
        raise ValueError("aa_dag='cueq' does not support species-embedding (TrACE) models yet: TrACE support: not yet")
    nz = (model.ctilde if model.folded else model.WB).shape[1]
    if nz != 1:
        raise ValueError("aa_dag='cueq' supports one species, or species-embedding (TrACE) models; "
                         f"this categorical model has {nz} species")


def n_A_of(specs):
    import numpy as np
    return 1 + max(int(np.asarray(s).max()) for s in specs if np.size(s))


def _build_poly(specs, n_A, coef, u):
    """SegmentedPolynomial for sum_v coef[v] prod A[s_v], coefficients baked into the paths.  `specs`: per-order
    (n_v, k) index arrays into A, concatenated in ctilde-row order."""
    import cuequivariance as cue
    import numpy as np
    specs = [np.asarray(s) for s in specs]
    A_op = cue.SegmentedOperand(ndim=1, segments=[(u,)] * n_A)
    out_op = cue.SegmentedOperand(ndim=1, segments=[(u,)])
    ops, off = [], 0
    for s in specs:
        if s.size == 0:
            continue
        k = s.shape[1]
        d = cue.SegmentedTensorProduct.from_subscripts(",".join(["u"] * (k + 1)))
        for o in range(k):
            for _ in range(n_A):
                d.add_segment(o, (u,))
        d.add_segment(k, (u,))
        for j, row in enumerate(s.tolist()):
            d.add_path(*row, 0, c=float(coef[off + j]))
        off += s.shape[0]
        ops.append((cue.Operation([0] * k + [1]), d))
    return cue.SegmentedPolynomial([A_op], [out_op], ops)


def polynomial(specs, coef, u=U, chunk=CHUNK):
    """`_build_poly` split into pieces of at most `chunk` monomials, as [(poly, cols)]: cols the global (ctilde-row)
    indices of the piece's monomials in its path order.  Why: the NVRTC compile of one uniform_1d kernel grows
    superlinearly with its path count (o4d16 order 4, gradient: 1000 paths 5.9 s, 2000 paths 29.8 s, RTX 4000 Ada)."""
    import numpy as np
    require()
    specs = [np.asarray(s) for s in specs]
    n_A = n_A_of(specs)
    coef = np.asarray(coef)
    rows, off = [], 0
    for s in specs:
        rows += [(s.shape[1], s[j], off + j) for j in range(s.shape[0])]
        off += s.shape[0]
    out = []
    for c0 in range(0, len(rows), chunk):
        part = rows[c0:c0 + chunk]
        sp, cols = [], []
        for k in sorted({k for k, _, _ in part}):
            sel = [r for r in part if r[0] == k]
            sp.append(np.stack([r[1] for r in sel]))
            cols += [r[2] for r in sel]
        cols = np.asarray(cols)
        out.append((_build_poly(sp, n_A, coef[cols], u), cols))
    return out


def product_energy(plans, A, u=U):
    """sum_v c_v prod A_i[s_v] per atom, A node-major (n, n_A); `plans` is `polynomial`'s [(poly, cols)] built
    with the same `u`, the chunks summed.  Atoms in the u mode: A zero-padded to G*U rows and laid out as
    (G, n_A*U), every path applied to U atoms at once; the padding rows are dropped from the output."""
    import cuequivariance_jax as cuex
    import jax
    import jax.numpy as jnp
    A = A[:, :plans[0][0].inputs[-1].num_segments]       # columns past the last one a monomial reads are unused
    n, n_A = A.shape
    G = -(-n // u)
    X = jnp.pad(A, ((0, G * u - n), (0, 0))).reshape(G, u, n_A).transpose(0, 2, 1).reshape(G, n_A * u)
    out = 0.0
    for p, _ in plans:
        [e] = cuex.segmented_polynomial(p, [X], [jax.ShapeDtypeStruct((G, u), A.dtype)], method="uniform_1d",
                                        name="ace_product_u")
        out = out + e
    return out.reshape(-1)[:n]


_nvrtc_warned = False


def with_cueq(model):
    """`model` (a folded single-species ACEModel, after `lean`) with its product basis evaluated by cueq: plans for
    `aa_specs` and, if blocked, `blk_aa_specs`, coefficients ctilde baked in (so a readout change needs a new plan);
    the DAG plans are cleared.  `eval.aa_dag.with_aa_dag(model, "cueq")` calls this."""
    global _nvrtc_warned
    import dataclasses
    import os
    import warnings
    import numpy as np
    require()
    check_supported(model)
    if not os.environ.get("CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR") and not _nvrtc_warned:
        _nvrtc_warned = True
        warnings.warn("cueq kernels compile on first use (minutes for large models); "
                      "set CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR to reuse them", stacklevel=2)
    coef = np.asarray(model.ctilde[:, 0])
    plan = lambda specs: _Static(polynomial(specs, coef))
    return dataclasses.replace(model, aa_dag=None, blk_aa_dag=None, cueq_plan=plan(model.aa_specs),
                               blk_cueq_plan=None if model.blk_aa_specs is None else plan(model.blk_aa_specs))


# ------------------------------------------------------------------ lammps-jax bundles (export.lammps)
def check_export(model, platform=None, cueq=True):
    """The export-time refusals for aa_dag="cueq" (`export.lammps.export_lammps`, `aj export`), before any work:
    the model (`check_supported`, on the model as given: `lean` folds it later), a non-CUDA `platform`, then
    (`cueq`) cueq itself.  The model check comes first so that it needs no cueq."""
    check_supported(model, folded=False)
    if platform not in (None, "cuda"):
        raise ValueError(f"aa_dag='cueq' runs cuequivariance's CUDA kernels: it needs platform='cuda', got {platform!r}")
    if cueq:
        require()


def version():
    import cuequivariance
    return cuequivariance.__version__


def recorder():
    """lammps-jax's contrib/ffi-replay recorder (`ffi_replay_record`, not part of the lammps-jax package): from
    $LAMMPS_JAX_CONTRIB (that directory), else contrib/ffi-replay of the lammps-jax checkout the installed package
    runs from (an editable install).  One module per process: its RECORDS are the process's plan compiles."""
    import importlib.util
    import os
    import pathlib
    import sys
    if "ffi_replay_record" in sys.modules:
        return sys.modules["ffi_replay_record"]
    d = os.environ.get("LAMMPS_JAX_CONTRIB")
    if d is None:
        try:
            import lammps_jax
            d = pathlib.Path(lammps_jax.__file__).resolve().parents[2] / "contrib" / "ffi-replay"
        except ImportError:
            d = None
    f = None if d is None else pathlib.Path(d) / "ffi_replay_record.py"
    if f is None or not f.is_file():
        raise CueqUnavailable("a cueq bundle records its plan compiles with lammps-jax's contrib/ffi-replay "
                              "recorder, which is not in the lammps-jax package: set LAMMPS_JAX_CONTRIB to the "
                              "contrib/ffi-replay directory of a lammps-jax checkout"
                              + ("" if f is None else f" (no {f})"))
    spec = importlib.util.spec_from_file_location("ffi_replay_record", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["ffi_replay_record"] = mod
    return mod


def _ops_library():
    import cuequivariance_ops_jax._common as cc
    return cc.library


def install_recorder():
    """Install the recorder on libcue_ops_jax's plan compile (before any cueq lowering in this process); returns
    the recorder module.  One bundle per process: the recorded handles are plan-registry indices, which the
    replay reproduces only from a fresh registry (`check_replayable`), so a recorder already holding plans (an
    earlier export in this process) is refused."""
    rec = recorder()
    if rec.RECORDS:
        raise RuntimeError(f"the ffi-replay recorder already holds {len(rec.RECORDS)} cueq plans from an earlier "
                           "export: export one cueq bundle per process")
    rec.install(_ops_library(), COMPILE)
    return rec


def check_replayable(records):
    """Raise unless replaying `records` (the recorder's [(args, handle)]) from an empty plan registry reproduces
    their handles: registry indices count from 0 and a recompiled identical plan returns its old handle, so a
    handle must be one seen before or the next new one.  A first handle above 0 means this process compiled
    cueq plans before the recorder was installed (e.g. a cueq ACECalculator)."""
    top = -1
    for _, h in records:
        if h > top + 1:
            raise RuntimeError(f"cueq plan handle {h} after {top + 1} recorded plans: this process compiled cueq "
                               "plans before the export, which a LAMMPS replay cannot reproduce; export the "
                               "bundle in a fresh process")
        top = max(top, h)


def shim(rec):
    """The replay shim the bundle's LAMMPS_JAX_FFI_HANDLERS names: next to the recorder."""
    import pathlib
    return pathlib.Path(rec.__file__).resolve().parent / SHIM


def handlers(shim_path):
    return f"{TARGET}={shim_path}:ffi_replay_handler"


def run_env(record):
    """The environment LAMMPS needs for a cueq bundle, from its `ace_jax.cueq` record: {name: value}, plus the
    directories LD_LIBRARY_PATH needs (libcue_ops and the pip CUDA libraries nvrtc / cublas, which libcue_ops
    finds by no rpath) under "LD_LIBRARY_PATH" (prepend them)."""
    import importlib.util
    import pathlib
    env = {"LAMMPS_JAX_FFI_HANDLERS": record["handlers"], "FFI_REPLAY_FILE": record["replay"],
           "CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR": record["cache"]}
    libs = []
    try:
        jl = pathlib.Path(importlib.util.find_spec("cuequivariance_ops_jax").origin).parent / "lib"
        env["FFI_REPLAY_LIB"] = str(jl / "libcue_ops_jax.so")
        site = jl.parent.parent
        libs = [str(site / "cuequivariance_ops" / "lib")] + [str(site / "nvidia" / d / "lib")
                                                             for d in ("cuda_nvrtc", "cublas")]
    except (ImportError, AttributeError, TypeError):
        env["FFI_REPLAY_LIB"] = "<site-packages>/cuequivariance_ops_jax/lib/libcue_ops_jax.so"
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    return env
