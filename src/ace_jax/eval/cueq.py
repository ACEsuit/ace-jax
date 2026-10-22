"""cuequivariance (NVIDIA cueq) as a GPU form of the lean ACE product basis (`aa_dag="cueq"`).

Design: docs/dev/specs/2026-10-22-cueq-product-basis-design.md.  The folded site energy
e_i = sum_v ctilde[v, z_i] prod_t A_i[s_v,t] is a cueq SegmentedPolynomial: one SegmentedTensorProduct per order
k ([A]*k -> out), one path per monomial, evaluated by cuequivariance_jax.segmented_polynomial(method="uniform_1d")
with atoms in the u mode (U atoms per group: the obvious batch-axis layout is 6x slower) and the polynomial split
into kernels of at most CHUNK monomials (NVRTC compile grows superlinearly with paths).  Species-embedding
(TrACE) models: `trace_polynomial`, channel-wise paths with per-centre weights.  cueq is optional: nothing here
imports it at module level."""
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


def require_descriptors():
    """The pure-Python descriptor package only: building plans (`polynomial`, `trace_polynomial`) needs no GPU, so
    it must work where cuequivariance_jax's CUDA ops cannot load (no driver: libcuda.so.1 missing raises OSError)."""
    try:
        import cuequivariance  # noqa: F401
    except ImportError:
        raise CueqUnavailable("aa_dag='cueq' needs cuequivariance: pip install 'ace-jax[cueq]' (Linux, CUDA 12)") from None


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
    if getattr(model, "rnl_embedding", None) is not None:   # TrACE: channel-wise paths (`trace_polynomial`)
        return
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
    require_descriptors()                          # building a plan needs no GPU (with_cueq requires the ops)
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


def product_energy(plans, A, u=U, node_z=None):
    """sum_v c_v prod A_i[s_v] per atom, A node-major (n, n_A); `plans` is `polynomial`'s [(poly, cols)] built
    with the same `u`, the chunks summed.  Atoms in the u mode: A zero-padded to G*U rows and laid out as
    (G, n_A*U), every path applied to U atoms at once; the padding rows are dropped from the output.  A
    `TracePlan` (species-embedding model) goes to `trace_energy`, which needs `node_z`; a single-species plan
    ignores it."""
    if isinstance(plans, TracePlan):
        return trace_energy(plans, A, node_z, u)
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


# ------------------------------------------------------------------ TrACE (species-embedding) models
# An embedding model's radial column (n'k, l) is Q_n'(x) emb[z_j, k] and its many-body spec is channel-diagonal:
# every factor of a monomial has the same embedding channel k (`basis.spec.build_embedding_spec`).  With
# A'[s, k] = A[(n', l, m), k] (s the channel-free key (n', l, m)) the folded site energy is
#     e_i = sum_k sum_p W[p, k, z_i] prod_t A'_i[s_pt, k]
# over channel-free monomials p: one path per p, applied channel-wise, with per-(path, channel) weights picked
# by the centre species -- the shape of cueq's `symmetric_contraction`, but on ace-jax's own monomials (its
# descriptor builds coupled-irreps paths, a superset of the degree-truncated basis).  Layout "ach": u = U atoms
# x d channels, the weights gathered per atom and laid out like A' (u = d with atoms in the batch, weights
# indexed per atom, was 2-6x slower: small u).  The spike's bench/cueq_product/trace_cueq.py, parity-tested.


class TracePlan:
    """`trace_polynomial`'s plan: polys [(poly, cols)] (cols: path indices), gather (n_seg, d) the A column of
    each (key, channel) or -1, W (NZ, n_paths, d) float64 weights, stats."""
    def __init__(self, polys, gather, W, stats):
        self.polys, self.gather, self.W, self.stats = polys, gather, W, stats


def channel_map(model, blocked):
    """{A column: (channel-free key (n', l, m), channel k)} of an embedding model: blocked, the lean l-blocked
    layout (`model.block_dense`, never species-compact for a factorised radial), else the aspec layout."""
    import numpy as np
    nidx, kidx = np.asarray(model.rnl_emb_nidx), np.asarray(model.rnl_emb_kidx)
    out = {}
    if blocked:
        if model.blk_compact:
            raise ValueError("aa_dag='cueq' on an embedding model needs the non-compact l-blocked layout")
        base = 0
        for l, off, w in model.blk:
            nm = 2 * l + 1
            for kk in range(w):
                r = off + kk
                for mm in range(nm):
                    out[base + kk * nm + mm] = ((int(nidx[r]), l, mm), int(kidx[r]))
            base += w * nm
    else:
        ar, ay = np.asarray(model.aspec_r), np.asarray(model.aspec_y)
        for a, (r, y) in enumerate(zip(ar.tolist(), ay.tolist())):
            l = int(np.floor(np.sqrt(y)))
            out[a] = ((int(nidx[r]), l, y - l * l), int(kidx[r]))
    return out


def trace_plan(model, blocked):
    """Channel-free monomials and weights of a folded embedding model's product basis (numpy, no cueq):
    (gather (n_seg, d): the A column per (key, channel) or -1, paths [sorted segment-index tuples, by order],
    W (n_paths, d, NZ): ctilde summed onto (path, channel), stats).  Raises if a monomial mixes channels."""
    import numpy as np
    specs = [np.asarray(s) for s in (model.blk_aa_specs if blocked else model.aa_specs)]
    cmap = channel_map(model, blocked)
    d = 1 + max(k for _, k in cmap.values())
    keys = sorted({key for key, _ in cmap.values()})
    sidx = {key: i for i, key in enumerate(keys)}
    gather = -np.ones((len(keys), d), np.int64)
    for a, (key, k) in cmap.items():
        gather[sidx[key], k] = a
    ct = np.asarray(model.ctilde, np.float64)
    W, off, mixed = {}, 0, 0
    for s in specs:
        for j, row in enumerate(s.tolist()):
            ks = {cmap[a][1] for a in row}
            if len(ks) != 1:
                mixed += 1
                continue
            p = tuple(sorted(sidx[cmap[a][0]] for a in row))
            W.setdefault(p, np.zeros((d, ct.shape[1])))[ks.pop()] += ct[off + j]
        off += s.shape[0]
    if mixed:
        raise ValueError(f"aa_dag='cueq': {mixed} monomials of this embedding model mix embedding channels")
    paths = sorted(W, key=lambda p: (len(p), p))
    st = {"n_AA": off, "n_paths": len(paths), "d": d, "n_seg": len(keys)}
    return gather, paths, np.stack([W[p] for p in paths]), st


def trace_size(model):
    """(n_paths, d) of an embedding model's cueq plan (`trace_plan`), from its spec alone: no readout, so the
    full model before `lean` gives the lean plan's counts (the channel-free monomials are layout-independent)."""
    import numpy as np
    cmap = channel_map(model, False)
    paths = {tuple(sorted(cmap[a][0] for a in row)) for s in model.aa_specs for row in np.asarray(s).tolist()}
    return len(paths), 1 + max(k for _, k in cmap.values())


def trace_weight_bytes_per_row(model, itemsize):
    """Device bytes per row of a TrACE plan's per-atom weights (`trace_energy`: W[node_z] and its laid-out copy,
    which the backward pass keeps as a residual of every chunk, so all chunks are live): 2 x n_paths x d x
    itemsize; 0 for a model without an embedding.  `export.lammps.cueq_block_rows_for` adds it per block row."""
    if getattr(model, "rnl_embedding", None) is None:
        return 0
    n_paths, d = trace_size(model)
    return 2 * n_paths * d * itemsize


def _build_trace_poly(paths, n_seg, u):
    """sum_p w[p] prod A'[s_p] -> out, every operand in mode u (channel-wise); w an input operand."""
    import cuequivariance as cue
    w_op = cue.SegmentedOperand(ndim=1, segments=[(u,)] * len(paths))
    A_op = cue.SegmentedOperand(ndim=1, segments=[(u,)] * n_seg)
    out_op = cue.SegmentedOperand(ndim=1, segments=[(u,)])
    by, ops = {}, []
    for j, p in enumerate(paths):
        by.setdefault(len(p), []).append((j, p))
    for nu, items in sorted(by.items()):
        d = cue.SegmentedTensorProduct.from_subscripts(",".join(["u"] * (nu + 2)))
        for _ in range(len(paths)):
            d.add_segment(0, (u,))
        for o in range(1, nu + 1):
            for _ in range(n_seg):
                d.add_segment(o, (u,))
        d.add_segment(nu + 1, (u,))
        for j, p in items:
            d.add_path(j, *p, 0, c=1.0)
        ops.append((cue.Operation([0] + [1] * nu + [2]), d))
    return cue.SegmentedPolynomial([w_op, A_op], [out_op], ops)


def trace_polynomial(model, blocked, u=U, chunk=CHUNK):
    """`TracePlan` of a folded embedding model (`trace_plan`), in kernels of at most `chunk` paths, mode
    u = `u` atoms x d channels.  The weights are the readout (a readout change needs a new plan)."""
    import numpy as np
    require_descriptors()                          # building a plan needs no GPU (with_cueq requires the ops)
    gather, paths, W, st = trace_plan(model, blocked)
    uu = u * gather.shape[1]
    polys = [(_build_trace_poly(paths[c0:c0 + chunk], gather.shape[0], uu), np.arange(c0, min(c0 + chunk, len(paths))))
             for c0 in range(0, len(paths), chunk)]
    return TracePlan(polys, gather, W.transpose(2, 0, 1), st)


def trace_energy(plan, A, node_z, u=U):
    """Site energies of a `TracePlan` from node-major A (n, n_A): A' (n, n_seg, d) gathered from A, laid out
    (G, n_seg * u * d) with u atoms per group (zero padding rows dropped), each chunk's weights W[z_i] gathered
    per atom and laid out alike, the channels summed."""
    import cuequivariance_jax as cuex
    import jax
    import jax.numpy as jnp
    import numpy as np
    if node_z is None:
        raise ValueError("a species-embedding cueq plan needs node_z (the centre species per atom)")
    n = A.shape[0]
    n_seg, d = plan.gather.shape
    Ap = jnp.concatenate([A, jnp.zeros((n, 1), A.dtype)], axis=1)
    Xc = Ap[:, jnp.asarray(np.where(plan.gather < 0, A.shape[1], plan.gather))]       # (n, n_seg, d)
    G = -(-n // u)
    pad = lambda M: jnp.pad(M, ((0, G * u - n),) + ((0, 0),) * (M.ndim - 1))
    X = pad(Xc).reshape(G, u, n_seg, d).transpose(0, 2, 1, 3).reshape(G, n_seg * u * d)   # segment: u atoms x d
    Wt = jnp.asarray(plan.W, A.dtype)                                                     # (NZ, n_paths, d)
    out = 0.0
    for p, cols in plan.polys:
        Wa = Wt[:, cols, :][node_z]                                                       # (n, n_p, d)
        Wx = pad(Wa).reshape(G, u, len(cols), d).transpose(0, 2, 1, 3).reshape(G, len(cols) * u * d)
        [e] = cuex.segmented_polynomial(p, [Wx, X], [jax.ShapeDtypeStruct((G, u * d), A.dtype)],
                                        method="uniform_1d", name="ace_trace_ach")
        out = out + e.reshape(G * u, d).sum(-1)
    return out[:n]


_nvrtc_warned = False


def with_cueq(model):
    """`model` (a folded single-species or species-embedding ACEModel, after `lean`) with its product basis
    evaluated by cueq: plans for `aa_specs` and, if blocked, `blk_aa_specs`, the readout ctilde in the plan
    (single species: baked into the paths; embedding: `trace_polynomial`'s weights), so a readout change needs a
    new plan; the DAG plans are cleared.  `eval.aa_dag.with_aa_dag(model, "cueq")` calls this."""
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
    if model.rnl_embedding is not None:          # TrACE: channel-wise paths, per-centre weights
        plan, blk = _Static(trace_polynomial(model, False)), model.blk_aa_specs is not None
        return dataclasses.replace(model, aa_dag=None, blk_aa_dag=None, cueq_plan=plan,
                                   blk_cueq_plan=_Static(trace_polynomial(model, True)) if blk else None)
    coef = np.asarray(model.ctilde[:, 0])
    plan = lambda specs: _Static(polynomial(specs, coef))
    return dataclasses.replace(model, aa_dag=None, blk_aa_dag=None, cueq_plan=plan(model.aa_specs),
                               blk_cueq_plan=None if model.blk_aa_specs is None else plan(model.blk_aa_specs))


# ------------------------------------------------------------------ lammps-jax bundles (export.lammps)
def check_export(model, platform=None, cueq=True):
    """The export-time refusals for aa_dag="cueq" (`export.lammps.export_lammps`, `aj export`), before any work:
    the model (`check_supported`, on the model as given: `lean` folds it later), a non-CUDA `platform`, then
    (`cueq`) a host whose JAX backend is not a GPU (the export lowers on the default device) and cueq itself.
    The model check comes first so that it needs no cueq; cueq=False imports neither jax nor cueq."""
    check_supported(model, folded=False)
    if platform not in (None, "cuda"):
        raise ValueError(f"aa_dag='cueq' runs cuequivariance's CUDA kernels: it needs platform='cuda', got {platform!r}")
    if cueq:
        if (b := backend()) != "gpu":
            raise ValueError("aa_dag='cueq' exports on a CUDA GPU (it lowers and compiles cuequivariance's kernels "
                             f"there); the JAX backend here is {b!r}")
        require()


def backend():
    """jax.default_backend(): the export's GPU check (a seam, so CPU tests can stub it)."""
    import jax
    return jax.default_backend()


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


def run_env(record, bundle_path):
    """The environment LAMMPS needs for the cueq bundle at `bundle_path`, from its `ace_jax.cueq` record:
    {name: value}, plus the directories LD_LIBRARY_PATH needs (libcue_ops and the pip CUDA libraries nvrtc /
    cublas, which libcue_ops finds by no rpath) under "LD_LIBRARY_PATH" (prepend them).  The record's replay
    and cache are relative to the bundle's directory (bundle, sidecar and cache travel together), resolved
    here.  The shim is machine-specific: $LAMMPS_JAX_CONTRIB/ffi_replay_shim.so when that is set here, else
    the record's `handlers`, which names the export host's shim."""
    import importlib.util
    import os
    import pathlib
    d = pathlib.Path(bundle_path).absolute().parent
    h = record["handlers"]
    if os.environ.get("LAMMPS_JAX_CONTRIB"):
        h = handlers(pathlib.Path(os.environ["LAMMPS_JAX_CONTRIB"]).absolute() / SHIM)
    env = {"LAMMPS_JAX_FFI_HANDLERS": h, "FFI_REPLAY_FILE": str(d / record["replay"]),
           "CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR": str(d / record["cache"])}
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
