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


def check_supported(model):
    from .model import ACEModel
    if not isinstance(model, ACEModel):
        raise ValueError(f"aa_dag='cueq' needs a linear ACEModel, got {type(model).__name__}")
    if not model.folded:
        raise ValueError("aa_dag='cueq' needs the folded (lean) model: apply lean() first")
    if getattr(model, "rnl_embedding", None) is not None:  # TODO(Task 8): remove with TrACE support
        raise ValueError("aa_dag='cueq' does not support species-embedding (TrACE) models yet: TrACE support: not yet")
    if model.ctilde.shape[1] != 1:
        raise ValueError("aa_dag='cueq' supports one species, or species-embedding (TrACE) models; "
                         f"this categorical model has {model.ctilde.shape[1]} species")


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
