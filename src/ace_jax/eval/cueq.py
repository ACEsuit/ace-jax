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
