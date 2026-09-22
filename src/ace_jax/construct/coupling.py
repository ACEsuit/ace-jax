"""In-process EquivariantTensors coupling shim via JuliaCall.

`couple(mb_spec, Rnl_spec, Ylm_spec)` calls
`EquivariantTensors.sparse_equivariant_tensor(L=0, ...)` and returns the coupling
in ace-jax's export layout (see `Coupling`).  Requires the optional `authoring`
extra (juliacall + juliapkg, which auto-provisions Julia + EquivariantTensors);
the import is lazy so the core package never depends on it.

juliacall note: dependency discovery scans `sys.path` for `juliapkg.json`, so a
standalone script must put the repo root on `PYTHONPATH` (pytest and
`python -m acegp.cli` do this implicitly).
"""

from typing import NamedTuple


class Coupling(NamedTuple):
    """Everything the ET call returns, in ace-jax's export layout.

    A2B: dense (n_B, n_AA) float64 coupling, one nonzero per column.
    aa_sig: per-column (n, l, m) tuples in A2B column order -- the column
        *identity*, from meta `𝔸spec` in raw row order (NOT aabasis.specs:
        SparseSymmProd re-sorts its input, so the evaluation order is a
        different permutation; see the build_spec/couple docstrings in earlier
        findings).
    aspec: 0-based (Rnl_idx, Ylm_idx) per A function, in abasis.spec order --
        the index space A2B columns and aa_specs entries share.
    aa_specs: per-order (n_v, order) 0-based A-column index arrays, aabasis
        evaluation order (consumed as `A[:, g]` in eval).
    nnll_spec: per-B-row body lists from ET's `get_nnll_spec` -- exactly the
        exporter's meta["nnll"] source (per-row, not the input echo: the
        tensor meta also carries "mb_spec", which is the ET *input* echo).
    """

    A2B: object
    aa_sig: tuple
    aspec: tuple
    aa_specs: tuple
    nnll_spec: tuple


def subspace_residual(A, B):
    """Max-abs difference of the orthogonal projectors onto the row spaces of `A`
    and `B` (both shape (m, k)): ``||A⁺A_proj - B⁺B_proj||_max`` computed as
    ``|Qa Qaᵀ - Qb Qbᵀ|`` with Q an orthonormal basis of the row space (from a
    QR of the transpose).

    This is the invariant used to compare a DEGENERATE nnll coupling block across
    two basis constructions: at order >= 4 a block of multiplicity m has a
    symmetrisation basis defined only up to a within-block ORTHOGONAL rotation, so
    two enumerations give ``A`` and ``B = O A`` for some orthogonal ``O``.  The
    row-space projector is invariant to that rotation (and to row permutation and
    per-row sign), so a correct coupling gives ~0 here while the raw entries can
    differ substantially.  For multiplicity 1 it degenerates to the (sign- and
    scale-invariant) 1-D projector.  Pure numpy -- no Julia."""
    import numpy as np
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.shape != B.shape:
        raise ValueError(f"block shapes differ: {A.shape} vs {B.shape}")
    Qa = np.linalg.qr(A.T)[0]
    Qb = np.linalg.qr(B.T)[0]
    return float(np.abs(Qa @ Qa.T - Qb @ Qb.T).max())


def _jl():
    try:
        from juliacall import Main as jl
    except ModuleNotFoundError as e:  # pragma: no cover - exercised only with the extra
        raise ModuleNotFoundError(
            "coupling generation needs the 'authoring' extra: pip install ace-jax[authoring]"
        ) from e
    # Activate the juliapkg-provisioned project explicitly before `using`: under
    # some harnesses (pytest capture) the ambient project is not the one juliapkg
    # resolved EquivariantTensors into, giving "Package ... not found in current path".
    import juliapkg
    jl.seval("import Pkg")
    jl.seval(f'Pkg.activate(raw"{juliapkg.project()}"; io=devnull)')
    jl.seval("using EquivariantTensors")
    return jl


def couple(mb_spec, Rnl_spec, Ylm_spec):
    """mb_spec: list of list of (n, l); Rnl_spec: list of (n, l); Ylm_spec: list
    of (l, m).  Returns a `Coupling`.

    `aa_sig[j]` is the identity of A2B COLUMN j.  It is taken from the tensor's
    meta `𝔸spec` (the readable spec returned *alongside* the symmetrisation
    matrix), NOT from `aabasis.specs`: `SparseSymmProd` re-sorts its input, so the
    aabasis EVALUATION order (`specs`/`reconstruct_spec`) is a different
    permutation from the A2B column order.  Under `𝔸spec` order A2B is exactly
    block-diagonal in nnll; under `specs` order it is not.  This distinction is
    what makes end-to-end parity work for an arbitrary (Python-generated) mb_spec
    rather than only for the oracle's ordering."""
    import numpy as np
    jl = _jl()
    ET = jl.EquivariantTensors
    # Build CONCRETELY-typed Julia vectors (juliacall would otherwise pass Python
    # lists as Vector{Any}, which _make_idx_A_spec cannot dispatch on).  The Julia
    # comprehensions re-type from the numpy int arrays.
    ai = lambda xs: np.asarray(xs, np.int64)
    rnl = jl.seval("(ns,ls)->[(n=Int(n),l=Int(l)) for (n,l) in zip(ns,ls)]")(
        ai([n for n, l in Rnl_spec]), ai([l for n, l in Rnl_spec]))
    ylm = jl.seval("(ls,ms)->[(l=Int(l),m=Int(m)) for (l,m) in zip(ls,ms)]")(
        ai([l for l, m in Ylm_spec]), ai([m for l, m in Ylm_spec]))
    mb = jl.seval("()->Vector{Vector{@NamedTuple{n::Int,l::Int}}}()")()
    push = jl.seval("(v,ns,ls)->push!(v, [(n=Int(n),l=Int(l)) for (n,l) in zip(ns,ls)])")
    for bb in mb_spec:
        push(mb, ai([n for n, l in bb]), ai([l for n, l in bb]))
    tensor = ET.sparse_equivariant_tensor(L=0, mb_spec=mb, Rnl_spec=rnl,
                                          Ylm_spec=ylm, basis=jl.real)
    A2B = np.asarray(jl.Matrix(jl.getindex(tensor.A2Bmaps, 1)), float)    # (n_B, n_AA)
    tomat = jl.seval("s -> permutedims(reduce(hcat, collect.(s)))")       # Vector{NTuple} -> (n, k), 1-based
    # Per-column (n,l,m) signatures in A2B COLUMN order, straight from meta 𝔸spec
    # (raw row order -- sorting here would lose the original body channel order
    # that get_nnll_spec dumps; consumers sort when comparing).
    aaspec = tensor.meta["𝔸spec"]
    col_sig = jl.seval(
        "row -> [(Int(b.n), Int(b.l), Int(b.m)) for b in row]")
    aa_sig = tuple(
        tuple((int(n), int(l), int(m)) for n, l, m in col_sig(jl.getindex(aaspec, k)))
        for k in range(1, int(jl.length(aaspec)) + 1))
    aspec_arr = np.asarray(tomat(tensor.abasis.spec), np.int64) - 1        # (n_A, 2): (Rnl_idx, Ylm_idx)
    aspec = [(int(r), int(y)) for r, y in aspec_arr]
    # AA basis in EVALUATION order, split by correlation order (aa_lens), and
    # the per-row body dump.  Both are exporter-artifact sources.
    aa_specs = tuple(
        np.asarray(tomat(jl.getindex(tensor.aabasis.specs, k)), np.int64) - 1
        for k in range(1, int(jl.length(tensor.aabasis.specs)) + 1))
    nnll = jl.seval("t -> EquivariantTensors.get_nnll_spec(t, 1)")(tensor)
    row_sig = jl.seval(
        "row -> [(Int(b.n), Int(b.l)) for b in row]")
    nnll_spec = tuple(
        tuple((int(n), int(l)) for n, l in row_sig(jl.getindex(nnll, k)))
        for k in range(1, int(jl.length(nnll)) + 1))
    return Coupling(A2B=A2B, aa_sig=aa_sig, aspec=aspec,
                    aa_specs=aa_specs, nnll_spec=nnll_spec)
