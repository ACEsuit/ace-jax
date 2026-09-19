"""In-process EquivariantTensors coupling shim via JuliaCall.

`couple(mb_spec, Rnl_spec, Ylm_spec)` calls
`EquivariantTensors.sparse_equivariant_tensor(L=0, ...)` and returns the coupling
in ace-jax's export layout: dense A2B (n_B, n_AA), per-order aa_spec (0-based int
arrays), and the A-basis spec (0-based).  Requires the optional `authoring` extra
(juliacall + juliapkg, which auto-provisions Julia + EquivariantTensors); the
import is lazy so the core package never depends on it.
"""


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
    of (l, m).  Returns (A2B dense (n_B, n_AA) float64, aa_spec tuple of 0-based
    int arrays, aspec list of 0-based (Rnl_idx, Ylm_idx))."""
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
    specs = tensor.aabasis.specs
    aa_spec = tuple(np.asarray(tomat(jl.getindex(specs, k)), np.int64) - 1  # 0-based (n_v, order)
                    for k in range(1, int(jl.length(specs)) + 1))
    aspec_arr = np.asarray(tomat(tensor.abasis.spec), np.int64) - 1        # (n_A, 2): (Rnl_idx, Ylm_idx)
    aspec = [(int(r), int(y)) for r, y in aspec_arr]
    return A2B, aa_spec, aspec
