# Julia/ACEpotentials coupling ORACLE (CI-only; the runtime shim is ET-only).
# Builds an ace1_model for a shape and dumps the coupling A2B / aa_spec plus the
# integer specs (mb_spec bodies, Rnl_spec, Ylm_spec) and per-B / per-AA alignment
# signatures, so the Python ET-bridge can be numerically validated against it.
#
#   julia --project=<tooling> julia/coupling_reference.jl out.npz
# shape via ENV: ACE_ELEMENTS, ACE_ORDER, ACE_TOTALDEGREE, ACE_WL
using ACEpotentials, NPZ, LinearAlgebra, SparseArrays
M = ACEpotentials.Models
const EquivariantTensors = M.EquivariantTensors

els   = Symbol.(split(get(ENV, "ACE_ELEMENTS", "Si,Ge"), ","))
order = parse(Int, get(ENV, "ACE_ORDER", "2"))
tdeg  = parse(Int, get(ENV, "ACE_TOTALDEGREE", "6"))
wL    = parse(Float64, get(ENV, "ACE_WL", "1.5"))
out   = length(ARGS) >= 1 ? ARGS[1] : "coupling_ref.npz"

model = ACEpotentials.ace1_model(elements = els, order = order, totaldegree = tdeg, wL = wL)
t = model.model.tensor
r_spec = [(b.n, b.l) for b in model.model.rbasis.spec]           # (n,l) per Rnl idx (1-based)
maxl   = maximum(l for (n, l) in r_spec)
y_spec = [(l, m) for l in 0:maxl for m in -l:l]                  # real Ylm (l,m) order
abasis = [(tt[1], tt[2]) for tt in t.abasis.spec]               # (Rnl_idx, Ylm_idx) per A (1-based)

# per-AA (n,l) bodies (drop m), used ONLY to extract the mb_spec (the ET input).
aa_specs = t.aabasis.specs
bodies = Vector{Vector{Tuple{Int,Int}}}()                        # per AA function, (n,l)
for s in aa_specs, aa in s
    push!(bodies, [r_spec[abasis[a][1]] for a in aa])
end
mb_spec = unique(bodies)                                          # the ET input, in this extracted order

# Recompute the coupling from the EXACT extracted mb_spec so the dumped A2B, its
# column identities (aa_sig) and row identities (nnll) are all self-consistent
# with each other and with the dumped mb_spec order.  The Python bridge fed this
# same mb_spec reproduces THIS A2B bit-for-bit on multiplicity-1 nnll blocks; on
# degenerate blocks (order >= 4) it reproduces the row SUBSPACE (the block basis
# is unique only up to a within-block orthogonal rotation) -- see the parity test.
mb_nt  = [[(n=nl[1], l=nl[2]) for nl in bb] for bb in mb_spec]
rnl_nt = [(n=nl[1], l=nl[2]) for nl in r_spec]
ylm_nt = [(l=lm[1], m=lm[2]) for lm in y_spec]
t2 = EquivariantTensors.sparse_equivariant_tensor(L=0, mb_spec=mb_nt, Rnl_spec=rnl_nt,
                                                   Ylm_spec=ylm_nt, basis=real)
A2B = Matrix(t2.A2Bmaps[1]); rows, cols, vals = findnz(sparse(A2B))
nnll = M.get_nnll_spec(t2)                                        # per-B (n,l), in A2B ROW order
# per-AA (n,l,m) signature in A2B COLUMN order.  Taken from meta 𝔸spec (the spec
# returned WITH the symmetrisation matrix), NOT aabasis.specs: SparseSymmProd
# re-sorts its input, so the aabasis evaluation order is a different permutation
# from the A2B column order (this mislabels columns within/across nnll blocks).
aa_sig = [sort([(Int(b.n), Int(b.l), Int(b.m)) for b in row]) for row in t2.meta["𝔸spec"]]

# flatten ragged specs to (data, lengths, offsets) for npz
flat(v) = (reduce(vcat, [collect(Iterators.flatten(x)) for x in v]), Int32[length(x) for x in v])
mb_flat, mb_len = flat([[(nl...,) for nl in bb] for bb in mb_spec])   # each body: n,l pairs
nnll_flat, nnll_len = flat([[(b.n, b.l) for b in bb] for bb in nnll])
aasig_flat, aasig_len = flat(aa_sig)                                  # each: n,l,m triples

npzwrite(out, Dict{String,Any}(
    "n_elements" => length(els),
    "order" => order, "totaldegree" => tdeg, "wL" => wL, "maxl" => maxl,
    "r_spec" => Int32.(reduce(hcat, [collect(t) for t in r_spec])' ),      # (n_rnl, 2), 1-based n but l is l
    "y_spec" => Int32.(reduce(hcat, [collect(t) for t in y_spec])' ),      # (n_ylm, 2) = (l,m)
    "A2B_rows" => Int32.(rows .- 1), "A2B_cols" => Int32.(cols .- 1),
    "A2B_vals" => vals, "A2B_shape" => Int32[size(A2B,1), size(A2B,2)],
    "mb_flat" => Int32.(mb_flat), "mb_len" => mb_len,                      # mb_spec bodies (n,l)*
    "nnll_flat" => Int32.(nnll_flat), "nnll_len" => nnll_len,              # per-B (n,l)
    "aasig_flat" => Int32.(aasig_flat), "aasig_len" => aasig_len,          # per-AA (n,l,m) sorted
))
println("wrote $out  n_B=$(size(A2B,1)) n_AA=$(size(A2B,2)) n_mb=$(length(mb_spec)) nnz=$(length(vals))")
