# Julia/ACEpotentials smoothness-prior ORACLE (CI-only; the Python port is
# src/ace_jax/construct/prior.py).  Builds an ace1_model for a shape and dumps
# the algebraic smoothness prior diagonal plus the FULL per-column nnll basis
# (tensor bodies + pair singletons, per species -- the layout _nnll_basis
# feeds the prior), so the Python port can be validated against it.
#
#   julia --project=<tooling> julia/smoothness_reference.jl out.npz
# shape via ENV: ACE_ELEMENTS, ACE_ORDER, ACE_TOTALDEGREE, ACE_WL
#
# The prior is a pure function of the per-column nnll (smoothness_priors.jl:
#   gamma[i] = sum((b.l/wl)^p + (b.n/wn)^p for b in nnll[i]);  p=4, wl=2/3,
#   wn=1.0 defaults), independent of radials/splines, so the fixture pins the
# FORMULA and the (tensor|pair)xNZ column layout -- not any radial data.
using ACEpotentials, NPZ, LinearAlgebra
M = ACEpotentials.Models

els   = Symbol.(split(get(ENV, "ACE_ELEMENTS", "Si,Ge"), ","))
order = parse(Int, get(ENV, "ACE_ORDER", "2"))
tdeg  = parse(Int, get(ENV, "ACE_TOTALDEGREE", "6"))
wL    = parse(Float64, get(ENV, "ACE_WL", "1.5"))
out   = length(ARGS) >= 1 ? ARGS[1] : "smoothness_ref.npz"

# ACEpotentials DefaultHypers.bond_len crashes for elements absent from its
# VASP length-scale table (e.g. Mn, z=25); the prior is bond-length-independent,
# so inject a placeholder like coupling_reference.jl does.
let DH = ACEpotentials.DefaultHypers
    for sym in els
        z = Int(DH.atomic_number(DH.ChemicalSpecies(sym)))
        haskey(DH._lengthscales, z) && continue
        DH._lengthscales[z] = Dict{Any,Any}("bond_len" => Any[2.5, "placeholder (ace-jax)"],
                                            "min_bond_len" => Any[1.8, "placeholder (ace-jax)"])
        @info "smoothness_reference: injected placeholder bond_len for $sym (z=$z)"
    end
end

model = ACEpotentials.ace1_model(elements = els, order = order, totaldegree = tdeg, wL = wL)
m = model.model                                  # the inner model, as export_model.jl:102

# The fixtures pin the plain algebraic prior: Diagonal, no coupling scalings
# (those are commented out in smoothness_priors.jl:122) and no embedded-model
# n-folding (meta["embedding"] is only set by ace_embedding_model, never by
# ace1_model -- assert here so the fixture cannot silently drift; the port's
# fold is unit-tested from a synthetic meta["embedding"] instead).
@assert !haskey(m.meta, "embedding") "embedded model: the oracle fixtures are for the unfolded ace1_model prior"
P = M.algebraic_smoothness_prior(m)
@assert P isa Diagonal "algebraic_smoothness_prior is not Diagonal"
gamma = collect(diag(P))
@assert all(gamma .> 0) "non-positive prior diagonal"
@assert length(gamma) == M._basis_length(m)

# The FULL per-column nnll the prior consumes (smoothness_priors.jl:_nnll_basis):
# per species, the tensor B-functions then the pair singletons.
nnll  = M._nnll_basis(m)
nflat = reduce(vcat, [[Int(b.n), Int(b.l)] for bb in nnll for b in bb])
nlen  = Int32.(length.(nnll))

# Column layout contract: per-column kind (0=tensor, 1=pair), both specs
# species-independent, so every (kind, pattern) occurs exactly NZ times.  The
# BLOCK ORDER is a property of the model's global basis indexing -- dump it
# (nnll_kind) rather than assert it: the fixture is the layout truth, and the
# production wire-in rebuilds gamma from ace-jax's own design assembly anyway.
nz         = M._get_nz(m)
len_tensor = length(M.get_nnll_spec(m.tensor))
len_pair   = length(m.pairbasis.spec)
kind = fill(Int32(1), length(nnll))
for iz in 1:nz
    kind[M.get_basis_inds(m, M._i2z(m, iz))] .= Int32(0)
    kind[M.get_pairbasis_inds(m, M._i2z(m, iz))] .= Int32(1)
end
@assert sum(kind .== 0) == len_tensor * nz && sum(kind .== 1) == len_pair * nz

npzwrite(out, Dict{String,Any}(
    "gamma" => gamma,
    "nnll_flat" => Int32.(nflat), "nnll_len" => nlen,      # per-column (n,l) bodies
    "nnll_kind" => kind,                                   # 0=tensor, 1=pair
    "n_tensor" => Int32(len_tensor), "n_pair" => Int32(len_pair),
    "n_basis" => Int32(length(gamma)), "n_species" => Int32(length(els)),
    "order" => Int32(order), "totaldegree" => Int32(tdeg), "wL" => wL,
    "prior_p" => 4.0, "prior_wl" => 2/3, "prior_wn" => 1.0,
    # NPZ can't write bare strings: byte vectors, as export_model.jl's meta_json
    "acepotentials_version" => Vector{UInt8}(string(pkgversion(ACEpotentials))),
    "julia_version" => Vector{UInt8}(string(VERSION)),
))
println("wrote $out  n_basis=$(length(gamma))  n_cols=$(length(nnll))")
