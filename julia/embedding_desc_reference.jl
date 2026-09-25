# Julia/ACEpotentials ORACLE for Python embedded-model authoring (CI-only; the port is
# construct/model.py build_embedding_model).  Builds ace_embedding_model for a shape
# and dumps site descriptors on several random structures (enough sites per species
# to pin the descriptor SUBSPACE, which is what makes the parity test immune to
# basis ordering), the reduced embedding rows it used, and the smoothness prior with
# each column's nnll.
#
#   julia --project=<tooling providing ACEpotentials with ace_embedding_model>
#         julia/embedding_desc_reference.jl out.npz
# ENV: ACE_ELEMENTS (symbols), ACE_ORDER, ACE_TOTALDEGREE, ACE_DMAX (optional),
#      ACE_EMBEDDING (JSON table), ACE_LATTICE (fcc|diamond), ACE_A0, ACE_NSTRUCT
using ACEpotentials, NPZ, LinearAlgebra, Random, AtomsBuilder, AtomsBase, Unitful, StaticArrays
M = ACEpotentials.Models

els   = Symbol.(split(get(ENV, "ACE_ELEMENTS", "Si,Ge"), ","))
order = parse(Int, get(ENV, "ACE_ORDER", "2"))
tdeg  = parse(Int, get(ENV, "ACE_TOTALDEGREE", "6"))
dmax  = haskey(ENV, "ACE_DMAX") ? parse(Int, ENV["ACE_DMAX"]) : nothing
lat   = Symbol(get(ENV, "ACE_LATTICE", "diamond"))
a0    = parse(Float64, get(ENV, "ACE_A0", "5.5"))
nstr  = parse(Int, get(ENV, "ACE_NSTRUCT", "10"))
out   = length(ARGS) >= 1 ? ARGS[1] : "embedding_desc_ref.npz"

emb = M.read_mace_embedding(ENV["ACE_EMBEDDING"])
model = M.ace_embedding_model(elements = tuple(els...), order = order, totaldegree = tdeg,
                              embedding = emb, d_max = dmax, maxl = 6)
calc = model
zlist = [Int(M.atomic_number(z)) for z in model.model.rbasis._i2z]
dmb = model.model.meta["embedding"]["d_max"]
rows = M.embedding_rows(emb, zlist; d = dmb)

Random.seed!(20260924)
P, C, Z, Dsc, nat = Matrix{Float64}[], Matrix{Float64}[], Vector{Int32}[], Matrix{Float64}[], Int32[]
for s = 1:nstr
    sys = lat == :fcc ? bulk(:Ni, cubic = true, a = a0 * u"Å") * (2, 2, 2) : bulk(:Si, cubic = true, a = a0 * u"Å") * (2, 2, 2)
    n = length(sys)
    F = I + 0.03 * (2 .* rand(3, 3) .- 1); F = (F + F') / 2
    cell = reduce(hcat, [ustrip.(v) for v in cell_vectors(sys)])' * F       # rows = lattice vectors
    pos = reduce(hcat, [ustrip.(position(sys, i)) for i = 1:n])' * F .+ 0.08 .* randn(n, 3)
    zs = rand(zlist, n)
    atoms = [Atom(zs[i], SVector{3}(pos[i, :]...) .* u"Å") for i = 1:n]
    sysr = periodic_system(atoms, Tuple(SVector{3}(cell[k, :]...) .* u"Å" for k = 1:3))
    push!(P, pos); push!(C, cell); push!(Z, Int32.(zs)); push!(nat, Int32(n))
    push!(Dsc, reduce(hcat, ACEpotentials.site_descriptors(sysr, calc)))    # (len_basis, n)
end

m = model.model
gamma = collect(diag(M.algebraic_smoothness_prior(m)))
npzwrite(out, Dict{String, Any}(
    "zlist" => Int32.(zlist), "rows" => rows, "widths" => Int32.(model.model.meta["embedding"]["widths"]),
    "nat" => nat, "pos" => reduce(vcat, P), "cells" => reduce(vcat, C), "Z" => reduce(vcat, Z),
    "desc" => reduce(hcat, Dsc), "gamma" => gamma,
    "order" => Int32(order), "totaldegree" => Int32(tdeg), "dmax" => Int32(dmax === nothing ? 0 : dmax)))
println("wrote ", out, "  desc ", size(reduce(hcat, Dsc)))
