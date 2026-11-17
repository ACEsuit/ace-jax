# ACEhamiltonians.jl's off-site (bond-environment) model on the bonds of
# bench/ham/offsite_accuracy.py (`export`): the same training and test bonds, blocks in
# FHI-aims' own phases.  One sub-model per shell pair s1 <= s2, built as `Model` builds them
# (a dual-basis AnisoSubModel when the shells differ), fitted with its own `fit!` and predicted
# with its own `predict`.
#   julia +1.10 --startup-file=no --project=<ACEhamiltonians.jl checkout> bench/ham/aceh_offsite_accuracy.jl \
#         <al_offsite.h5> <order> <degree> <bond cutoff> <environment cutoff> [channel = H] [n timed = 100]
using ACEhamiltonians, HDF5, JuLIP, StaticArrays, LinearAlgebra, Statistics
using ACEhamiltonians.Bases: off_site_ace_basis, AHSubModel, envelope
using ACEhamiltonians.States: get_state
using ACEhamiltonians.DataSets: DataSet
using ACEhamiltonians.Fitting: fit!
using ACEhamiltonians.Predicting: predict

function structures(f, tag)
   out = []
   for k in sort(collect(keys(f[tag])))
      g = f["$tag/$k"]
      pos = read(g["positions"]); cell = read(g["cell"])            # HDF5.jl reverses numpy's axes
      X = [SVector{3, Float64}(pos[:, a]) for a in 1:size(pos, 2)]
      atoms = Atoms(; X = X, Z = fill(AtomicNumber(13), length(X)), cell = Matrix(cell'), pbc = true)
      bonds = read(g["bonds"])                                        # (5, n): Rx, Ry, Rz, i, j (0-based)
      push!(out, (atoms, bonds, g))
   end
   return out
end

blocks(g, ch) = permutedims(read(g[ch]), (3, 2, 1))                   # (bond, row, column)

function states_of(set, env)
   st = Vector{Any}()
   for (atoms, bonds, _) in set, e in 1:size(bonds, 2)
      push!(st, get_state(bonds[4, e] + 1, bonds[5, e] + 1, atoms, env, bonds[1:3, e]))
   end
   return [s for s in st]
end

function main(args)
   path, ν, deg = args[1], parse(Int, args[2]), parse(Int, args[3])
   bcut, ecut = parse(Float64, args[4]), parse(Float64, args[5])
   ch = length(args) >= 6 ? args[6] : "H"
   ntime = length(args) >= 7 ? parse(Int, args[7]) : 100
   f = h5open(path)
   shells = Int.(read(f["shells"]))
   train, test = structures(f, "train"), structures(f, "test")
   Htr = cat([blocks(g, ch) for (_, _, g) in train]...; dims = 1)
   Hte = cat([blocks(g, ch) for (_, _, g) in test]...; dims = 1)
   idx(set) = hcat([vcat(b[4:5, :] .+ 1, ones(Int, 1, size(b, 2))) for (_, b, _) in set]...)
   offs = cumsum([0; 2 .* shells .+ 1])
   sq = Dict{String, Vector{Float64}}(); sqtr = 0.0; ntr = 0.0
   tpred = 0.0; nfun = 0; tfit = 0.0
   str = nothing; ste = nothing
   for a in 1:length(shells), b in a:length(shells)
      l1, l2 = shells[a], shells[b]
      ra, rb = (offs[a] + 1):offs[a + 1], (offs[b] + 1):offs[b + 1]
      basis = off_site_ace_basis(l1, l2, ν, deg, bcut, ecut)
      sub = a == b ? AHSubModel(basis, (13, 13, a, b)) :
                     AHSubModel(basis, off_site_ace_basis(l2, l1, ν, deg, bcut, ecut), (13, 13, a, b))
      if str === nothing                                              # the states depend only on the envelope
         str = states_of(train, envelope(sub)); ste = states_of(test, envelope(sub))
         println(length(str), " training bonds, ", length(ste), " test bonds, ",
                 round(mean(length, str) - 1; digits = 1), " environment atoms per bond")
      end
      vals = permutedims(Htr[:, ra, rb], (2, 3, 1))                   # (n1, n2, bond)
      tfit += @elapsed fit!(sub, DataSet(vals, idx(train), str))
      ptr = predict(sub, str); pte = predict(sub, ste)
      for i in 1:length(str); sqtr += sum(abs2, ptr[:, :, i] - Htr[i, ra, rb]); ntr += length(ra) * length(rb); end
      acc = get!(sq, string("spd"[l1 + 1], "spd"[l2 + 1]), [0.0, 0.0])
      for i in 1:length(ste); acc[1] += sum(abs2, pte[:, :, i] - Hte[i, ra, rb]); acc[2] += length(ra) * length(rb); end
      tpred += minimum(@elapsed(predict(sub, ste[1:ntime])) for _ in 1:2) / ntime
      nfun += length(basis) * (a == b ? 1 : 2)
   end
   tot = sqrt(sum(v[1] for v in values(sq)) / sum(v[2] for v in values(sq)))
   println("ACEhamiltonians off-site $ch, order $ν degree $deg, bond cutoff $bcut, environment cutoff $ecut: ",
           "$nfun basis functions over ", div(length(shells) * (length(shells) + 1), 2), " sub-models; fit ",
           round(tfit; digits = 0), " s; predict ", round(1e6 * tpred; digits = 0),
           " us per bond for the canonical shell pairs (1 thread, states prebuilt)")
   println("   RMSE over canonical blocks (meV; S x 1e3): train ", round(1e3 * sqrt(sqtr / ntr); digits = 4),
           " test ", round(1e3 * tot; digits = 4))
   println("   test by block type: ", Dict(k => round(1e3 * sqrt(v[1] / v[2]); digits = 4) for (k, v) in sq))
end

main(ARGS)
