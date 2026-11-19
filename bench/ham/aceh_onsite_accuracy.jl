# ACEhamiltonians.jl on the on-site accuracy benchmark of bench/ham/onsite_accuracy.py: the same
# training and test atoms (environments and blocks from `onsite_accuracy.py export`, FHI-aims' own
# phases, which is what ACEhamiltonians.jl is written for), one sub-model per shell pair, fitted
# as ACEhamiltonians' `fit!(submodel, data; enable_mean = true)` does (same assembly, mean, scaling
# and solver), then predicted with its `predict`.
#   julia +1.10 --startup-file=no --project=<ACEhamiltonians.jl checkout> bench/ham/aceh_onsite_accuracy.jl \
#         <al_onsite.h5> <order> <degree> [solver = LSQR|QR] [n_atoms_timed = 64]
using ACEhamiltonians, HDF5, StaticArrays, LinearAlgebra, Statistics, ACE
using ACEhamiltonians.Bases: on_site_ace_basis, SubModel
using ACEhamiltonians.States: AtomState
using ACEhamiltonians.Predicting: predict
using ACEhamiltonians.Fitting: _evaluate_real, solve_ls
using ACE: ACEConfig, evaluate, scaling

function load(f, tag)
   V = read(f["$tag/vectors"]); off = read(f["$tag/offsets"]); B = read(f["$tag/blocks"])
   V = size(V, 1) == 3 ? V : permutedims(V)            # HDF5.jl reverses numpy's axes: (3, n_edges)
   B = permutedims(B, (3, 2, 1))                       # (atom, row, column)
   states = [[AtomState(SVector{3, Float64}(V[:, k])) for k in (off[i] + 1):off[i + 1]] for i in 1:(length(off) - 1)]
   return states, B
end

function main(args)
   path, ν, deg = args[1], parse(Int, args[2]), parse(Int, args[3])
   solver = length(args) >= 4 ? args[4] : "LSQR"
   ntime = length(args) >= 5 ? parse(Int, args[5]) : 64
   f = h5open(path)
   shells = Int.(read(f["shells"]))
   train, Htrain = load(f, "train"); test, Htest = load(f, "test")
   offs = cumsum([0; 2 .* shells .+ 1])
   sq = Dict{String, Vector{Float64}}(); sqtr = 0.0; ntr = 0
   tpred = 0.0; nfun = 0; tfit = 0.0
   for a in 1:length(shells), b in a:length(shells)
      l1, l2 = shells[a], shells[b]
      ra, rb = (offs[a] + 1):offs[a + 1], (offs[b] + 1):offs[b + 1]
      basis = on_site_ace_basis(l1, l2, ν, deg, 9.0, 2.5)
      sub = SubModel(basis, (13, a, b))
      tfit += @elapsed begin
         A = permutedims(reduce(hcat, _evaluate_real.(evaluate.(Ref(basis), ACEConfig.(train)))), (2, 1))
         Y = [Htrain[i, ra, rb] for i in 1:length(train)]
         x̄ = l1 == l2 ? mean(diag(mean(Y))) * I(2l1 + 1) : zeros(2l1 + 1, 2l2 + 1)   # fit!'s enable_mean
         Y .-= Ref(x̄)
         sub.mean .= x̄
         sub.coefficients .= collect(solve_ls(A, Y, 1e-7, Diagonal(scaling(basis, 2)), solver))
      end
      ptr = predict(sub, train); pte = predict(sub, test)
      for i in 1:length(train); sqtr += sum(abs2, ptr[:, :, i] - Htrain[i, ra, rb]); ntr += length(ra) * length(rb); end
      key = string("spd"[l1 + 1], "spd"[l2 + 1])
      acc = get!(sq, key, [0.0, 0.0])
      for i in 1:length(test); acc[1] += sum(abs2, pte[:, :, i] - Htest[i, ra, rb]); acc[2] += length(ra) * length(rb); end
      tpred += minimum(@elapsed(predict(sub, test[1:ntime])) for _ in 1:2) / ntime
      nfun += length(basis)
   end
   tot = sqrt(sum(v[1] for v in values(sq)) / sum(v[2] for v in values(sq)))        # shell pairs s1 <= s2 once
   println("ACEhamiltonians order $ν degree $deg solver $solver: $nfun basis functions over ",
           div(length(shells) * (length(shells) + 1), 2), " sub-models; fit ", round(tfit; digits = 1), " s; ",
           "predict ", round(1e6 * tpred; digits = 1), " us per atom (1 thread, states prebuilt)")
   println("   test RMSE by block type (meV): ", Dict(k => round(1e3 * sqrt(v[1] / v[2]); digits = 3) for (k, v) in sq))
   println("   RMSE over shell pairs s1 <= s2 (meV): train ", round(1e3 * sqrt(sqtr / ntr); digits = 3),
           " test ", round(1e3 * tot; digits = 3))
end

main(ARGS)
