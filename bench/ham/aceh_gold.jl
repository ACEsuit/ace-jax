# ACEhamiltonians.jl fitted to the bulk gold data of bench/ham/aceh_gold.py (`export`) and its
# predictions of every block of the test structures, for `aceh_gold.py collect`.  One sub-model
# per shell pair s1 <= s2, on-site (fitted as `fit!(...; enable_mean = true)` does: its assembly,
# mean, scaling and LSQR, lambda 1e-7) and off-site (its own `fit!` and `predict`; a dual basis
# when the shells differ), as bench/ham/aceh_onsite_accuracy.jl and aceh_offsite_accuracy.jl.
#   julia +1.10 -t 1 --startup-file=no --project=<ACEhamiltonians.jl checkout> bench/ham/aceh_gold.jl \
#         <gold.h5> <predictions.h5> <on-site order> <on-site degree> <off-site order> <off-site degree> \
#         [bond cutoff = 10.0] [environment cutoff = 5.0] [r0 = 2.9] [n test = all]
# Timing: the first test structure, states (its neighbour search) and every sub-model's predict,
# on the threads Julia was started with; the smaller of two runs.
using ACEhamiltonians, HDF5, JuLIP, StaticArrays, LinearAlgebra, Statistics, ACE
using ACEhamiltonians.Bases: on_site_ace_basis, off_site_ace_basis, SubModel, AHSubModel, envelope
using ACEhamiltonians.States: AtomState, get_state
using ACEhamiltonians.DataSets: DataSet
using ACEhamiltonians.Fitting: fit!, _evaluate_real, solve_ls
using ACEhamiltonians.Predicting: predict
using ACE: ACEConfig, evaluate, scaling

function structure(g)
   pos = read(g["positions"]); cell = read(g["cell"]); Z = Int.(read(g["numbers"]))     # HDF5.jl reverses numpy's axes
   X = [SVector{3, Float64}(pos[:, a]) for a in 1:size(pos, 2)]
   atoms = Atoms(; X = X, Z = AtomicNumber.(Z), cell = Matrix(cell'), pbc = true)
   V = read(g["vectors"]); off = read(g["offsets"])
   return (atoms = atoms, bonds = read(g["bonds"]), V = V, off = off, g = g, Z = Z[1])
end

blocks(g, name) = permutedims(read(g[name]), (3, 2, 1))               # (item, row, column)
site_states(s) = [[AtomState(SVector{3, Float64}(s.V[:, k])) for k in (s.off[i] + 1):s.off[i + 1]] for i in 1:(length(s.off) - 1)]
bond_states(s, env) = [st for st in Any[get_state(s.bonds[4, e] + 1, s.bonds[5, e] + 1, s.atoms, env, s.bonds[1:3, e]) for e in 1:size(s.bonds, 2)]]

function main(args)
   path, out = args[1], args[2]
   νon, don, νoff, doff = parse.(Int, args[3:6])
   bcut = length(args) >= 7 ? parse(Float64, args[7]) : 10.0
   ecut = length(args) >= 8 ? parse(Float64, args[8]) : 5.0
   r0 = length(args) >= 9 ? parse(Float64, args[9]) : 2.9
   f = h5open(path)
   shells = Int.(read(f["shells"])); rcut = read(attributes(f)["rcut"])
   train = [structure(f["train/$k"]) for k in sort(collect(keys(f["train"])))]
   test = [structure(f["test/$k"]) for k in sort(collect(keys(f["test"])))]
   length(args) >= 10 && (test = test[1:parse(Int, args[10])])
   z = train[1].Z
   offs = cumsum([0; 2 .* shells .+ 1]); no = offs[end]
   pairs = [(a, b) for a in 1:length(shells) for b in a:length(shells)]
   # on-site
   st_on = reduce(vcat, site_states.(train)); Hon = cat([blocks(s.g, "onsite") for s in train]...; dims = 1)
   on = Dict(); nfun = 0; tfit = 0.0
   for (a, b) in pairs
      l1, l2 = shells[a], shells[b]; ra, rb = (offs[a] + 1):offs[a + 1], (offs[b] + 1):offs[b + 1]
      basis = on_site_ace_basis(l1, l2, νon, don, rcut, r0)
      sub = SubModel(basis, (z, a, b))
      tfit += @elapsed begin
         A = permutedims(reduce(hcat, _evaluate_real.(evaluate.(Ref(basis), ACEConfig.(st_on)))), (2, 1))
         Y = [Hon[i, ra, rb] for i in 1:length(st_on)]
         x̄ = l1 == l2 ? mean(diag(mean(Y))) * I(2l1 + 1) : zeros(2l1 + 1, 2l2 + 1)
         Y .-= Ref(x̄); sub.mean .= x̄
         sub.coefficients .= collect(solve_ls(A, Y, 1e-7, Diagonal(scaling(basis, 2)), "LSQR"))
      end
      on[(a, b)] = sub; nfun += length(basis)
   end
   # off-site
   off = Dict(); st_tr = nothing; env = nothing
   Hoff = cat([blocks(s.g, "H") for s in train]...; dims = 1)
   idx = hcat([vcat(s.bonds[4:5, :] .+ 1, ones(Int, 1, size(s.bonds, 2))) for s in train]...)
   for (a, b) in pairs
      l1, l2 = shells[a], shells[b]; ra, rb = (offs[a] + 1):offs[a + 1], (offs[b] + 1):offs[b + 1]
      basis = off_site_ace_basis(l1, l2, νoff, doff, bcut, ecut)
      sub = a == b ? AHSubModel(basis, (z, z, a, b)) : AHSubModel(basis, off_site_ace_basis(l2, l1, νoff, doff, bcut, ecut), (z, z, a, b))
      if st_tr === nothing
         env = envelope(sub); st_tr = reduce(vcat, bond_states.(train, Ref(env)))
      end
      tfit += @elapsed fit!(sub, DataSet(permutedims(Hoff[:, ra, rb], (2, 3, 1)), idx, st_tr))
      off[(a, b)] = sub; nfun += length(basis) * (a == b ? 1 : 2)
   end
   println("ACEhamiltonians.jl gold: on-site order $νon degree $don (cutoff $rcut), off-site order $νoff degree $doff (bond $bcut, environment $ecut); ",
           "$nfun basis functions over ", 2 * length(pairs), " sub-models; ", length(st_on), " atoms and ", length(st_tr), " bonds; fit ", round(tfit; digits = 0), " s")
   flush(stdout)
   function predict_structure(s)
      Pon = zeros(length(s.off) - 1, no, no); Poff = zeros(size(s.bonds, 2), no, no)
      son = site_states(s); sb = bond_states(s, env)
      for (a, b) in pairs
         ra, rb = (offs[a] + 1):offs[a + 1], (offs[b] + 1):offs[b + 1]
         p = predict(on[(a, b)], son); for i in 1:length(son); Pon[i, ra, rb] = p[:, :, i]; end
         p = predict(off[(a, b)], sb); for i in 1:length(sb); Poff[i, ra, rb] = p[:, :, i]; end
      end
      return Pon, Poff
   end
   t = minimum(@elapsed(predict_structure(test[1])) for _ in 1:2)
   n = length(test[1].off) - 1
   println("predict: ", round(t; digits = 2), " s per structure, ", round(1e3 * t / n; digits = 1), " ms per atom (", Threads.nthreads(),
           " thread, ", n, " atoms, ", size(test[1].bonds, 2), " bonds, states included)")
   flush(stdout)
   h5open(out, "w") do o
      for (k, s) in enumerate(test)
         Pon, Poff = predict_structure(s)
         g = create_group(o, "test/" * lpad(k - 1, 3, "0"))
         attributes(g)["name"] = read(attributes(s.g)["name"])
         g["bonds"] = permutedims(s.bonds)                                # written (n, 5) for numpy
         g["pred_on"] = permutedims(Pon, (3, 2, 1)); g["pred_off"] = permutedims(Poff, (3, 2, 1))
         e = Poff .- blocks(s.g, "H"); m = 0.0; c = 0
         for (a, b) in pairs
            ra, rb = (offs[a] + 1):offs[a + 1], (offs[b] + 1):offs[b + 1]; m += sum(abs, e[:, ra, rb]); c += length(e[:, ra, rb])
         end
         println("  test $k: off-site MAE over shell pairs s1 <= s2 ", round(1e3 * m / c; digits = 3), " meV"); flush(stdout)
      end
   end
end

main(ARGS)
