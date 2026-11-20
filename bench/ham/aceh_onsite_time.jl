# On-site prediction time of ACEhamiltonians.jl, for comparison with bench/ham/onsite_compare.py.
# ACEhamiltonians has one sub-model (its own ACE basis) per shell pair, so the cost of an atom is
# the sum over shell pairs.  Each distinct (l1, l2) is timed once, on the atoms of the structure
# file, and weighted by the number of shell pairs of that type; coefficients are random.
#   julia +1.10 --startup-file=no --project=<ACEhamiltonians.jl checkout> bench/ham/aceh_onsite_time.jl \
#         <structure.txt> <order> <degree> <rcut> [n_atoms_timed]
# structure.txt: 3 lattice-vector lines, then one "x y z" line per (silicon) atom.
using ACEhamiltonians, JuLIP, StaticArrays, LinearAlgebra, Random
using ACEhamiltonians.Bases: on_site_ace_basis, SubModel
using ACEhamiltonians.States: get_state
using ACEhamiltonians.Predicting: predict

const SHELLS = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]          # the Si basis of Quoll's SiC data (30 orbitals)

function main(args)
   path, ν, deg, rcut = args[1], parse(Int, args[2]), parse(Int, args[3]), parse(Float64, args[4])
   lines = [parse.(Float64, split(l)) for l in readlines(path) if !isempty(strip(l))]
   cell = hcat(lines[1:3]...)'                         # rows are lattice vectors
   X = [SVector{3}(l) for l in lines[4:end]]
   atoms = Atoms(; X = X, Z = fill(AtomicNumber(14), length(X)), cell = Matrix(cell), pbc = true)
   ntime = length(args) >= 5 ? parse(Int, args[5]) : length(X)
   states = [get_state(i, atoms; r = rcut) for i in 1:ntime]
   println(length(X), " atoms, ", sum(length, states) / ntime, " neighbours, timing ", ntime,
           " atoms, threads ", Threads.nthreads(), ", BLAS threads ", BLAS.get_num_threads())
   count = Dict{Tuple{Int, Int}, Int}()
   for a in 1:length(SHELLS), b in a:length(SHELLS)
      k = (SHELLS[a], SHELLS[b]); count[k] = get(count, k, 0) + 1
   end
   total, nfun = 0.0, 0
   rng = MersenneTwister(0)
   for (l1, l2) in sort(collect(keys(count)))
      basis = on_site_ace_basis(l1, l2, ν, deg, rcut, 2.5)
      sub = SubModel(basis, (14, 1, 1))
      sub.coefficients .= randn(rng, length(sub.coefficients))
      predict(sub, states[1:1])                        # compile
      t = minimum(@elapsed(predict(sub, states)) for _ in 1:3) / ntime
      total += count[(l1, l2)] * t; nfun += count[(l1, l2)] * length(basis)
      println("  (l1, l2) = ($l1, $l2): ", length(basis), " basis functions, ", count[(l1, l2)],
              " shell pairs, ", round(1e6 * t; digits = 1), " us per atom each")
   end
   println("on-site H, all ", sum(values(count)), " shell pairs: ", round(1e6 * total; digits = 1),
           " us per atom (", nfun, " basis functions in total)")
end

main(ARGS)
