# Oracle for etc_couple_equivariant (ABI v2): the same ET as reference.jl (upstream
# main plus ONLY the nullspace_solver commit, untrimmed, :dense), through the
# Integer API that upstream ships: symmetrisation_matrix(L, mb; refl_sym).  The
# fork's Val(L) path and its refl_sym forwarding are what is being checked.
#   julia +1.13 --project=coupling/julia/reference coupling/julia/reference/reference_equivariant.jl \
#         coupling/python/tests/data/cases_equivariant.json coupling/python/tests/data/et_reference_equivariant.npz
import EquivariantTensors as ET
using JSON, NPZ, SparseArrays

rows(v::Vector{Vector{Int64}}, k) = isempty(v) ? zeros(Int64, 0, k) : permutedims(reduce(hcat, v))

cases = JSON.parsefile(ARGS[1])
out = Dict{String, Any}()
for name in sort(collect(keys(cases)))
   c = cases[name]
   mb = [[(n = Int(b[1]), l = Int(b[2])) for b in bb] for bb in c["mb"]]
   R = [(n = Int(b[1]), l = Int(b[2])) for b in c["R"]]
   Y = [(l = Int(b[1]), m = Int(b[2])) for b in c["Y"]]
   for L in 0:Int(c["Lmax"]), (parity, rs) in ((1, :sym), (-1, :asym))
      key = "$(name)__L$(L)_p$(parity == 1 ? "e" : "o")"
      symm, sig = ET.symmetrisation_matrix(L, mb; prune = true, PI = true, basis = real,
                                           nullspace_solver = :dense, refl_sym = rs)
      if isempty(sig)
         out["$(key)__empty"] = Int64[1]
         println(rpad(key, 28), " empty")
         continue
      end
      Aspec = sort(unique(reduce(vcat, sig)))
      Aspec_raw = ET._make_idx_A_spec(Aspec, R, Y)
      AAspec_raw = ET._make_idx_AA_spec(sig, Aspec)
      aa = [collect(s) for s in vcat(ET.SparseSymmProd(AAspec_raw).specs...)]
      I, J, V = findnz(symm)
      Ie = Int64[]; Je = Int64[]; Me = Int64[]; Ve = Float64[]
      for k in eachindex(V), m in 1:(2L + 1)
         x = V[k][m]
         (L > 0 && abs(x) <= 1e-12) && continue       # == ETCouple.ZERO_TOL; L = 0 keeps etc_couple's entries
         push!(Ie, I[k] - 1); push!(Je, J[k] - 1); push!(Me, m - 1); push!(Ve, x)
      end
      first_col = fill(typemax(Int), size(symm, 1))
      for k in eachindex(I)
         J[k] < first_col[I[k]] && (first_col[I[k]] = J[k])
      end
      nnll = [[Int64[b.n, b.l] for b in sig[first_col[i]]] for i in 1:size(symm, 1)]
      cs(v) = Int64[0; cumsum(length.(v))]
      out["$(key)__A2B_shape"] = Int64[size(symm)...]
      out["$(key)__A2B_rows"] = Ie
      out["$(key)__A2B_cols"] = Je
      out["$(key)__A2B_m"] = Me
      out["$(key)__A2B_vals"] = Ve
      out["$(key)__aa_sig_off"] = cs(sig)
      out["$(key)__aa_sig"] = rows([Int64[b.n, b.l, b.m] for bb in sig for b in bb], 3)
      out["$(key)__aspec"] = rows([Int64[r - 1, y - 1] for (r, y) in Aspec_raw], 2)
      out["$(key)__aa_off"] = cs(aa)
      out["$(key)__aa_idx"] = Int64[i - 1 for s in aa for i in s]
      out["$(key)__nnll_off"] = cs(nnll)
      out["$(key)__nnll"] = rows([b for bb in nnll for b in bb], 2)
      println(rpad(key, 28), " A2B ", size(symm), " nnz=", length(Ve))
   end
end
npzwrite(ARGS[2], out)
