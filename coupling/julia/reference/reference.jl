# Oracle for ace-jax-coupling: UNPATCHED upstream EquivariantTensors at BASE_SHA
# (pinned in ./Project.toml [sources]).
#   julia +1.13 --project=coupling/julia/reference coupling/julia/reference/reference.jl \
#         coupling/python/tests/data/cases.json coupling/python/tests/data/et_reference.npz
import EquivariantTensors as ET
using JSON, NPZ, SparseArrays

# (k, n) columns -> (n, k) matrix, so numpy reads one row per entry
rows(v::Vector{Vector{Int64}}, k) = isempty(v) ? zeros(Int64, 0, k) : permutedims(reduce(hcat, v))

cases = JSON.parsefile(ARGS[1])
out = Dict{String, Any}()
for name in sort(collect(keys(cases)))
   c = cases[name]
   mb = [[(n = Int(b[1]), l = Int(b[2])) for b in bb] for bb in c["mb"]]
   R = [(n = Int(b[1]), l = Int(b[2])) for b in c["R"]]
   Y = [(l = Int(b[1]), m = Int(b[2])) for b in c["Y"]]
   t = ET.sparse_equivariant_tensor(L = 0, mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
   A2B = sparse(t.A2Bmaps[1])
   I, J, V = findnz(A2B)
   sig = t.meta["𝔸spec"]
   aa = [collect(s) for s in vcat(t.aabasis.specs...)]
   nnll = ET.get_nnll_spec(t, 1)
   cs(v) = Int64[0; cumsum(length.(v))]
   out["$(name)__A2B_shape"] = Int64[size(A2B)...]
   out["$(name)__A2B_rows"] = Int64.(I .- 1)
   out["$(name)__A2B_cols"] = Int64.(J .- 1)
   out["$(name)__A2B_vals"] = Float64.(V)
   out["$(name)__aa_sig_off"] = cs(sig)
   out["$(name)__aa_sig"] = rows([Int64[b.n, b.l, b.m] for bb in sig for b in bb], 3)
   out["$(name)__aspec"] = rows([Int64[r - 1, y - 1] for (r, y) in t.abasis.spec], 2)
   out["$(name)__aa_off"] = cs(aa)
   out["$(name)__aa_idx"] = Int64[i - 1 for s in aa for i in s]
   out["$(name)__nnll_off"] = cs(nnll)
   out["$(name)__nnll"] = rows([Int64[b.n, b.l] for bb in nnll for b in bb], 2)
   println(rpad(name, 18), " A2B ", size(A2B), " nnz=", length(V))
end
npzwrite(ARGS[2], out)
