# Julia/ACEpotentials element-embedding reduction ORACLE (CI-only; the Python port
# is src/ace_jax/construct/embedding.py).  Dumps ACEpotentials' embedding_rows
# (:pca and :truncate, normalised or not) for several widths d -- below, at and
# above the rank, which exercises _pca_reduce's generic-frame branch -- plus
# _generic_frame itself and embedding_widths, so the port can be checked
# bit-for-bit where the maths is deterministic.
#
#   julia --project=<tooling providing ACEpotentials with Models.embedding_rows>
#         julia/embedding_reference.jl out.npz
# (ace_embedding_model / embedding_rows live on ACEpotentials' pr/element-embeddings
#  line; acejax/julia in that repo devs a checkout that has them.)
# ENV: ACE_EMBEDDING = JSON table (Z, emb) or "identity"; ACE_ELEMENTS (atomic
# numbers, comma-separated); ACE_DS widths (comma-separated).
using ACEpotentials, NPZ, LinearAlgebra
M = ACEpotentials.Models

zlist = parse.(Int, split(get(ENV, "ACE_ELEMENTS", "24,25,26,27,28"), ","))
src   = get(ENV, "ACE_EMBEDDING", "identity")
ds    = parse.(Int, split(get(ENV, "ACE_DS", "2,4,5,8,15"), ","))
out   = length(ARGS) >= 1 ? ARGS[1] : "embedding_ref.npz"

emb = src == "identity" ? M.identity_embedding(zlist) : M.read_mace_embedding(src)
idx = [findfirst(==(z), emb.Z) for z in zlist]
D = Dict{String, Any}("zlist" => Int32.(zlist), "table" => emb.emb[idx, :],
                      "ds" => Int32.(ds))
for d in ds
    D["pca_norm_d$d"]   = M.embedding_rows(emb, zlist; d = d, reduction = :pca, normalise = true)
    D["pca_raw_d$d"]    = M.embedding_rows(emb, zlist; d = d, reduction = :pca, normalise = false)
    if d <= size(emb.emb, 2)
        D["trunc_norm_d$d"] = M.embedding_rows(emb, zlist; d = d, reduction = :truncate, normalise = true)
    end
end
# the raw principal coordinates P = U S of the normalised block (rank-truncated):
# Julia's svd leaves each singular vector's SIGN to LAPACK, so rows for d > rank
# (P * frame) are LAPACK-dependent; the Python port fixes the sign (largest-|.|
# entry of each column of U positive) and is checked against canon(P) * frame.
let R = emb.emb[idx, :]
    R = R ./ [norm(R[i, :]) for i = 1:size(R, 1)]
    F = svd(R)
    tol = maximum(size(R)) * eps(Float64) * F.S[1]
    r = count(>(tol), F.S)
    D["pca_P"] = F.U[:, 1:r] .* F.S[1:r]'
end
# the generic frame on its own, for (r, d) with d > r
for (r, d) in ((3, 5), (5, 15), (5, 35), (2, 7))
    D["frame_r$(r)_d$(d)"] = M._generic_frame(r, d)
end
# per-order widths
S = length(zlist)
for order in (2, 3, 4), dmax in (0, 4, 16)
    D["widths_o$(order)_dmax$(dmax)"] = Int32.(M.embedding_widths(S, order; d_max = dmax == 0 ? nothing : dmax))
end
npzwrite(out, D)
println("wrote ", out)
