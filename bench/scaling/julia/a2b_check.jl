# The A2B identity check (common.jl `build_splined`), on its own so the test suite can
# exercise it without loading ACEpotentials.
#
# The coupling coefficients that fill A2B differ at the ULP level between Julia 1.11 and
# 1.12 (ace_SiGe_large.npz, written on 1.11.9: 72 of 3860 entries, <= 8.3e-16 relative,
# deterministic run to run, the basis not permuted).  So A2B is compared to roundoff, not
# bit for bit: entries with |v| < A2B_NOISE on a side are cancellation noise (structural
# zeros, which some coefficients only reach to roundoff) and are dropped; what remains must
# have the same sparsity pattern and agree within A2B_ULPS * eps * max|A2B|.  Every other
# identity field (specs, nnll, splines, weights) stays exact.
using SparseArrays

const A2B_NOISE = 1e-12
const A2B_ULPS = 4

function _kept(A)
    I, J, V = findnz(sparse(A))
    k = abs.(V) .>= A2B_NOISE
    return sparse(I[k], J[k], V[k], Base.size(A)...)
end

"""
    a2b_compare(A, B) -> (ok, max_abs, max_rel)

`ok`: same shape, the same pattern once |v| < A2B_NOISE entries are dropped, and
max |A - B| <= A2B_ULPS * eps() * max|B|.  max_abs and max_rel (per entry, over the kept
entries of B) are reported either way; Inf when the shape or pattern differs.
"""
function a2b_compare(A, B)
    Base.size(A) == Base.size(B) || return (false, Inf, Inf)
    Ak, Bk = _kept(A), _kept(B)
    (ia, ja, va), (ib, jb, vb) = findnz(Ak), findnz(Bk)
    (ia == ib && ja == jb) || return (false, Inf, Inf)
    d = abs.(va .- vb)
    scale = maximum(abs.(vb); init = 0.0)
    max_abs = maximum(d; init = 0.0)
    max_rel = maximum(d ./ abs.(vb); init = 0.0)
    # the dropped noise entries must stay noise: compare the full matrices too
    max_abs_all = maximum(abs.(sparse(A) .- sparse(B)); init = 0.0)
    ok = max_abs <= A2B_ULPS * eps(Float64) * scale && max_abs_all <= A2B_NOISE * 2
    return (ok, max(max_abs, max_abs_all), max_rel)
end
