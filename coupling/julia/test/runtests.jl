# Uncompiled checks of the shim: `compute` against the real (patched-fork) ET
# objects, the validation codes, and the two-phase C entry called from Julia.
#   julia +1.13 --project=coupling/julia coupling/julia/test/runtests.jl
using Test
import EquivariantTensors as ET
include(joinpath(@__DIR__, "..", "src", "ETCouple.jl"))
using .ETCouple: validate, compute, required_sizes, etc_couple, OK, BUFFERS_TOO_SMALL, INVALID_INPUT, ORDER_TOO_HIGH

nl(n, l) = (n = n, l = l); lm(l, m) = (l = l, m = m)
ylm(L) = [lm(l, m) for l in 0:L for m in -l:l]
function bodies(maxn, maxl, ord)                     # sorted bodies, even ∑l
   ch = [nl(n, l) for l in 0:maxl for n in 1:maxn]
   mb = Vector{ETCouple.NL}[]; bb = ETCouple.NL[]
   function rec(s)
      !isempty(bb) && iseven(sum(b.l for b in bb)) && push!(mb, copy(bb))
      length(bb) == ord && return
      for i in s:length(ch); push!(bb, ch[i]); rec(i); pop!(bb); end
   end
   rec(1)
   return mb
end
CASES = [(2, 2, 2), (2, 2, 3), (1, 2, 4), (1, 0, 8)]

@testset "compute == sparse_equivariant_tensor" begin
   for (maxn, maxl, ord) in CASES
      mb = bodies(maxn, maxl, ord); R = [nl(n, l) for l in 0:maxl for n in 1:maxn]; Y = ylm(maxl)
      @test validate(mb, R, Y) == OK
      r = compute(mb, R, Y)
      t = ET.sparse_equivariant_tensor(L = 0, mb_spec = mb, Rnl_spec = R, Ylm_spec = Y, basis = real)
      A = Matrix(t.A2Bmaps[1])
      B = zeros(r.nB, r.nAA); for k in eachindex(r.V); B[r.I[k], r.J[k]] = r.V[k]; end
      @test B == A
      @test r.sig == t.meta["𝔸spec"]
      @test r.aspec == collect(t.abasis.spec)
      @test r.aa == [collect(s) for s in vcat(t.aabasis.specs...)]      # SparseSymmProd order
      @test r.nnll == ET.get_nnll_spec(t, 1)
   end
end

@testset "validate" begin
   R = [nl(1, 0), nl(1, 1)]; Y = ylm(1)
   @test validate([[nl(1, 0)]], R, Y) == OK
   @test validate(Vector{ETCouple.NL}[], R, Y) == INVALID_INPUT          # empty mb
   @test validate([ETCouple.NL[]], R, Y) == INVALID_INPUT                # empty body
   @test validate([[nl(2, 0)]], R, Y) == INVALID_INPUT                   # (2,0) not in Rnl
   @test validate([[nl(1, 1), nl(1, 1)]], R, [lm(0, 0), lm(1, 0)]) == INVALID_INPUT   # (1,±1) missing
   @test validate([[nl(1, 0)]], [nl(1, 0), nl(1, 0)], Y) == INVALID_INPUT             # duplicate Rnl
   @test validate([[nl(1, 0)]], R, [lm(0, 0), lm(0, 0)]) == INVALID_INPUT             # duplicate Ylm
   @test validate([[nl(1, 0)]], R, [lm(1, 2)]) == INVALID_INPUT                       # |m| > l
   @test validate([[nl(1, -1)]], [nl(1, -1)], Y) == INVALID_INPUT                     # l < 0
   @test validate([[nl(1, 0) for _ in 1:9]], R, Y) == ORDER_TOO_HIGH
end

@testset "etc_couple two-phase from Julia" begin
   mb = [[nl(1, 0)], [nl(1, 1), nl(1, 1)]]; R = [nl(1, 0), nl(1, 1), nl(2, 0)]; Y = ylm(1)
   off = Int64[0, 1, 3]; mn = Int64[1, 1, 1]; ml = Int64[0, 1, 1]
   rn = Int64[1, 1, 2]; rl = Int64[0, 1, 0]; yl = Int64[y.l for y in Y]; ym = Int64[y.m for y in Y]
   sizes = zeros(Int64, 7); z = zeros(Int64, 1); zf = zeros(Float64, 1)
   call(ai, aj, av, so, sg, asp, ao, ax, no, nn) = GC.@preserve off mn ml rn rl yl ym sizes ai aj av so sg asp ao ax no nn etc_couple(
      Int64(2), pointer(off), pointer(mn), pointer(ml), Int64(3), pointer(rn), pointer(rl),
      Int64(length(Y)), pointer(yl), pointer(ym), pointer(sizes),
      pointer(ai), pointer(aj), pointer(av), pointer(so), pointer(sg), pointer(asp),
      pointer(ao), pointer(ax), pointer(no), pointer(nn))
   @test call(z, z, zf, z, z, z, z, z, z, z) == BUFFERS_TOO_SMALL
   r = compute(mb, R, Y)
   @test Tuple(sizes) == required_sizes(r)
   nnz, nB, nAA, nsig, nA, naa, nnl = sizes
   bufs = (zeros(Int64, nnz), zeros(Int64, nnz), zeros(nnz), zeros(Int64, nAA + 1), zeros(Int64, 3nsig),
           zeros(Int64, 2nA), zeros(Int64, nAA + 1), zeros(Int64, naa), zeros(Int64, nB + 1), zeros(Int64, 2nnl))
   @test call(bufs...) == OK
   @test bufs[1] == r.I .- 1 && bufs[2] == r.J .- 1 && bufs[3] == r.V
   @test bufs[4][end] == nsig && bufs[7][end] == naa && bufs[9][end] == nnl
   bad_off = Int64[0, 2, 1]                                               # non-monotone offsets (in bounds)
   @test GC.@preserve bad_off mn ml rn rl yl ym sizes etc_couple(Int64(2), pointer(bad_off), pointer(mn), pointer(ml),
      Int64(3), pointer(rn), pointer(rl), Int64(length(Y)), pointer(yl), pointer(ym), pointer(sizes),
      Ptr{Int64}(C_NULL), Ptr{Int64}(C_NULL), Ptr{Float64}(C_NULL), Ptr{Int64}(C_NULL), Ptr{Int64}(C_NULL),
      Ptr{Int64}(C_NULL), Ptr{Int64}(C_NULL), Ptr{Int64}(C_NULL), Ptr{Int64}(C_NULL), Ptr{Int64}(C_NULL)) == INVALID_INPUT
end
