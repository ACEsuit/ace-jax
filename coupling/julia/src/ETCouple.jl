"""
    ETCouple

C entry point (compiled with `juliac --trim=safe`, see ../build.jl) for the L = 0,
real-basis SO(3) coupling of EquivariantTensors.  Stateless: every call computes
from its inputs and writes only into caller-owned buffers; no Julia object
outlives a call.  Two-phase: a call whose capacities (`sizes`) are too small
returns BUFFERS_TOO_SMALL with the required sizes written back.  Inputs are
validated first so no exception can unwind across the C boundary.
ABI: docs/coupling-etshim-spec.md.
"""
module ETCouple

import EquivariantTensors as ET
using SparseArrays: findnz

const ABI_VERSION = Cint(1)
const MAX_ORDER = 8                       # == ET.O3.MAX_STATIC_ORDER
const OK = Cint(0)
const BUFFERS_TOO_SMALL = Cint(1)
const INVALID_INPUT = Cint(2)
const ORDER_TOO_HIGH = Cint(3)
const NO_INVARIANTS = Cint(4)             # no body of mb_spec admits an L = 0 invariant
const INTERNAL_ERROR = Cint(5)            # anything else thrown by ET, or inconsistent sizes
const NSIZES = 7                          # [nnz, n_B, n_AA, n_sig, n_A, n_aaidx, n_nnll]

const NL = @NamedTuple{n::Int, l::Int}
const LM = @NamedTuple{l::Int, m::Int}
const NLM = @NamedTuple{n::Int, l::Int, m::Int}

"""Everything the ABI returns, 1-based (converted to 0-based on write)."""
struct Result
   nB::Int
   nAA::Int
   I::Vector{Int}                         # A2B triplets, findnz order
   J::Vector{Int}
   V::Vector{Float64}
   sig::Vector{Vector{NLM}}               # per A2B column: meta["𝔸spec"]
   aspec::Vector{Tuple{Int, Int}}         # abasis.spec: (Rnl_idx, Ylm_idx)
   aa::Vector{Vector{Int}}                # aabasis evaluation order (SparseSymmProd)
   nnll::Vector{Vector{NL}}               # get_nnll_spec(tensor, 1)
end

function validate(mb::Vector{Vector{NL}}, rnl::Vector{NL}, ylm::Vector{LM})::Cint
   (isempty(mb) || isempty(rnl) || isempty(ylm)) && return INVALID_INPUT
   length(unique(rnl)) == length(rnl) || return INVALID_INPUT
   length(unique(ylm)) == length(ylm) || return INVALID_INPUT
   for r in rnl
      r.l >= 0 || return INVALID_INPUT
   end
   for y in ylm
      (y.l >= 0 && abs(y.m) <= y.l) || return INVALID_INPUT
   end
   rset = Set{NL}(rnl)
   yset = Set{LM}(ylm)
   for bb in mb
      isempty(bb) && return INVALID_INPUT
      length(bb) > MAX_ORDER && return ORDER_TOO_HIGH
      for b in bb
         b.l >= 0 || return INVALID_INPUT
         b in rset || return INVALID_INPUT
         for m in -b.l:b.l
            (l = b.l, m = m) in yset || return INVALID_INPUT
         end
      end
   end
   return OK
end

# `nothing` when no body has an L = 0 invariant (ET would then throw reducing
# over an empty 𝔸spec; an exception escaping the C ABI aborts the host process)
function compute(mb::Vector{Vector{NL}}, rnl::Vector{NL}, ylm::Vector{LM})::Union{Result, Nothing}
   symm, 𝔸spec = ET.symmetrisation_matrix(Val(0), mb; prune = true, PI = true, basis = real)
   isempty(𝔸spec) && return nothing
   t = ET._tensor_specs(symm, 𝔸spec, rnl, ylm)      # == sparse_equivariant_tensor_spec(Val(0); ...)
   I, J, V = findnz(t.symm)
   nB, nAA = size(t.symm)
   # SparseSymmProd(𝔸spec_raw) evaluation order: the constructor's own
   # `sort(spec, by = length)` (stable); each body is already sorted by
   # _make_idx_AA_spec.
   aa = sort(t.𝔸spec_raw; by = length)
   # get_nnll_spec: (n, l) of each row's first stored column
   first_col = fill(typemax(Int), nB)
   for k in eachindex(I)
      J[k] < first_col[I[k]] && (first_col[I[k]] = J[k])
   end
   nnll = Vector{NL}[NL[(n = b.n, l = b.l) for b in t.𝔸spec[first_col[i]]] for i in 1:nB]
   aspec = Tuple{Int, Int}[(Int(p[1]), Int(p[2])) for p in t.Aspec_raw]
   return Result(nB, nAA, I, J, Vector{Float64}(V), t.𝔸spec, aspec, aa, nnll)
end

required_sizes(r::Result) = (length(r.V), r.nB, r.nAA, sum(length, r.sig; init = 0),
                             length(r.aspec), sum(length, r.aa; init = 0), sum(length, r.nnll; init = 0))

# read mb from CSR offsets; `nothing` if the offsets are malformed
function _read_mb(n_mb::Int64, off::Ptr{Int64}, pn::Ptr{Int64}, pl::Ptr{Int64})
   unsafe_load(off, 1) == 0 || return nothing
   mb = Vector{Vector{NL}}(undef, n_mb)
   for b in 1:n_mb
      lo = unsafe_load(off, b); hi = unsafe_load(off, b + 1)
      hi >= lo || return nothing
      body = Vector{NL}(undef, hi - lo)
      for k in 1:(hi - lo)
         body[k] = (n = Int(unsafe_load(pn, lo + k)), l = Int(unsafe_load(pl, lo + k)))
      end
      mb[b] = body
   end
   return mb
end

Base.@ccallable function etc_abi_version()::Cint
   return ABI_VERSION
end

Base.@ccallable function etc_max_order()::Cint
   return Cint(MAX_ORDER)
end

Base.@ccallable function etc_couple(
      n_mb::Int64, mb_off::Ptr{Int64}, mb_n::Ptr{Int64}, mb_l::Ptr{Int64},
      n_r::Int64, r_n::Ptr{Int64}, r_l::Ptr{Int64},
      n_y::Int64, y_l::Ptr{Int64}, y_m::Ptr{Int64},
      sizes::Ptr{Int64},
      a2b_i::Ptr{Int64}, a2b_j::Ptr{Int64}, a2b_v::Ptr{Float64},
      sig_off::Ptr{Int64}, sig::Ptr{Int64}, aspec::Ptr{Int64},
      aa_off::Ptr{Int64}, aa_idx::Ptr{Int64},
      nnll_off::Ptr{Int64}, nnll::Ptr{Int64})::Cint
   (n_mb < 1 || n_r < 1 || n_y < 1) && return INVALID_INPUT
   mb = _read_mb(n_mb, mb_off, mb_n, mb_l)
   mb === nothing && return INVALID_INPUT
   rnl = NL[(n = Int(unsafe_load(r_n, k)), l = Int(unsafe_load(r_l, k))) for k in 1:n_r]
   ylm = LM[(l = Int(unsafe_load(y_l, k)), m = Int(unsafe_load(y_m, k))) for k in 1:n_y]
   code = validate(mb, rnl, ylm)
   code == OK || return code
   r = try
      compute(mb, rnl, ylm)
   catch
      return INTERNAL_ERROR
   end
   r === nothing && return NO_INVARIANTS
   # the offset arrays below are sized from nAA / nB: never trust ET's shapes blindly
   (length(r.sig) == r.nAA && length(r.aa) == r.nAA && length(r.nnll) == r.nB) || return INTERNAL_ERROR
   need = required_sizes(r)
   short = false
   for k in 1:NSIZES
      unsafe_load(sizes, k) < need[k] && (short = true)
      unsafe_store!(sizes, need[k], k)
   end
   short && return BUFFERS_TOO_SMALL
   for k in eachindex(r.V)
      unsafe_store!(a2b_i, r.I[k] - 1, k); unsafe_store!(a2b_j, r.J[k] - 1, k); unsafe_store!(a2b_v, r.V[k], k)
   end
   p = 0; unsafe_store!(sig_off, 0, 1)
   for (c, bb) in enumerate(r.sig)
      for b in bb
         unsafe_store!(sig, b.n, 3p + 1); unsafe_store!(sig, b.l, 3p + 2); unsafe_store!(sig, b.m, 3p + 3)
         p += 1
      end
      unsafe_store!(sig_off, p, c + 1)
   end
   for (k, (ri, yi)) in enumerate(r.aspec)
      unsafe_store!(aspec, ri - 1, 2k - 1); unsafe_store!(aspec, yi - 1, 2k)
   end
   p = 0; unsafe_store!(aa_off, 0, 1)
   for (c, s) in enumerate(r.aa)
      for i in s
         p += 1; unsafe_store!(aa_idx, i - 1, p)
      end
      unsafe_store!(aa_off, p, c + 1)
   end
   p = 0; unsafe_store!(nnll_off, 0, 1)
   for (i, bb) in enumerate(r.nnll)
      for b in bb
         unsafe_store!(nnll, b.n, 2p + 1); unsafe_store!(nnll, b.l, 2p + 2)
         p += 1
      end
      unsafe_store!(nnll_off, p, i + 1)
   end
   return OK
end

end # module
