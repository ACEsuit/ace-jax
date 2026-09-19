# Exporter: fit a model in Julia, write it to npz for the JAX
# evaluator.  See docs/plans/jax_ace_port_plan.md.
#
#   julia --project=julia julia/export_model.jl [output.npz] [ace1|ace]
#
# Exports the SPLINED radial branch, which is what ace1_model actually builds
# (src/ace1_compat.jl:283) and therefore what a fitted production model is.
# Reproducing Julia's own splines means JAX and Julia agree bit-for-bit rather
# than differing by a re-tabulation error.
#
# npz, not JSON: Julia writes matrices column-major, so 2-D arrays round-trip
# through JSON transposed.  Every array below is written in the orientation the
# Python loader expects and checked by tests/test_roundtrip.py.

using ACEpotentials, NPZ, JSON, StaticArrays, LinearAlgebra, Random, Printf
using AtomsCalculators, ACEfit
using SparseArrays
using Lux
using LazyArtifacts, ExtXYZ, AtomsBase, Unitful
M = ACEpotentials.Models

# usage: export_model.jl [out.npz] [ace1|ace|embedding]
#
# `embedding` builds a frozen-element-embedding model (ACE_EMBEDDING must point
# at the artefact; ACE_DMAX optionally caps the channel width).  Nothing on the
# JAX side needs to know: the embedding is baked into the radial splines, so the
# export is an ordinary splined model with a wider radial basis.
const OUT  = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "si_fitted.npz")
const KIND = length(ARGS) >= 2 ? ARGS[2] : "ace1"
@assert KIND in ("ace1", "ace", "embedding") "model kind must be ace1, ace or embedding"

# ---------------------------------------------------------------- fit
# overridable, so a benchmark can match a reference potential
elements = Symbol.(split(get(ENV, "ACE_ELEMENTS", "Si"), ","))
order       = parse(Int, get(ENV, "ACE_ORDER", "3"))
totaldegree = parse(Int, get(ENV, "ACE_TOTALDEGREE", "10"))
rcut_kw     = haskey(ENV, "ACE_RCUT") ? parse(Float64, ENV["ACE_RCUT"]) : nothing
# maxl matters: real ACE potentials use lmax 3-6 with correlation order 4-5, not
# high lmax at low order.  ace1_model derives lmax from totaldegree; ace_model
# lets it be set, which is why the benchmark series uses ace_model.
maxl_kw     = parse(Int, get(ENV, "ACE_MAXL", "6"))
# ACE_EREF="Si:-158.54496821,Ge:-..." sets Eref so E0 is the isolated-atom
# reference energy (the tutorial setting); without it E0 = 0 and an isolated
# atom in the training set carries a residual no edge-based model can absorb.
eref_kw = haskey(ENV, "ACE_EREF") ?
    [Symbol(split(p, ":")[1]) => parse(Float64, split(p, ":")[2]) for p in split(ENV["ACE_EREF"], ",")] :
    nothing
if KIND == "ace1"
    @info "building ace1_model(elements=$elements, order=$order, totaldegree=$totaldegree)"
    ace1_kwargs = Dict{Symbol, Any}(:elements => elements, :order => order, :totaldegree => totaldegree)
    rcut_kw === nothing || (ace1_kwargs[:rcut] = rcut_kw)
    eref_kw === nothing || (ace1_kwargs[:Eref] = eref_kw)
    model = ace1_model(; ace1_kwargs...)
elseif KIND == "ace"
    # ace_model: LEARNABLE (analytic) rbasis, solid harmonics.  Its pair basis is
    # still splined (ace_heuristics.jl:213), so the branches are per-basis.
    @info "building ace_model(elements=$elements, order=$order, max_level=$totaldegree, Ytype=:solid)"
    rcut0 = rcut_kw === nothing ? 5.5 : rcut_kw
    ri = M._default_rin0cuts(tuple(elements...))
    ri = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut0)).(ri)
    raw = M.ace_model(; elements = tuple(elements...), order = order, Ytype = :solid,
                      level = M.TotalDegree(), max_level = totaldegree, maxl = maxl_kw,
                      pair_maxn = totaldegree, rin0cuts = ri,
                      init_WB = :glorot_normal, init_Wpair = :glorot_normal)
    ps0, st0 = Lux.setup(MersenneTwister(1234), raw)
    model = M.ACEPotential(raw, ps0, st0)
end

if KIND == "embedding"
    haskey(ENV, "ACE_EMBEDDING") ||
        error("KIND=embedding needs ACE_EMBEDDING pointing at the artefact")
    emb = M.read_mace_embedding(ENV["ACE_EMBEDDING"])
    d_max = haskey(ENV, "ACE_DMAX") ? parse(Int, ENV["ACE_DMAX"]) : nothing
    @info "building ace_embedding_model(elements=$elements, order=$order, " *
          "totaldegree=$totaldegree, d_max=$(d_max === nothing ? "lossless" : d_max))"
    model = M.ace_embedding_model(elements = tuple(elements...), order = order,
                                  totaldegree = totaldegree, embedding = emb,
                                  d_max = d_max, maxl = maxl_kw)
end

# For a throughput benchmark the coefficients are irrelevant -- cost depends on
# the basis, not its weights -- and at a few thousand functions a fit to Si_tiny
# would be wildly underdetermined as well as slow.  ACE_NOFIT=1 keeps the random
# initialisation.  Do not use such a model for anything but timing.
if get(ENV, "ACE_NOFIT", "0") == "1"
    # ace1_model initialises WB to ZERO.  Left that way, XLA can fold the
    # readout to a constant and eliminate the descriptor entirely, so the
    # benchmark would measure nothing.  Randomise explicitly.
    Random.seed!(11)
    model.ps.WB .= 0.02 .* randn(size(model.ps.WB))
    model.ps.Wpair .= 0.02 .* randn(size(model.ps.Wpair))
    @info "ACE_NOFIT=1: RANDOM weights, NOT fitted -- timing only"
else
    data = ACEpotentials.example_dataset("Si_tiny").train
    @info "fitting on Si_tiny with acefit!"
    acefit!(data, model;
            energy_key = "dft_energy", force_key = "dft_force",
            virial_key = "dft_virial", solver = ACEfit.BLR(), verbose = false)
    @info "fit done"
end

m = model.model; ps = model.ps; st = model.st
NZ = length(m.rbasis._i2z)
i2z = collect(m.rbasis._i2z)

# ---------------------------------------------------------------- radial spline
# SplineRnlrzzBasis evaluates  Rnl(r) = spline(x) * envelope(r, x),  x = T(r).
# The spline is Interpolations.cubic_spline_interpolation over a uniform grid on
# [-1,1]: 100 nodes, 102 B-spline coefficients (one pad each side).
function spline_arrays(basis)
    NZ = length(basis._i2z)
    s11 = basis.splines[1, 1]
    ncoef = length(s11.itp.itp.coefs)
    LEN = length(basis.spec)
    rng = s11.itp.ranges[1]
    coefs = zeros(NZ, NZ, ncoef, LEN)
    for iz = 1:NZ, jz = 1:NZ
        co = basis.splines[iz, jz].itp.itp.coefs
        for (p, c) in enumerate(co)          # p = 1..ncoef  <->  offset index 0..ncoef-1
            coefs[iz, jz, p, :] .= c
        end
        r = basis.splines[iz, jz].itp.ranges[1]
        @assert first(r) ≈ first(rng) && step(r) ≈ step(rng) && length(r) == length(rng)
    end
    return coefs, Float64(first(rng)), Float64(step(rng)), length(rng)
end

_is_spline(b) = b isa M.SplineRnlrzzBasis
rkind  = _is_spline(m.rbasis)    ? "spline" : "analytic"
pkind  = _is_spline(m.pairbasis) ? "spline" : "analytic"
@info "radial branches: rbasis=$rkind pairbasis=$pkind"

# `Wnlq` stays a live parameter for the analytic branch -- training would need it
# trainable, and splines are not differentiable w.r.t. what generated them.
function analytic_arrays(basis, ps_b)
    NZ = length(basis._i2z)
    W = zeros(NZ, NZ, length(basis.spec), length(basis.polys))
    for iz = 1:NZ, jz = 1:NZ
        W[iz, jz, :, :] = ps_b.Wnlq[:, :, iz, jz]
    end
    rs = basis.polys.refstate
    return W, collect(rs.A), collect(rs.B), collect(rs.C)
end

# ---------------------------------------------------------------- transforms
# NormalizedTransform(GeneralizedAgnesiTransform):
#   y = r <= rin ? 1 : 1/(1 + a s^q/(1 + s^(q-p))),  s = (r-rin)/(r0-rin)
#   x = clamp(-1 + 2 (y - yin)/(ycut - yin), -1, 1)
function transform_params(basis)
    NZ = length(basis._i2z)
    P = zeros(NZ, NZ, 7)     # p q a rin r0 yin ycut
    for iz = 1:NZ, jz = 1:NZ
        t = basis.transforms[iz, jz]
        g = t.trans
        @assert g isa M.GeneralizedAgnesiTransform "unsupported transform $(typeof(g))"
        P[iz, jz, :] = [g.p, g.q, g.a, g.rin, g.r0, t.yin, t.ycut]
    end
    P
end

# envelopes
# many-body : PolyEnvelope2sX(x1,x2,p1,p2,s)  env = s (x-x1)^p1 (x2-x)^p2 on (x1,x2)
# pair      : ACE1_PolyEnvelope1sR(rcut,r0,p) env = s^-p - sc^-p + p sc^(-p-1)(s-sc)
function env2sx_params(basis)
    NZ = length(basis._i2z); P = zeros(NZ, NZ, 5)
    for iz = 1:NZ, jz = 1:NZ
        e = basis.envelopes[iz, jz]
        @assert e isa M.PolyEnvelope2sX "unsupported envelope $(typeof(e))"
        P[iz, jz, :] = [e.x1, e.x2, e.p1, e.p2, e.s]
    end
    P
end
# ace1_model's pair basis uses ACE1_PolyEnvelope1sR (rcut, r0, p);
# ace_model's uses PolyEnvelope1sR (rcut, p) -- a different formula, not just
# different parameters.
function env1sr_params(basis)
    NZ = length(basis._i2z)
    e1 = basis.envelopes[1, 1]
    if e1 isa M.ACE1_PolyEnvelope1sR
        P = zeros(NZ, NZ, 3)
        for iz = 1:NZ, jz = 1:NZ
            e = basis.envelopes[iz, jz]
            P[iz, jz, :] = [e.rcut, e.r0, e.p]
        end
        return P, "ace1_poly1sr"
    elseif e1 isa M.PolyEnvelope1sR
        P = zeros(NZ, NZ, 2)
        for iz = 1:NZ, jz = 1:NZ
            e = basis.envelopes[iz, jz]
            P[iz, jz, :] = [e.rcut, e.p]
        end
        return P, "poly1sr"
    end
    error("unsupported pair envelope $(typeof(e1))")
end

# ---------------------------------------------------------------- tensor
aspec = m.tensor.abasis.spec                       # Vector{Tuple{Int,Int}} (Rnl idx, Ylm idx)
aspec_r = Int32[t[1] - 1 for t in aspec]           # 0-based for python
aspec_y = Int32[t[2] - 1 for t in aspec]
aa_specs = m.tensor.aabasis.specs                  # tuple of per-order specs
A2B = Matrix(m.tensor.A2Bmaps[1])

# rin0cuts (rcut per species pair)
rcuts = [m.rbasis.rin0cuts[iz, jz].rcut for iz = 1:NZ, jz = 1:NZ]
pair_rcuts = [m.pairbasis.rin0cuts[iz, jz].rcut for iz = 1:NZ, jz = 1:NZ]

# WB / Wpair, and E0
WB = Matrix(ps.WB)                                  # (n_B, NZ)
Wpair = Matrix(ps.Wpair)                            # (n_pair, NZ)
# ace_model may carry no Vref; treat that as zero one-body energies
E0 = (m.Vref === nothing) ? zeros(length(i2z)) :
     [Float64(get(m.Vref.E0, z, 0.0)) for z in i2z]

# ---------------------------------------------------------------- probe values
# Per-stage reference values so a mismatch localises to a stage rather than
# only showing up at the end (this idea earned its keep in an earlier prototype).
Random.seed!(20260909)
n_probe = 24
probe_r = collect(range(0.9, maximum(rcuts) - 1e-6, length = n_probe))
probe_dirs = [normalize(SVector{3}(randn(3))) for _ in 1:n_probe]
probe_rij = reduce(hcat, [collect(probe_r[i] * probe_dirs[i]) for i in 1:n_probe])  # (3, n)
z = i2z[1]
probe_x   = [m.rbasis.transforms[1,1](probe_r[i]) for i in 1:n_probe]
probe_env = [M.evaluate(m.rbasis.envelopes[1,1], probe_r[i], probe_x[i]) for i in 1:n_probe]
probe_Rnl = Matrix(M.evaluate_batched(m.rbasis, probe_r, z, fill(z, n_probe), ps.rbasis, st.rbasis))
probe_Rpair = Matrix(M.evaluate_batched(m.pairbasis, probe_r, z, fill(z, n_probe), ps.pairbasis, st.pairbasis))
import Polynomials4ML as P4ML
probe_Ylm = Matrix(P4ML.evaluate(m.ybasis, [SVector{3}(probe_rij[:, i]) for i in 1:n_probe]))

# ---------------------------------------------------------------- test system
# The test system must contain EVERY element the model was built for, or the
# species-pair machinery (Wnlq[:,:,iz,jz], the SelectLinL-equivalent indexing,
# per-species E0) is exported but never evaluated -- a single-species test
# system would agree perfectly while a two-species convention was wrong.
using AtomsBuilder
sys = AtomsBuilder.bulk(:Si, cubic = true) * 2
if length(elements) > 1
    # Deterministic round-robin substitution: every element appears, and the
    # assignment does not depend on a seed or on iteration order.
    zs = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in elements]
    ats = [AtomsBase.Atom(zs[mod1(i, length(zs))], AtomsBase.position(sys, i))
           for i in 1:length(sys)]
    sys = AtomsBase.FlexibleSystem(ats, AtomsBase.cell(sys))
    @info "multi-element test system: $(elements), counts $( [count(==(z), [AtomsBase.atomic_number(sys,i) for i in 1:length(sys)]) for z in zs] )"
end
rattle!(sys, 0.15u"Å")
nat = length(sys)
test_pos = reduce(hcat, [ustrip.(u"Å", p) for p in AtomsBase.position(sys, :)])   # (3, nat)
test_cell = reduce(hcat, [ustrip.(u"Å", v) for v in AtomsBase.cell_vectors(sys)]) # (3, 3) columns
test_Z = Int32[AtomsBase.atomic_number(sys, i) for i in 1:nat]

# per-site energies + total, via the model's own site evaluation on a full nlist
import AtomsCalculatorsUtilities.SitePotentials as SP
calc = model
nlist = SP.PairList(sys, SP.cutoff_radius(calc))
site_E = zeros(nat)
edge_i = Int32[]; edge_j = Int32[]; edge_rij = Vector{Float64}[]
for i = 1:nat
    Js, Rs, Zs, z0 = SP.get_neighbours(sys, calc, nlist, i)
    site_E[i] = M.evaluate(m, Rs, Zs, z0, ps, st)
    for (jj, R) in zip(Js, Rs)
        push!(edge_i, Int32(i - 1)); push!(edge_j, Int32(jj - 1))
        push!(edge_rij, collect(ustrip.(R)))
    end
end
test_edge_rij = reduce(hcat, edge_rij)     # (3, n_edges)
# Site descriptors: the parity target for acejax.site_descriptors.
# Layout is species-blocked, (n_B + n_pair) * NZ per site, with the centre
# species selecting which block is populated -- exactly what the readout
# contracts against (src/models/ace.jl:544-566).
test_desc = reduce(hcat, ACEpotentials.site_descriptors(sys, calc))   # (len_basis, nat)
@printf("descriptors: %d per site, %d sites\n", size(test_desc,1), size(test_desc,2))

efv = AtomsCalculators.energy_forces_virial(sys, calc)
test_E = ustrip(u"eV", efv.energy)
test_F = reduce(hcat, [ustrip.(u"eV/Å", f) for f in efv.forces])   # (3, nat)
# Julia convention (AtomsCalculatorsUtilities sitepotentials/assembly.jl:6):
#   site_virial(dV, Rs) = - sum(dv_i * r_i')   i.e.  V = -sum_e dE/dr_e (x) r_e
test_V = Matrix(ustrip.(u"eV", efv.virial))                       # (3,3)
test_pbc = Int32[Bool(b) for b in AtomsBase.periodicity(sys)]
@printf("test system: %d atoms, %d edges, E = %.10f eV\n", nat, length(edge_i), test_E)
@printf("  sum(site_E) = %.10f   |diff| = %.2e\n", sum(site_E), abs(sum(site_E) - test_E))
@printf("  virial trace = %.6f eV,  |V - V'| = %.2e (symmetry check)\n",
        test_V[1,1]+test_V[2,2]+test_V[3,3], maximum(abs.(test_V .- test_V')))

# ---------------------------------------------------------------- branch data
rnl_coefs = rnl_W = rnl_pA = rnl_pB = rnl_pC = nothing
rnl_spl_meta = nothing
if rkind == "spline"
    rnl_coefs, rx0, rh, rn = spline_arrays(m.rbasis)
    rnl_spl_meta = Dict("x0"=>rx0, "h"=>rh, "n"=>rn, "ncoef"=>size(rnl_coefs,3))
else
    rnl_W, rnl_pA, rnl_pB, rnl_pC = analytic_arrays(m.rbasis, ps.rbasis)
end
pair_coefs = pair_W = pair_pA = pair_pB = pair_pC = nothing
pair_spl_meta = nothing
if pkind == "spline"
    pair_coefs, px0, ph, pn = spline_arrays(m.pairbasis)
    pair_spl_meta = Dict("x0"=>px0, "h"=>ph, "n"=>pn, "ncoef"=>size(pair_coefs,3))
else
    pair_W, pair_pA, pair_pB, pair_pC = analytic_arrays(m.pairbasis, ps.pairbasis)
end
pair_env, pair_env_kind = env1sr_params(m.pairbasis)

# ---------------------------------------------------------------- meta
meta = Dict(
  "schema_version" => 1,
  "source" => "ACEpotentials.jl $(KIND) + acefit!(Si_tiny, BLR)",
  "embedding" => (KIND == "embedding" ?
                  JSON.json(m.meta["embedding"]) : ""),
  "acepotentials_version" => string(pkgversion(ACEpotentials)),
  "julia_version" => string(VERSION),
  "elements" => i2z,
  "order" => order, "totaldegree" => totaldegree,
  "radial_kind" => rkind, "pair_radial_kind" => pkind,
  "transform_kind" => "agnesi_normalized",
  "envelope_kind" => "poly2sx",
  "pair_envelope_kind" => pair_env_kind,
  "ybasis_kind" => (occursin("Solid", string(typeof(m.ybasis.scbasis))) ?
                    "real_solidharmonics" : "real_sphericalharmonics"),
  "lmax" => Int(isqrt(length(m.ybasis)) - 1),
  "n_rnl" => length(m.rbasis.spec), "n_pair" => length(m.pairbasis.spec),
  "n_ylm" => length(m.ybasis), "n_A" => length(aspec),
  "n_AA" => sum(length, aa_specs), "n_B" => size(A2B, 1),
  "len_basis" => M.length_basis(m),
  "aa_orders" => [length(s[1]) for s in aa_specs],
  "aa_lens" => [length(s) for s in aa_specs],
  "rnl_spline" => rnl_spl_meta, "pair_spline" => pair_spl_meta,
  "rcut" => maximum(rcuts),
  "nnll" => [[ [b.n, b.l] for b in bb ] for bb in M.get_nnll_spec(m.tensor)],
)

D = Dict{String, Any}(
  "meta_json" => Vector{UInt8}(JSON.json(meta)),
  "rnl_transform" => transform_params(m.rbasis),      # (NZ,NZ,7)
  "pair_transform" => transform_params(m.pairbasis),
  "rnl_envelope" => env2sx_params(m.rbasis),          # (NZ,NZ,5)
  "pair_envelope" => pair_env,
  "rcuts" => rcuts, "pair_rcuts" => pair_rcuts,
  "aspec_r" => aspec_r, "aspec_y" => aspec_y,
  # A2B is stored as triplets: at n_B = 1429 the dense form is 125 MB with
  # 11474 nonzeros (0.009% occupied).  The loader densifies.
  "A2B_rows" => Int32.(findnz(sparse(A2B))[1] .- 1),
  "A2B_cols" => Int32.(findnz(sparse(A2B))[2] .- 1),
  "A2B_vals" => findnz(sparse(A2B))[3],
  "A2B_shape" => Int32[size(A2B, 1), size(A2B, 2)],
  "WB" => WB, "Wpair" => Wpair, "E0" => E0,
  "elements" => Int32.(i2z),
  # smoothness prior diagonal, ACEfit convention: acefit! solves W (A / P) with
  # P = Diagonal(gamma), so the coefficient prior is N(0, sigma_c^2 P^-2).
  "gamma" => collect(diag(M.algebraic_smoothness_prior(m))),
  # probes
  "probe_r" => probe_r, "probe_rij" => probe_rij, "probe_x" => probe_x,
  "probe_env" => probe_env, "probe_Rnl" => probe_Rnl,
  "probe_Rpair" => probe_Rpair, "probe_Ylm" => probe_Ylm,
  # test system
  "test_pos" => test_pos, "test_cell" => test_cell, "test_Z" => test_Z,
  "test_V" => test_V, "test_pbc" => test_pbc, "test_desc" => test_desc,
  "test_edge_i" => edge_i, "test_edge_j" => edge_j, "test_edge_rij" => test_edge_rij,
  "test_site_E" => site_E, "test_E" => [test_E], "test_F" => test_F,
)
# only the populated radial branch is written; the loader defaults the other
if rkind == "spline"
    # A frozen embedding with UNIFORM cutoffs makes this table separable:
    #     coefs[z1, z2, :, i] == emb[z2, k(i)] * C[:, n'(i)]
    # so the (NZ, NZ, ncoef, n_rnl) array is NZ^2 scaled copies of one
    # (ncoef, n1) table.  Storing it once is O(1) in the number of elements
    # instead of O(S^2) -- 0.2 MB rather than 896 MB at S=75 -- and removes the
    # per-edge (E, ncoef, n_rnl) gather that dominates many-element evaluation.
    # Detected numerically rather than assumed: if the factorisation does not
    # hold to 1e-12 (per-pair cutoffs, a non-embedding model) the dense table is
    # written and nothing changes.
    fact = nothing
    if KIND == "embedding"
        dmb = m.meta["embedding"]["d_max"]
        n1  = size(rnl_coefs, 4) ÷ dmb
        embrows = M.embedding_rows(emb, [Int(z) for z in i2z]; d = dmb)
        C = [ rnl_coefs[1, 1, q, (np_ - 1) * dmb + 1] / embrows[1, 1]
              for q = 1:size(rnl_coefs, 3), np_ = 1:n1 ]
        pred = [ embrows[z2, mod1(i, dmb)] * C[q, div(i - 1, dmb) + 1]
                 for z1 = 1:NZ, z2 = 1:NZ, q = 1:size(rnl_coefs, 3),
                     i = 1:size(rnl_coefs, 4) ]
        err = maximum(abs.(pred .- rnl_coefs)) / max(maximum(abs.(rnl_coefs)), eps())
        @info "radial factorisation residual = $err"
        err < 1e-12 && (fact = (C, embrows, dmb, n1))
    end
    if fact === nothing
        D["rnl_spline_coefs"] = rnl_coefs
    else
        C, embrows, dmb, n1 = fact
        D["rnl_spline_coefs_single"] = C            # (ncoef, n1)
        D["rnl_embedding"] = embrows                # (NZ, d)
        D["rnl_emb_nidx"] = Int32[ div(i - 1, dmb) for i = 1:size(rnl_coefs, 4) ]
        D["rnl_emb_kidx"] = Int32[ mod1(i, dmb) - 1 for i = 1:size(rnl_coefs, 4) ]
        @info "radial stored FACTORISED: $(size(C)) + $(size(embrows)) " *
              "instead of $(size(rnl_coefs))"
    end
else
    D["rnl_Wnlq"] = rnl_W; D["polys_A"] = rnl_pA; D["polys_B"] = rnl_pB; D["polys_C"] = rnl_pC
end
if pkind == "spline"
    D["pair_spline_coefs"] = pair_coefs
else
    D["pair_Wnlq"] = pair_W
    D["pair_polys_A"] = pair_pA; D["pair_polys_B"] = pair_pB; D["pair_polys_C"] = pair_pC
end

for (k, s) in enumerate(aa_specs)
    D["aa_spec_$(k)"] = Int32.(reduce(hcat, [collect(t) for t in s])' .- Int32(1))  # (n_v, order) 0-based
end
npzwrite(OUT, D)
@printf("wrote %s  (%.1f MB)\n", OUT, filesize(OUT)/2^20)

# ---------------------------------------------------------------- GP fixtures
# ACE_GP_FIXTURES=1 also writes the Si_tiny training set and ACEfit's design
# matrix for the SAME model, so acegp's assembly can be checked row-for-row
# (tests/test_gp_rows.py) and its BLR limit against a ridge solve
# (tests/test_gp_objective.py).
if get(ENV, "ACE_GP_FIXTURES", "0") == "1"
    using Artifacts
    fixdir = dirname(OUT)
    toml = joinpath(pkgdir(ACEpotentials), "Artifacts.toml")
    src = joinpath(artifact_path(artifact_hash("Si_tiny_dataset", toml)), "Si_tiny.xyz")
    cp(src, joinpath(fixdir, "si_tiny_train.xyz"); force = true)
    train = ACEpotentials.example_dataset("Si_tiny").train
    A, Y, W = ACEpotentials.assemble(train, model;
                energy_key = "dft_energy", force_key = "dft_force",
                virial_key = "dft_virial",
                weights = Dict("default" => Dict("E" => 1.0, "F" => 1.0, "V" => 1.0)))
    npzwrite(joinpath(fixdir, "si_tiny_design.npz"), Dict{String, Any}(
        "A" => Matrix(A), "Y" => collect(Y), "W" => collect(W),
        "natoms" => Int32[length(at) for at in train],
        "gamma" => collect(diag(M.algebraic_smoothness_prior(m))),
        "E0" => E0))
    @info "GP fixtures written" size(A)
end
