# Build the PR 309 LAMMPS trim library of a model ace-jax BUILT AND FITTED (not an
# ACEpotentials export): the accuracy-cost `acepotentials-trim` rows.
#
#   julia --project=bench/scaling/julia build_trim_acejax.jl <model.npz> <gate.json> <out.so>
#
# build_trim.jl starts from an ace1_model export.  An ace-jax `aj fit` model is instead
# the `ace_model` that `ace_jax.basis.build_model` reproduces (julia/built_basis_reference.jl):
# solid harmonics, Legendre radials over agnesi(2, 2) with the poly2sx envelope, the
# TotalDegree(NZ, 1/wL) level, and the analytic pair basis (agnesi(1, 4), poly1sr), with
# ace-jax's own frozen radial coefficients.  So:
# 1. ace_model rebuilt from the npz's meta_json (`basis`: wL, r0, rin, rcut, pair_maxn),
#    pair_learnable = true (the unsplined pair basis: ace-jax's is analytic); the rebuilt
#    basis must have the npz's A and AA specifications, and the npz's WB is mapped onto
#    its B (the B rows differ: see below);
# 2. that WB and the npz's Wpair, rnl_Wnlq, pair_Wnlq and E0 set; gate: the ACEPotential
#    vs ace-jax's E / F on <gate.json>'s structure (written by the Python side), tight;
# 3. convert2et_full, gate: ETACE stack vs the ACEPotential, tight;
# 4. export_ace_model(stack; for_library = true) -> <out>.jl, gate: the generated module
#    vs the ETACE stack; then juliac --trim=safe -> <out>.so.
# gate.json: {"xyz": path, "E": eV, "F": [[fx, fy, fz], ...]}.  Prints one JSON line.
include(joinpath(@__DIR__, "common.jl"))
const EXPORT = joinpath(pkgdir(ACEpotentials), "export")
include(joinpath(EXPORT, "src", "export_ace_model.jl"))
import AtomsCalculatorsUtilities.SitePotentials as SPt
using StaticArrays: SMatrix

const TOL = (1e-10, 1e-9)                  # |dE|/atom (eV), max|dF| (eV/Å)

function gate(name, E1, F1, E2, F2, nat; tol = TOL)
    dE, dF = abs(E1 - E2) / nat, maxdiff(F1, F2)
    ok = dE <= tol[1] && dF <= tol[2]
    ok || error("$name: |dE|/atom = $dE eV, max|dF| = $dF eV/Å, over $tol")
    return Dict("dE_per_atom" => dE, "max_dF" => dF, "tol" => collect(tol))
end

"E, F of the generated module (loaded in-process) by summing its site energies / forces."
function exported_ef(ex, sys, ref)
    nat = length(sys); nl = SPt.PairList(sys, SPt.cutoff_radius(ref))
    E, F = 0.0, zeros(3, nat)
    for i in 1:nat
        Js, Rs, Zs, z0 = SPt.get_neighbours(sys, ref, nl, i)
        Rs = [SVector{3,Float64}(ustrip.(R)) for R in Rs]
        Ei, Fi = Base.invokelatest(ex.site_energy_forces, Rs, Int.(AtomsBase.atomic_number.(Zs)),
                                   Int(AtomsBase.atomic_number(z0)))
        E += Ei
        for (k, j) in enumerate(Js)
            F[:, j] .+= Fi[k]; F[:, i] .-= Fi[k]
        end
    end
    return E, F
end

"ace-jax's (NZ, NZ, n_nl, n_q) radial coefficients as ACEpotentials' Wnlq (n_nl, n_q, NZ, NZ);
the q columns past the Julia basis's maxq must be zero (ace-jax pads the pair basis)."
function to_wnlq(W, target)
    nl, nq = Base.size(target, 1), Base.size(target, 2)
    Base.size(W, 3) == nl || error("radial: $(Base.size(W, 3)) nl channels in the npz, $nl in Julia")
    all(iszero, W[:, :, :, nq+1:end]) || error("radial: nonzero npz q columns past Julia's maxq = $nq")
    return permutedims(W[:, :, :, 1:nq], (3, 4, 1, 2))
end

npz, gj, so = ARGS[1], JSON.parsefile(ARGS[2]), abspath(ARGS[3])
D = npzread(npz); mt = npz_meta(D); b = mt["basis"]
elements = [Symbol(AtomsBase.element_symbol(ChemicalSpecies(Int(z)))) for z in mt["elements"]]
NZ = length(elements)
r0 = b["r0"]; r0 = r0 isa AbstractVector ? reduce(hcat, r0)' : fill(Float64(r0), NZ, NZ)
ri = M._default_rin0cuts(elements)
ri = SMatrix{NZ,NZ}([(rin = Float64(b["rin"]), r0 = Float64(r0[i, j]), rcut = Float64(b["rcut"]))
                     for i in 1:NZ, j in 1:NZ])
E0s = Dict(elements[i] => D["E0"][i] * u"eV" for i in 1:NZ)
model = M.ace_model(; elements = elements, order = Int(mt["order"]), Ytype = :solid,
                    level = M.TotalDegree(1.0 * NZ, 1 / b["wL"]), max_level = Int(mt["totaldegree"]),
                    pair_maxn = Int(b["pair_maxn"]), rin0cuts = ri, maxq_fact = b["n_q_factor"],
                    init_Wradial = :onehot, pair_learnable = true, E0s = E0s)
ps, st = Lux.setup(MersenneTwister(1), model)

# --- 1. the rebuilt basis spans the npz's: the same A (aspec) and AA (as sets), and the
# npz energy, sum_B WB A2B AA = c . AA, re-expressed in the Julia B by least squares.  The
# coupling library (ET main) orders and rescales B rows differently from the ET release
# ACEpotentials pins, and at order >= 4 a multiplicity-m block is defined only up to an
# m x m rotation (tests/test_built_basis_parity.py), so WB is mapped, never copied.  The
# two AA lists may differ by a few functions no B row uses (o4d20: 4 of 47681 order-4 AAs
# only in the npz, with coupled weight ~1e-24 of the largest; 3 only in Julia): those
# must carry no weight, and the least-squares residual is checked.
aspec = model.tensor.abasis.spec; aa = model.tensor.aabasis.specs
key(t) = Tuple(sort(collect(t)))
jl_index = Dict{Any,Int}(); off = 0
for k in 1:length(aa)
    for (i, t) in enumerate(aa[k]); jl_index[key(t .- 1)] = off + i; end
    global off += length(aa[k])
end
npz_aa = vcat([[key(Tuple(D["aa_spec_$k"][i, :])) for i in 1:Base.size(D["aa_spec_$k"], 1)]
               for k in 1:length(aa) if haskey(D, "aa_spec_$k")]...)
perm = [get(jl_index, t, 0) for t in npz_aa]     # npz AA column -> Julia AA column
A2Bj = sparse(Matrix(model.tensor.A2Bmaps[1]))
A2Bn = sparse(D["A2B_rows"] .+ 1, D["A2B_cols"] .+ 1, D["A2B_vals"], D["A2B_shape"]...)
id = Dict{String,Any}(
    "n_pair" => Base.size(ps.Wpair) == Base.size(D["Wpair"]),
    "aspec" => [t[1] - 1 for t in aspec] == D["aspec_r"] && [t[2] - 1 for t in aspec] == D["aspec_y"],
    "aa_unmatched_npz" => count(==(0), perm), "aa_unmatched_julia" => off - count(>(0), perm),
    "aa_perm" => allunique(filter(>(0), perm)),
    "n_B" => Base.size(A2Bj, 1) == Base.size(A2Bn, 1))
bad = [k for (k, v) in id if v === false]
isempty(bad) || error("rebuilt ace_model is NOT the npz basis $npz: $(join(sort(bad), ", ")) differ")
WBj = zeros(Base.size(ps.WB))
for iz in 1:NZ
    cn = A2Bn' * D["WB"][:, iz]                                 # AA coefficients, npz order
    lost = maximum(abs.(cn[perm .== 0]); init = 0.0) / maximum(abs.(cn))
    lost <= 1e-14 || error("npz AA functions the Julia basis lacks carry weight (species $iz): $lost")
    c = zeros(off); c[perm[perm .> 0]] .= cn[perm .> 0]          # Julia order
    w = sparse(A2Bj') \ c                         # SPQR least squares (dense: GBs at o4d20)
    r = maximum(abs.(A2Bj' * w .- c)) / maximum(abs.(c))
    r <= 1e-12 || error("WB does not map onto the Julia B (species $iz): relative residual $r")
    WBj[:, iz] .= w; id["WB_map_residual_$iz"] = r; id["aa_unmatched_weight_$iz"] = lost
end

# --- 2. coefficients in; gate against ace-jax
ps.WB .= WBj; ps.Wpair .= D["Wpair"]
ps.rbasis.Wnlq .= to_wnlq(D["rnl_Wnlq"], ps.rbasis.Wnlq)
ps.pairbasis.Wnlq .= to_wnlq(D["pair_Wnlq"], ps.pairbasis.Wnlq)
calc = M.ACEPotential(model, ps, st)
sys = read_xyz(gj["xyz"]); nat = length(sys)
Fref = reduce(hcat, [Float64.(f) for f in gj["F"]])
Ea, Fa, _ = efv(sys, calc)
gates = Dict{String,Any}("ace_model_vs_acejax" => gate("ace_model vs ace-jax", Ea, Fa, Float64(gj["E"]), Fref, nat))

# --- 3./4. ETACE, export, juliac
et = ETM.convert2et_full(model, ps, st)
Et, Ft, _ = efv(sys, et)
gates["etace_vs_ace_model"] = gate("ETACE stack vs ace_model", Et, Ft, Ea, Fa, nat)
outdir = dirname(so); mkpath(outdir); jl = replace(so, r"\.so$" => ".jl")
t_export = @elapsed Base.invokelatest(export_ace_model, et, jl; for_library = true)
ex = Module(:Exported); Base.include(ex, jl)
Ex, Fx = exported_ef(ex, sys, calc)
gates["exported_vs_etace"] = gate("exported module vs ETACE stack", Ex, Fx, Et, Ft, nat)
juliac = joinpath(dirname(Sys.BINDIR), "share", "julia", "juliac", "juliac.jl")
cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $juliac
       --output-lib $so --experimental --trim=safe --compile-ccallable $jl`
rm(so; force = true)
t_juliac = @elapsed cd(() -> run(cmd), outdir)
isfile(so) || error("juliac produced no $so")
emit(Dict("so" => so, "jl" => jl, "build_id" => string(export_build_id(jl); base = 16),
          "so_bytes" => filesize(so), "jl_bytes" => filesize(jl), "export_s" => t_export,
          "juliac_s" => t_juliac, "gates" => gates, "identity" => id,
          "gate_structure" => Dict("path" => abspath(gj["xyz"]), "n_atoms" => nat),
          "cpu_target" => get(ENV, "JULIA_CPU_TARGET", "native"), "versions" => versions()))
