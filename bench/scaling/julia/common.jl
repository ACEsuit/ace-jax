# Shared by the ACEpotentials.jl benchmark drivers (run_standalone.jl, eval_ef.jl,
# build_trim.jl): rebuild a benchmark linear ACE model in this env, prove it is the
# model the ace-jax `.npz` holds, and build its exact (unsplined) twin as ETACE.
#
# The model spec (a JSON object, from bench/scaling/models.py `ace1_spec`):
#   {"npz": path, "elements": ["Si", "Ge"], "order": 4, "totaldegree": 5,
#    "rcut": 5.0 | null, "r0": 2.54 | null}
# null / absent keys keep ace1_model's default, as julia/export_model.jl does.

using ACEpotentials, NPZ, JSON, StaticArrays, LinearAlgebra, Random, SparseArrays, Printf
using AtomsBase, AtomsCalculators, ExtXYZ, Unitful, Lux, LuxCore
import Pkg
const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

"ace1_model keyword arguments for a model spec (as julia/export_model.jl builds them)."
function ace1_kwargs(spec)
    kw = Dict{Symbol,Any}(:elements => Symbol.(spec["elements"]), :order => Int(spec["order"]),
                          :totaldegree => Int(spec["totaldegree"]))
    get(spec, "rcut", nothing) === nothing || (kw[:rcut] = Float64(spec["rcut"]))
    get(spec, "r0", nothing) === nothing || (kw[:r0] = Float64(spec["r0"]))
    return kw
end

npz_meta(D) = JSON.parse(String(UInt8.(D["meta_json"])))

"Spline coefficient table (NZ, NZ, ncoef, L) of a splined radial basis, as the npz stores it."
function spline_table(basis)
    NZ = length(basis._i2z); nc = length(basis.splines[1, 1].itp.itp.coefs); L = length(basis.spec)
    C = zeros(NZ, NZ, nc, L)
    for iz in 1:NZ, jz in 1:NZ, (p, c) in enumerate(basis.splines[iz, jz].itp.itp.coefs)
        C[iz, jz, p, :] .= c
    end
    return C
end

maxdiff(a, b) = Base.size(a) == Base.size(b) ? maximum(abs.(a .- b); init = 0.0) : Inf

"(E, F (3, nat), V (3, 3)) in eV, eV/Å, eV."
function efv(sys, calc)
    r = AtomsCalculators.energy_forces_virial(sys, calc)
    return (ustrip(u"eV", r.energy), reduce(hcat, [ustrip.(u"eV/Å", f) for f in r.forces]),
            Matrix(ustrip.(u"eV", r.virial)))
end

function npz_test_system(D)
    pos, cellm, Z = D["test_pos"], D["test_cell"], D["test_Z"]
    atoms = [AtomsBase.Atom(Int(Z[i]), SVector{3}(pos[:, i]) * u"Å") for i in 1:Base.size(pos, 2)]
    return periodic_system(atoms, [SVector{3}(cellm[:, k]) * u"Å" for k in 1:3])
end

"""
    build_splined(spec) -> (calc, identity)

`ace1_model` rebuilt exactly as `julia/export_model.jl` (ACE_NOFIT=1) built the npz:
Random.seed!(11), WB / Wpair .= 0.02 randn.  Then the npz weights are copied in and the
rebuilt model is ASSERTED identical to the npz: basis sizes, the A, AA and nnll
specifications, the A2B map, both spline tables, the seed-11 weights, and E/F/V on the
npz's own test structure (evaluated when the npz was written, possibly on another Julia).
Any difference throws: these lines must evaluate the model ace-jax evaluates.
"""
function build_splined(spec)
    D = npzread(spec["npz"]); mt = npz_meta(D)
    calc = ace1_model(; ace1_kwargs(spec)...)
    Random.seed!(11)
    calc.ps.WB .= 0.02 .* randn(Base.size(calc.ps.WB))
    calc.ps.Wpair .= 0.02 .* randn(Base.size(calc.ps.Wpair))
    m = calc.model
    aspec = m.tensor.abasis.spec
    aa = m.tensor.aabasis.specs
    A2B = sparse(Matrix(m.tensor.A2Bmaps[1]))
    A2Bn = sparse(D["A2B_rows"] .+ 1, D["A2B_cols"] .+ 1, D["A2B_vals"], D["A2B_shape"]...)
    id = Dict{String,Any}(
        "n_B" => Base.size(calc.ps.WB, 1) == mt["n_B"] && Base.size(calc.ps.WB) == Base.size(D["WB"]),
        "n_pair" => Base.size(calc.ps.Wpair) == Base.size(D["Wpair"]),
        "aspec" => [t[1] - 1 for t in aspec] == D["aspec_r"] && [t[2] - 1 for t in aspec] == D["aspec_y"],
        "aa_specs" => all(k -> haskey(D, "aa_spec_$k") &&
                               Int32.(reduce(hcat, [collect(t) for t in aa[k]])' .- 1) == D["aa_spec_$k"],
                          1:length(aa)),
        "nnll" => [[[b.n, b.l] for b in bb] for bb in M.get_nnll_spec(m.tensor)] == mt["nnll"],
        "A2B" => maxdiff(A2B, A2Bn) == 0.0,
        "rnl_spline" => maxdiff(spline_table(m.rbasis), D["rnl_spline_coefs"]) == 0.0,
        "pair_spline" => maxdiff(spline_table(m.pairbasis), D["pair_spline_coefs"]) == 0.0,
        "WB_seed11" => maxdiff(calc.ps.WB, D["WB"]) == 0.0,
        "Wpair_seed11" => maxdiff(calc.ps.Wpair, D["Wpair"]) == 0.0)
    calc.ps.WB .= D["WB"]; calc.ps.Wpair .= D["Wpair"]
    tsys = npz_test_system(D)
    E, F, V = efv(tsys, calc)
    nat = length(tsys)
    dE, dF, dV = abs(E - D["test_E"][1]) / nat, maxdiff(F, D["test_F"]), maxdiff(V, D["test_V"])
    id["test_EFV"] = dE <= 1e-10 && dF <= 1e-9 && dV <= 1e-9
    id["test_dE_per_atom"], id["test_max_dF"], id["test_max_dV"] = dE, dF, dV
    bad = [k for (k, v) in id if v === false]
    isempty(bad) || error("rebuilt ace1_model is NOT the npz model $(spec["npz"]): $(join(sort(bad), ", ")) " *
                          "differ (npz: ACEpotentials $(mt["acepotentials_version"]), Julia $(mt["julia_version"]); " *
                          "now ACEpotentials $(pkgversion(ACEpotentials)), Julia $VERSION)")
    id["npz_julia"], id["npz_acepotentials"] = mt["julia_version"], mt["acepotentials_version"]
    return calc, id
end

# PR 309 ships the twin builder as test code (not package API): ace1_exact_twin,
# ace1_twin_matches
include(joinpath(pkgdir(ACEpotentials), "test", "etmodels", "ace1_exact_twin.jl"))

"""
    build_twin(spec, spl) -> (exact::ACEPotential, et::StackedCalculator)

The exact (unsplined) twin of the splined model `spl` (PR 309's
`test/etmodels/ace1_exact_twin.jl`; asserted to have the same basis), with the same
weights, and its ETModels stack (`convert2et_full`: one-body + ETPairModel + ETACE).
"""
function build_twin(spec, spl)
    twin = ace1_exact_twin(; ace1_kwargs(spec)...)
    ace1_twin_matches(twin, spl.model) ||
        error("exact twin basis differs from the splined ace1_model")
    ps, st = Lux.setup(MersenneTwister(1), twin)
    ps.WB .= spl.ps.WB; ps.Wpair .= spl.ps.Wpair
    return M.ACEPotential(twin, ps, st), ETM.convert2et_full(twin, ps, st)
end

"Julia, ACEpotentials (and its git revision) versions for a row."
function versions()
    out = Dict{String,Any}("julia" => string(VERSION), "ACEpotentials" => string(pkgversion(ACEpotentials)))
    for (_, d) in Pkg.dependencies()
        if d.name == "ACEpotentials"
            out["ACEpotentials_rev"] = something(d.git_revision, "")
        elseif d.name in ("EquivariantTensors", "Polynomials4ML", "JuliaC")
            out[d.name] = string(d.version)
        end
    end
    return out
end

"Read an extxyz (the structure Python wrote and re-read, so both see the same bits)."
read_xyz(path) = ExtXYZ.load(path)

"A periodic system with `sys`'s species and cell at new positions (n, 3) in Å."
function with_positions(sys, X)
    cell = AtomsBase.cell_vectors(sys)
    atoms = [AtomsBase.Atom(AtomsBase.atomic_number(sys, i), SVector{3}(X[i, 1], X[i, 2], X[i, 3]) * u"Å")
             for i in 1:length(sys)]
    return periodic_system(atoms, cell)
end

"Print the result as one JSON line on stdout (the Python side reads the last `{` line)."
emit(d) = (println(stdout, JSON.json(d)); flush(stdout))
