# Build the `acepotentials-trim` library of one benchmark model: the PR 309 LAMMPS export.
#
#   julia --project=bench/scaling/julia build_trim.jl <spec.json> <structure.extxyz> <out.so>
#
# 1. the splined ace1_model the npz holds (build_splined: asserted identical to the npz);
# 2. its exact twin (same basis and weights, unsplined radials) as an ETModels stack;
#    gate: ETACE stack vs the exact ACEModel, tight;
# 3. export_ace_model(stack; for_library = true) -> <out>.jl; gate: the generated module,
#    loaded in this process, vs the ETACE stack, tight (an exporter that cannot handle the
#    model -- it refuses, or gives other numbers -- fails here, loudly);
# 4. juliac --trim=safe -> <out>.so (JULIA_CPU_TARGET, when set, picks the CPU target;
#    otherwise juliac's default, this host's CPU).
# The exact twin and the splined model differ by ace1_model's spline error, reported as
# `spline_error` (not gated).  Prints one JSON line.
include(joinpath(@__DIR__, "common.jl"))
const EXPORT = joinpath(pkgdir(ACEpotentials), "export")
include(joinpath(EXPORT, "src", "export_ace_model.jl"))
import AtomsCalculatorsUtilities.SitePotentials as SPt

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

spec = JSON.parsefile(ARGS[1]); sys = read_xyz(ARGS[2]); so = abspath(ARGS[3])
nat = length(sys); outdir = dirname(so); mkpath(outdir)
jl = replace(so, r"\.so$" => ".jl")
spl, id = build_splined(spec)
exact, et = build_twin(spec, spl)
Es, Fs, _ = efv(sys, spl); Ee, Fe, _ = efv(sys, exact); Et, Ft, _ = efv(sys, et)
gates = Dict{String,Any}("etace_vs_exact" => gate("ETACE stack vs exact ACEModel", Et, Ft, Ee, Fe, nat))
t_export = @elapsed Base.invokelatest(export_ace_model, et, jl; for_library = true)
ex = Module(:Exported); Base.include(ex, jl)
Ex, Fx = exported_ef(ex, sys, exact)
gates["exported_vs_etace"] = gate("exported module vs ETACE stack", Ex, Fx, Et, Ft, nat)
spline_error = Dict("dE_per_atom" => abs(Ee - Es) / nat, "max_dF" => maxdiff(Fe, Fs))
juliac = joinpath(dirname(Sys.BINDIR), "share", "julia", "juliac", "juliac.jl")
cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $juliac
       --output-lib $so --experimental --trim=safe --compile-ccallable $jl`
rm(so; force = true)
t_juliac = @elapsed cd(() -> run(cmd), outdir)
isfile(so) || error("juliac produced no $so")
emit(Dict("so" => so, "jl" => jl, "build_id" => string(export_build_id(jl); base = 16),
          "so_bytes" => filesize(so), "jl_bytes" => filesize(jl), "export_s" => t_export,
          "juliac_s" => t_juliac, "gates" => gates, "spline_error" => spline_error,
          "gate_structure" => Dict("path" => abspath(ARGS[2]), "n_atoms" => nat),
          "cpu_target" => get(ENV, "JULIA_CPU_TARGET", "native"),
          "identity" => id, "versions" => versions()))
