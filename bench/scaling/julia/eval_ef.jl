# Energy and forces of a benchmark model on one structure, for the parity gates.
#
#   julia --project=bench/scaling/julia eval_ef.jl <spec.json> <structure.extxyz> <which>
#
# which = "splined": the ace1_model the npz holds (what ace-jax evaluates);
#         "etace":   its exact twin as an ETModels stack (what the trim library compiles).
# Prints one JSON line: {"E", "F" (nat x 3, rows = atoms), "identity", "versions"}.
include(joinpath(@__DIR__, "common.jl"))

spec = JSON.parsefile(ARGS[1]); sys = read_xyz(ARGS[2]); which = ARGS[3]
spl, id = build_splined(spec)
calc = which == "splined" ? spl : which == "etace" ? build_twin(spec, spl)[2] :
       error("which must be splined or etace, not $which")
E, F, _ = efv(sys, calc)
emit(Dict("E" => E, "F" => [F[:, i] for i in 1:Base.size(F, 2)], "which" => which,
          "identity" => id, "versions" => versions()))
