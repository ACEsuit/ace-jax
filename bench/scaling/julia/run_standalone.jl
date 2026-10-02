# One standalone ACEpotentials.jl case (the `acepotentials` line): direct evaluation of the
# splined ace1_model the npz holds, through AtomsCalculators.energy_forces_virial
# (energy + forces + virial, neighbour list included, as every standalone line).
#
#   JULIA_NUM_THREADS=<n> julia --project=bench/scaling/julia run_standalone.jl \
#       <spec.json> <structure.extxyz> <positions.npy>
#
# positions.npy: (reps, nat, 3) positions in Å, one per timed call -- the MD-like random
# walk run_standalone.py generates for every standalone code.  The first (compile) call,
# and `energy`, use the undisplaced structure; a second untimed call follows it, then the
# timed calls.  Julia's GC makes single calls vary (it can be a large share of one call),
# so each call's GC time is recorded and the median call is reported.
# Prints one JSON line.
include(joinpath(@__DIR__, "common.jl"))
using Statistics: median

spec = JSON.parsefile(ARGS[1]); sys0 = read_xyz(ARGS[2]); X = npzread(ARGS[3])
t0 = time(); calc, id = build_splined(spec); t_build = time() - t0
c = @timed efv(sys0, calc)                         # compile + first call
E0 = c.value[1]
efv(sys0, calc)                                    # untimed: past compilation
calls = []
for k in 1:Base.size(X, 1)
    sys = with_positions(sys0, X[k, :, :])        # setup, untimed
    s = @timed efv(sys, calc)
    push!(calls, (time = s.time, gc = s.gctime, bytes = s.bytes))
end
ts = [x.time for x in calls]
emit(Dict("call_s" => median(ts), "call_s_min" => minimum(ts), "call_s_max" => maximum(ts),
          "gc_frac" => median([x.gc / x.time for x in calls]),
          "gc_s" => sum(x.gc for x in calls), "alloc_bytes" => median([x.bytes for x in calls]),
          "compile_s" => c.time, "energy" => E0, "model_build_s" => t_build,
          "peak_bytes" => Int(Sys.maxrss()), "julia_threads" => Threads.nthreads(),
          "identity" => id, "versions" => versions(), "n_calls" => length(calls)))
