# Convert the output of an electronic-structure code to the DeepH layout with
# Quoll.jl, which owns the conventions: orbital order and signs, units, sparsity.
# ace-jax then reads the DeepH directory in Python (ace_jax.ham.deeph) and keeps
# it as an npz cache (ace_jax.ham.cache); no Julia after this step.
#   julia --startup-file=no --project=ham/julia ham/julia/ingest.jl <format> <in dir> <out dir> [H,S]
# <format>: fhi-aims (rs_hamiltonian.h5 / .out, rs_overlap, basis-indices.out,
# geometry.in) or deeph (a round trip through Quoll's canonical form).
# Quoll reads FHI-aims geometries with ASE through CondaPkg, which puts a ~400 MB conda
# environment in <project>/.CondaPkg unless JULIA_CONDAPKG_ENV names another directory.
# Configurations is held at 0.17.6 in Project.toml: the release after it breaks
# Quoll's precompile (an ambiguous from_dict method).
using Quoll

const FORMATS = Dict("fhi-aims" => Quoll.FHIaimsCSCNoSpinRealMetadata,
                     "deeph" => Quoll.DeepHBlockNoSpinRealMetadata)
const KINDS = Dict("H" => () -> Quoll.Hamiltonian(; source = :ref), "S" => () -> Quoll.Overlap(; source = :ref))

function main(args)
   3 <= length(args) <= 4 || error("usage: ingest.jl <fhi-aims|deeph> <in dir> <out dir> [H,S]")
   fmt, indir, outdir = args[1], args[2], args[3]
   haskey(FORMATS, fmt) || error("unknown format $fmt; one of $(sort(collect(keys(FORMATS))))")
   names = split(length(args) == 4 ? args[4] : "H,S", ",")
   ops = map(names) do name
      haskey(KINDS, name) || error("unknown operator $name; one of H, S")
      op = Quoll.load_operator(Quoll.Operator, FORMATS[fmt], indir, KINDS[name]())
      canon = Quoll.convert_operator(Quoll.KeyedOperator, Quoll.CanonicalBlockRealMetadata, op; hermitian = true)
      Quoll.convert_operator(Quoll.Operator, Quoll.DeepHBlockNoSpinRealMetadata, canon; hermitian = false)
   end
   Quoll.write_operators(Quoll.DeepHBlockNoSpinRealMetadata, outdir, ops)
   println("wrote ", join(names, ", "), " to ", outdir)
end

main(ARGS)
