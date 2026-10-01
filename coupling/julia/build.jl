# Build the ace-jax-coupling native library.
#   julia +1.13 --project=coupling/julia/build coupling/julia/build.jl coupling/build
# Produces <out>/bundle/{lib/libetcouple.<ext>, lib/julia/..., share/..., build_info.json}.
VERSION >= v"1.13" || error("build needs Julia >= 1.13 (got $VERSION)")
using JuliaC, TOML

const HERE = @__DIR__
const LIBPROJ = HERE                                   # coupling/julia (the shim project)
const ENTRY = joinpath(HERE, "src", "ETCouple.jl")
# Build with exactly the Julia the Manifests were resolved with: the bundled runtime
# libraries (THIRD_PARTY_NOTICES.md versions) come with it, so a patch release
# picked up by a loose "1.13" would ship different builds unannounced.
for m in (joinpath(LIBPROJ, "Manifest.toml"), joinpath(HERE, "build", "Manifest.toml"))
   want = TOML.parsefile(m)["julia_version"]
   string(VERSION) == want || error("build needs Julia $want (julia_version in $m), got $VERSION")
end
out = abspath(length(ARGS) >= 1 ? ARGS[1] : joinpath(HERE, "..", "build"))
bundle = joinpath(out, "bundle")
rm(bundle; force = true, recursive = true); mkpath(out)

ENV["JULIA_CPU_TARGET"] = "generic"
Sys.isapple() && (ENV["MACOSX_DEPLOYMENT_TARGET"] = "11.0")

run(`$(Base.julia_cmd()) --project=$LIBPROJ -e "using Pkg; Pkg.instantiate()"`)
JuliaC.main(["--output-lib", joinpath(out, "libetcouple"), "--project=$LIBPROJ",
             "--trim=safe", "--compile-ccallable", "--experimental",
             "--jl-option", "handle-signals=no", "--privatize",
             "--bundle", bundle, ENTRY])

ext = Sys.isapple() ? "dylib" : Sys.iswindows() ? "dll" : "so"
lib = joinpath(bundle, "lib", "libetcouple.$ext")
if Sys.iswindows() && !isfile(lib)            # Windows bundles keep DLLs next to each other in bin/
   lib = joinpath(bundle, "bin", "libetcouple.$ext")
end
if !isfile(lib)
   found = [joinpath(r, f) for (r, _, fs) in walkdir(bundle) for f in fs if startswith(f, "libetcouple")]
   error("expected $lib; libetcouple files in the bundle: $found")
end

# privatize/bundling rewrites load commands, which invalidates macOS signatures;
# arm64 macOS refuses to load unsigned modified code, so re-sign ad hoc.
if Sys.isapple()
   for (root, _, files) in walkdir(bundle), f in files
      p = joinpath(root, f)
      (!islink(p) && endswith(f, ".dylib")) && run(`codesign --force --sign - $p`)
   end
end

src = TOML.parsefile(joinpath(LIBPROJ, "Project.toml"))["sources"]["EquivariantTensors"]
info = Dict("abi" => 1, "max_order" => 8, "et_repo" => src["url"], "et_rev" => src["rev"],
            "julia" => string(VERSION), "juliac" => string(pkgversion(JuliaC)),
            "platform" => "$(Sys.KERNEL)-$(Sys.ARCH)", "cpu_target" => "generic")
open(joinpath(bundle, "build_info.json"), "w") do io
   print(io, "{", join(["\"$k\": " * (v isa Number ? string(v) : "\"$v\"") for (k, v) in sort(collect(info); by = first)], ", "), "}")
end
println("built $lib")
