#!/usr/bin/env bash
# The ACEpotentials.jl trim line on Sulis: the accuracy-cost `acepotentials-trim` CPU rows
# (bench/accuracy_cost/build_trim.py, timing.py).  Run on the login node (it downloads):
#
#   bash bench/scaling/envs/sulis-acepot.sh [step ...]      # default: all steps
#
# Steps:
#   julia       the official Julia $JV tarball, checked against julialang's sha256 list -> $ROOT/julia-$JV
#   julia_env   instantiate the pinned PR 309 env (bench/scaling/julia, ACEpotentials v0.10.2) into its
#               own depot, $ROOT/julia-pr309 (never ~/.julia); precompiling is left to the build job
#   ace_plugin  PR 309's export/lammps/plugin built with mpicxx against the lammps-dev tree that
#               build-kk-cpu was built from (scaling/envs/sulis-lammps-cpu-build.sbatch) -> $ROOT/ace-plugin-cpu
#   wrapper     $ROOT/lmp-ace-cpu.sh: build-kk-cpu's lmp with its modules (the input `plugin load`s the plugin)
# accuracy_cost/envs/sulis-cpu-trim.json names these paths.  The trim libraries themselves are built
# on a compute node (juliac targets the build CPU): accuracy_cost/envs/sulis-cpu-trim.sbatch.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ACEJAX=$(cd "$HERE/../../.." && pwd)                 # this checkout ($ROOT/ace-jax-trim)
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
JV=1.12.6
JDIR=$ROOT/julia-$JV
JULIA=$JDIR/bin/julia
JULIA_DEPOT=$ROOT/julia-pr309
JULIA_PROJECT_DIR=$ACEJAX/bench/scaling/julia
MODS="GCC/12.3.0 OpenMPI/4.1.5 CMake/3.26.3"        # build-kk-cpu's (sulis-lammps-cpu-build.sbatch)
LAMMPS_DEV=$ROOT/lammps-dev
PLUGIN_BUILD=$ROOT/ace-plugin-cpu
source /etc/profile.d/modules.sh 2>/dev/null || true

julia() {
  [ -x "$JULIA" ] && return 0
  local tgz=julia-$JV-linux-x86_64.tar.gz url=https://julialang-s3.julialang.org/bin
  local tmp; tmp=$(mktemp -d "$ROOT/julia-dl.XXXX")
  curl -fsSL -o "$tmp/$tgz" "$url/linux/x64/${JV%.*}/$tgz"
  curl -fsSL -o "$tmp/sums" "$url/checksums/julia-$JV.sha256"
  (cd "$tmp" && grep " $tgz\$" sums | sha256sum -c -)
  mkdir -p "$JDIR" && tar -xzf "$tmp/$tgz" -C "$JDIR" --strip-components=1
  rm -rf "$tmp"
  "$JULIA" --version
}

julia_env() {
  JULIA_DEPOT_PATH=$JULIA_DEPOT JULIA_PKG_PRECOMPILE_AUTO=0 "$JULIA" --startup-file=no \
    --project="$JULIA_PROJECT_DIR" -e 'using Pkg; Pkg.instantiate(); println(Pkg.project().path)'
}

ace_plugin() {
  module purge; module load $MODS
  local src
  src=$(JULIA_DEPOT_PATH=$JULIA_DEPOT "$JULIA" --startup-file=no --project="$JULIA_PROJECT_DIR" \
        -e 'print(dirname(dirname(Base.locate_package(Base.PkgId(Base.UUID("3b96b61c-0fcc-4693-95ed-1ef9f35fcc53"), "ACEpotentials")))))')/export/lammps/plugin/cmake
  [ -f "$PLUGIN_BUILD/aceplugin.so" ] && [ "$(cat "$PLUGIN_BUILD/.src" 2>/dev/null)" = "$src" ] && return 0
  rm -rf "$PLUGIN_BUILD"
  cmake -S "$src" -B "$PLUGIN_BUILD" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=mpicxx \
    -D LAMMPS_SOURCE_DIR="$LAMMPS_DEV/src"
  cmake --build "$PLUGIN_BUILD" -j 8
  echo "$src" > "$PLUGIN_BUILD/.src"
  ls -la "$PLUGIN_BUILD/aceplugin.so"
}

wrapper() {
  local b=$LAMMPS_DEV/build-kk-cpu
  cat > "$ROOT/lmp-ace-cpu.sh" <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load $MODS
export LD_LIBRARY_PATH=$b:\${LD_LIBRARY_PATH:-}
exec $b/lmp "\$@"
EOF
  chmod +x "$ROOT/lmp-ace-cpu.sh"
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(julia julia_env ace_plugin wrapper)
for s in "${steps[@]}"; do echo "== $s"; "$s"; done
