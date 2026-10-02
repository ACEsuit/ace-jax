#!/usr/bin/env bash
# Benchmark environment for the lestrade-cpu host (i9-14900K; run on lestrade).
#
#   bash bench/scaling/envs/lestrade.sh [step ...]    # default: all steps
#
# Reuses moriarty's environment through the shared home, read-only: its Python venv,
# its `lmp.sh` (the patch_10Sep2025 tree, which runs ML-PACE in plain CPU mode here)
# and its lammps-dev headers.  Nothing under ~/bench-scaling is written.  What is
# lestrade-specific lives under $ROOT on /storage: the PR 309 LAMMPS plugin, built
# against lammps-dev (`ace_plugin`), the `lmp-ace.sh` wrapper (`wrapper`), and the
# git-ignored envs/lestrade-cpu.json (`env_json`).  The trim libraries are built on
# lestrade too (`models.py acepotentials-trim`, into this checkout's git-ignored
# models/trim/): juliac targets the build CPU.
#
# Steps: ace_plugin wrapper env_json
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ACEJAX=$(cd "$HERE/../../.." && pwd)                 # this checkout
ROOT=${LESTRADE_ROOT:-/storage/eng/essswb/bench-scaling-lestrade}
MORIARTY=${MORIARTY_ROOT:-$HOME/bench-scaling}       # read-only
LAMMPS_DEV=$MORIARTY/lammps-dev
BUILD_DEV=$LAMMPS_DEV/build-kk
JULIA=${ACEPOT_JULIA:-$HOME/.juliaup/bin/julia +1.12.6}
JULIA_DEPOT=${ACEPOT_JULIA_DEPOT:-/storage/eng/essswb/cache/julia-pr309}
mkdir -p "$ROOT"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load gompi/2023a CUDA/12.9.0                   # lammps-dev's modules (mpicxx: OpenMPI 4.1.5)

ace_plugin() {         # moriarty.sh's step, with its outputs redirected here
  BENCH_ROOT=$ROOT ACEJAX_SRC=$ACEJAX LAMMPS_DEV=$LAMMPS_DEV BUILD_DEV=$BUILD_DEV \
    ACE_PLUGIN_BUILD=$ROOT/ace-plugin ACEPOT_JULIA="$JULIA" ACEPOT_JULIA_DEPOT=$JULIA_DEPOT \
    bash "$HERE/moriarty.sh" ace_plugin
}

wrapper() {            # the trim line: lammps-dev on the CPU (the input loads the plugin)
  cat > "$ROOT/lmp-ace.sh" <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load gompi/2023a CUDA/12.9.0
export LD_LIBRARY_PATH=$BUILD_DEV:\${LD_LIBRARY_PATH:-}
exec $BUILD_DEV/lmp "\$@"
EOF
  chmod +x "$ROOT/lmp-ace.sh"
}

env_json() {           # mpirun is not on a non-login shell's PATH: os_env puts OpenMPI's there
  local mpibin
  mpibin=$(dirname "$(command -v mpirun)")
  cat > "$HERE/lestrade-cpu.json" <<EOF
{"lmp": "$MORIARTY/lmp.sh", "pythonpath": "$ACEJAX/bench:$ACEJAX/src",
 "python": "$MORIARTY/venv/bin/python", "root": "$ROOT",
 "lmp_ace": "$ROOT/lmp-ace.sh", "ace_plugin": "$ROOT/ace-plugin/aceplugin.so",
 "julia": "$JULIA", "julia_depot": "$JULIA_DEPOT", "julia_project": "$ACEJAX/bench/scaling/julia",
 "os_env": {"PATH": "$mpibin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"}}
EOF
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(ace_plugin wrapper env_json)
for s in "${steps[@]}"; do
  echo "=== $s"
  "$s"
done
echo "=== done"
