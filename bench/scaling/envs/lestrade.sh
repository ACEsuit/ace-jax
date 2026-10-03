#!/usr/bin/env bash
# Benchmark environment for the lestrade-cpu host (i9-14900K; run on lestrade).
#
#   bash bench/scaling/envs/lestrade.sh [step ...]    # default: all steps
#
# Reuses moriarty's environment through the shared home, read-only: its Python venv,
# its lammps-dev headers, and its pinned lammps + symmetrix checkouts (copied, see
# below).  The ML-PACE rows recorded before `lmp-cpu.sh` existed ran moriarty's
# `lmp.sh` (the patch_10Sep2025 tree, ML-PACE in plain CPU mode).  Nothing under ~/bench-scaling is written.  What is
# lestrade-specific lives under $ROOT on /storage: the PR 309 LAMMPS plugin, built
# against lammps-dev (`ace_plugin`), the `lmp-ace.sh` and `lmp-cpu.sh` wrappers (`wrapper`), and the
# git-ignored envs/lestrade-cpu.json (`env_json`).  The trim libraries are built on
# lestrade too (`models.py acepotentials-trim`, into this checkout's git-ignored
# models/trim/): juliac targets the build CPU.
#
# MACE in LAMMPS needs its own tree: moriarty's Symmetrix build dies here with
# SIGILL (libsymmetrix is compiled -march=native, i.e. AVX-512 on moriarty's Cascade
# Lake; the i9-14900K has none).  `symmetrix_src` copies moriarty's pinned checkouts
# (lammps patch_10Sep2025 9792f6a, symmetrix 0d86e1e: envs/moriarty-VERSIONS) to
# $ROOT/src and re-runs pair_symmetrix's install.sh against the copy; `lammps_cpu`
# builds it CPU-only (no Kokkos: plain pair_style symmetrix/mace needs none) with
# ML-PACE, BLAS from the system FlexiBLAS (as on moriarty) and CMake >= 3.27
# (sphericart; /usr/bin/cmake is 3.31).  `lmp-cpu.sh` wraps it, and env_json's
# `lmp` points there, so the mace and mlpace LAMMPS lines both run it.
#
# Steps: ace_plugin symmetrix_src lammps_cpu wrapper env_json
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ACEJAX=$(cd "$HERE/../../.." && pwd)                 # this checkout
ROOT=${LESTRADE_ROOT:-/storage/eng/essswb/bench-scaling-lestrade}
MORIARTY=${MORIARTY_ROOT:-$HOME/bench-scaling}       # read-only
LAMMPS_DEV=$MORIARTY/lammps-dev
BUILD_DEV=$LAMMPS_DEV/build-kk
LAMMPS_CPU=$ROOT/src/lammps                          # copy of moriarty's lammps + pair_symmetrix
SYMMETRIX=$ROOT/src/symmetrix
BUILD_CPU=$LAMMPS_CPU/build-cpu
LAMMPS_REV=9792f6a9a32517780a8276c8ab201f17cae37d6b   # patch_10Sep2025
SYMMETRIX_REV=0d86e1e4467640f752ec8569c0be00b2c7b467a4
JOBS=${JOBS:-8}                                      # agent scopes cap memory: keep -j low
BUILD_CPUS=${BUILD_CPUS:-16-31}                      # build on the E-cores
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

symmetrix_src() {      # copies of the pinned trees; pair_symmetrix installed against the copy
  mkdir -p "$ROOT/src"
  [ -d "$LAMMPS_CPU/.git" ] || rsync -a --exclude=/build-kk "$MORIARTY/lammps/" "$LAMMPS_CPU/"
  [ -d "$SYMMETRIX/.git" ] || rsync -a --exclude=/symmetrix/build "$MORIARTY/symmetrix/" "$SYMMETRIX/"
  [ "$(git -C "$LAMMPS_CPU" rev-parse HEAD)" = $LAMMPS_REV ] || { echo "lammps not at $LAMMPS_REV"; exit 1; }
  [ "$(git -C "$SYMMETRIX" rev-parse HEAD)" = $SYMMETRIX_REV ] || { echo "symmetrix not at $SYMMETRIX_REV"; exit 1; }
  # the copied CMakeLists.txt and src/ symlinks point at moriarty's symmetrix: redo them
  if ! grep -q "$SYMMETRIX/pair_symmetrix" "$LAMMPS_CPU/cmake/CMakeLists.txt"; then
    git -C "$LAMMPS_CPU" checkout -q cmake/CMakeLists.txt
    rm -f "$LAMMPS_CPU"/src/pair_symmetrix_mace.{h,cpp} "$LAMMPS_CPU"/src/KOKKOS/pair_symmetrix_mace_kokkos.{h,cpp}
    (cd "$SYMMETRIX/pair_symmetrix" && ./install.sh "$LAMMPS_CPU")
  fi
  { git -C "$LAMMPS_CPU" log -1 --format='lammps %h %cd'
    git -C "$SYMMETRIX" log -1 --format='symmetrix %h %cd'; } > "$ROOT/src/VERSIONS"
}

lammps_cpu() {         # CPU-only: MPI + OpenMP, ML-PACE, Symmetrix without Kokkos
  [ -x "$BUILD_CPU/lmp" ] && return 0
  module unload CUDA
  rm -rf "$BUILD_CPU"
  # no global -march: like moriarty's tree, only libsymmetrix gets -march=native
  # (its own CMakeLists), which here is Raptor Lake (AVX2, no AVX-512)
  cmake -S "$LAMMPS_CPU/cmake" -B "$BUILD_CPU" \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON -D PKG_KOKKOS=OFF \
    -D PKG_ML-PACE=ON -D SYMMETRIX_KOKKOS=OFF -D SYMMETRIX_SPHERICART_CUDA=OFF \
    -D BLA_VENDOR=FlexiBLAS
  taskset -c "$BUILD_CPUS" cmake --build "$BUILD_CPU" -j "$JOBS"
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
  # the mace + mlpace lines: the lestrade CPU tree (no CUDA module)
  cat > "$ROOT/lmp-cpu.sh" <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load gompi/2023a
export LD_LIBRARY_PATH=$BUILD_CPU:\${LD_LIBRARY_PATH:-}
exec $BUILD_CPU/lmp "\$@"
EOF
  chmod +x "$ROOT/lmp-cpu.sh"
}

env_json() {           # mpirun is not on a non-login shell's PATH: os_env puts OpenMPI's there
  local mpibin
  mpibin=$(dirname "$(command -v mpirun)")
  cat > "$HERE/lestrade-cpu.json" <<EOF
{"lmp": "$ROOT/lmp-cpu.sh", "pythonpath": "$ACEJAX/bench:$ACEJAX/src",
 "python": "$MORIARTY/venv/bin/python", "root": "$ROOT",
 "lmp_ace": "$ROOT/lmp-ace.sh", "ace_plugin": "$ROOT/ace-plugin/aceplugin.so",
 "julia": "$JULIA", "julia_depot": "$JULIA_DEPOT", "julia_project": "$ACEJAX/bench/scaling/julia",
 "os_env": {"PATH": "$mpibin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"}}
EOF
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(ace_plugin symmetrix_src lammps_cpu wrapper env_json)
for s in "${steps[@]}"; do
  echo "=== $s"
  "$s"
done
echo "=== done"
