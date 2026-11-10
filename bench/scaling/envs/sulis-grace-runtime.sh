#!/usr/bin/env bash
# grace-runtime on Sulis: LAMMPS from the yury-lysogorskiy/lammps `grace-runtime` branch (c90c3a5, 2026-11-05) with
# ICAMS/grace-runtime 0.1.4 (the revision that branch pins) and ML-PACE = lammps-user-pace v.2025.12.4.p1 (its
# ML-PACE.cmake pin; already on Sulis for sulis-grace.sh).  Pair styles grace/runtime(/kk) (GRACE-1L/2L/3L
# kokkos.npz: fp64 geometry and sums, fp32 network and radial tables -- the library has no float64 mode) and
# pace/runtime(/kk) (pacemaker's B-basis .yaml; float64, radial tables of 1024 intervals by default).
#
# The CPU styles are plain MPI + OpenMP (no KOKKOS), compiled -march=native: build on a compute node (AMD EPYC 7742,
# the timing nodes).  The GPU build is Kokkos CUDA (the fork's kokkos-cuda preset, Ampere80); CUDA 12.8 as for
# sulis-grace.sh (sm_80 SASS, so driver 560 runs it).  grace-runtime's docs: GCC >= 12 (older GCC vectorizes its
# CPU kernels poorly), CMake >= 3.22.
#
# Steps (login node first: the clones need the network, the build jobs do not):
#   bash sulis-grace-runtime.sh src
#   sbatch sulis-grace-runtime-build-cpu.sbatch; sbatch sulis-grace-runtime-build-gpu.sbatch
# Wrappers: $ROOT/lmp-gracert-cpu.sh, $ROOT/lmp-gracert.sh (accuracy_cost/envs/sulis-{cpu,gpu}-gracert.json `lmp`).
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SRC=$ROOT/lammps-gracert
GR=$ROOT/grace-runtime-0.1.4
PACE=$ROOT/lammps-user-pace-v.2025.12.4.p1
LAMMPS_REV=c90c3a5fd477ed49d13e743fed7ba20a9a81121d
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-16}}
MODS_CPU="GCC/12.3.0 OpenMPI/4.1.5 CMake/3.26.3"
MODS_GPU="GCC/12.3.0 OpenMPI/4.1.5 CUDA/12.8.0 CMake/3.26.3"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; set -u

wrapper() {   # wrapper <name> <modules> <build>
  printf '%s\n' '#!/usr/bin/env bash' 'source /etc/profile.d/modules.sh 2>/dev/null || true' \
    "module purge; module load $2" "export LD_LIBRARY_PATH=$3:\${LD_LIBRARY_PATH:-}" \
    "exec $3/lmp \"\$@\"" > "$ROOT/$1"
  chmod +x "$ROOT/$1"
}

src() {       # login node: the LAMMPS fork at its tested commit, grace-runtime at its pinned tag
  if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"; git -C "$SRC" remote add origin https://github.com/yury-lysogorskiy/lammps.git
  fi
  git -C "$SRC" fetch -q --depth 1 origin "$LAMMPS_REV"; git -C "$SRC" checkout -q FETCH_HEAD
  [ -d "$GR/.git" ] || git clone -q --depth 1 -b 0.1.4 https://github.com/ICAMS/grace-runtime.git "$GR"
  [ -d "$PACE" ] || { echo "missing $PACE (staged for sulis-grace.sh)"; exit 1; }
  grep -q '_grace_runtime_pin "0.1.4"' "$SRC/cmake/Modules/Packages/ML-PACE.cmake" || { echo "pin moved"; exit 1; }
  echo "lammps-gracert $(git -C "$SRC" rev-parse --short HEAD), grace-runtime $(git -C "$GR" describe --tags)"
}

cpu() {
  local B=$SRC/build-cpu
  set +u; module purge; module load $MODS_CPU; set -u
  rm -rf "$B"
  cmake -S "$SRC/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ -D CMAKE_CXX_FLAGS="-march=native" \
    -D BUILD_MPI=yes -D BUILD_OMP=yes -D PKG_ML-PACE=yes -D GRACE_BLAS=OFF \
    -D LOCAL_ML-PACE="$PACE" -D LOCAL_GRACE_RUNTIME="$GR"
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-gracert-cpu.sh "$MODS_CPU" "$B"
}

gpu() {
  local B=$SRC/build-cuda
  set +u; module purge; module load $MODS_GPU; set -u
  export NVCC_WRAPPER_DEFAULT_COMPILER=g++
  rm -rf "$B"
  cmake -S "$SRC/cmake" -B "$B" -C "$SRC/cmake/presets/kokkos-cuda.cmake" -D CMAKE_BUILD_TYPE=Release \
    -D Kokkos_ARCH_AMPERE80=yes -D BUILD_MPI=yes -D PKG_ML-PACE=yes -D GRACE_BLAS=OFF \
    -D LOCAL_ML-PACE="$PACE" -D LOCAL_GRACE_RUNTIME="$GR"
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-gracert.sh "$MODS_GPU" "$B"
}

for step in "$@"; do "$step"; done
