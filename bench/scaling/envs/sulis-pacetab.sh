#!/usr/bin/env bash
# LAMMPS (the lammps-dev tree, develop ec02ed0) with ML-PACE built from a local lammps-user-pace carrying the
# tabulated-radial patch (`radbasename: Tabulated`, docs/dev/ml-pace-tabulated-radials.md), for timing ace-jax
# ACE models exported exactly by write_yace (ace_to_yace.py --radial tabulated).  Configured as lammps-dev's build-kk-cpu
# (Kokkos OpenMP, Zen 2) and build-kk (Kokkos CUDA, A100) are, only LOCAL_ML-PACE differs; the patch is on
# top of the release that tree pins (v.2025.12.4.p1), so stock ML-PACE rows time the same code otherwise.
#
#   lammps-user-pace v.2025.12.4.p1 + pace_ref/tabulated/0001-tabulated-radials.patch (upstream:
#   ICAMS/lammps-user-pace#31), then
#   rsync -a <that checkout>/ sulis:bench-scaling/lammps-user-pace-tab/
#   (from bench/scaling/envs) sbatch sulis-pacetab-build-cpu.sbatch; sbatch sulis-pacetab-build-gpu.sbatch
#
# Wrappers: $ROOT/lmp-pacetab-cpu.sh, $ROOT/lmp-pacetab.sh (the env jsons' `lmp`).
#
# `cpufast`: the same CPU build from a copy of that tree carrying pace_ref/fast/0002-recursive-evaluator-fast.patch
# with -DNO_DB_FILL -DFAST_DAG -DFAST_AF (the recursive evaluator only: no dB_flatten clear the recursive path never
# reads, register-carried DAG accumulation, Y / dY cached once per neighbour; same E/F to ~1e-13).  pace/kk is
# untouched.  Wrapper $ROOT/lmp-pacefast-cpu.sh (accuracy_cost/envs/sulis-cpu-pacefast.json).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SRC=$ROOT/lammps-dev
PACE=$ROOT/lammps-user-pace-tab
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

cpu() {
  local B=$SRC/build-kk-cpu-pacetab
  set +u; module purge; module load $MODS_CPU; set -u
  rm -rf "$B"
  cmake -S "$SRC/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_ZEN2=ON \
    -D PKG_ML-PACE=ON -D LOCAL_ML-PACE="$PACE"
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-pacetab-cpu.sh "$MODS_CPU" "$B"
}

cpufast() {
  local B=$SRC/build-kk-cpu-pacefast F=$ROOT/lammps-user-pace-fast
  rm -rf "$F"; cp -a "$PACE" "$F"
  patch -d "$F" -p1 < "$HERE/../../../pace_ref/fast/0002-recursive-evaluator-fast.patch"
  set +u; module purge; module load $MODS_CPU; set -u
  rm -rf "$B"
  cmake -S "$SRC/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ \
    -D CMAKE_CXX_FLAGS="-DNO_DB_FILL -DFAST_DAG -DFAST_AF" \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_ZEN2=ON \
    -D PKG_ML-PACE=ON -D LOCAL_ML-PACE="$F"
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-pacefast-cpu.sh "$MODS_CPU" "$B"
}

gpu() {
  local B=$SRC/build-kk-pacetab
  set +u; module purge; module load $MODS_GPU; set -u
  export NVCC_WRAPPER_DEFAULT_COMPILER=g++
  rm -rf "$B"
  cmake -S "$SRC/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER="$SRC/lib/kokkos/bin/nvcc_wrapper" \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ARCH_AMPERE80=ON -D PKG_ML-PACE=ON -D LOCAL_ML-PACE="$PACE"
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-pacetab.sh "$MODS_GPU" "$B"
}

git -C "$PACE" log -1 --format="lammps-user-pace-tab %h %s" 2>/dev/null || cat "$PACE/REV"
for step in "$@"; do "$step"; done
