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
# Wrappers: $ROOT/lmp-pacetab-cpu.sh, $ROOT/lmp-pacetab.sh (the env jsons' `lmp`), and with the gpu_mixed
# step $ROOT/lmp-pacetab-mixed.sh (KOKKOS_PREC=mixed; the env jsons' `lmp_mixed`).
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

gpu_mixed() {   # the same with KOKKOS_PREC=mixed (float compute, double accumulation): the mixed-precision ML-PACE rows
  local B=$SRC/build-kk-pacetab-mixed
  set +u; module purge; module load $MODS_GPU; set -u
  export NVCC_WRAPPER_DEFAULT_COMPILER=g++
  rm -rf "$B"
  cmake -S "$SRC/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER="$SRC/lib/kokkos/bin/nvcc_wrapper" \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ARCH_AMPERE80=ON -D PKG_ML-PACE=ON -D LOCAL_ML-PACE="$PACE" -D KOKKOS_PREC=mixed
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-pacetab-mixed.sh "$MODS_GPU" "$B"
}

# gpufork_mixed: main's gpufork (pace_ref/fast 0003, 0005, 0006, 0007 on a copy of lammps-dev, ML-PACE from
# ICAMS/lammps-user-pace#31 in $ROOT/lammps-user-pace-fork; PACE_KK_RHOW at run time) built with KOKKOS_PREC=mixed.
# The patches keep A, rho and the weights in KK_FLOAT and accumulate forces in KK_ACC_FLOAT, as stock pace/kk does.
# Wrapper $ROOT/lmp-pacekk-fork-mixed.sh (accuracy_cost/envs/sulis-gpu-fp32.json `lmp_mixed`).
gpufork_mixed() { PREC=mixed _gpupatched lammps-pacekk-fork-mixed "$ROOT/lammps-user-pace-fork" lmp-pacekk-fork-mixed.sh \
                       0003-pacekk-team-rho-weights.patch 0005-pacekk-summed-neighbour-A.patch \
                       0006-pacekk-flat-offset-rho-weights.patch 0007-pacekk-segmented-weights.patch; }

_gpupatched() {   # <tree name> <ML-PACE> <wrapper> <patch...>; $PREC: KOKKOS_PREC (default double)
  local T=$ROOT/$1 P=$2 W=$3 B; shift 3
  rm -rf "$T"; mkdir -p "$T"
  (cd "$SRC" && tar --exclude='./build*' --exclude=./.git --exclude=./doc --exclude=./examples -cf - .) | tar -xf - -C "$T"
  for p in "$@"; do patch -d "$T" -p1 < "$HERE/../../../pace_ref/fast/$p"; done
  B=$T/build-kk
  set +u; module purge; module load $MODS_GPU; set -u
  export NVCC_WRAPPER_DEFAULT_COMPILER=g++
  cmake -S "$T/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER="$T/lib/kokkos/bin/nvcc_wrapper" \
    -D CMAKE_CXX_FLAGS="-lineinfo" -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ARCH_AMPERE80=ON -D PKG_ML-PACE=ON -D LOCAL_ML-PACE="$P" -D KOKKOS_PREC="${PREC:-double}"
  cmake --build "$B" -j "$JOBS"
  wrapper "$W" "$MODS_GPU" "$B"
}

git -C "$PACE" log -1 --format="lammps-user-pace-tab %h %s" 2>/dev/null || cat "$PACE/REV"
for step in "$@"; do "$step"; done
