#!/usr/bin/env bash
# NEP for the accuracy-cost benchmark on Sulis (bench/accuracy_cost/fit_nep.py):
#   gpumd  GPUMD's `nep` trainer (A100, sm_80)                      -> $ROOT/GPUMD/src/nep
#   cpu    LAMMPS with NEP_CPU's USER-NEP (`pair_style nep`, float64) -> $ROOT/lmp-nep-cpu.sh
# The CPU LAMMPS is the lammps-dev tree (develop ec02ed0, as every other CPU row) copied to lammps-nep, so the
# shared tree's CMakeLists stays as it is; configured as sulis-pacetab.sh's CPU build, minus ML-PACE.
#
#   rsync -a /storage/eng/essswb/src/{GPUMD,NEP_CPU} sulis:bench-scaling/      (GPUMD v5.9.1, NEP_CPU af615ce)
#   (from bench/scaling/envs) sbatch sulis-nep-build-gpu.sbatch; sbatch sulis-nep-build-cpu.sbatch
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SRC=$ROOT/lammps-dev
DST=$ROOT/lammps-nep
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-16}}
MODS_CPU="GCC/12.3.0 OpenMPI/4.1.5 CMake/3.26.3"
MODS_GPU="GCC/12.3.0 CUDA/12.8.0"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; set -u

gpumd() {
  set +u; module purge; module load $MODS_GPU; set -u
  cd "$ROOT/GPUMD/src"
  make clean >/dev/null 2>&1 || true
  make -j "$JOBS" CUDA_ARCH=-arch=sm_80 nep
  ./nep --help >/dev/null 2>&1 || true          # nep reads nep.in from cwd; this only checks it loads
  printf '%s\n' '#!/usr/bin/env bash' 'source /etc/profile.d/modules.sh 2>/dev/null || true' \
    "module purge; module load $MODS_GPU" "exec $ROOT/GPUMD/src/nep \"\$@\"" > "$ROOT/nep.sh"
  chmod +x "$ROOT/nep.sh"
}

cpu() {
  set +u; module purge; module load $MODS_CPU; set -u
  rsync -a --delete --exclude 'build*' --exclude doc --exclude examples --exclude bench --exclude unittest \
    --exclude .git "$SRC/" "$DST/"
  mkdir -p "$DST/src/USER-NEP"
  cp "$ROOT/NEP_CPU/src/"* "$ROOT/NEP_CPU/interface/lammps/USER-NEP/"* "$DST/src/USER-NEP/"
  cp "$ROOT/NEP_CPU/interface/lammps/USER-NEP.cmake" "$DST/cmake/Modules/Packages/"
  grep -q "USER-NEP" "$DST/cmake/CMakeLists.txt" || {                # NEP_CPU's README registration
    sed -i '/foreach(PKG_WITH_INCL / s/)/ USER-NEP)/' "$DST/cmake/CMakeLists.txt"
    sed -i '/set(STANDARD_PACKAGES/,/)/ s/)/  \n  USER-NEP)/' "$DST/cmake/CMakeLists.txt"
  }
  local B=$DST/build-cpu-nep
  rm -rf "$B"
  cmake -S "$DST/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON -D PKG_USER-NEP=ON
  cmake --build "$B" -j "$JOBS"
  printf '%s\n' '#!/usr/bin/env bash' 'source /etc/profile.d/modules.sh 2>/dev/null || true' \
    "module purge; module load $MODS_CPU" "export LD_LIBRARY_PATH=$B:\${LD_LIBRARY_PATH:-}" \
    "exec $B/lmp \"\$@\"" > "$ROOT/lmp-nep-cpu.sh"
  chmod +x "$ROOT/lmp-nep-cpu.sh"
  "$ROOT/lmp-nep-cpu.sh" -h | grep -q "^nep\b\|[[:space:]]nep[[:space:]]" && echo "pair_style nep: built" \
    || { echo "pair_style nep missing from $B/lmp"; exit 1; }
}

"$@"
