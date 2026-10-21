#!/usr/bin/env bash
# GRACE on Sulis: the gracemaker venv (fits, gate references) and LAMMPS from the yury-lysogorskiy/lammps `grace`
# branch (b1f4a6c, 2026-10-15) with ML-PACE = lammps-user-pace v.2025.12.4.p1 (the release that branch's ML-PACE.cmake
# pins), built without TensorFlow (-D NO_GRACE_TF=ON): pair_style grace/1l/kk, grace/1l/cpu/kk (gracemaker
# export_kokkos .npz; full float64 by default) and grace/fs, grace/fs/kk (C++ YAML).  The TensorFlow pair_style
# grace (SavedModel) is not built: no timed model needs it.
#
# Staged from the desktop (no clone on Sulis), then the steps:
#   rsync -a /storage/eng/essswb/projects/ace-jax/grace-lammps/{lammps-grace,lammps-user-pace-v.2025.12.4.p1} sulis:bench-scaling/
#   rsync -a --exclude .git --exclude .venv <worktree>/ sulis:bench-scaling/ace-jax-grace/
#   bash sulis-grace.sh venv                      (login node: uv sync --frozen of accuracy_cost/envs/grace)
#   sbatch sulis-grace-build-cpu.sbatch; sbatch sulis-grace-build-gpu.sbatch
# Wrappers: $ROOT/lmp-grace-cpu.sh, $ROOT/lmp-grace.sh (accuracy_cost/envs/sulis-{cpu,gpu}-grace.json `lmp`).
#
# Kokkos 1L limits (gracemaker _kokkos_export CAPS_1L): n_rad_max <= 32, MLP widths <= 64; and the kernel reads an
# order-4 (AA2) block unconditionally, so max_order 3 models export but die of SIGFPE in copy_weights_to_device.
# gracemaker's docs: CUDA >= 12.8 (12.2-12.6 can hang for hours compiling pair_grace_2l_kokkos.cpp).
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SRC=$ROOT/lammps-grace
PACE=$ROOT/lammps-user-pace-v.2025.12.4.p1
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

venv() {      # login node (needs the network): the grace venv from the committed lock
  export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/uv-cache}
  UV_PROJECT_ENVIRONMENT=$ROOT/venv-grace ~/.local/bin/uv sync --frozen -p 3.12 \
    --project "$ROOT/ace-jax-grace/bench/accuracy_cost/envs/grace"
  "$ROOT/venv-grace/bin/python" -c "import tensorflow as tf, importlib.metadata as m; print('tensorflow', tf.__version__, 'tensorpotential', m.version('tensorpotential'))"
}

cpu() {
  local B=$SRC/build-kk-cpu
  set +u; module purge; module load $MODS_CPU; set -u
  rm -rf "$B"
  cmake -S "$SRC/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_ZEN2=ON \
    -D PKG_ML-PACE=ON -D LOCAL_ML-PACE="$PACE" -D NO_GRACE_TF=ON
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-grace-cpu.sh "$MODS_CPU" "$B"
}

gpu() {
  local B=$SRC/build-kk
  set +u; module purge; module load $MODS_GPU; set -u
  export NVCC_WRAPPER_DEFAULT_COMPILER=g++
  rm -rf "$B"
  cmake -S "$SRC/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER="$SRC/lib/kokkos/bin/nvcc_wrapper" \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ARCH_AMPERE80=ON -D PKG_ML-PACE=ON -D LOCAL_ML-PACE="$PACE" -D NO_GRACE_TF=ON
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-grace.sh "$MODS_GPU" "$B"
}

echo "lammps-grace $(cat "$SRC/REV" 2>/dev/null || echo b1f4a6c)"
for step in "$@"; do "$step"; done
