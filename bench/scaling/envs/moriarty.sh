#!/usr/bin/env bash
# Benchmark environment on moriarty (Rocky 9, RTX A4500): one LAMMPS tree with
# KOKKOS (CUDA sm_86 + OpenMP), ML-PACE and Symmetrix; the lammps-jax plugin;
# and a Python venv (JAX CUDA, ace-jax, lammps-jax, mace-torch, symmetrix).
# Idempotent: re-running skips finished steps.  Everything lives under $ROOT in
# the shared home, so other nodes see it.
#
#   bash bench/scaling/envs/moriarty.sh [step ...]    # default: all steps
#
# Two LAMMPS trees, because no single version builds both add-ons today:
#   lammps      patch_10Sep2025 + Symmetrix + ML-PACE   (MACE, ML-PACE rows)
#   lammps-dev  develop + ML-PACE + the lammps-jax plugin (ace-jax rows; the
#               plugin needs Pair::eflag_only, newer than 10 Sep 2025, while
#               Symmetrix does not compile against current develop)
#
# Steps: sources venv neighbours lammps lmp_python symmetrix_py lammps_dev plugin env_json
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
ACEJAX=${ACEJAX_SRC:-$ROOT/ace-jax}          # synced checkout of this branch
LAMMPS=$ROOT/lammps
BUILD=$LAMMPS/build-kk
LAMMPS_DEV=$ROOT/lammps-dev
BUILD_DEV=$LAMMPS_DEV/build-kk
VENV=$ROOT/venv
JOBS=${JOBS:-24}
mkdir -p "$ROOT"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load gompi/2023a CUDA/12.4.0          # GCC 12.3 + OpenMPI; C++20 for Symmetrix
# the lammps-jax plugin needs a newer nvcc (parenthesised aggregate init in
# emplace_back fails on 12.4's front end); the dev tree + plugin use this one
CUDA_DEV=CUDA/12.9.0
export PATH=$HOME/.local/bin:$PATH

sources() {
  cd "$ROOT"
  [ -d lammps ] || git clone --depth 1 -b patch_10Sep2025 https://github.com/lammps/lammps.git   # pinned: Symmetrix fails on current develop
  [ -d symmetrix ] || git clone --recursive https://github.com/wcwitt/symmetrix.git
  [ -d lammps-jax ] || git clone https://github.com/abhijeetgangan/lammps-jax.git
  [ -d lammps-dev ] || git clone --depth 1 -b develop https://github.com/lammps/lammps.git lammps-dev
  # patch LAMMPS with pair_symmetrix (copies sources into src/)
  (cd symmetrix/pair_symmetrix && ./install.sh "$LAMMPS")
  git -C "$LAMMPS" log -1 --format='lammps %h %cd' > "$ROOT/VERSIONS"
  git -C symmetrix log -1 --format='symmetrix %h %cd' >> "$ROOT/VERSIONS"
  git -C lammps-jax log -1 --format='lammps-jax %h %cd' >> "$ROOT/VERSIONS"
  git -C lammps-dev log -1 --format='lammps-dev %h %cd' >> "$ROOT/VERSIONS"
}

venv() {
  [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
  # pip too: LAMMPS's install-python target calls `python -m pip`
  uv pip install --python "$VENV/bin/python" -q pip "jax[cuda12]" matscipy ase matplotlib pyyaml psutil \
    mace-torch cuequivariance-torch cuequivariance-ops-torch-cu12 \
    -e "$ACEJAX" -e "$ROOT/lammps-jax"
}

lammps() {
  [ -x "$BUILD/lmp" ] && return 0
  cmake -S "$LAMMPS/cmake" -B "$BUILD" \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_CXX_COMPILER="$LAMMPS/lib/kokkos/bin/nvcc_wrapper" \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_AMPERE86=ON \
    -D PKG_ML-PACE=ON -D PKG_PYTHON=ON -D Python_EXECUTABLE="$VENV/bin/python" \
    -D SYMMETRIX_KOKKOS=ON -D SYMMETRIX_SPHERICART_CUDA=ON
  cmake --build "$BUILD" -j "$JOBS"
}

lmp_python() {
  cmake --build "$BUILD" --target install-python
}

lammps_dev() {          # develop + ML-PACE + PLUGIN, host for the lammps-jax plugin
  module swap CUDA/12.4.0 "$CUDA_DEV"
  # PLUGIN: LAMMPS_PLUGIN_PATH auto-loading of lammps_jaxplugin.so needs it
  local stamp="$CUDA_DEV PLUGIN"
  [ -x "$BUILD_DEV/lmp" ] && [ "$(cat "$BUILD_DEV/.cuda" 2>/dev/null)" = "$stamp" ] && return 0
  # a different nvcc: start clean (CMake drops the -D values on a compiler change)
  [ "$(cut -d' ' -f1 "$BUILD_DEV/.cuda" 2>/dev/null)" = "$CUDA_DEV" ] || rm -rf "$BUILD_DEV"
  cmake -S "$LAMMPS_DEV/cmake" -B "$BUILD_DEV" \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER="$LAMMPS_DEV/lib/kokkos/bin/nvcc_wrapper" \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_AMPERE86=ON -D PKG_ML-PACE=ON -D PKG_PLUGIN=ON
  cmake --build "$BUILD_DEV" -j "$JOBS"
  echo "$stamp" > "$BUILD_DEV/.cuda"
}

neighbours() {        # matscipy-neighbours with CUDA: the calculator's dense graph on the GPU (DLPack)
  [ "$("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('matscipy-neighbours'))" 2>/dev/null)" = "1.0.0" ] && return 0
  module swap CUDA/12.4.0 "$CUDA_DEV" 2>/dev/null || true
  CC=gcc CXX=g++ uv pip install --python "$VENV/bin/python" \
    -C cmake.define.ENABLE_CUDA=ON -C cmake.define.CMAKE_CUDA_ARCHITECTURES=86 \
    "matscipy-neighbours==1.0.0"
}

symmetrix_py() {        # for symmetrix_extract_mace (model export only)
  # not `import symmetrix`: from $ROOT the source dir imports as a namespace package
  [ -x "$VENV/bin/symmetrix_extract_mace" ] || \
    CC=gcc CXX=g++ SKBUILD_CMAKE_ARGS="-DKokkos_ENABLE_CUDA=OFF;-DSYMMETRIX_SPHERICART_CUDA=OFF;-DKokkos_ARCH_NATIVE=OFF" \
    uv pip install --python "$VENV/bin/python" "$ROOT/symmetrix/symmetrix"   # CPU-only: runs on any node
}

plugin() {
  local inc pjrt
  module swap CUDA/12.4.0 "$CUDA_DEV" 2>/dev/null || module load "$CUDA_DEV"
  inc=$("$VENV/bin/python" -c "import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")
  # always configure fresh: changing CMAKE_CXX_COMPILER on an existing cache makes
  # CMake wipe it and silently drop the -D values given here
  rm -rf "$ROOT/lammps-jax/build-plugin-gpu-pjrt"
  cmake -S "$ROOT/lammps-jax/cpp" -B "$ROOT/lammps-jax/build-plugin-gpu-pjrt" \
    -D CMAKE_CXX_COMPILER="$LAMMPS_DEV/lib/kokkos/bin/nvcc_wrapper" -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_FLAGS="-fno-lto -fopenmp" -D CMAKE_SHARED_LINKER_FLAGS="-fno-lto -fopenmp" \
    -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -D LAMMPS_HEADER_DIR="$LAMMPS_DEV/src" -D JAXLIB_INCLUDE_DIR="$inc" \
    -D KOKKOS_CONFIG_INCLUDE_DIR="$BUILD_DEV/lib/kokkos"
  cmake --build "$ROOT/lammps-jax/build-plugin-gpu-pjrt" -j "$JOBS"
}

env_json() {
  local pjrt
  pjrt=$("$VENV/bin/python" -c "import jax_plugins.xla_cuda12 as p, os; print(os.path.join(os.path.dirname(p.__file__), 'xla_cuda_plugin.so'))")
  for host in moriarty-gpu moriarty-cpu; do
    cat > "$ACEJAX/bench/scaling/envs/$host.json" <<EOF
{"lmp": "$ROOT/lmp.sh", "lmp_jax": "$ROOT/lmp-jax.sh", "pjrt": "$pjrt", "pythonpath": "$ACEJAX/bench",
 "python": "$VENV/bin/python", "root": "$ROOT"}
EOF
  done
  # lmp wrappers: modules (+ plugin path), so every LAMMPS call sees the same env
  cat > "$ROOT/lmp.sh" <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load gompi/2023a CUDA/12.4.0
export LD_LIBRARY_PATH=$BUILD:\${LD_LIBRARY_PATH:-}
exec $BUILD/lmp "\$@"
EOF
  cat > "$ROOT/lmp-jax.sh" <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load gompi/2023a $CUDA_DEV
export LAMMPS_PLUGIN_PATH=$ROOT/lammps-jax/build-plugin-gpu-pjrt
export LD_LIBRARY_PATH=$BUILD_DEV:\${LD_LIBRARY_PATH:-}
exec $BUILD_DEV/lmp "\$@"
EOF
  chmod +x "$ROOT/lmp.sh" "$ROOT/lmp-jax.sh"
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(sources venv neighbours lammps lmp_python symmetrix_py lammps_dev plugin env_json)
for s in "${steps[@]}"; do
  echo "=== $s"
  "$s"
done
echo "=== done"
