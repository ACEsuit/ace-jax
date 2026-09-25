#!/usr/bin/env bash
# Benchmark environment on moriarty (Rocky 9, RTX A4500): one LAMMPS tree with
# KOKKOS (CUDA sm_86 + OpenMP), ML-PACE and Symmetrix; the lammps-jax plugin;
# and a Python venv (JAX CUDA, ace-jax, lammps-jax, mace-torch, symmetrix).
# Idempotent: re-running skips finished steps.  Everything lives under $ROOT in
# the shared home, so other nodes see it.
#
#   bash bench/scaling/envs/moriarty.sh [step ...]    # default: all steps
#
# Steps: sources lammps venv lmp_python symmetrix_py plugin env_json
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
ACEJAX=${ACEJAX_SRC:-$ROOT/ace-jax}          # synced checkout of this branch
LAMMPS=$ROOT/lammps
BUILD=$LAMMPS/build-kk
VENV=$ROOT/venv
JOBS=${JOBS:-24}
mkdir -p "$ROOT"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load gompi/2023a CUDA/12.4.0          # GCC 12.3 + OpenMPI; C++20 for Symmetrix
export PATH=$HOME/.local/bin:$PATH

sources() {
  cd "$ROOT"
  [ -d lammps ] || git clone --depth 1 -b patch_10Sep2025 https://github.com/lammps/lammps.git   # pinned: Symmetrix fails on current develop
  [ -d symmetrix ] || git clone --recursive https://github.com/wcwitt/symmetrix.git
  [ -d lammps-jax ] || git clone https://github.com/abhijeetgangan/lammps-jax.git
  # patch LAMMPS with pair_symmetrix (copies sources into src/)
  (cd symmetrix/pair_symmetrix && ./install.sh "$LAMMPS")
  git -C "$LAMMPS" log -1 --format='lammps %h %cd' > "$ROOT/VERSIONS"
  git -C symmetrix log -1 --format='symmetrix %h %cd' >> "$ROOT/VERSIONS"
  git -C lammps-jax log -1 --format='lammps-jax %h %cd' >> "$ROOT/VERSIONS"
}

venv() {
  [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
  # pip too: LAMMPS's install-python target calls `python -m pip`
  uv pip install --python "$VENV/bin/python" -q pip "jax[cuda12]" matscipy ase matplotlib pyyaml \
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

symmetrix_py() {        # for symmetrix_extract_mace (model export only)
  "$VENV/bin/python" -c "import symmetrix" 2>/dev/null || \
    uv pip install --python "$VENV/bin/python" "$ROOT/symmetrix/symmetrix"
}

plugin() {
  local inc pjrt
  inc=$("$VENV/bin/python" -c "import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")
  cmake -S "$ROOT/lammps-jax/cpp" -B "$ROOT/lammps-jax/build-plugin-gpu-pjrt" \
    -D CMAKE_CXX_COMPILER="$LAMMPS/lib/kokkos/bin/nvcc_wrapper" -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_FLAGS="-fno-lto" -D CMAKE_SHARED_LINKER_FLAGS="-fno-lto" \
    -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -D LAMMPS_HEADER_DIR="$LAMMPS/src" -D JAXLIB_INCLUDE_DIR="$inc" \
    -D KOKKOS_CONFIG_INCLUDE_DIR="$BUILD/lib/kokkos"
  cmake --build "$ROOT/lammps-jax/build-plugin-gpu-pjrt" -j "$JOBS"
}

env_json() {
  local pjrt
  pjrt=$("$VENV/bin/python" -c "import jax_plugins.xla_cuda12 as p, os; print(os.path.join(os.path.dirname(p.__file__), 'xla_cuda_plugin.so'))")
  for host in moriarty-gpu moriarty-cpu; do
    cat > "$ACEJAX/bench/scaling/envs/$host.json" <<EOF
{"lmp": "$ROOT/lmp.sh", "pjrt": "$pjrt", "pythonpath": "$ACEJAX/bench",
 "python": "$VENV/bin/python", "root": "$ROOT"}
EOF
  done
  # lmp wrapper: modules + plugin path, so every LAMMPS call sees the same env
  cat > "$ROOT/lmp.sh" <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load gompi/2023a CUDA/12.4.0
export LAMMPS_PLUGIN_PATH=$ROOT/lammps-jax/build-plugin-gpu-pjrt
export LD_LIBRARY_PATH=$BUILD:\${LD_LIBRARY_PATH:-}
exec $BUILD/lmp "\$@"
EOF
  chmod +x "$ROOT/lmp.sh"
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(sources venv lammps lmp_python symmetrix_py plugin env_json)
for s in "${steps[@]}"; do
  echo "=== $s"
  "$s"
done
echo "=== done"
