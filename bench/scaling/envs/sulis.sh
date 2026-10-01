#!/usr/bin/env bash
# Benchmark environment on Sulis (RHEL 8, A100 sm_80): the moriarty recipe
# (envs/moriarty.sh) with Sulis modules.  One LAMMPS tree with KOKKOS (CUDA
# sm_80 + OpenMP), ML-PACE and Symmetrix; the lammps-jax plugin on a second
# (develop) tree; and a Python venv (JAX CUDA, ace-jax, lammps-jax, mace-torch,
# symmetrix).  Idempotent: re-running skips finished steps.
#
#   bash bench/scaling/envs/sulis.sh [step ...]    # default: all steps
#
# Where to run it: `sources` and `venv` only clone and download, so they run on
# the login node; every other step compiles and runs inside a GPU allocation
# (envs/sulis-build.sbatch), since nvcc and Kokkos arch detection want the A100.
#
# Everything lives under $ROOT in home (/gpfs/home: 2 TiB / 2M-file quota,
# `mmlsquota`; /gpfs/scratch has no per-user directory).  The sources are
# pinned to the commits moriarty ran (envs/moriarty-VERSIONS): the ace-jax
# neighbour-matrix bundle layout needs lammps-jax 4a7f4fb.
#
# Two LAMMPS trees, as on moriarty:
#   lammps      patch_10Sep2025 + Symmetrix + ML-PACE   (MACE, ML-PACE rows)
#   lammps-dev  develop ec02ed0 + ML-PACE + PLUGIN      (ace-jax rows, via lammps-jax)
#
# nvcc: the lammps-jax plugin needs nvcc > 12.4 (parenthesised aggregate init in
# emplace_back); Sulis modules stop at CUDA 12.8.0, which is what both trees and
# the plugin use.  If 12.8 ever rejects the plugin, `nvcc129` installs a CUDA
# 12.9 toolkit with micromamba and CUDA_DEV=nvcc129 rebuilds lammps-dev and the
# plugin against it (CUDA_DEV=nvcc129 bash sulis.sh nvcc129 lammps_dev plugin env_json).
#
# Steps: sources venv neighbours lammps lmp_python symmetrix_py lammps_dev plugin env_json
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
ACEJAX=${ACEJAX_SRC:-$ROOT/ace-jax}          # rsynced from a checkout (src bench fixtures julia pyproject.toml)
LAMMPS=$ROOT/lammps
BUILD=$LAMMPS/build-kk
LAMMPS_DEV=$ROOT/lammps-dev
BUILD_DEV=$LAMMPS_DEV/build-kk
VENV=$ROOT/venv
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-16}}
ARCH=AMPERE80                                  # A100
MODS="GCC/12.3.0 OpenMPI/4.1.5 CUDA/12.8.0 CMake/3.26.3"   # gompi/2023a + CUDA; C++20 for Symmetrix
CUDA_DEV=${CUDA_DEV:-CUDA/12.8.0}             # the dev tree + plugin; "nvcc129": the micromamba toolkit
CUDA129=$ROOT/cuda-12.9
LAMMPS_REV=9792f6a9a32517780a8276c8ab201f17cae37d6b       # patch_10Sep2025
SYMMETRIX_REV=0d86e1e4467640f752ec8569c0be00b2c7b467a4
LAMMPS_JAX_REV=4a7f4fb27ecc689abd35c2a91833590e613edb51
LAMMPS_DEV_REV=ec02ed0f8b0b347d3693d96892a87499b24f40f9
mkdir -p "$ROOT"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load $MODS; set -u
export PATH=$HOME/.local/bin:$PATH
export CC=gcc CXX=g++

use_cuda_dev() {          # the nvcc for lammps-dev and the plugin
  if [ "$CUDA_DEV" = nvcc129 ]; then
    export CUDA_HOME=$CUDA129 CUDA_PATH=$CUDA129 CUDAToolkit_ROOT=$CUDA129
    export PATH=$CUDA129/bin:$PATH LD_LIBRARY_PATH=$CUDA129/lib:${LD_LIBRARY_PATH:-}
  else
    set +u; module swap CUDA/12.8.0 "$CUDA_DEV" 2>/dev/null || true; set -u
  fi
  export NVCC_WRAPPER_DEFAULT_COMPILER=g++
}

pin() {                   # pin <dir> <url> <sha>: a checkout of exactly that commit
  local dir=$1 url=$2 sha=$3
  if [ ! -d "$dir/.git" ]; then
    git init -q "$dir"; git -C "$dir" remote add origin "$url"
  fi
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$sha" ]; then
    git -C "$dir" fetch -q --depth 1 origin "$sha"
    git -C "$dir" checkout -q --detach FETCH_HEAD
  fi
}

sources() {
  cd "$ROOT"
  pin lammps https://github.com/lammps/lammps.git $LAMMPS_REV
  pin lammps-dev https://github.com/lammps/lammps.git $LAMMPS_DEV_REV
  pin lammps-jax https://github.com/abhijeetgangan/lammps-jax.git $LAMMPS_JAX_REV
  pin symmetrix https://github.com/wcwitt/symmetrix.git $SYMMETRIX_REV
  git -C symmetrix submodule update -q --init --recursive --depth 1
  # patch LAMMPS with pair_symmetrix (copies sources into src/)
  [ -f "$LAMMPS/src/pair_symmetrix_mace.cpp" ] || (cd symmetrix/pair_symmetrix && ./install.sh "$LAMMPS")
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

nvcc129() {               # fallback toolkit: CUDA 12.9 from conda-forge, via micromamba
  [ -x "$CUDA129/bin/nvcc" ] && return 0
  local mm=$ROOT/bin/micromamba
  if [ ! -x "$mm" ]; then
    mkdir -p "$ROOT/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$ROOT" bin/micromamba
  fi
  MAMBA_ROOT_PREFIX=$ROOT/mamba "$mm" create -y -q -p "$CUDA129" -c conda-forge \
    cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-driver-dev=12.9 cuda-nvtx-dev=12.9 \
    libcublas-dev=12.9 libcusparse-dev=12.9 libcufft-dev=12.9 libcurand-dev=12.9 "gxx_linux-64=12.*"
}

lammps() {
  [ -x "$BUILD/lmp" ] && return 0
  cmake -S "$LAMMPS/cmake" -B "$BUILD" \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_CXX_COMPILER="$LAMMPS/lib/kokkos/bin/nvcc_wrapper" \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_$ARCH=ON \
    -D PKG_ML-PACE=ON -D PKG_PYTHON=ON -D Python_EXECUTABLE="$VENV/bin/python" \
    -D SYMMETRIX_KOKKOS=ON -D SYMMETRIX_SPHERICART_CUDA=ON
  cmake --build "$BUILD" -j "$JOBS"
}

lmp_python() {
  cmake --build "$BUILD" --target install-python
}

lammps_dev() {          # develop + ML-PACE + PLUGIN, host for the lammps-jax plugin
  use_cuda_dev
  # PLUGIN: LAMMPS_PLUGIN_PATH auto-loading of lammps_jaxplugin.so needs it
  local stamp="$CUDA_DEV PLUGIN $ARCH"
  [ -x "$BUILD_DEV/lmp" ] && [ "$(cat "$BUILD_DEV/.cuda" 2>/dev/null)" = "$stamp" ] && return 0
  # a different nvcc: start clean (CMake drops the -D values on a compiler change)
  [ "$(cat "$BUILD_DEV/.cuda" 2>/dev/null)" = "$stamp" ] || rm -rf "$BUILD_DEV"
  cmake -S "$LAMMPS_DEV/cmake" -B "$BUILD_DEV" \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER="$LAMMPS_DEV/lib/kokkos/bin/nvcc_wrapper" \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_$ARCH=ON -D PKG_ML-PACE=ON -D PKG_PLUGIN=ON
  cmake --build "$BUILD_DEV" -j "$JOBS"
  echo "$stamp" > "$BUILD_DEV/.cuda"
}

neighbours() {        # matscipy-neighbours with CUDA: the calculator's dense graph on the GPU (DLPack)
  [ "$("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('matscipy-neighbours'))" 2>/dev/null)" = "1.0.0" ] && return 0
  uv pip install --python "$VENV/bin/python" \
    -C cmake.define.ENABLE_CUDA=ON -C cmake.define.CMAKE_CUDA_ARCHITECTURES=80 \
    "matscipy-neighbours==1.0.0"
}

symmetrix_py() {        # for symmetrix_extract_mace (model export only)
  [ -x "$VENV/bin/symmetrix_extract_mace" ] || \
    SKBUILD_CMAKE_ARGS="-DKokkos_ENABLE_CUDA=OFF;-DSYMMETRIX_SPHERICART_CUDA=OFF;-DKokkos_ARCH_NATIVE=OFF" \
    uv pip install --python "$VENV/bin/python" "$ROOT/symmetrix/symmetrix"   # CPU-only
}

plugin() {
  local inc
  use_cuda_dev
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
  local pjrt dev_env
  pjrt=$("$VENV/bin/python" -c "import jax_plugins.xla_cuda12 as p, os; print(os.path.join(os.path.dirname(p.__file__), 'xla_cuda_plugin.so'))")
  cat > "$ACEJAX/bench/scaling/envs/sulis-a100.json" <<EOF
{"lmp": "$ROOT/lmp.sh", "lmp_jax": "$ROOT/lmp-jax.sh", "pjrt": "$pjrt", "pythonpath": "$ACEJAX/bench",
 "python": "$VENV/bin/python", "root": "$ROOT"}
EOF
  if [ "$CUDA_DEV" = nvcc129 ]; then
    dev_env="module purge; module load ${MODS/CUDA\/12.8.0/}
export PATH=$CUDA129/bin:\$PATH LD_LIBRARY_PATH=$CUDA129/lib:\${LD_LIBRARY_PATH:-}"
  else
    dev_env="module purge; module load ${MODS/CUDA\/12.8.0/$CUDA_DEV}"
  fi
  # lmp wrappers: modules (+ plugin path), so every LAMMPS call sees the same env.
  # Without the Symmetrix tree (its build failed), lmp.sh runs the dev tree:
  # ML-PACE is there too, and the MACE gate then fails, blocking only MACE.
  if [ -x "$BUILD/lmp" ]; then
    cat > "$ROOT/lmp.sh" <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load $MODS
export LD_LIBRARY_PATH=$BUILD:\${LD_LIBRARY_PATH:-}
exec $BUILD/lmp "\$@"
EOF
  else
    echo "no $BUILD/lmp: lmp.sh uses the dev tree (no Symmetrix)"
    cat > "$ROOT/lmp.sh" <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
$dev_env
export LD_LIBRARY_PATH=$BUILD_DEV:\${LD_LIBRARY_PATH:-}
exec $BUILD_DEV/lmp "\$@"
EOF
  fi
  # the PJRT plugin must load the pip CUDA libraries it was built against
  # (venv nvidia/*/lib), not the module's 12.8 ones: with those first on the
  # path, XLA segfaults compiling the model (ConfigAssigner, at pair_coeff)
  local nv
  nv=$(ls -d "$VENV"/lib/python3.12/site-packages/nvidia/*/lib | grep -v /cu13/ | tr '\n' :)
  cat > "$ROOT/lmp-jax.sh" <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
$dev_env
export LAMMPS_PLUGIN_PATH=$ROOT/lammps-jax/build-plugin-gpu-pjrt
export LD_LIBRARY_PATH=$nv$BUILD_DEV:\${LD_LIBRARY_PATH:-}
exec $BUILD_DEV/lmp "\$@"
EOF
  chmod +x "$ROOT/lmp.sh" "$ROOT/lmp-jax.sh"
  cp "$ROOT/VERSIONS" "$ACEJAX/bench/scaling/envs/sulis-VERSIONS"
  { echo "modules $MODS"; echo "cuda_dev $CUDA_DEV"; } >> "$ACEJAX/bench/scaling/envs/sulis-VERSIONS"
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(sources venv neighbours lammps lmp_python symmetrix_py lammps_dev plugin env_json)
for s in "${steps[@]}"; do
  echo "=== $s $(date +%T)"
  "$s"
done
echo "=== done"
