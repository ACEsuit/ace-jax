#!/usr/bin/env bash
# Symmetrix v0.3.0 (wcwitt/symmetrix fc7e532, 2026-11-09) on Sulis: the accuracy-cost benchmark's "mace-symx3" code,
# against the "mace" rows' Symmetrix 0d86e1e (2026-08-13; envs/sulis.sh).  v0.1.0-v0.2.0 reworked the generic
# symmetrix/mace (flattened kernels, splines in-kernel, OpenMP threads per rank on the CPU, fast math and one thread
# per edge on CUDA); v0.3.0 adds symmetrix/mace/foundation, which takes only preset shapes (96-256 channels: none of
# the benchmark's 8-64-channel fits), NequIP and polar MACE.  So the generic style is what is timed.
#
# Its own LAMMPS tree: the release 0165101 ("30 Sep 2026", as envs/sulis-xl.sh), whose bundled Kokkos 5.2.1 matches
# the kokkos-kernels 5.1.1 Symmetrix vendors (the "mace" tree, patch_10Sep2025, has Kokkos 4.6.2).  The model .json
# format changed too, so each MACE fit is extracted again with v0.3.0's symmetrix_extract_mace (model-symx3.json).
#
#   bash sulis-symx3.sh sources venv              # login node (network): clones, Python deps
#   sbatch sulis-symx3-build-cpu.sbatch            # py models lammps_cpu  (compute node: -march=native, EPYC 7742)
#   sbatch sulis-symx3-build-gpu.sbatch            # lammps_cuda           (an A100 node)
# Wrappers: $ROOT/lmp-symx3-cpu.sh, $ROOT/lmp-symx3.sh (accuracy_cost/envs/sulis-{cpu,gpu}-symx3.json `lmp_symx3`).
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SX=$ROOT/symmetrix-0.3.0
SYMMETRIX_REV=fc7e53276baa7728f08bee0ddd865a20f82ff998   # tag v0.3.0 (resolved by `sources`)
LAMMPS=$ROOT/lammps-symx3
LAMMPS_REV=0165101ce505734e8fb249694b22e295a7ab4182      # release branch, "30 Sep 2026"
VENV=$ROOT/venv-symx3
MACE_TORCH=0.3.16                                        # as the fits
MODS="GCC/12.3.0 OpenMPI/4.1.5 OpenBLAS/0.3.23"
CUDA_MOD=CUDA/12.8.0
MACE_DIR=${MACE_DIR:-$ROOT/ac/runs/accuracy_cost/mace}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-16}}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/uv-cache}
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge; module load $MODS; set -u
export CC=gcc CXX=g++

pin() {                   # pin <dir> <url> <sha>
  local dir=$1 url=$2 sha=$3
  [ -d "$dir/.git" ] || { git init -q "$dir"; git -C "$dir" remote add origin "$url"; }
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$sha" ]; then
    git -C "$dir" fetch -q --depth 1 origin "$sha"; git -C "$dir" checkout -q FETCH_HEAD
  fi
}

sources() {
  [ "$(git ls-remote https://github.com/wcwitt/symmetrix.git 'refs/tags/v0.3.0^{}' | cut -f1)" = "$SYMMETRIX_REV" ] \
    || { echo "v0.3.0 moved"; exit 1; }
  pin "$SX" https://github.com/wcwitt/symmetrix.git $SYMMETRIX_REV
  git -C "$SX" submodule update -q --init --recursive --depth 1
  pin "$LAMMPS" https://github.com/lammps/lammps.git $LAMMPS_REV
  grep -q libsymmetrix "$LAMMPS/cmake/CMakeLists.txt" || (cd "$SX/pair_symmetrix" && ./install.sh "$LAMMPS")
  { git -C "$SX" log -1 --format='symmetrix %h %cd (v0.3.0)'; git -C "$SX" submodule status
    git -C "$LAMMPS" log -1 --format='lammps %h %cd'; echo "modules $MODS (+ $CUDA_MOD for cuda)"; } > "$ROOT/VERSIONS-symx3"
}

venv() {                  # login node: the extractor's dependencies (the package itself builds in `py`, offline)
  [ -x "$VENV/bin/python" ] || ~/.local/bin/uv venv -q -p 3.12 "$VENV"
  ~/.local/bin/uv pip install -q --python "$VENV/bin/python" "mace-torch==$MACE_TORCH" "cmake>=3.27,<4" \
    "scikit-build-core>=1.0" pybind11 numpy scipy ase \
    torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match
}

py() {                    # symmetrix (Python, CPU-only) for symmetrix_extract_mace
  export PATH=$VENV/bin:$PATH
  SKBUILD_CMAKE_ARGS="-DKokkos_ENABLE_CUDA=OFF;-DSYMMETRIX_SPHERICART_CUDA=OFF;-DKokkos_ARCH_NATIVE=OFF" \
    CMAKE_BUILD_PARALLEL_LEVEL=$JOBS ~/.local/bin/uv pip install -q --python "$VENV/bin/python" \
    --no-build-isolation --no-deps --reinstall-package symmetrix "$SX/symmetrix"
  "$VENV/bin/python" -c "import symmetrix, importlib.metadata as m; print('symmetrix', m.version('symmetrix'))"
}

models() {                # one model-symx3.json per MACE fit (Si), from its CPU copy where there is one
  local d m
  for d in "$MACE_DIR"/*/; do
    m=$d/model-cpu.model; [ -f "$m" ] || m=$d/model.model; [ -f "$m" ] || continue
    [ -s "$d/model-symx3.json" ] && [ "$d/model-symx3.json" -nt "$m" ] && continue
    "$VENV/bin/symmetrix_extract_mace" --model "$m" --atomic-numbers 14 --output "$d/model-symx3.json" \
      > "$d/symx3-extract.log" 2>&1 && echo "extracted $d" || echo "FAILED $d (see symx3-extract.log)"
  done
}

wrapper() {   # wrapper <name> <modules> <build>
  printf '%s\n' '#!/usr/bin/env bash' 'source /etc/profile.d/modules.sh 2>/dev/null || true' \
    "module purge; module load $2" "export LD_LIBRARY_PATH=$3:\${LD_LIBRARY_PATH:-}" \
    "exec $3/lmp \"\$@\"" > "$ROOT/$1"
  chmod +x "$ROOT/$1"
}

lammps_cpu() {            # plain symmetrix/mace (OpenMP per rank) + its Kokkos OpenMP twin; -march=native on EPYC
  local B=$LAMMPS/build-cpu
  export PATH=$VENV/bin:$PATH        # CMake >= 3.27 (the module is 3.26)
  rm -rf "$B"
  cmake -S "$LAMMPS/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON -D CMAKE_CXX_COMPILER=g++ -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_ZEN2=ON \
    -D SYMMETRIX_KOKKOS=ON -D SYMMETRIX_SPHERICART_CUDA=OFF
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-symx3-cpu.sh "$MODS" "$B"
}

lammps_cuda() {
  local B=$LAMMPS/build-kk
  export PATH=$VENV/bin:$PATH NVCC_WRAPPER_DEFAULT_COMPILER=g++
  set +u; module load $CUDA_MOD; set -u
  # The foundation styles are left out of the GPU build: they take only 96-256-channel presets (not the benchmark's
  # fits), and nvcc 12.8 rejects nequip_foundation_cuda.hpp ("expression must have a constant value", job 2280020).
  # install.sh linked them into src/ and src/KOKKOS/; LAMMPS globs those, so the links are removed before configuring
  # (the CPU tree, built first, keeps them).  KEEP_BUILD=1 reuses an existing build dir (incremental).
  rm -f "$LAMMPS"/src/pair_symmetrix_{mace,nequip}_foundation.{h,cpp} \
        "$LAMMPS"/src/KOKKOS/pair_symmetrix_{mace,nequip}_foundation_kokkos.{h,cpp}
  [ "${KEEP_BUILD:-0}" = 1 ] || rm -rf "$B"
  # Symmetrix's own GPU recipe (pair_symmetrix/README.md): g++ as the C++ compiler, not nvcc_wrapper.  Under
  # nvcc_wrapper every .cpp is CUDA, so sphericart 2's gpu-lite.hpp (plain C++ that loads CUDA at run time) meets the
  # real cuda_runtime.h: "type dim3 has already been defined" (job 2280017)
  cmake -S "$LAMMPS/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON -D CMAKE_CXX_COMPILER=g++ -D CMAKE_CXX_FLAGS="-march=native -ffast-math" \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ARCH_NATIVE=ON -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -D Kokkos_ARCH_AMPERE80=ON -D SYMMETRIX_KOKKOS=ON -D SYMMETRIX_SPHERICART_CUDA=ON
  cmake --build "$B" -j "$JOBS"
  wrapper lmp-symx3.sh "$MODS $CUDA_MOD" "$B"
}

for step in "$@"; do "$step"; done
