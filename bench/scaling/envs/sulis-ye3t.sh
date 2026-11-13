#!/usr/bin/env bash
# YE3T (Goff and Thompson, arXiv:2609.31895) for the accuracy-cost benchmark on Sulis:
#   cpu    LAMMPS with ML-YE3T (`pair_style ye3t`, float64, MPI) and stock ML-PACE in one binary -> $ROOT/lmp-ye3t-cpu.sh
#   venv   ye3t + ye3t-methods (the fitted models' reference evaluator for the timing gate)     -> $ROOT/venv-ye3t
# The LAMMPS is the lammps-dev tree (develop ec02ed0, "2 Sep 2026", as every other CPU row) copied to lammps-ye3t and
# patched by ye3t-lammps's own installer (tools/patch_lammps.sh; upstream tests stable_22Jul2025_update4, and its
# --check passes on this tree).  Configured as sulis-pacetab.sh's CPU build (Kokkos OpenMP, Zen 2: the same
# compiler flags for pace and ye3t), with stock lammps-user-pace v.2025.12.4.p1.  ML-YE3T needs yaml-cpp with a
# CMake config (the copy lammps-user-pace bundles has its install rules removed): yaml-cpp 0.8.0, fetched by `fetch`
# on the login node and built here; lammps-user-pace's own find_package then takes the same library.
#
#   rsync -a --exclude .git $ACEJAX_AC_ROOT/ext/{ye3t,ye3t-lammps,ye3t-methods} sulis:bench-scaling/
#   (ye3t f8ff96d, ye3t-lammps 06100f9, ye3t-methods 6b85c54)
#   (from bench/scaling/envs) bash sulis-ye3t.sh fetch; sbatch sulis-ye3t-build-cpu.sbatch
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SRC=$ROOT/lammps-dev
DST=$ROOT/lammps-ye3t
PACE=$ROOT/lammps-user-pace-v.2025.12.4.p1
YAML=$ROOT/yaml-cpp-ye3t
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-16}}
MODS_CPU="GCC/12.3.0 OpenMPI/4.1.5 CMake/3.26.3"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; set -u

fetch() {   # login node (network): the yaml-cpp release
  [ -d "$ROOT/yaml-cpp-0.8.0" ] || curl -Ls https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz | tar -xz -C "$ROOT"
  ls "$ROOT/yaml-cpp-0.8.0/CMakeLists.txt"
}

cpu() {
  set +u; module purge; module load $MODS_CPU; set -u
  rsync -a --delete --exclude '/build*' --exclude /doc --exclude /bench --exclude /unittest --exclude /.git \
    --exclude /.ye3t-uninstall-backups "$SRC/" "$DST/"            # anchored: lib/kokkos/cmake/build_env_info.cmake stays
  rm -rf "$DST/doc" "$DST/.ye3t-uninstall-backups"                     # a re-run: the rsync restored CMakeLists.txt, so
  mkdir -p "$DST/doc/src" "$DST/examples/PACKAGES"                     # the installer must find no earlier doc page
  sh "$ROOT/ye3t-lammps/tools/patch_lammps.sh" --apply --lammps-source "$DST"
  if [ ! -f "$YAML/lib64/cmake/yaml-cpp/yaml-cpp-config.cmake" ] && [ ! -f "$YAML/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]; then
    cmake -S "$ROOT/yaml-cpp-0.8.0" -B "$YAML-build" -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX="$YAML" \
      -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D YAML_CPP_INSTALL=ON -D YAML_CPP_BUILD_TESTS=OFF -D YAML_CPP_BUILD_TOOLS=OFF -D YAML_BUILD_SHARED_LIBS=OFF
    cmake --build "$YAML-build" -j "$JOBS" --target install
  fi
  local B=$DST/build-kk-cpu-ye3t
  rm -rf "$B"
  cmake -S "$DST/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ \
    -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ARCH_ZEN2=ON \
    -D PKG_ML-PACE=ON -D LOCAL_ML-PACE="$PACE" \
    -D PKG_ML-YE3T=ON -D ML_YE3T_RUNTIME_SOURCE="$ROOT/ye3t" -D CMAKE_PREFIX_PATH="$YAML"
  cmake --build "$B" -j "$JOBS"
  printf '%s\n' '#!/usr/bin/env bash' 'source /etc/profile.d/modules.sh 2>/dev/null || true' \
    "module purge; module load $MODS_CPU" "export LD_LIBRARY_PATH=$B:\${LD_LIBRARY_PATH:-}" \
    "exec $B/lmp \"\$@\"" > "$ROOT/lmp-ye3t-cpu.sh"
  chmod +x "$ROOT/lmp-ye3t-cpu.sh"
  help=$("$ROOT/lmp-ye3t-cpu.sh" -h)                              # (not piped into grep -q: under pipefail lmp's SIGPIPE fails)
  grep -qw ye3t <<< "$help" && echo "pair_style ye3t: built" || { echo "pair_style ye3t missing from $B/lmp"; exit 1; }
  { echo "lammps-ye3t: lammps-dev ec02ed0 (2 Sep 2026) + ye3t-lammps 06100f9 (tools/patch_lammps.sh) + ye3t f8ff96d runtime"
    echo "ML-PACE: lammps-user-pace v.2025.12.4.p1 (stock); yaml-cpp 0.8.0 (static, for both packages)"
    echo "modules: $MODS_CPU; cmake: Release, Kokkos OpenMP+Serial ARCH_ZEN2, BUILD_MPI, BUILD_OMP"
    echo "built $(date -Is) on $(hostname)"; } > "$ROOT/VERSIONS-ye3t"
}

venv() {   # the gate's reference for fitted models: ye3t-methods' ASE calculator (C++20 build: GCC 12.3)
  set +u; module purge; module load $MODS_CPU; set -u
  local V=$ROOT/venv-ye3t
  [ -d "$V" ] || uv venv -q --python 3.12 "$V"
  uv pip install -q --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cpu torch
  uv pip install -q --python "$V/bin/python" 'setuptools>=77,<82' wheel cmake numpy ase sympy pyyaml psutil pip
  (cd "$ROOT" && "$V/bin/python" -m pip install --no-build-isolation ./ye3t && "$V/bin/python" -m pip install --no-build-isolation ./ye3t-methods)
  "$V/bin/python" -c "import ye3t, ye3t_methods; print('ye3t venv ok')"
}

"$@"
