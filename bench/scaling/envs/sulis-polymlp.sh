#!/usr/bin/env bash
# Polynomial MLPs (Seko) for the accuracy-cost benchmark on Sulis (bench/accuracy_cost/fit_polymlp.py):
#   cpu    LAMMPS with lammps-polymlp-package's POLYMLP (`pair_style polymlp`, float64, OpenMP) -> $ROOT/lmp-polymlp-cpu.sh
#   env    pypolymlp from conda-forge (the timing gate's reference evaluator)                    -> $ROOT/env-polymlp
# The LAMMPS is the lammps-dev tree (develop ec02ed0, as every other CPU row) copied to lammps-polymlp.  The package
# ships only the legacy make glue (src/POLYMLP/Install.sh, lib/polymlp/Makefile.lammps: -fopenmp); for CMake the
# package directory is registered in STANDARD_PACKAGES, BUILD_OMP=ON supplies OpenMP and the Boost module its headers
# (polymlp_basis_function.h uses boost::math).  -march=znver2 as the Kokkos ARCH_ZEN2 builds of the other codes.
# pair_style polymlp has NO MPI support (README: "MPI is not currently available"; its arrays are indexed by atom tag
# and sized by the local atom count): run one rank, OMP_NUM_THREADS threads.
#
#   rsync -a --exclude .git $ACEJAX_AC_ROOT/ext/lammps-polymlp-package sulis:bench-scaling/     (046eaa4, VERSION 0.3.4)
#   (from bench/scaling/envs) sbatch sulis-polymlp-build-cpu.sbatch;  bash sulis-polymlp.sh env  (login node: network)
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
SRC=$ROOT/lammps-dev
DST=$ROOT/lammps-polymlp
PKG=$ROOT/lammps-polymlp-package
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-16}}
MODS_CPU="GCC/12.3.0 OpenMPI/4.1.5 CMake/3.26.3 ${BOOST_MODULE:-Boost/1.82.0}"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; set -u

cpu() {
  set +u; module purge; module load $MODS_CPU; set -u
  rsync -a --delete --exclude 'build*' --exclude doc --exclude examples --exclude bench --exclude unittest \
    --exclude .git "$SRC/" "$DST/"
  mkdir -p "$DST/src/POLYMLP"
  cp "$PKG/src/POLYMLP/"*.cpp "$PKG/src/POLYMLP/"*.h "$DST/src/POLYMLP/"
  grep -q "POLYMLP" "$DST/cmake/CMakeLists.txt" || \
    sed -i '/set(STANDARD_PACKAGES/,/)/ s/)/  \n  POLYMLP)/' "$DST/cmake/CMakeLists.txt"
  local B=$DST/build-cpu-polymlp
  rm -rf "$B"
  cmake -S "$DST/cmake" -B "$B" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ \
    -D CMAKE_CXX_FLAGS="-march=znver2" -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON -D PKG_POLYMLP=ON
  cmake --build "$B" -j "$JOBS"
  printf '%s\n' '#!/usr/bin/env bash' 'source /etc/profile.d/modules.sh 2>/dev/null || true' \
    "module purge; module load $MODS_CPU" "export LD_LIBRARY_PATH=$B:\${LD_LIBRARY_PATH:-}" \
    "exec $B/lmp \"\$@\"" > "$ROOT/lmp-polymlp-cpu.sh"
  chmod +x "$ROOT/lmp-polymlp-cpu.sh"
  help=$("$ROOT/lmp-polymlp-cpu.sh" -h)                              # (not piped into grep -q: under pipefail lmp's SIGPIPE fails)
  grep -qw polymlp <<< "$help" && echo "pair_style polymlp: built" || { echo "pair_style polymlp missing from $B/lmp"; exit 1; }
  { echo "lammps-polymlp: lammps-dev ec02ed0 (2 Sep 2026) + lammps-polymlp-package 046eaa4 (VERSION 0.3.4), src/POLYMLP registered for CMake"
    echo "modules: $MODS_CPU; cmake: Release, -march=znver2, BUILD_MPI, BUILD_OMP (no KOKKOS)"
    echo "built $(date -Is) on $(hostname)"; } > "$ROOT/VERSIONS-polymlp"
}

env() {   # pypolymlp has no wheels and its sdist's build requirement `eigen` resolves to an unrelated PyPI package: conda-forge
  mkdir -p "$ROOT/bin"
  [ -x "$ROOT/bin/micromamba" ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$ROOT" bin/micromamba
  MAMBA_ROOT_PREFIX=$ROOT/micromamba "$ROOT/bin/micromamba" create -y -q -p "$ROOT/env-polymlp" -c conda-forge \
    --override-channels python=3.12 pypolymlp=0.22.1 ase pyyaml phonopy symfc   # pypolymlp.api.pypolymlp_calc imports symfc
  (cd /tmp && "$ROOT/env-polymlp/bin/python" -c "import pypolymlp; print('pypolymlp', pypolymlp.__version__)")
}

"$@"
