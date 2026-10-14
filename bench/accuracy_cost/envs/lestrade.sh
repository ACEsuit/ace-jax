#!/bin/bash
# Accuracy-cost timing environment on lestrade (RTX 4000 Ada, sm_89, 32 cores):
# bench/scaling's moriarty build with the Ada Kokkos arch and ML-QUIP (GAP's
# pair quip) added to the Symmetrix/ML-PACE tree, and quippy-ase in the venv.
# Build trees on /storage.  The CUDA/gompi modules are the same as moriarty's.
#   bash bench/accuracy_cost/envs/lestrade.sh [step ...]
set -euo pipefail
[ "$(hostname -s)" = lestrade ] || { echo "run on lestrade (Kokkos arch is per GPU)"; exit 1; }
export BENCH_ROOT=${BENCH_ROOT:-/storage/eng/essswb/bench-accuracy-cost-lestrade}
export ACEJAX_SRC=${ACEJAX_SRC:-$HOME/projects/ace-jax}
export KOKKOS_ARCH=ADA89 CUDA_ARCH=89
export EXTRA_PKGS="-D PKG_ML-QUIP=ON -D DOWNLOAD_QUIP=ON"
# LAMMPS's ML-QUIP.cmake sets F95FLAGS, but QUIP public (2026) reads F90FLAGS; without
# -ffree-line-length-none libAtoms/System.F90 fails.  make imports env vars, and += appends.
export F90FLAGS="-x f95-cpp-input -ffree-line-length-none -ffree-form -fno-second-underscore -fPIC"
export EXTRA_PIP=quippy-ase
export ENV_HOSTS="lestrade-gpu lestrade-cpu" ENV_JSON_DIR=$ACEJAX_SRC/bench/accuracy_cost/envs
export JOBS=${JOBS:-8}
exec bash "$ACEJAX_SRC/bench/scaling/envs/moriarty.sh" "$@"
