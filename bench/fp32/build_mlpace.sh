#!/bin/bash
# Plan step 1.6: KOKKOS ML-PACE on moriarty's RTX A4500 (Ampere 86) at KOKKOS_PREC=double|mixed|single,
# from the lestrade lammps-dev source (develop, Oct 2026), which is only read: builds go under $O.
set -e
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load gompi/2023a CUDA/12.4.0
SRC=/storage/eng/essswb/bench-accuracy-cost-lestrade/lammps-dev
PACE_TGZ=$SRC/build-kk/libpace.tar.gz
O=${OUT:-/storage/eng/essswb/projects/ace-jax/runs/fp32_step1/mlpace}
mkdir -p $O
for prec in ${PRECS:-double mixed single}; do
  B=$O/build-$prec
  mkdir -p $B
  cmake -S $SRC/cmake -B $B -D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=on \
    -D CMAKE_CXX_COMPILER=$SRC/lib/kokkos/bin/nvcc_wrapper \
    -D PKG_KOKKOS=on -D Kokkos_ENABLE_CUDA=on -D Kokkos_ENABLE_OPENMP=on -D Kokkos_ARCH_AMPERE86=on \
    -D PKG_ML-PACE=on -D KOKKOS_PREC=$prec \
    -D PACELIB_URL=file://$PACE_TGZ -D PACELIB_SHA256=$(sha256sum $PACE_TGZ | cut -d' ' -f1) > $B/cmake.log
  cmake --build $B -j ${JOBS:-6} > $B/build.log 2>&1
  echo "$prec: $(ls -la $B/lmp)"
done
