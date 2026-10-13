#!/usr/bin/env bash
# lammps-jax on the CPU (Sulis): a host-Kokkos LAMMPS (OpenMP, MPI, PLUGIN) from the lammps-dev tree, and the
# lammps-jax pair style (lammps-jax-cpu: upstream main + the cpu-fixes branch) built against it.
#   sbatch bench/scaling/envs/sulis-lammps-cpu-build.sbatch      (from ~/bench-scaling/ace-jax-cpu)
# Gives $ROOT/lammps-dev/build-kk-cpu/lmp, $ROOT/lammps-jax-cpu/build-plugin-cpu-pjrt and $ROOT/lmp-jax-cpu.sh.
#SBATCH --job-name=lj-cpu-build
#SBATCH --account=su007-jrk
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:58:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
MODS="GCC/12.3.0 OpenMPI/4.1.5 CMake/3.26.3"
source /etc/profile.d/modules.sh; module purge; module load $MODS
SRC=$ROOT/lammps-dev; B=$SRC/build-kk-cpu; LJ=$ROOT/lammps-jax-cpu; P=$LJ/build-plugin-cpu-pjrt
INC=$("$ROOT/venv/bin/python" -c 'import jaxlib, pathlib; print(pathlib.Path(jaxlib.__file__).parent / "include")')
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is); jaxlib include $INC"
cmake -S $SRC/cmake -B $B -C $SRC/cmake/presets/kokkos-openmp.cmake -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=g++ -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON -D PKG_PLUGIN=ON \
  -D PKG_ML-PACE=ON -D Kokkos_ENABLE_IMPL_VIEW_LEGACY=ON -D Kokkos_ARCH_ZEN2=ON
cmake --build $B -j 64
rm -rf $P       # a changed compiler or Kokkos config is silently ignored by a cached build dir
cmake -S $LJ/cpp -B $P -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ -D CMAKE_CXX_FLAGS="-fno-lto" \
  -D CMAKE_SHARED_LINKER_FLAGS="-fno-lto" -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -D LAMMPS_HEADER_DIR=$SRC/src -D JAXLIB_INCLUDE_DIR=$INC -D KOKKOS_CONFIG_INCLUDE_DIR=$B/lib/kokkos
cmake --build $P -j 16
cat > $ROOT/lmp-jax-cpu.sh <<EOF
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load $MODS
export LD_LIBRARY_PATH=$B:\${LD_LIBRARY_PATH:-}
export LAMMPS_PLUGIN_PATH=$P
exec $B/lmp "\$@"
EOF
chmod +x $ROOT/lmp-jax-cpu.sh
git -C $SRC log -1 --format="lammps-dev %h %cd (build-kk-cpu)" > $ROOT/VERSIONS-cpu
echo "lammps-jax-cpu upstream 880d409 + cpu-fixes (client options, export platform)" >> $ROOT/VERSIONS-cpu
cat $ROOT/pjrt-cpu-xla-91888df/VERSION >> $ROOT/VERSIONS-cpu
echo "done $(date -Is)"; ls -la $B/lmp $P/lammps_jaxplugin.so
