#!/usr/bin/env bash
# ace-jax's C++ FFI kernel on the CPU (ace_jax.eval.ffi): the lammps-jax pair style with the plugin fix that registers
# LAMMPS_JAX_FFI_HANDLERS for the build's platform (lammps-jax 0.2.0 registers them for CUDA only; $ROOT/lammps-jax-ffi-platform.patch),
# in its own tree against the same LAMMPS (sulis-lammps-cpu-build.sbatch's build-kk-cpu), and the kernel compiled on a compute
# node for its CPU (EPYC 7742) against the venv jax's FFI headers (jax 0.11.2, as the XLA CPU plugin, 91888df).
#   sbatch accuracy_cost/envs/sulis-cpu-ffi-build.sbatch      (from ~/bench-scaling/ace-jax-ffi/bench)
# Gives $ROOT/lammps-jax-ffi/build-plugin-cpu-pjrt, $ROOT/lmp-jax-ffi-cpu.sh and $ROOT/ace-jax-ffi/libace_jax_ffi.so.
#SBATCH --job-name=ffi-build
#SBATCH --account=su007-jrk
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
MODS="GCC/12.3.0 OpenMPI/4.1.5 CMake/3.26.3"
source /etc/profile.d/modules.sh; module purge; module load $MODS
B=$ROOT/lammps-dev/build-kk-cpu; LJ=$ROOT/lammps-jax-ffi; P=$LJ/build-plugin-cpu-pjrt
INC=$("$ROOT/venv/bin/python" -c 'import jaxlib, pathlib; print(pathlib.Path(jaxlib.__file__).parent / "include")')
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"
rm -rf $LJ && rsync -a --exclude 'build-*' $ROOT/lammps-jax-cpu/ $LJ/
patch -d $LJ -p1 < $ROOT/lammps-jax-ffi-platform.patch
cmake -S $LJ/cpp -B $P -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++ -D CMAKE_CXX_FLAGS="-fno-lto" \
  -D CMAKE_SHARED_LINKER_FLAGS="-fno-lto" -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -D LAMMPS_HEADER_DIR=$ROOT/lammps-dev/src -D JAXLIB_INCLUDE_DIR=$INC -D KOKKOS_CONFIG_INCLUDE_DIR=$B/lib/kokkos
cmake --build $P -j 16
cat > $ROOT/lmp-jax-ffi-cpu.sh <<EOS
#!/usr/bin/env bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge; module load $MODS
export LD_LIBRARY_PATH=$B:\${LD_LIBRARY_PATH:-}
export LAMMPS_PLUGIN_PATH=$P
exec $B/lmp "\$@"
EOS
chmod +x $ROOT/lmp-jax-ffi-cpu.sh
PYTHONPATH=$ROOT/ace-jax-ffi/src JAX_PLATFORMS=cpu "$ROOT/venv/bin/python" -c "
from ace_jax.eval import ffi; print(ffi.build('$ROOT/ace-jax-ffi/libace_jax_ffi.so', cxx='g++', flags=('-O3', '-march=znver2')))"
echo "done $(date -Is)"; ls -la $P/lammps_jaxplugin.so $ROOT/ace-jax-ffi/libace_jax_ffi.so
