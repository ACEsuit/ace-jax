#!/usr/bin/env bash
# The device-scaled block target on a Sulis A100: what device_memory the export sees, and the
# auto (0) block each model gets, with its step time (compare sulis_blocks.sbatch's sweep).
#SBATCH --job-name=fp32-blocks-auto
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -uo pipefail
ROOT=$HOME/bench-scaling
HERE=$ROOT/ace-jax-fp32
cd $HERE
export PYTHONPATH=$HERE/src:$HERE/bench
export BUNDLE_PYTHON=$ROOT/venv/bin/python
export LMP_JAX=$ROOT/lmp-jax.sh
export PJRT_PLUGIN=$ROOT/venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export BUNDLE_TMP=$HERE/tmp BUNDLE_STEPS=50 XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p $BUNDLE_TMP
nvidia-smi --query-gpu=name,memory.total --format=csv
$BUNDLE_PYTHON -c "
import jax; jax.config.update('jax_enable_x64', True)
from ace_jax.calc.point import device_memory_bytes
from ace_jax.export.lammps import block_target_elems
m = device_memory_bytes(); print('device_memory', m, 'target', block_target_elems(m))"
M=$HERE/models; C=$ROOT/ace-jax/bench/scaling/models
run() { echo "# $*"; $BUNDLE_PYTHON bench/fp32/diag_blocks.py "$@" 2>&1 >/dev/null | grep '^{'; }
for dt in float32 float64; do
  run $M/o4d16.npz 32768 $dt 0
  run $M/o3d8.npz 32768 $dt 0
  run $M/f1352.yace 32768 $dt 0
done
export DIAG_SYSTEM=Cantor
run $C/ace_Cantor_medium.npz 32768 float64 0
run $C/ace_Cantor_medium.npz 32768 float32 0
run $C/pace_Cantor_medium.yace 32768 float64 0
run $C/ace_Cantor_large.npz 32768 float32 0
