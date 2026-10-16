#!/usr/bin/env bash
# Step-2 follow-up: the bundle block-size rule (export.lammps.BUNDLE_BLOCK_TARGET_ELEMS) rechecked on a
# Sulis A100: LAMMPS step time against block_rows (diag_blocks.py), auto (0) = the new rule.
#   sbatch bench/fp32/sulis_blocks.sbatch      (from ~/bench-scaling/ace-jax-fp32)
#SBATCH --job-name=fp32-blocks
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
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
M=$HERE/models; C=$ROOT/ace-jax/bench/scaling/models
run() { echo "# $*"; $BUNDLE_PYTHON bench/fp32/diag_blocks.py "$@" 2>&1 >/dev/null | grep '^{'; }
for dt in float32 float64; do
  run $M/o4d16.npz 32768 $dt 1024 2048 4096 8192 16384 0
  run $M/o3d8.npz 32768 $dt 1024 4096 32768 0
  run $M/f1352.yace 32768 $dt 1024 2048 4096 32768 0
done
run $M/o4d16.npz 262144 float32 1024 16384 0
export DIAG_SYSTEM=Cantor
run $C/ace_Cantor_medium.npz 32768 float64 1024 2048 4096 8192 16384 0
run $C/ace_Cantor_medium.npz 32768 float32 1024 2048 4096 8192 32768 0
run $C/pace_Cantor_medium.yace 32768 float64 1024 2048 4096 8192 32768 0
run $C/ace_Cantor_large.npz 32768 float32 1024 2048 4096 8192 32768 0
