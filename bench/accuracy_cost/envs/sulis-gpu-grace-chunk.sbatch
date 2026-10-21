#!/usr/bin/env bash
# GRACE-1L chunk-size sweep on a Sulis A100: grace/1l/kk float64 at chunksize 4096 (the default, the published rows),
# 8192, 16384 and 32768 atoms per kernel batch, for c16l4 and c64l4 (seed 0), each parity-gated as in
# sulis-gpu-grace.sbatch.  Rows: results/sulis-gpu-grace-chunk<N>.jsonl (one file per setting; the published
# sulis-gpu-grace.jsonl is untouched).  run_lammps._pair reads $GRACE_CHUNKSIZE.
#   sbatch accuracy_cost/envs/sulis-gpu-grace-chunk.sbatch
#SBATCH --job-name=ac-gpu-gracechunk
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-grace/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0
export PYTHONPATH=$ROOT/ace-jax-grace/bench:$ROOT/ace-jax-ac/src ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16
[ -n "${ACEJAX_AC_GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$ACEJAX_AC_GPU
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for n in 4096 8192 16384 32768; do
  for c in grace/c16l4_s0 grace/c64l4_s0; do
    GRACE_CHUNKSIZE=$n "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c \
        --env accuracy_cost/envs/sulis-gpu-grace.json --results accuracy_cost/results/sulis-gpu-grace-chunk$n.jsonl \
        --work "$ROOT/ac/runs/accuracy_cost/timing/sulis-gpu-grace-chunk$n" || true
  done
done
echo "done $(date -Is)"
