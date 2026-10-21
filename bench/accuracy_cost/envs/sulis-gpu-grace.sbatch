#!/usr/bin/env bash
# GRACE in LAMMPS on a Sulis A100 (grace/1l/kk, grace/fs/kk; bench/scaling/envs/sulis-grace.sh gpu), float64,
# each model gated (64 atoms) against gracemaker's TPCalculator in the grace venv (env json `grace_python`).
# Rows: results/sulis-gpu-grace.jsonl.  Args: cases (family or family/size_s0; default every grace and grace_fs size, seed 0).
#   sbatch accuracy_cost/envs/sulis-gpu-grace.sbatch [grace/c16l4_s0 ...]
#SBATCH --job-name=ac-gpu-grace
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-grace/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0
export PYTHONPATH=$ROOT/ace-jax-grace/bench:$ROOT/ace-jax-ac/src ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=16
[ -n "${ACEJAX_AC_GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$ACEJAX_AC_GPU
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for c in ${@:-grace grace_fs}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c \
      --env accuracy_cost/envs/sulis-gpu-grace.json --results accuracy_cost/results/sulis-gpu-grace.jsonl || true
done
echo "done $(date -Is)"
