#!/usr/bin/env bash
# GRACE foundation models (family grace_fm) on an A100 through grace-runtime's pair_style grace/runtime/kk (as
# sulis-gpu-gracert.sbatch), gated against the model's SavedModel in gracemaker's TPCalculator.
# Rows: results/sulis-gpu-grace-fm.jsonl.  A shared node: --export=ALL,ACEJAX_AC_MAX_LOAD1=9999.
#   sbatch accuracy_cost/envs/sulis-gpu-grace-fm.sbatch [grace_fm/2l-omat-large ...]   (from ~/bench-scaling/ace-jax-fdn/bench)
#SBATCH --job-name=ac-gpu-grace-fm
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
cd "$ROOT/ace-jax-fdn/bench"
source /etc/profile.d/modules.sh; module purge; module load GCC/12.3.0
export PYTHONPATH=$ROOT/ace-jax-fdn/bench:$ROOT/ace-jax-fdn/src ACEJAX_AC_ROOT=$ROOT/ac ACEJAX_AC_MAX_LOAD1=${ACEJAX_AC_MAX_LOAD1:-16}
RESULTS=${RESULTS:-accuracy_cost/results/sulis-gpu-grace-fm.jsonl}
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for c in ${@:-grace_fm}; do
  "$ROOT/venv/bin/python" -m accuracy_cost.timing gpu --only $c --codes grace-rt \
      --env accuracy_cost/envs/sulis-gpu-grace-fm.json --results "$RESULTS" || true
done
echo "done $(date -Is)"
