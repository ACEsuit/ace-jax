#!/usr/bin/env bash
# Fine-tune one GRACE foundation model (family grace_fm, fit_grace_fm.py) on a Sulis GPU (an L40 is enough; the
# fit's hardware is not part of the benchmark).  The models are in gracemaker's cache (grace_models download /
# checkpoint, on the login node).
#   sbatch accuracy_cost/envs/sulis-grace-fm.sbatch 2l-omat-large-ft        (from ~/bench-scaling/ace-jax-fdn/bench)
#SBATCH --job-name=ac-grace-fm
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-fdn/bench"
export PYTHONPATH=$ROOT/ace-jax-fdn/bench ACEJAX_AC_ROOT=$ROOT/ac TF_CPP_MIN_LOG_LEVEL=1
export ACEJAX_AC_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "job $SLURM_JOB_ID grace_fm $1 on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv
"$ROOT/venv-grace/bin/python" -m accuracy_cost.fit_grace_fm fit "$1"
echo "done $(date -Is)"
