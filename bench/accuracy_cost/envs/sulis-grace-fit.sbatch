#!/usr/bin/env bash
# GRACE fits on one Sulis A100 (fit_grace.py in the grace venv, $ROOT/venv-grace: bench/scaling/envs/sulis-grace.sh
# venv), one after another: FAMILY SIZE [SEED ...] (default seeds 0 1 2), runs under $ACEJAX_AC_ROOT/runs/accuracy_cost.
# Wall time per fit: the A100 probe (sulis-grace-probe.sbatch); pass --time to sbatch for the large sizes.
#   sbatch accuracy_cost/envs/sulis-grace-fit.sbatch grace c16l4 0 1 2
#SBATCH --job-name=ac-grace
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/${GRACE_TREE:-ace-jax-grace}/bench"
FAM=$1 SIZE=$2; shift 2
export PYTHONPATH=$ROOT/${GRACE_TREE:-ace-jax-grace}/bench ACEJAX_AC_ROOT=$ROOT/ac TF_CPP_MIN_LOG_LEVEL=1
export ACEJAX_AC_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "job $SLURM_JOB_ID grace $FAM $SIZE seeds ${*:-0 1 2} on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv
"$ROOT/venv-grace/bin/python" -m accuracy_cost.fit_grace "$FAM" "$SIZE" --seed ${@:-0 1 2}
echo "done $(date -Is)"
