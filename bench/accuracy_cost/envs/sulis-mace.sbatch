#!/usr/bin/env bash
# One MACE training run on a Sulis A100 (bench/scaling's venv: mace-torch 0.3.16, as the Modal fits), then its
# Symmetrix export: python -m accuracy_cost.fit_mace SIZE, the data and runs under $ACEJAX_AC_ROOT.
#   sbatch envs/sulis-mace.sbatch c16L0n1
#SBATCH --job-name=ac-mace
#SBATCH --account=su007-jrk-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
ROOT=${BENCH_ROOT:-$HOME/bench-scaling}
cd "$ROOT/ace-jax-ac/bench"
set +u; source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; set -u
export LD_LIBRARY_PATH=$ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/ace-jax-ac/bench:$ROOT/ace-jax-ac/src ACEJAX_AC_ROOT=$ROOT/ac
echo "job $SLURM_JOB_ID mace $1 on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv
"$ROOT/venv/bin/python" -m accuracy_cost.fit_mace "$1"
